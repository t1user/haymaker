from __future__ import annotations

import asyncio
import itertools
import logging
import operator as op
from collections import deque
from datetime import date, datetime
from functools import cached_property
from typing import Literal

import eventkit as ev  # type: ignore
import ib_insync as ibi

from .base import Atom

log = logging.getLogger(__name__)

_counter = itertools.count().__next__


class BarAggregator(Atom):
    """
    Aggregate recieved data bars into new bars based on the criteria
    specified in :attr:`filter'.  Store processed data.

    When future contract changes, data already in the filter will be
    adjusted.  However, when system is started afresh right after
    contract changed, all back data will be for the new contract,
    which in some cases might be incorrect.

    :class:`BarAggregator` works best for strategies that don't
    require large amounts of back data.

    Args:
    -----

    filter: one of :class:`CountBars`, :class:`VolumeBars`,
    :class:`TimeBars`, :class:`NoFilter`, determines how input bars
    are grouped into output bars

    future_adjust_type: one of: "add" or "mul" on future contract
    change, how price bars currently in the filter are to be adjusted;
    resulting adjusted series will created by splicing two price
    series using either addition or multiplication.
    """

    def __init__(
        self,
        filter: CountBars | VolumeBars | TimeBars | NoFilter,
        future_adjust_type: Literal["add", "mul", None] = "add",
    ):
        Atom.__init__(self)
        self.filter = filter
        self.filter += self.onDataBar
        self.future_adjust_type = future_adjust_type
        # if future needs to be adjusted; set by onContractChanged
        self._future_adjust_flag = False
        # reference point for last bar processed
        self._last_data_point: datetime | date | None = None
        # data queued during long backfills
        self._queue: asyncio.Queue = asyncio.Queue()
        # task that processes queue where all data bars are put
        self._worker_task: asyncio.Task | None = None
        # used to determine if backfill in progress
        self._backfill_event: asyncio.Event = asyncio.Event()
        # start with cleared state (will not block)
        self._backfill_event.set()

    def onStart(self, data: dict, *args) -> None:
        """Syncing contract with streamer."""
        super().onStart(data, *args)
        if (streamer := data.get("streamer")) and self.contract is None:
            self._contract_blueprint = streamer._contract_blueprint
            self.which_contract = streamer.which_contract

    def onDataBar(self, bars, *args) -> None:
        """
        This is connected to `self.filter`, will emit whatever comes
        from filter.  Additional filtering/adjustment/conversion logic
        can be put here.

        Backfill means this is stale data, it should not be emitted,
        because we don't want this data to be treated as if it was
        current.  It's up to other system components to determine how
        we want to generate signals.  However :class:`BarAggregator`
        emits only when latest data becomes available and passes
        past data as such.
        """
        if self._backfill_event.is_set():
            self.dataEvent.emit(bars)

    async def onData(self, data: ibi.BarDataList, *args) -> None:
        """
        The purpose of the queue is to avoid a race condition when
        `onData` receives several datapoints during a backfill and
        then has to process this data in the correct order when
        backfill is finished.
        """
        if self._future_adjust_flag:
            self.adjust_future(data)

        await self._queue.put(data)
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(
                self._process_queue(), name=f"{self!s}_queue_worker"
            )

    async def _process_queue(self) -> None:
        while True:
            data = await self._queue.get()
            await self._process(data)
            self._queue.task_done()

    async def _process(self, data_: ibi.BarDataList, *args) -> None:
        """
        Pass correct data to `self.filter`.

        If there is any more new data than the last bar, then it needs
        to be backfilled (the reason may be: 1.  fresh start 2.
        restart).  If after restart contract changed, data we already
        have in the filter needs to be adjusted.  If we have just one
        new data bar (as determined comparing timestamps on the
        available data), it should be just passed to the filter.
        """

        # Guard against empty data
        if len(data_) == 0:
            log.warning(f"{self!s} received empty BarDataList, skipping")
            return

        data, last_bar = data_[:-1], data_[-1]

        # wait for any ongoing backfill to complete
        await self._backfill_event.wait()

        # this is a fresh start
        if len(self.filter.bars) == 0:
            await self.backfill(data)

        # this is backfill after restart
        elif self._last_data_point and (data[-1].date > self._last_data_point):
            # we already have some data in the filter, we expect to add
            # only last few bars so its faster to iterate backwards to
            # fine the new data
            accumulator: deque[ibi.BarData] = deque()
            for bar in reversed(data):
                if bar.date > self._last_data_point:
                    accumulator.appendleft(bar)
                else:
                    break

            if len(accumulator) > 0:
                await self.backfill(accumulator)

        if self._last_data_point is None or last_bar.date > self._last_data_point:
            # always reflects last bar processed
            self._last_data_point = last_bar.date
            # regular emit of a new datapoint
            self.filter.on_source(last_bar)

    async def backfill(self, bars: deque[ibi.BarData] | list[ibi.BarData]) -> None:
        try:
            log.debug(
                f"BACKFILL: {self!s} ({len(bars)} bars), "
                f"from: {self._last_data_point} "
                f"to: {bars[-1].date if bars else 'empty'}"
            )
        except Exception:
            pass  # don't let logging kill backfill
        try:
            self._backfill_event.clear()
            stream = ev.Sequence(bars).connect(self.filter)
            await stream.list()
            self._backfill_event.set()
        except Exception:
            # in case of error _backfill_event will stay cleared
            # disabling processing of faulty signals
            log.exception(f"{self!s} error in backfill.")
            raise

    @cached_property
    def operator(self):
        return {"add": op.add, "mul": op.mul, None: None}.get(self.future_adjust_type)

    @cached_property
    def reverse_operator(self):
        return {"add": op.sub, "mul": op.truediv, None: None}.get(
            self.future_adjust_type
        )

    def adjust_future(self, bars: ibi.BarDataList):
        """
        Create continuous future price series on future contract
        change.

        Args:
        -----

        bars: new price bars not currently included in :meth:`self.filter`
        for current (post-roll) contract that the old series needs to
        be adjusted to
        """

        log.warning(f"{self} adjusting future.")

        old_bar = self.filter.bars[-1]
        for bar_ in reversed(bars):
            if bar_.date == old_bar.date:
                new_bar = bar_
                break
        assert new_bar, f"{self} failed future adjustment: non-overlapping series."

        value = self.reverse_operator(new_bar.close, old_bar.close)
        log.warning(
            f"FUTURES ADJUSTMENT | old bar: {old_bar} | new bar: {new_bar} "
            f"| adjustment basis: {value} | operator: {self.operator}"
        )

        for bar in self.filter.bars:
            for field in ("open", "high", "low", "close", "average"):
                setattr(bar, field, self.operator(getattr(bar, field), value))

        self._future_adjust_flag = False

    def onContractChanged(
        self, old_contract: ibi.Contract, new_contract: ibi.Contract
    ) -> None:
        self._future_adjust_flag = True
        log.debug(f"Contract on {self} reset from {old_contract} to {new_contract}")

    @cached_property
    def _id(self) -> int:
        return _counter()

    def __str__(self):
        contract_symbol = self.contract.localSymbol if self.contract else "NoContract"
        return (
            f"{self.__class__.__name__}<{contract_symbol}><{self._id}><{self.filter}>"
        )


class CountBars(ev.Op):
    """
    Group input bars into new bars corresponding to a fixed number
    of source bars.

    Args:
    -----

    count: number of source bars that constitue one output bar
    """

    __slots__ = ("_count", "bars", "label")

    bars: ibi.BarDataList

    def __init__(self, count: int, source: ev.Event | None = None, *, label: str = ""):
        ev.Op.__init__(self, source)
        self._count = count
        self.bars = ibi.BarDataList()
        self.label = label

    def on_source(self, new_bar: ibi.BarData, *args) -> None:
        if not self.bars or self.bars[-1].barCount == self._count:
            bar = new_bar
            bar.average = new_bar.average * new_bar.volume
            bar.barCount = 1
            self.bars.append(bar)
        else:
            bar = self.bars[-1]
            bar.high = max(bar.high, new_bar.high)
            bar.low = min(bar.low, new_bar.low)
            bar.close = new_bar.close
            bar.volume += new_bar.volume
            bar.average = new_bar.average * new_bar.volume
            bar.barCount += 1
        if bar.barCount == self._count:
            bar.average = bar.average / bar.volume
            self.bars.updateEvent.emit(self.bars, True)
            self.emit(self.bars)

    def __str__(self) -> str:
        return f"CountBars({self.label})"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(count={self._count}, "
            f"source={self._source}, label={self.label})"
        )


class VolumeBars(ev.Op):
    """
    Group input bars into new bars so that each output corresponds to
    the same volume.

    Args:
    -----

    volume: desired volume of each output bar
    """

    __slots__ = ("_volume", "bars", "label")

    bars: ibi.BarDataList

    def __init__(
        self, volume: int, source: ev.Event | None = None, *, label: str = ""
    ) -> None:
        ev.Op.__init__(self, source)
        self._volume = volume
        self.bars = ibi.BarDataList()
        self.label = label

    def on_source(self, new_bar: ibi.BarData, *args) -> None:
        # filter out faulty bars emitted as first daily bars
        if new_bar.volume < 0 or new_bar.barCount < 0:
            return
        if not self.bars or self.bars[-1].volume >= self._volume:
            bar = new_bar
            bar.average = bar.average * bar.volume
            self.bars.append(bar)
        else:
            bar = self.bars[-1]
            bar.high = max(bar.high, new_bar.high)
            bar.low = min(bar.low, new_bar.low)
            bar.close = new_bar.close
            bar.volume += new_bar.volume
            bar.average = new_bar.average * new_bar.volume
            bar.barCount += new_bar.barCount
        log.log(
            5,
            f"{new_bar.date} --> accumulated volume: "
            f"{bar.volume:6.0f} /{self._volume:6.0f} += {new_bar.volume:5.0f},"
            f" --> {self.label}",
        )
        if bar.volume >= self._volume:
            bar.average = bar.average / bar.volume
            log.log(5, f"New candle for {self.label} -> {self.bars[-1]}")
            self.bars.updateEvent.emit(self.bars, True)
            self.emit(self.bars)

    def __str__(self) -> str:
        return f"VolumeBars({self.label})"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(volume={self._volume}, "
            f"source={self._source}, label={self.label})"
        )


class TickBars(ev.Op):
    """
    Group input bars into new bars so that each output bar corresponds
    to the same number of ticks.

    Args:
    ----

    count: desired number of ticks for every output bar
    """

    __slots__ = ("_count", "bars", "label")

    bars: ibi.BarDataList

    def __init__(
        self, count: int, source: ev.Event | None = None, *, label: str = ""
    ) -> None:
        ev.Op.__init__(self, source)
        self._count = count
        self.bars = ibi.BarDataList()
        self.label = label

    def on_source(self, new_bar: ibi.BarData, *args) -> None:
        if not self.bars or self.bars[-1].barCount == self._count:
            bar = new_bar
            new_bar.average = new_bar.average * new_bar.volume
            self.bars.append(bar)
        else:
            bar = self.bars[-1]
            bar.high = max(bar.high, new_bar.high)
            bar.low = min(bar.low, new_bar.low)
            bar.close = new_bar.close
            bar.volume += new_bar.volume
            bar.average = new_bar.average * new_bar.volume
            bar.barCount += new_bar.barCount
        if bar.barCount == self._count:
            bar.average = bar.average / bar.volume
            self.bars.updateEvent.emit(self.bars, True)
            self.emit(self.bars)

    def __str__(self) -> str:
        return f"TickBars({self.label})"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(count={self._count}, "
            f"source={self._source}, label={self.label})"
        )


class TimeBars(ev.Op):
    """
    Group input bars into new bars so that every output bar
    corresponds to the same time period.

    Args:
    -----

    timer: :class:`eventkit.create.Timer` corresponding to desired
    duration of output bars
    """

    __slots__ = ("_timer", "bars", "_running_price_volume", "label")

    bars: ibi.BarDataList

    def __init__(
        self, timer: ev.Timer, source: ev.Event | None = None, *, label: str = ""
    ) -> None:
        ev.Op.__init__(self, source)
        self._timer = timer
        self._timer.connect(self._on_timer, None, self._on_timer_done)
        self.bars = ibi.BarDataList()
        self._running_price_volume = 0.0
        self.label = label

    def on_source(self, new_bar: ibi.BarData, *args) -> None:
        if not self.bars:
            return
        bar = self.bars[-1]
        if bar.open == 0:
            bar = new_bar
        self._running_price_volume += new_bar.average * new_bar.volume
        bar.high = max(bar.high, new_bar.high)
        bar.low = min(bar.low, new_bar.low)
        bar.close = new_bar.close
        bar.volume += new_bar.volume
        bar.average = self._running_price_volume / bar.volume
        bar.barCount += new_bar.barCount

        self.bars.updateEvent.emit(self.bars, False)

    def _on_timer(self, time) -> None:
        if self.bars:
            bar = self.bars[-1]
            if bar.close == 0 and len(self.bars) > 1:
                bar.open = bar.high = bar.low = bar.close = self.bars[-2].close
            self.bars.updateEvent.emit(self.bars, True)
            self.emit(bar)
        self.bars.append(ibi.BarData(time))

    def _on_timer_done(self, timer) -> None:
        self._timer = None
        self.set_done()

    def __str__(self) -> str:
        return f"TimeBars({self.label})"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(timer={self._timer}, "
            f"source={self._source}, label={self.label})"
        )


class NoFilter(ev.Op):
    """
    Accumulate input bars to ensure no bars are lost during restarts.
    Every input bar and output bar is the same.
    """

    __slots__ = ("bars", "label")

    bars: ibi.BarDataList

    def __init__(self, source: ev.Event | None = None, *, label: str = "") -> None:
        ev.Op.__init__(self, source)
        self.bars = ibi.BarDataList()
        self.label = label

    def on_source(self, new_bar: ibi.BarData, *args) -> None:
        self.bars.append(new_bar)
        log.log(5, f"New candle for {self.label} -> {self.bars[-1]}")
        self.bars.updateEvent.emit(self.bars, True)
        self.emit(self.bars)

    def __str__(self) -> str:
        return f"NoFilter({self.label})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(source={self._source}, label={self.label})"
