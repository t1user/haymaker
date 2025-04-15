from __future__ import annotations

import logging
import operator as op
from functools import cached_property
from typing import Literal

import eventkit as ev  # type: ignore
import ib_insync as ibi

from .base import Atom

log = logging.getLogger(__name__)


class BarAggregator(Atom):
    """
    Aggregate recieved data bars into new bars based on the
    criteria specified in :attr:`filter'. Store processed data.
    """

    def __init__(
        self,
        filter: "CountBars | VolumeBars | TimeBars | NoFilter",
        incremental_only: bool = False,
        future_adjust_type: Literal["add", "mul", None] = "add",
        debug: bool = False,
    ):
        Atom.__init__(self)
        self._incremental_only = incremental_only
        self._filter = filter
        self._filter += self.onDataBar
        self._log_level = log.level
        self._debug = debug
        self._future_adjust_type = future_adjust_type
        # if future needs to be adjusted; set by onStart
        self._future_adjust_flag = False

    def onStart(self, data, *args):
        if isinstance(data, dict):
            startup = data.get("startup")
            # prevent logging messages during startup phase
            if startup and not self._debug:
                log.debug(f"Startup: {self}")
                log.setLevel(logging.ERROR)
            else:
                log.setLevel(self._log_level)
            self._future_adjust_flag = data.get("future_adjust_flag", False)
        super().onStart(data, *args)

        if self._future_adjust_flag:
            log.warning(f"{self} onStart knows future needs adjust")

    def onDataBar(self, bars, *args):
        if self._incremental_only:
            try:
                self.dataEvent.emit(bars[-1])
            except KeyError:
                log.debug(f"Empty input from filter {self._filter}")
        else:
            self.dataEvent.emit(bars)

    def onData(self, data: ibi.BarData | ibi.BarDataList, *args) -> None:
        # data is just single BarData object (if incremental_only=True,
        # which is default and currently only mode supported)
        if self._future_adjust_flag:
            log.warning(f"{self} will adjust futures")
            log.debug(f"{data=}")
            # first data point is just to determine adjustment basis
            # should be passed to adjust_future
            try:
                assert isinstance(data, ibi.BarData)
                self.adjust_future(data)
            except Exception as e:
                log.exception(e)
            # adjust mode should be switched off for subsequent emits
            self._future_adjust_flag = False
            # and it shouldn't be emitted further down the chain
            return
        if isinstance(data, ibi.BarDataList):
            # Streamers with incremental_only=False have not been properly tested!
            log.critical(
                "WE SHOULD NOT BE HERE; DONT USE PROCESSOR WITH `incremental_only=True`"
            )
            try:
                data = data[-1]
            except KeyError:
                log.debug("Empty input from streamer.")
        assert isinstance(data, ibi.BarData)
        self._filter.on_source(data)

    @cached_property
    def operator(self):
        return {"add": op.add, "mul": op.mul, None: None}.get(self._future_adjust_type)

    @cached_property
    def reverse_operator(self):
        return {"add": op.sub, "mul": op.truediv, None: None}.get(
            self._future_adjust_type
        )

    def adjust_future(self, new_bar: ibi.BarData):
        """Create continuous future price series on future contract
        change.  IB mechanism cannot be trusted.  This feature can be
        turned off by passing `future_adjust_type=None` while
        initiating the class.
        """
        log.warning(f"{self} inside future adjust")
        if self.operator is None:
            log.error(f"Skipping futures adjustment on {self}")
            return
        assert self.reverse_operator is not None

        old_bar = self._filter.bars[-1]

        if old_bar.date != new_bar.date:
            log.error("Cannot back-adjust a future because dates don't match.")
            return

        value = self.reverse_operator(new_bar.close, old_bar.close)
        log.warning(
            f"FUTURES ADJUSTMENT | old bar: {old_bar} | new bar: {new_bar} "
            f"| adjustment basis: {value} | operator: {self.operator}"
        )

        for bar in self._filter.bars:
            for field in ("open", "high", "low", "close", "average"):
                setattr(bar, field, self.operator(getattr(bar, field), value))


class BarList(list[ibi.BarData]):
    def __init__(self, *args):
        super().__init__(*args)
        self.updateEvent = ibi.Event("updateEvent")

    def __eq__(self, other):
        return self is other

    def __hash__(self):  # type: ignore
        return id(self)


class CountBars(ev.Op):
    __slots__ = ("_count", "bars", "label")

    bars: BarList

    def __init__(self, count: int, source: ev.Event | None = None, *, label: str = ""):
        ev.Op.__init__(self, source)
        self._count = count
        self.bars = BarList()

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
    __slots__ = ("_volume", "bars", "label")

    bars: BarList

    def __init__(
        self, volume: int, source: ev.Event | None = None, *, label: str = ""
    ) -> None:
        ev.Op.__init__(self, source)
        self._volume = volume
        self.bars = BarList()
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
    __slots__ = ("_count", "bars", "label")

    bars: BarList

    def __init__(
        self, count: int, source: ev.Event | None = None, *, label: str = ""
    ) -> None:
        ev.Op.__init__(self, source)
        self._count = count
        self.bars = BarList()
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
    __slots__ = ("_timer", "bars", "_running_price_volume", "label")

    bars: BarList

    def __init__(
        self, timer, source: ev.Event | None = None, *, label: str = ""
    ) -> None:
        ev.Op.__init__(self, source)
        self._timer = timer
        self._timer.connect(self._on_timer, None, self._on_timer_done)
        self.bars = BarList()
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
    This works as an accumulator making sure that no bars are lost
    during restarts.
    """

    __slots__ = ("bars", "label")

    bars: BarList

    def __init__(self, source: ev.Event | None = None, *, label: str = "") -> None:
        ev.Op.__init__(self, source)
        self.bars = BarList()
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
