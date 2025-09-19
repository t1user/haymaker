from __future__ import annotations

import asyncio
import itertools
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from typing import Awaitable, ClassVar

import eventkit as ev  # type: ignore
import ib_insync as ibi

from haymaker.base import Atom
from haymaker.timeout import Timeout

log = logging.getLogger(__name__)


_counter = itertools.count().__next__


class Streamer(Atom, ABC):
    instances: ClassVar[list["Streamer"]] = []

    def __new__(cls, *args, **kwargs):
        # Keep track of all :class:`.Streamer` instances created so that they
        # can be re-started on reboot.
        obj = super().__new__(cls)
        cls.instances.append(obj)
        return obj

    @classmethod
    def awaitables(cls) -> list[Awaitable]:
        """
        Coroutines from all instantiated streamers.  Can be passed to
        :py:func:`asyncio.gather`
        """
        return [s.run() for s in cls.instances]

    @abstractmethod
    def streaming_func(self):
        raise NotImplementedError

    async def run(self) -> None:
        """
        Start subscription and start emitting data.  This is the main
        entry point into the streamer.
        """
        self.onStart({})
        while self.ib.isConnected():
            await asyncio.sleep(0)

    def onStart(self, data, *args) -> None:
        ticker = self.streaming_func()
        # automatically monitor updateEvent on streaming_func for
        # stale data
        Timeout.from_atom(self, ticker.updateEvent, "ticks")
        ticker.updateEvent += self.dataEvent
        # relies on superclass to emit startEvent
        super().onStart(data)

    @cached_property
    def _id(self) -> int:
        return _counter()

    def __str__(self) -> str:
        identifier = [str(self._id)]
        if contract := getattr(self, "contract", None):
            identifier.append(contract.symbol)
        if name := getattr(self, "name", None):
            identifier.append(name)
        return f"{self.__class__.__name__}<{"><".join(identifier)}>"


@dataclass
class HistoricalDataStreamer(Streamer):
    contract: ibi.Contract
    durationStr: str
    barSizeSetting: str
    whatToShow: str
    useRTH: bool = False
    formatDate: int = 2  # don't change
    incremental_only: bool = True
    startup_seconds: float = 5
    _last_bar_date: datetime | None = None
    _future_adjust_flag: bool = False  # flag that future needs to be adjusted
    _adjusted: list[ibi.Future] = field(
        default_factory=list
    )  # already adjusted futures

    def __post_init__(self):
        Atom.__init__(self)

    def streaming_func(self) -> Awaitable:
        return self.ib.reqHistoricalDataAsync(
            self.contract,
            endDateTime="",
            durationStr=self.durationStr,
            barSizeSetting=self.barSizeSetting,
            whatToShow=self.whatToShow,
            useRTH=self.useRTH,
            formatDate=self.formatDate,
            keepUpToDate=True,
            timeout=0,
        )

    def date_to_delta(self, date: datetime) -> int:
        """
        Return number of bars (as per barSizeSetting) since date. Used to determine
        number of bars required to backfill since last reset.
        """
        secs = (datetime.now(date.tzinfo) - date).seconds
        bar_size = int(self.barSizeSetting.split(" ")[0])
        bars = secs // bar_size
        # with some margin for delays, duplicated bars will be filtered out
        duration = max((bars + 3) * bar_size, 30)
        return duration

    def onStart(self, data, *args) -> None:
        # this starts subscription so that current price is readily available from ib
        stream = self.ib.reqMktData(self.contract, "221")
        Timeout.from_atom(self, stream.updateEvent, "ticks")
        # bypass onStart in class Streamer
        Atom.onStart(self, data)

    async def backfill(self, bars):
        # this will switch processor into backfill mode
        self.startEvent.emit(
            {"startup": True, "future_adjust_flag": self._future_adjust_flag}
        )
        # release control to allow for adjustments down the chain
        # await asyncio.sleep(0)
        log.debug(
            f"Starting backfill {self!s}, pulled {len(bars)} bars, "
            f"last bar: {bars[-1]}"
        )
        if self._last_bar_date and self._future_adjust_flag:
            # this is a restart and contract has changed, emitting data since
            # last emitted point, but:
            # include one additional point in emit
            # which is for the same time point as the value already emitted
            # but for new, rolled future
            # this way processor will be be able to calculate necessary adjustment
            log.warning(
                f"{self!s} data requires roll adjustment "
                f"last bar: {self._last_bar_date}"
            )

            stream = (
                ev.Sequence(bars[:-1])
                .pipe(
                    ev.Filter(lambda x: x.date >= self._last_bar_date)  # type: ignore
                )
                .connect(self.dataEvent)
            )
            self._future_adjust_flag = False
        elif self._last_bar_date:
            # this is a regular restart; backfilling only data since last emitted point
            stream = (
                ev.Sequence(bars[:-1])
                .pipe(ev.Filter(lambda x: x.date > self._last_bar_date))  # type: ignore
                .connect(self.dataEvent)
            )

        else:
            # this is not a restart, just getting all new data
            stream = ev.Sequence(bars[:-1]).connect(self.dataEvent)
        await stream
        await asyncio.sleep(self.startup_seconds)  # time in which backfill must happen
        log.info(
            f"{self!s} backfilled from {self._last_bar_date or bars[0].date} to "
            f"{bars[-2].date}"
        )
        # let processor know backfill is finished
        self.startEvent.emit({"startup": False})

    async def run(self) -> None:
        log.debug(f"Requesting bars for {self.contract.localSymbol}")
        self.onStart({})
        if self._last_bar_date:
            self.durationStr = f"{self.date_to_delta(self._last_bar_date)} S"
            log.debug(f"{self!s} duration str: {self.durationStr}")

        bars = await self.streaming_func()
        Timeout.from_atom(self, bars.updateEvent, "bars")
        log.debug(f"Historical bars received for {self.contract.localSymbol}")

        backfill_predicate = (
            (not self._last_bar_date)
            or (self._last_bar_date < bars[-2].date)
            or self._future_adjust_flag
        )

        if bars and backfill_predicate:
            await self.backfill(bars)
        else:
            log.debug(f"{self!s}: No backfill needed.")

        try:
            async for bars_, hasNewBar in bars.updateEvent:
                if hasNewBar and (
                    (not self._last_bar_date) or (bars[-2].date > self._last_bar_date)
                ):
                    self._last_bar_date = bars_[-2].date
                    self.on_new_bar(bars_[:-1])
        except ValueError as e:
            log.debug(f"Empty emit for {self}: {e}")

    def on_new_bar(self, bars: ibi.BarDataList) -> None:
        if any(
            (
                bars[-1].close <= 0,
                bars[-1].open <= 0,
                bars[-1].high <= 0,
                bars[-1].low <= 0,
                bars[-1].average <= 0,
            )
        ):
            return
        elif self.incremental_only:
            self.dataEvent.emit(bars[-1])
        else:
            self.dataEvent.emit(bars)

    def onContractChanged(self, old_contract: ibi.Contract, new_contract: ibi.Contract):
        if isinstance(old_contract, ibi.Future) and isinstance(
            new_contract, ibi.Future
        ):
            log.warning(f"{self!s} will adjust for rolled future.")
            if old_contract not in self._adjusted:
                log.warning(f"{self!s} set to adjust future")
                self._future_adjust_flag = True
                self._adjusted.append(old_contract)
                # don't move to backfill, because then aggregator doesn't get control
                # until backfill is finished
                # self.startEvent.emit({"future_adjust_flag": True})
            else:
                log.warning(
                    f"{self!s} abandoning attempt to roll future "
                    f"{old_contract.localSymbol} for the second time"
                )
        else:
            log.warning("onContractChanged triggered on non-future Contract")


@dataclass
class MktDataStreamer(Streamer):
    contract: ibi.Contract
    tickList: str

    def __post_init__(self):
        Atom.__init__(self)

    def streaming_func(self) -> ibi.Ticker:
        return self.ib.reqMktData(self.contract, self.tickList)


@dataclass
class RealTimeBarsStreamer(Streamer):
    contract: ibi.Contract
    whatToShow: str
    useRTH: bool
    incremental_only: bool = True

    def __post_init__(self):
        Atom.__init__(self)

    def streaming_func(self):
        return self.ib.reqRealTimeBars(
            self.contract,
            5,
            self.whatToShow,
            self.useRTH,
        )

    def onStart(self, data, *args) -> None:
        self._run()
        Atom.onStart(self, data, *args)

    def _run(self):
        bars = self.streaming_func()
        bars.updateEvent.clear()
        Timeout.from_atom(self, bars.updateEvent, "bars")
        bars.updateEvent += self.onUpdate

    def onUpdate(self, bars, hasNewBar):
        if hasNewBar:
            if self.incremental_only:
                self.dataEvent.emit(bars[-1])
            else:
                self.dataEvent.emit(bars)


@dataclass
class TickByTickStreamer(Streamer):
    contract: ibi.Contract
    tickType: str
    numberOfTicks: int = 0
    ignoreSize: bool = False

    def __post_init__(self):
        Atom.__init__(self)

    def streaming_func(self) -> ibi.Ticker:
        return self.ib.reqTickByTickData(
            contract=self.contract,
            tickType=self.tickType,
            numberOfTicks=self.numberOfTicks,
            ignoreSize=self.ignoreSize,
        )
