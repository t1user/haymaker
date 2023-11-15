from __future__ import annotations

import asyncio
import itertools
import logging
from abc import ABC, abstractmethod
from collections import UserDict, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable, ClassVar, Optional, cast

import eventkit as ev  # type: ignore
import ib_insync as ibi

from ib_tools import misc
from ib_tools.base import Atom

log = logging.getLogger(__name__)


class StaleDataError(Exception):
    pass


@dataclass
class Timeout:
    time: float
    event: ev.Event
    trading_hours: Optional[list[tuple[datetime, datetime]]] = None
    name: str = ""
    debug: bool = True

    def __post_init__(self):
        if self.time:
            self._set_timeout(self.event)

    is_active = staticmethod(misc.is_active)
    next_open = staticmethod(misc.next_open)

    def _on_timeout_error(self):
        log.error(f"{self!s} Timeout broken...")

    async def _timeout_callback(self, *args) -> None:
        log.debug(
            f"{self!s} - No data for {self.time} secs. Market active?: "
            f"{self.is_active(self.trading_hours)}"
        )
        if self.is_active(self.trading_hours):
            self.triggered_action()
        else:
            reactivate_time = self.next_open(self.trading_hours)

            sleep_time = (reactivate_time - datetime.now(tz=timezone.utc)).seconds
            log.debug(
                f"{self} will sleep till market reopen at: {reactivate_time} i.e. "
                f"{sleep_time}seconds"
            )
            await asyncio.sleep(sleep_time)
            self._set_timeout(self.event)

    def triggered_action(self):
        self._triggered = True
        if self.debug:
            log.error(f"Stale streamer {self!s} Reset?")
            return True
        else:
            pass
            # raise StaleDataError(f"Stale data for {self.name}")

    def _set_timeout(self, event: ev.Event) -> None:
        timeout = event.timeout(self.time)
        timeout.connect(self._timeout_callback)
        log.debug(f"Timeout set: {self!s}")

    def __str__(self) -> str:
        return f"Timeout <{self.time}s> for {self.name}"


class TimeoutContainerDefaultdict(defaultdict):
    def __missing__(self, key):
        self[key] = self.default_factory(key)
        return self[key]


class TimeoutContainer(UserDict):
    def __init__(self, streamer: "Streamer"):
        self.streamer = streamer
        super().__init__()

    def __setitem__(self, key: str, obj) -> None:
        if isinstance(obj, ev.Event):
            obj = Timeout(
                self.streamer.timeout,
                obj,
                self.streamer.trading_hours,
                f"{str(self.streamer)}-<<{key}>>",
            )
        super().__setitem__(key, obj)


class Streamer(Atom, ABC):
    instances: ClassVar[list["Streamer"]] = []
    timeout: float = 0
    _counter: ClassVar[Callable[[], int]] = itertools.count().__next__
    _name: Optional[str] = None
    _timers: TimeoutContainerDefaultdict = TimeoutContainerDefaultdict(
        cast(Callable, TimeoutContainer)
    )

    def __new__(cls, *args, **kwargs):
        """
        Keep track of all :class:`.Streamer` instances created so that they
        can be re-started on reboot.
        """
        obj = super().__new__(cls)
        cls.instances.append(obj)
        return obj

    @classmethod
    def awaitables(cls) -> list[Awaitable]:
        """
        Coroutines from all instantiated streamers.  Can be passed to
        :py:`asyncio.gather`
        """
        return [s.run() for s in cls.instances]

    @abstractmethod
    def streaming_func(self):
        raise NotImplementedError

    async def run(self):
        """
        Start subscription and start emitting data.  This is the main
        entry point into the streamer.
        """
        self.onStart({})
        while True:
            await asyncio.sleep(0)

    def onStart(self, data, *args) -> None:
        ticker = self.streaming_func()
        ticker.updateEvent += self.dataEvent
        self.startEvent.emit(data, self)

    @property
    def timers(self) -> dict[str, ev.Event]:
        return self._timers[self]

    @timers.setter
    def timers(self, event: ev.Event) -> None:
        raise ValueError(
            "Stream event cannot be overridden. "
            "Try using: `self.stream_event[key] = event`"
        )

    @property
    def name(self):
        if self._name:
            return self._name
        elif getattr(self, "contract", None):
            identifier = self.contract.symbol
        else:
            identifier = self._counter()
        return f"{self.__class__.__name__}<{identifier}>"

    @name.setter
    def name(self, value):
        self._name = value

    def __str__(self) -> str:
        return self.name


@dataclass(unsafe_hash=True)
class HistoricalDataStreamer(Streamer):
    contract: ibi.Contract
    durationStr: str
    barSizeSetting: str
    whatToShow: str
    useRTH: bool = False
    formatDate: int = 2
    incremental_only: bool = True
    startup_seconds: float = 5
    last_bar_date: Optional[datetime] = None
    timeout: float = 300  # 5 minutes without data

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
        duration = max((bars + 1) * bar_size, 30)
        return duration

    def onStart(self, data, *args) -> None:
        # this starts subscription so that current price is readily available from ib
        stream = self.ib.reqMktData(self.contract, "221")
        self.timers["ticks"] = stream.updateEvent
        self.startEvent.emit(data, self)

    async def backfill(self, bars):
        self.onStart({"startup": True})
        log.debug(f"Starting backfill {self.name}, pulled {len(bars)} bars.")
        log.debug(f"{self.name} last bar date: {self.last_bar_date}")
        log.debug(f"{self.name} last bar: {bars[-1]}")
        if self.last_bar_date:
            log.debug(f"{self.name} in")
            stream = (
                ev.Sequence(bars[:-1])
                .pipe(ev.Filter(lambda x: x.date > self.last_bar_date))
                .connect(self.dataEvent)
            )
            log.debug(f"{self.name} out")
        else:
            stream = ev.Sequence(bars[:-1]).connect(self.dataEvent)
        log.debug(f"{self.name} about to await stream")
        await stream
        log.debug(f"{self.name} stream awaited.")
        await asyncio.sleep(self.startup_seconds)  # time in which backfill must happen
        try:
            log.debug(f"{self.name}: bars[0]: {bars[0]}, bars[-2] {bars[-2]}")
            log.info(
                f"{self.name} backfilled from {self.last_bar_date or bars[0].date} to "
                f"{bars[-2].date}"
            )
            log.debug(f"Backfill completed {self.name}")
        except Exception:
            log.exception(f"Exception while trying to log bars: {bars}")
        self.onStart({"startup": False})

    async def run(self) -> None:
        log.debug(f"Requesting bars for {self.contract.localSymbol}")

        if self.last_bar_date:
            self.durationStr = f"{self.date_to_delta(self.last_bar_date)} S"
            log.debug(f"{self.name} duration str: {self.durationStr}")

        bars = await self.streaming_func()
        self.timers["bars"] = bars.updateEvent
        log.debug(f"Historical bars received for {self.contract.localSymbol}")

        backfill_predicate = (not self.last_bar_date) or (
            self.last_bar_date < bars[-2].date
        )
        if bars and backfill_predicate:
            log.debug(f"{self.name} first bar: {bars[0]}, last bar: {bars[-1]}")
            await self.backfill(bars)
        else:
            log.debug(f"{self!s}: No backfill needed.")

        async for bars_, hasNewBar in bars.updateEvent:
            if hasNewBar and (
                (not self.last_bar_date) or (bars[-2].date > self.last_bar_date)
            ):
                self.last_bar_date = bars_[-2].date
                self.on_new_bar(bars_[:-1])

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


@dataclass(unsafe_hash=True)
class MktDataStreamer(Streamer):
    contract: ibi.Contract
    tickList: str

    def __post_init__(self):
        Atom.__init__(self)

    def streaming_func(self) -> ibi.Ticker:
        return self.ib.reqMktData(self.contract, self.tickList)


@dataclass(unsafe_hash=True)
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
        self.startEvent.emit(data, self)
        self._run()

    def _run(self):
        bars = self.streaming_func()
        bars.updateEvent.clear()
        bars.updateEvent += self.onUpdate

    def onUpdate(self, bars, hasNewBar):
        if hasNewBar:
            if self.incremental_only:
                self.dataEvent.emit(bars[-1])
            else:
                self.dataEvent.emit(bars)


@dataclass(unsafe_hash=True)
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
