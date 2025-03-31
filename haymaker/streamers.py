from __future__ import annotations

import asyncio
import itertools
import logging
from abc import ABC, abstractmethod
from collections import UserDict, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, ClassVar, DefaultDict

import eventkit as ev  # type: ignore
import ib_insync as ibi

from .base import Atom, Details
from .config import CONFIG as config

log = logging.getLogger(__name__)


CONFIG = config.get("streamer") or {}
TIMEOUT_DEBUG = CONFIG["timeout"]["debug"]
TIMEOUT_TIME = CONFIG["timeout"]["time"]


@dataclass
class Timeout:
    time: float
    event: ev.Event
    ib: ibi.IB
    details: Details | None = None
    name: str = ""
    debug: bool = TIMEOUT_DEBUG
    _timeout: ev.Event | None = field(repr=False, default=None)
    _now: datetime | None = None  # for testing only

    def __post_init__(self):
        if self.time:
            self._set_timeout(self.event)

    def _on_timeout_error(self, *args):
        log.error(f"{self!s} Timeout broken... {args}")

    async def _timeout_callback(self, *args) -> None:
        if not self.details:
            log.error("Cannot set timout, no trading hours details.")
        else:
            log.log(
                5,
                f"{self!s} - No data for {self.time} secs. Market open?: "
                f"{self.details.is_open(self._now)}",
            )
            if self.details.is_open(self._now):
                self.triggered_action()
            else:
                reactivate_time = self.details.next_open(self._now)

                sleep_time = (reactivate_time - datetime.now(tz=timezone.utc)).seconds
                log.log(
                    5,
                    f"{self} will sleep till market reopen at: {reactivate_time} i.e. "
                    f"{sleep_time} seconds",
                )
                await asyncio.sleep(sleep_time)
                self._set_timeout(self.event)

    def triggered_action(self):
        if self.debug:
            log.error(f"Stale streamer {self!s} Reset?")
            return True
        else:
            log.debug(f"Stale streamer {self!s} will disconnect ib...")
            self.ib.disconnect()

    def _set_timeout(self, event: ev.Event) -> None:
        self._timeout = event.timeout(self.time)
        self._timeout.connect(self._timeout_callback)
        log.debug(f"Timeout set: {self!s}")

    def __str__(self) -> str:
        return f"Timeout <{self.time}s> for {self.name}  event id: {id(self._timeout)}"


class TimeoutContainer(UserDict):
    """
    Class that stores and creates timeouts based on parameters extracted from
    passed :class:`Streamer` instance.

    Args:
        streamer: :class:`Streamer` instance, whose streams are being kept alive


    Usage:
    .. code-block:: python

        class MyStreamer(Streamer):

            def some_meth(self):
                ticker = self.streaming_func()
                # this ensures streaming event is kept alive
                # and timer is added to TimeoutContainerDefaultdict
                self.timers["my_timer"] = ticker.updateEvent
                ticker.updateEvent += self.dataEvent

            def other_meth(self):
                # timer is accessible via:
                timer = self.timers["my_timer"]

        s = MyStreamer()
    """

    def __init__(self, streamer: "Streamer"):
        self.streamer = streamer
        super().__init__()

    def __setitem__(self, key: str, obj) -> None:
        if isinstance(obj, ev.Event):
            assert isinstance(self.streamer.details, Details)
            obj = Timeout(
                self.streamer.timeout,
                obj,
                self.streamer.ib,
                self.streamer.details,
                f"{str(self.streamer)}-<<{key}>>",
            )
        super().__setitem__(key, obj)


class TimeoutContainerDefaultdict(defaultdict):
    """
    Container for all timeouts.

    This is one container shared among all instances of :class:`Streamer`.
    * keys are :class:`Streamer` instances - each instance keeps its own dict of
      timeouts with unique names; it's essential that each :class:`Streamer` is hashable
    * values are :class:`TimeoutContainer`, which are modified dictionaries, where each
        :class:`Streamer` can keep its timeouts with keys unique for that class.

    Args:
        default_factory: essentially it has to be :type:`TimeoutContainer`
    """

    def __init__(self, default_factory: Callable = TimeoutContainer):
        super().__init__(default_factory)

    def __missing__(self, key: Any) -> Any:
        if self.default_factory:
            self[key] = self.default_factory(key)  # type: ignore
            return self[key]
        raise KeyError(key)


class Streamer(Atom, ABC):
    instances: ClassVar[list["Streamer"]] = []
    timeout: float = TIMEOUT_TIME
    _counter: ClassVar[Callable[[], int]] = itertools.count().__next__
    _name: str | None = None
    _timers: DefaultDict["Streamer", dict[str, ev.Event]] = (
        TimeoutContainerDefaultdict()
    )

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
        while self.ib.isConnected():
            await asyncio.sleep(0)

    def onStart(self, data, *args) -> None:
        ticker = self.streaming_func()
        self.timers["ticks"] = ticker.updateEvent
        ticker.updateEvent += self.dataEvent
        # relies on superclass to emit startEvent
        super().onStart(data)

    @property
    def timers(self) -> dict[str, ev.Event]:
        return self._timers[self]

    @timers.setter
    def timers(self, event: ev.Event) -> None:
        raise ValueError(
            "timers cannot be overridden, use: `self.timers[key] = event` to add timer"
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

    def __hash__(self):
        return hash(id(self))


@dataclass
class HistoricalDataStreamer(Streamer):
    contract: ibi.Contract
    durationStr: str
    barSizeSetting: str
    whatToShow: str
    useRTH: bool = False
    formatDate: int = 2  # don't change
    timeout: float = TIMEOUT_TIME
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
        self.timers["ticks"] = stream.updateEvent
        # bypass onStart in class Streamer
        Atom.onStart(self, data)

    async def backfill(self, bars):
        # this will switch processor into backfill mode
        self.startEvent.emit({"startup": True})
        # release control to allow for adjustments down the chain
        await asyncio.sleep(0)
        log.debug(
            f"Starting backfill {self.name}, pulled {len(bars)} bars, "
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
            f"{self.name} backfilled from {self._last_bar_date or bars[0].date} to "
            f"{bars[-2].date}"
        )
        # let processor know backfill is finished
        self.startEvent.emit({"startup": False})

    async def run(self) -> None:
        log.debug(f"Requesting bars for {self.contract.localSymbol}")
        self.onStart({})
        if self._last_bar_date:
            self.durationStr = f"{self.date_to_delta(self._last_bar_date)} S"
            log.debug(f"{self.name} duration str: {self.durationStr}")

        bars = await self.streaming_func()
        self.timers["bars"] = bars.updateEvent
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
                self.startEvent.emit({"_future_adjust_flag": True})
                self._adjusted.append(old_contract)
            else:
                log.warning(
                    f"{self!s} abandoning attempt to roll future "
                    f"{old_contract.localSymbol} for the second time"
                )

    def __hash__(self):
        return hash(id(self))


@dataclass
class MktDataStreamer(Streamer):
    contract: ibi.Contract
    tickList: str
    timeout: float = TIMEOUT_TIME

    def __post_init__(self):
        Atom.__init__(self)

    def streaming_func(self) -> ibi.Ticker:
        return self.ib.reqMktData(self.contract, self.tickList)

    def __hash__(self):
        return hash(id(self))


@dataclass
class RealTimeBarsStreamer(Streamer):
    contract: ibi.Contract
    whatToShow: str
    useRTH: bool
    incremental_only: bool = True
    timeout: float = TIMEOUT_TIME

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
        # self.startEvent.emit(data, self)
        self._run()
        Atom.onStart(self, data, *args)

    def _run(self):
        bars = self.streaming_func()
        bars.updateEvent.clear()
        self.timers["bars"] = bars.updateEvent
        bars.updateEvent += self.onUpdate

    def onUpdate(self, bars, hasNewBar):
        if hasNewBar:
            if self.incremental_only:
                self.dataEvent.emit(bars[-1])
            else:
                self.dataEvent.emit(bars)

    def __hash__(self):
        return hash(id(self))


@dataclass
class TickByTickStreamer(Streamer):
    contract: ibi.Contract
    tickType: str
    numberOfTicks: int = 0
    ignoreSize: bool = False
    timeout: float = TIMEOUT_TIME

    def __post_init__(self):
        Atom.__init__(self)

    def streaming_func(self) -> ibi.Ticker:
        return self.ib.reqTickByTickData(
            contract=self.contract,
            tickType=self.tickType,
            numberOfTicks=self.numberOfTicks,
            ignoreSize=self.ignoreSize,
        )

    def __hash__(self):
        return hash(id(self))
