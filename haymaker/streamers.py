from __future__ import annotations

import asyncio
import itertools
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from functools import cached_property
from typing import Any, Awaitable, ClassVar

import eventkit as ev  # type: ignore
import ib_insync as ibi

from haymaker.base import Atom
from haymaker.timeout import Timeout

log = logging.getLogger(__name__)


_counter = itertools.count().__next__


def bar_filter(bar: ibi.BarData) -> bool:
    """Return `true` if bar is faulty"""
    return any(
        (
            bar.close <= 0,
            bar.open <= 0,
            bar.high <= 0,
            bar.low <= 0,
            bar.average <= 0,
        )
    )


def date_to_delta(date: datetime, barSizeSetting: str) -> int:
    """
    Return number of bars (as per barSizeSetting) since date. Used to determine
    number of bars required to backfill since last reset.
    """
    secs = (datetime.now(date.tzinfo) - date).seconds
    bar_size = int(barSizeSetting.split(" ")[0])
    bars = secs // bar_size
    # with some margin for delays, duplicated bars will be filtered out
    duration = max((bars + 3) * bar_size, 30)
    return duration


class Streamer(Atom, ABC):
    instances: ClassVar[list["Streamer"]] = []
    timeout: bool = True

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

        self._set_timeout(ticker.updateEvent, "ticks")

        ticker.updateEvent += self.dataEvent
        data.update(self.params)
        # this is a contract blueprint, not actual contract
        assert data["contract"] is not None
        # relies on superclass to emit startEvent
        super().onStart(data)

    def _set_timeout(self, event: ev.Event, name: str) -> None:
        """
        Automatically monitor event for stale data.  Can be switched
        off by overriding class variable `set_timeout`
        """
        if self.timeout:
            Timeout.from_atom(self, event, name)

    @cached_property
    def _id(self) -> int:
        return _counter()

    def __str__(self) -> str:
        identifier = [str(self._id)]
        if contract := getattr(self, "contract", None):
            identifier.append(contract.symbol)
        if name := getattr(self, "name", None):
            identifier.append(name)
        return f"{self.__class__.__name__}<{'><'.join(identifier)}>"

    @property
    def params(self) -> dict[str, Any]:
        if is_dataclass(self.__class__):
            data = {field.name: getattr(self, field.name) for field in fields(self)}
            data["contract"] = self.__dict__.get("contract")
        else:
            # it contains some inherited items that are most likely
            # not helpful, but it doesn't matter
            data = {
                k: v for k, v in self.__dict__.items() if not isinstance(v, ev.Event)
            }
        data["streamer_class"] = self.__class__.__name__
        return data


@dataclass
class HistoricalDataStreamer(Streamer):
    contract: ibi.Contract
    durationStr: str
    barSizeSetting: str
    whatToShow: str
    useRTH: bool = False
    formatDate: int = 2  # don't change
    _last_bar_date: datetime | None = None

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

    def onStart(self, data, *args) -> None:
        # this starts subscription so that current price is readily available from ib
        stream = self.ib.reqMktData(self.contract, "221")
        self._set_timeout(stream.updateEvent, "ticks")
        data["contract"] = self.contract
        # bypass onStart in class Streamer
        Atom.onStart(self, data)

    async def run(self) -> None:
        if self._last_bar_date:
            self.durationStr = (
                f"{date_to_delta(self._last_bar_date, self.barSizeSetting)} S"
            )
        log.debug(f"{self!s} requesting bars {self.durationStr=}")

        bars = await self.streaming_func()
        log.debug(f"{self!s} received historical bars")

        self._set_timeout(bars.updateEvent, "bars")

        try:
            async for bars_, hasNewBar in bars.updateEvent:
                if hasNewBar and (
                    (not self._last_bar_date) or (bars[-2].date > self._last_bar_date)
                ):
                    self._last_bar_date = bars_[-2].date
                    self.on_new_bar(bars_[:-1])
        except ValueError as e:
            log.debug(f"Empty emit for {self!s}: {e}")

    def on_new_bar(self, bars: ibi.BarDataList) -> None:
        if bar_filter(bars[-1]):
            return
        else:
            self.dataEvent.emit(bars)

    def onContractChanged(
        self, old_contract: ibi.Contract, new_contract: ibi.Contract
    ) -> None:
        self._last_bar_date = None


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
        data["contract"] = self.contract
        self._run()
        Atom.onStart(self, data, *args)

    def _run(self):
        bars = self.streaming_func()
        bars.updateEvent.clear()
        Timeout.from_atom(self, bars.updateEvent, "bars")
        bars.updateEvent += self.onUpdate

    def onUpdate(self, bars, hasNewBar):
        if hasNewBar and not bar_filter(bars[-1]):
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
