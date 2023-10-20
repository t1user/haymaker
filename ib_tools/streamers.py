from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import Awaitable, ClassVar, Optional

import eventkit as ev  # type: ignore
import ib_insync as ibi

from ib_tools import misc
from ib_tools.base import Atom

log = logging.getLogger(__name__)


class StaleDataError(Exception):
    pass


class Streamer(Atom):
    instances: ClassVar[list["Streamer"]] = []

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
        Coroutines from all instantiated streamers.  Used to put in
        :py:`asyncio.gather`
        """
        return [s.run() for s in cls.instances]

    def streaming_func(self):
        raise NotImplementedError

    async def run(self):
        self.onStart(None, None)
        while True:
            await asyncio.sleep(0)

    def onStart(self, data, *args) -> None:
        ticker = self.streaming_func()
        ticker.updateEvent += self.dataEvent


@dataclass
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
    timeout: float = 0.0

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

    def _on_timeout_error(self):
        log.error("TIMEOUT ERROR")

    def _timeout_callback(self, *args) -> None:
        log.debug(
            f"No data for {self.timeout} secs. Market acitve?: "
            f"{misc.is_active(self.trading_hours)}"
        )
        if misc.is_active(self.trading_hours):
            raise StaleDataError

    async def _reset_timeout(
        self, timeout_event: ev.Event, event: ev.Event, *args
    ) -> None:
        log.debug("Will reset timout.")
        event.disconnect(timeout_event)
        self._set_timeout(event)

    def _set_timeout(self, event: ev.Event) -> None:
        timeout = event.timeout(self.timeout)
        timeout.connect(
            self._timeout_callback,
            error=self._on_timeout_error,
            done=partial(self._reset_timeout, event=event),
        )
        log.debug(f"Timeout connected {timeout}")
        # loop = asyncio.get_event_loop()
        # loop.call_soon(timeout)

        log.debug("Timeout set")

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
        # TODO: consider if it's needed
        # self.ib.reqMktData(self.contract, "221")
        self.startEvent.emit(data, self)

    async def run(self) -> None:
        log.debug(f"Requesting bars for {self.contract.localSymbol}")

        if self.last_bar_date:
            self.durationStr = f"{self.date_to_delta(self.last_bar_date)} S"

        # if self.bars:
        #    self.ib.cancelHistoricalData(self.bars)
        bars = await self.streaming_func()
        log.debug(f"Historical bars received for {self.contract.localSymbol}")

        if self.timeout:
            self._set_timeout(bars.updateEvent)

        # ### this is startup (backfill) ###
        self.onStart({"startup": True})
        if self.last_bar_date:
            stream = (
                ev.Sequence(bars[:-1])
                .pipe(ev.Filter(lambda x: x.date > self.last_bar_date))
                .connect(self.dataEvent)
            )
        else:
            stream = ev.Sequence(bars[:-1]).connect(self.dataEvent)
        await stream
        await asyncio.sleep(self.startup_seconds)  # time in which backfill must happen
        log.debug(f"bars[0]: {bars[0]}, bars[-2] {bars[-2]}")
        log.info(
            f"Backfilled from {self.last_bar_date or bars[0].date} to {bars[-2].date}"
        )
        self.onStart({"startup": False})
        # ### end of startup (backfill) ###

        async for bars_, hasNewBar in bars.updateEvent:
            if hasNewBar:
                self.last_bar_date = bars_[-2].date
                self.on_new_bar(bars_[:-1])

    def on_new_bar(self, bars: ibi.BarDataList) -> None:
        if self.incremental_only:
            self.dataEvent.emit(bars[-1])
        else:
            self.dataEvent.emit(bars)


@dataclass
class MktDataStreamer(Streamer):
    contract: ibi.Contract
    tickList: str

    def __post_init__(self):
        """Relevant only if inherited by a dataclass."""
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
        # this is a sync function
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
