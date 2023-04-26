import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Awaitable, List, Optional

import eventkit as ev  # type: ignore
import ib_insync as ibi
from logbook import Logger  # type: ignore

from ib_tools.base import Atom

log = Logger(__name__)


class Streamer(Atom):
    instances: List["Streamer"] = []

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        cls.instances.append(obj)
        return obj

    @classmethod
    def awaitables(cls) -> List[Awaitable]:
        """All instantiated streamers. Used to put in asyncio.gather"""
        return [s.run() for s in cls.instances]

    def __post_init__(self):
        """Relevant only if inherited by a dataclass."""
        Atom.__init__(self)

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
    last_bar_date: Optional[datetime] = None

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
        secs = (datetime.now(date.tzinfo) - date).seconds
        bar_size = int(self.barSizeSetting.split(" ")[0])
        bars = secs // bar_size
        duration = max((bars + 1) * bar_size, 30)
        return duration

    def onStart(self, data, *args) -> None:
        # this starts subscription so that current price is readily available from ib
        # TODO: consider if it's needed
        self.ib.reqMktData(self.contract, "221")

    async def run(self):
        self.onStart(None, None)
        log.debug(f"Requesting bars for {self.contract.localSymbol}")
        if self.last_bar_date:
            self.durationStr = f"{self.date_to_delta(self.last_bar_date)} S"
        bars = await self.streaming_func()
        log.debug(f"Historical bars received for {self.contract.localSymbol}")
        if self.last_bar_date:
            stream = (
                ev.Sequence(bars[:-1])
                .pipe(ev.Filter(lambda x: x.date > self.last_bar_date))
                .connect(self.dataEvent)
            )
        else:
            stream = ev.Sequence(bars[:-1]).connect(self.dataEvent)
        await stream

        async for bars, hasNewBar in bars.updateEvent:
            if hasNewBar:
                self.last_bar_date = bars[-2].date
                self.on_new_bar(bars[:-1])

    def on_new_bar(self, bars: ibi.BarDataList) -> None:
        if self.incremental_only:
            self.dataEvent.emit(bars[-1])
        else:
            self.dataEvent.emit(bars)


@dataclass
class MktDataStreamer(Streamer):
    contract: ibi.Contract
    tickList: str

    def streaming_func(self) -> ibi.Ticker:
        return self.ib.reqMktData(self.contract, self.tickList)


@dataclass
class RealTimeBarsStreamer(Streamer):
    contract: ibi.Contract
    whatToShow: str
    useRTH: bool
    incremental_only: bool = True

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

    def streaming_func(self) -> ibi.Ticker:
        return self.ib.reqTickByTickData(
            contract=self.contract,
            tickType=self.tickType,
            numberOfTicks=self.numberOfTicks,
            ignoreSize=self.ignoreSize,
        )
