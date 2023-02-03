from dataclasses import dataclass
from datetime import datetime

import ib_insync as ibi
from logbook import Logger  # type: ignore

from ib_tools.base import Atom

log = Logger(__name__)


@dataclass
class HistoricalDataStreamer(Atom):
    contract: ibi.Contract
    durationStr: str
    barSizeSetting: str
    whatToShow: str
    useRTH: bool
    formatDate: int
    incremental_only: bool = True

    def __post_init__(self):
        Atom.__init__(self)

    def streaming_func(self):
        return self.ib.reqHistoricalDataAsync(
            self.contract,
            endDateTime="",
            durationStr=self.durationStr,
            barSizeSetting=self.barSizeSetting,
            whatToShow=self.whatToShow,
            useRTH=self.useRTH,
            formatDate=1,
            keepUpToDate=True,
            timeout=0,
        )

    def onStart(self, start_date, source) -> None:
        log.debug(f"start_date: {start_date}")
        if start_date is not None:
            # 30s time-window to retrieve data
            # LOOK AT THIS
            # IT MAY CAUSE DOUBLING ON ONE DATA POINT
            self.durationStr = f"{self.date_to_delta(start_date) + 30} S"
        self.ib.run(self.run())
        self.ib.reqMktData(self.contract, "221")

    def onData(self, data, source: Atom) -> None:
        pass

    async def run(self):
        log.debug(f"Requesting bars for {self.contract.localSymbol}")
        bars = await self.streaming_func()
        log.debug(f"Historical bars received for {self.contract.localSymbol}")
        self.startEvent.emit(bars)
        async for bars, hasNewBar in bars.updateEvent:
            self.on_bars(bars, hasNewBar)

    def on_bars(self, bars, hasNewBar):
        if hasNewBar:
            self.on_new_bar(bars[:-1])

    def on_new_bar(self, bars: ibi.BarDataList, hasNewBar: bool) -> None:
        if self.incremental_only:
            self.dataEvent.emit(bars[-1])
        else:
            self.dataEvent.emit(bars)

    def date_to_delta(self, date: datetime) -> int:
        return (datetime.now() - date).seconds


@dataclass
class MktDataStreamer(Atom):
    contract: ibi.Contract
    tickList: str

    def __post_init__(self):
        Atom.__init__(self)

    def streaming_func(self) -> ibi.Ticker:
        return self.ib.reqMktData(self.contract, self.tickList)

    def onStart(self, data, source: Atom) -> None:
        ticker = self.streaming_func()
        ticker.updateEvent += self.dataEvent

    def onData(self, data, source: Atom) -> None:
        pass


@dataclass
class RealTimeBarsStreamer(Atom):
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

    def onStart(self, data, source: Atom) -> None:
        # this is a sync function
        self.run()

    def onData(self, data, source: Atom) -> None:
        pass

    def run(self):
        bars = self.streaming_func()
        bars.updateEvent += self.onUpdate

    def onUpdate(self, bars, hasNewBar):
        if hasNewBar:
            if self.incremental_only:
                self.dataEvent.emit(bars[-1])
            else:
                self.dataEvent.emit(bars)
