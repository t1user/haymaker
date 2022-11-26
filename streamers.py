from abc import ABC, abstractmethod
from typing import Optional, Deque, List
from collections import deque
from datetime import datetime

import pandas as pd  # type: ignore
from logbook import Logger  # type: ignore
from ib_insync import IB, util, Event, Contract, BarData, BarDataList

log = Logger(__name__)


class BarStreamer(ABC):

    durationStr = "12 D"
    barSizeSetting = "30 secs"
    whatToShow = "TRADES"
    useRTH = False
    contract = None

    """ TODO: backfilling leaves one duplicated data point on joining time series"""

    def __call__(
        self, ib, contract: Contract, start_date: Optional[datetime] = None
    ) -> None:
        self.ib = ib
        self.contract = contract
        log.debug(f"start_date: {start_date}")
        if start_date:
            # 30s time-window to retrieve data
            self.durationStr = f"{self.date_to_delta(start_date) + 30} S"
        while True:
            log.debug(f"Requesting bars for {self.contract.localSymbol}")
            self.bars = self.get_bars()
            if self.bars is not None:
                break
            else:
                util.sleep(5)
        log.debug(f"Bars received for {self.contract.localSymbol}")
        self.subscribe()

    def date_to_delta(self, date: datetime) -> int:
        return (self.now - date).seconds

    def get_bars(self) -> BarDataList:
        log.debug(
            f"reqHistoricalData params for {self.contract.localSymbol} "
            f"durationStr: {self.durationStr}"
        )
        return self.ib.reqHistoricalData(
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

    def subscribe(self) -> None:
        self.bars.updateEvent += self.onNewBar

    def onNewBar(self, bars: BarDataList, hasNewBar: bool) -> None:
        if hasNewBar:
            # latest bar is "under construction", [-2] is the latest
            # ready for consumption
            self.aggregate(bars[-2])

    @abstractmethod
    def aggregate(self, bar: BarData):
        pass


class StreamAggregator(BarStreamer):
    def __init__(self) -> None:
        self._createEvents()
        self.buffer: Deque[BarDataList] = deque()
        self.new_bars: List[BarData] = []
        self.backfill = True
        super().__init__()

    def _createEvents(self) -> None:
        self.newCandle = Event("newCandle")

    def __call__(self, ib: IB, contract: Contract) -> None:
        date = self.all_bars[-1].date if contract == self.contract else None
        super().__call__(ib, contract, date)
        self.process_back_data(date)

    def process_back_data(self, date: Optional[datetime] = None) -> None:
        # flag needed on re-connect
        self.backfill = True
        for counter, bar in enumerate(self.bars[:-1]):
            # date given on reconnect only
            if (date and (bar.date > date)) or not date:
                self.aggregate(bar)
                # prevent from blocking too long
                if counter % 10000 == 0:
                    log.debug(f"releasing control {self.contract.localSymbol}")
                    util.sleep(0)
        log.debug(f"startup data generated for {self.contract.localSymbol}")
        self.backfill = False
        self.clear_buffer()

    def create_candle(self) -> None:
        df = util.df(self.new_bars)
        df.date = df.date.astype("datetime64[ns]")
        df.set_index("date", inplace=True)
        df["volume_weighted"] = df.average * df.volume
        weighted_price = df.volume_weighted.sum() / df.volume.sum()
        self.newCandle.emit(
            {
                "backfill": self.backfill,
                "date": df.index[-1],
                "open": df.open[0],
                "high": df.high.max(),
                "low": df.low.min(),
                "close": df.close[-1],
                "weighted_price": weighted_price,
                # 'price': weighted_price,
                "price": df.close[-1],
                "volume": df.volume.sum(),
            }
        )

    def onNewBar(self, bars: BarDataList, hasNewBar: bool) -> None:
        if hasNewBar:
            if self.backfill:
                log.debug(f"buffering bar for {self.contract.localSymbol}")
                self.buffer.append(bars[-2])
            else:
                self.clear_buffer()
                self.aggregate(bars[-2])

    def clear_buffer(self) -> None:
        """Utilize bars that have been buffered while processing back data."""
        while self.buffer:
            log.debug(f"clearing buffer for {self.contract.localSymbol}")
            self.aggregate(self.buffer.popleft())


class VolumeStreamer(StreamAggregator):
    def __init__(
        self, volume: Optional[float] = None, avg_periods: Optional[float] = None
    ) -> None:
        super().__init__()
        self.all_bars = []
        self.volume = volume
        self.avg_periods = avg_periods
        self.aggregator = 0

    def __call__(self, ib: IB, contract: Contract) -> None:
        try:
            date = self.all_bars[-1].date if contract == self.contract else None
        except IndexError:
            date = None
        BarStreamer.__call__(self, ib, contract, date)
        if self.avg_periods:
            self.volume = self.reset_volume(self.avg_periods)
        else:
            self.volume = self.volume
        log.info(f"Volume for {contract.localSymbol}: {self.volume}")
        StreamAggregator.process_back_data(self, date)

    def reset_volume(self, avg_periods) -> int:
        # TODO: make span adjust to length of requested data
        bars = self.all_bars or self.bars
        if bars == self.bars:
            self.span = len(self.bars)
        df = util.df(bars)
        # last 5 days
        volume = df.iloc[-14100:].volume.rolling(avg_periods).sum().mean().round()
        log.debug(f"volume: {volume}")
        return volume

    @staticmethod
    def verify(bar: BarData) -> bool:
        """
        Faulty bar often comes with volume = -1 or barCount = -1.
        """
        return (bar.volume >= 0) and (bar.barCount >= 0)

    def aggregate(self, bar: BarData) -> None:
        if not self.verify(bar):
            log.warning(f"Faulty bar for: {self.contract.localSymbol} {bar}")
            return
        self.new_bars.append(bar)
        self.all_bars.append(bar)
        self.aggregator += bar.volume
        if not self.backfill:
            message = (
                f"{bar.date} {self.aggregator}/{self.volume}"
                f" {self.contract.localSymbol}"
            )
            log.debug(message)
        if self.aggregator >= self.volume:
            self.aggregator = 0
            self.create_candle()
            self.new_bars.clear()

    @property
    def all_bars_df(self):
        if self.all_bars:
            df = util.df(self.all_bars)
            df.date = df.date.astype("datetime64[ns]")
            df.set_index("date", inplace=True)
        else:
            df = pd.DataFrame()
        return df


class ResampledStreamer(StreamAggregator):
    def __init__(self, periods: int) -> None:
        self.periods = periods
        self.counter = 0
        super().__init__()

    def aggregate(self, bar) -> None:
        self.new_bars.append(bar)
        self.counter += 1
        if self.counter == self.periods:
            self.create_candle()
            self.counter = 0


class DirectStreamer(StreamAggregator):
    def aggregate(self, bar: BarData) -> None:
        self.new_bars.append(bar)
        self.create_candle()
