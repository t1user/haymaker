from abc import ABC, abstractmethod
from typing import Optional
from collections import deque
from datetime import datetime


from logbook import Logger
from ib_insync import IB, util, Event, Contract, BarData, BarDataList

log = Logger(__name__)


class BarStreamer(ABC):

    durationStr = '10 D'
    barSizeSetting = '30 secs'
    whatToShow = 'TRADES'
    useRTH = False

    def __call__(self, ib, contract: Contract,
                 start_date: Optional[datetime] = None) -> None:
        self.ib = ib
        self.contract = contract
        log.debug(f'start_date: {start_date}')
        if start_date:
            self.durationStr = f'{self.date_to_delta(start_date) + 10} S'
        while True:
            log.debug(f'Requesting bars for {self.contract.localSymbol}')
            self.bars = self.get_bars()
            if self.bars is not None:
                break
            else:
                util.sleep(5)
        log.debug(f'Bars received for {self.contract.localSymbol}')
        self.subscribe()

    @staticmethod
    def date_to_delta(date: datetime) -> int:
        return (datetime.now() - date).seconds

    def get_bars(self) -> BarDataList:
        log.debug(f'reqHistoricalData params for {self.contract.localSymbol} '
                  f'durationStr: {self.durationStr}')
        return self.ib.reqHistoricalData(
            self.contract,
            endDateTime='',
            durationStr=self.durationStr,
            barSizeSetting=self.barSizeSetting,
            whatToShow=self.whatToShow,
            useRTH=self.useRTH,
            formatDate=1,
            keepUpToDate=True)

    def subscribe(self) -> None:
        self.bars.updateEvent += self.onNewBar

    def onNewBar(self, bars: BarData, hasNewBar: bool) -> None:
        if hasNewBar:
            self.aggregate(bars[-2])

    @abstractmethod
    def aggregate(self, bar: BarData):
        pass


class StreamAggregator(BarStreamer):

    def __init__(self) -> None:
        self._createEvents()
        self.buffer = deque()
        self.new_bars = []
        super().__init__()

    def _createEvents(self) -> None:
        self.newCandle = Event('newCandle')

    def __call__(self, ib: IB, contract: Contract,
                 date: Optional[datetime] = None) -> None:
        super().__call__(ib, contract, date)
        self.process_back_data(date)

    def process_back_data(self, date: Optional[datetime] = None) -> None:
        # flag needed on re-connect
        self.backfill = True
        log.debug(f'generating startup data for {self.contract.localSymbol}')
        for counter, bar in enumerate(self.bars[:-2]):
            # date given on reconnect only
            if (date and bar.date >= date) or not date:
                self.aggregate(bar)
                # prevent from blocking too long
                if counter % 10000 == 0:
                    log.debug(f'releasing control {self.contract.localSymbol}')
                    util.sleep(0)
        log.debug(f'startup data generated for {self.contract.localSymbol}')
        self.backfill = False
        self.clear_buffer()

    def create_candle(self) -> None:
        df = util.df(self.new_bars)
        self.new_bars = []
        df.date = df.date.astype('datetime64')
        df.set_index('date', inplace=True)
        # df['backfill'] = True
        # df['volume_weighted'] = (df.close + df.open)/2 * df.volume
        # df['volume_weighted'] = df.close * df.volume
        # df['volume_weighted'] = df.average * df.volume
        # weighted_price = df.volume_weighted.sum() / df.volume.sum()
        self.newCandle.emit({'backfill': self.backfill,
                             'date': df.index[-1],
                             'open': df.open[0],
                             'high': df.high.max(),
                             'low': df.low.min(),
                             'close': df.close[-1],
                             # 'price': weighted_price,
                             'price': df.close[-1],
                             'volume': df.volume.sum()})

    def onNewBar(self, bars: BarData, hasNewBar: bool) -> None:
        if hasNewBar:
            if self.backfill:
                log.debug(f'buffering bar for {self.contract.localSymbol}')
                self.buffer.append(bars[-2])
            else:
                self.clear_buffer()
                self.aggregate(bars[-2])

    def clear_buffer(self) -> None:
        """Utilize bars that have been buffered during processing back data"""
        while self.buffer:
            log.debug(f'clearing buffer for {self.contract.localSymbol}')
            self.aggregate(self.buffer.popleft())


class VolumeStreamer(StreamAggregator):

    def __init__(self, volume: Optional[float] = None,
                 avg_periods: Optional[float] = None) -> None:
        super().__init__()
        self.volume = volume
        self.avg_periods = avg_periods
        self.aggregator = 0

    def __call__(self, ib: IB, contract: Contract,
                 date: Optional[datetime] = None) -> None:
        BarStreamer.__call__(self, ib, contract, date)
        self.volume = self.volume or self.reset_volume(self.avg_periods)
        StreamAggregator.process_back_data(self)

    def reset_volume(self, avg_periods) -> int:
        if self.bars:
            return util.df(self.bars).volume \
                .rolling(avg_periods).sum() \
                .mean().round()
        else:
            raise ValueError('No bars in VolumeStreamer')

    def aggregate(self, bar) -> None:
        self.new_bars.append(bar)
        self.aggregator += bar.volume
        if not self.backfill:
            message = (f'{bar.date} {self.aggregator}/{self.volume}'
                       f' {self.contract.localSymbol}')
            log.debug(message)
        if self.aggregator >= self.volume:
            self.aggregator = 0
            # self.aggregator -= self.volume
            self.create_candle()


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
