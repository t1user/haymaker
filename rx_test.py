import pandas as pd

from ib_insync import IB, util
from ib_insync.contract import ContFuture
import eventkit as ev

from indicators import (get_ATR, get_signals)
from trader import Consolidator


ib = IB()
ib.connect('127.0.0.1', 4002, clientId=20)


contract = ContFuture('NQ', 'GLOBEX')
ib.qualifyContracts(contract)

print('starting to load bars')

bars = ib.reqHistoricalData(
    contract,
    endDateTime='',
    durationStr='180 S',
    barSizeSetting='5 secs',
    whatToShow='TRADES',
    useRTH=False,
    formatDate=1,
    keepUpToDate=True)

volume = util.df(bars).volume.rolling(3).sum().mean().round()
print(f'volume: {volume}')


class VolumeCandle(Consolidator):

    def __init__(self, volume, bars):
        self.aggregator = 0
        super().__init__(volume, bars)

    def aggregate(self, bar):
        self.aggregator += bar.volume
        print(bar.date, end=' ')
        print('current volume: ', self.aggregator)
        if self.aggregator >= self.volume:
            self.aggregator -= self.volume
            self.create_candle()

    def create_candle(self):
        df = util.df(self.new_bars)
        self.new_bars = []
        df.date = df.date.astype('datetime64')
        df.set_index('date', inplace=True)
        df['volume_weighted'] = (df.close + df.open)/2 * df.volume
        weighted_price = df.volume_weighted.sum() / df.volume.sum()
        self.newCandle.emit({'date': df.index[-1],
                             'open': df.open[0],
                             'high': df.high.max(),
                             'low': df.low.min(),
                             'close': df.close[-1],
                             'price': weighted_price})


class Candle():

    periods = [10, 20, 40, 80, 160, ]
    ema_fast = 80  # number of periods for moving average filter
    sl_atr = 1  # stop loss in ATRs
    atr_periods = 80  # number of periods to calculate ATR on
    time_int = 30  # interval in minutes to be used to define volume candle

    def __init__(self):
        self.candles = []

    def append(self, candle):
        self.candles.append(candle)
        self.get_indicators()

    def get_indicators(self):
        self.df = pd.DataFrame(self.candles)
        self.df.set_index('date', inplace=True)
        self.df['ema_fast'] = self.df.price.ewm(span=self.ema_fast).mean()
        self.df['atr'] = get_ATR(self.df, self.atr_periods)
        self.df['signal'] = get_signals(self.df.price, self.periods)
        print(self.df)
        signal(self.df.iloc[-1])


#c = Consolidator(volume)
candle = Candle()
new_bar = ev.Event()
candle_event = ev.Event()
signal = ev.Event()


print('bars loaded')
print(util.df(bars))

# bars.updateEvent.pipe(onBarUpdate).pipe(c.aggregate).pipe(candle.append)

#start = ev.Event()
#pipe = start.pipe(Consolidator(volume)).pipe(Candle)
#bars.updateEvent += start

#new_bar += c.aggregate
#c += candle.append

c = VolumeCandle(volume, bars)
c += candle.append


signal += print
ib.run()
