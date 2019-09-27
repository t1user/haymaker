import pandas as pd
from pprint import pprint
from collections import namedtuple

from ib_insync import IB, util, MarketOrder, StopOrder
from ib_insync.contract import ContFuture
from eventkit import Event
from eventkit.ops.op import Op

from indicators import (get_ATR, get_signals)
from trader import Consolidator, BaseStrategy


ib = IB()
ib.connect('127.0.0.1', 4002, clientId=20)


contract = ContFuture('NQ', 'GLOBEX')
ib.qualifyContracts(contract)
print(contract)

print('starting to load bars')

bars = ib.reqHistoricalData(
    contract,
    endDateTime='',
    durationStr='410 S',
    barSizeSetting='5 secs',
    whatToShow='TRADES',
    useRTH=False,
    formatDate=1,
    keepUpToDate=True)

# volume = util.df(bars).volume.rolling(3).sum().mean().round()
# print(f'volume: {volume}')


class VolumeCandle(Consolidator):

    def __init__(self, bars, avg_periods=3, span=80):
        self.span = span
        self.avg_periods = avg_periods
        self.bars = bars
        self.aggregator = 0
        self.reset_volume()
        super().__init__(bars)

    @classmethod
    def create(cls, contract):
        cls.contract = contract
        return cls

    def reset_volume(self):
        self.volume = util.df(self.bars).volume \
            .rolling(self.avg_periods).sum() \
            .ewm(span=self.span).mean().iloc[-1].round()
        print(f'volume: {self.volume}')

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
        print('newCandle emitted')


class Candle():

    periods = [10, 20, 40, 80, 160, ]
    ema_fast = 80  # number of periods for moving average filter
    sl_atr = 1  # stop loss in ATRs
    atr_periods = 80  # number of periods to calculate ATR on
    time_int = 30  # interval in minutes to be used to define volume candle

    def __init__(self):
        print('candle init')
        self.candles = []
        self.counter = 0

    @classmethod
    def create(cls, contract):
        cls.contract = contract
        return cls()

    def append(self, candle):
        self.candles.append(candle)
        print('current contract: ', self.contract)
        self.get_indicators()

    def get_indicators(self):
        self.df = pd.DataFrame(self.candles)
        self.df.set_index('date', inplace=True)
        self.df['ema_fast'] = self.df.price.ewm(span=self.ema_fast).mean()
        self.df['atr'] = get_ATR(self.df, self.atr_periods)
        self.df['signal'] = get_signals(self.df.price, self.periods)
        # print(self.df.iloc[-1])
        # self.process()
        print('candle emitting event signal')
        print('>>>>>>>>>positions>>>>>>>>>')
        print(ib.positions())
        print('>>>>>>>>>orders>>>>>>>>>')
        print(ib.orders())
        signal(1)

    def process(self):
        if self.df.signal[-1]:
            if self.df.signal[-1] * (self.df.price[-1] - self.df.ema_fast[-1]):
                signal.entry(self.contract, self.df.signal[1], self.df.atr)
            elif self.df.signal[-1]:
                signal.breakout(self.contract)


class SignalProcessor:
    def __init__(self):
        self._createEvents()

    def positions(self):
        positions = ib.reqPositions()
        return {p.contract: p.position for p in positions}

    def _createEvents(self):
        self.entrySignal = Event('entrySignal')
        self.closeSignal = Event('closeSignal')

    def entry(contract, signal, atr):
        positions = self.positions()
        if contract in positions:
            pass


class Trader:

    def __init__(self):
        pass

    def onSignal(self, signal):
        print(signal)
        if signal == 1:
            order = MarketOrder('BUY', 1)
        elif signal == -1:
            order = MarketOrder('Sell', 1)
        trade = ib.placeOrder(contract, order)
        trade.filledEvent += self.attach_sl
        print('order placed')

    def onExit(self, signal):
        pass

    def attach_sl(self, trade):
        side = {'BUY': 'SELL', 'SELL': 'BUY'}
        direction = {'BUY': -1, 'SELL': 1}
        contract = trade.contract
        action = trade.order.action
        amount = trade.orderStatus.filled
        price = trade.orderStatus.avgFillPrice
        print('TRADE PRICE: ', price)
        sl_points = 20
        sl_price = price + sl_points * direction[action]
        print('STOP LOSS PRICE: ', sl_price)
        order = StopOrder(side[action], amount, sl_price,
                          outsideRth=True, tif='GTC')
        ib.placeOrder(contract, order)
        print('stop loss attached')

    def remove_sl(self, trade):
        pass


signal = Event()

c = VolumeCandle.create(contract)(bars)
candle = Candle.create(contract)
t = Trader()


# print(util.df(bars))
print('bars loaded')


c += candle.append
signal += t.onSignal
# signal += print


class Strategy(BaseStrategy):
    pass


strategy = Strategy(ib)

# util.patchAsyncio()
ib.run()
