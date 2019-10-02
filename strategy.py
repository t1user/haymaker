import pandas as pd
import numpy as np
from pprint import pprint
from collections import namedtuple, defaultdict

from ib_insync import IB, util, MarketOrder, StopOrder
from ib_insync.contract import ContFuture, Future
from eventkit import Event
from eventkit.ops.op import Op

from indicators import (get_ATR, get_signals)
from trader import Consolidator, BaseStrategy


ib = IB()
ib.connect('127.0.0.1', 4002, clientId=0)


contract = ContFuture('NQ', 'GLOBEX')
ib.qualifyContracts(contract)
id = contract.conId
contract = Future(conId=id)
ib.qualifyContracts(contract)
print(contract)

print('starting to load bars')

bars = ib.reqHistoricalData(
    contract,
    endDateTime='',
    durationStr='2 D',
    barSizeSetting='30 secs',
    whatToShow='TRADES',
    useRTH=False,
    formatDate=1,
    keepUpToDate=True)

# volume = util.df(bars).volume.rolling(3).sum().mean().round()
# print(f'volume: {volume}')


class VolumeCandle(Consolidator):

    def __init__(self, bars, avg_periods=30, span=5500):
        self.span = span
        self.avg_periods = avg_periods
        self.bars = bars
        self.aggregator = 0
        print(len(self.bars))
        self.reset_volume()
        # self.minTick = ib.reqContractDetails[0].minTick
        super().__init__(bars)

    @classmethod
    def create(cls, contract):
        cls.contract = contract
        return cls

    def reset_volume(self):
        self.volume = util.df(self.bars).volume \
            .rolling(self.avg_periods).sum() \
            .ewm(span=self.span).mean().iloc[-1].round()
        """
        self.volume = util.df(self.bars).volume \
            .rolling(self.avg_periods).sum() \
            .mean().round()
        """
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
        df['backfill'] = True
        df['volume_weighted'] = (df.close + df.open)/2 * df.volume
        weighted_price = df.volume_weighted.sum() / df.volume.sum()
        self.newCandle.emit({'backfill': self.backfill,
                             'date': df.index[-1],
                             'open': df.open[0],
                             'high': df.high.max(),
                             'low': df.low.min(),
                             'close': df.close[-1],
                             'price': weighted_price})
        print('newCandle emitted')


class Candle():

    periods = [10, 20, 40, 80, 160, ]
    ema_fast = 80  # number of periods for moving average filter
    sl_atr = 5  # stop loss in ATRs
    atr_periods = 80  # number of periods to calculate ATR on
    time_int = 30  # interval in minutes to be used to define volume candle

    def __init__(self):
        print('candle init')
        print('contract: ', self.contract)
        self.candles = []
        self.counter = 0

    @classmethod
    def create(cls, contract):
        cls.contract = contract
        return cls()

    def append(self, candle):
        self.candles.append(candle)
        self.get_indicators()

    def get_indicators(self):
        self.df = pd.DataFrame(self.candles)
        self.df.set_index('date', inplace=True)
        self.df['ema_fast'] = self.df.price.ewm(span=self.ema_fast).mean()
        self.df['atr'] = get_ATR(self.df, self.atr_periods)
        self.df['signal'] = get_signals(self.df.price, self.periods)
        print('>>>>>>>>>positions>>>>>>>>>')
        print(ib.positions())
        print('>>>>>>>>>orders>>>>>>>>>')
        print('number of orders: ', len(ib.orders()))
        print('trades: ', ib.openTrades())
        print('>>>>>>>>>last row<<<<<<<<<')
        print(self.df.iloc[-1])
        # signal(1)
        self.process()

    def process(self):
        if self.df.backfill[-1]:
            return
        if self.df.signal[-1]:
            if self.df.signal[-1] * (self.df.price[-1] - self.df.ema_fast[-1]) > 0:
                signal.entry(
                    self.contract, self.df.signal[-1], self.df.atr[-1])
            elif self.df.signal[-1]:
                signal.breakout(self.contract, self.df.signal[-1])


class SignalProcessor:
    def __init__(self):
        self._createEvents()

    def positions(self):
        positions = ib.positions()
        return {p.contract: p.position for p in positions}

    def _createEvents(self):
        self.entrySignal = Event('entrySignal')
        self.closeSignal = Event('closeSignal')

    def entry(self, contract, signal, atr):
        positions = self.positions()
        if not positions.get(contract):
            print('entry signal emitted')
            print(f'contract: {contract}, signal: {signal}, atr: {atr}')
            self.entrySignal.emit(contract, signal, atr)
        else:
            self.breakout(contract, signal)

    def breakout(self, contract, signal):
        positions = self.positions()
        if positions.get(contract):
            if positions.get(contract) * signal < 0:
                print('close signal emitted')
                self.closeSignal.emit(contract, signal)


class Trader:

    def __init__(self):
        self.atr_dict = {}

    def onEntry(self, contract, signal, atr):
        print('entry singnal handled for: ', contract, signal, atr)
        self.atr_dict[contract] = atr
        trade = self.trade(contract, signal)
        trade.filledEvent += self.attach_sl
        print('entry order placed')

    def onClose(self, contract, signal):
        print('close signal handled for: ', contract, signal)
        positions = {p.contract: p.position for p in ib.positions()}
        if contract in positions:
            trade = self.trade(contract, signal)
            trade.filledEvent += self.remove_sl
            print('closing order placed')

    def trade(self, contract, signal):
        if signal == 1:
            print('entering buy order')
            order = MarketOrder('BUY', 1)
        elif signal == -1:
            print('entering sell order')
            order = MarketOrder('Sell', 1)
        trade = ib.placeOrder(contract, order)
        return trade

    def attach_sl(self, trade):
        side = {'BUY': 'SELL', 'SELL': 'BUY'}
        direction = {'BUY': -1, 'SELL': 1}
        contract = trade.contract
        action = trade.order.action
        amount = trade.orderStatus.filled
        price = trade.orderStatus.avgFillPrice
        print('TRADE PRICE: ', price)
        sl_points = self.atr_dict[contract]
        # TODO round to the nearest tick
        sl_price = round(price + sl_points * direction[action])
        print('STOP LOSS PRICE: ', sl_price)
        order = StopOrder(side[action], amount, sl_price,
                          outsideRth=True, tif='GTC')
        ib.placeOrder(contract, order)
        print('stop loss attached')

    def remove_sl(self, trade):
        contract = trade.contract
        open_trades = ib.openTrades()
        # open_orders = ib.reqAllOpenOrders()
        orders = defaultdict(list)
        for t in open_trades:
            orders[t.contract].append(t.order)
        for order in orders[contract]:
            if order.orderType == 'STP':
                ib.cancelOrder(order)
                print('stop loss removed')


candle = Candle.create(contract)
c = VolumeCandle.create(contract)(bars)
c += candle.append


signal = SignalProcessor()
t = Trader()

signal.entrySignal += t.onEntry
signal.closeSignal += t.onClose


class Strategy(BaseStrategy):
    pass


strategy = Strategy(ib)

util.patchAsyncio()
ib.run()
