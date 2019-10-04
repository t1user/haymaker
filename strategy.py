import sys
from pprint import pprint
from collections import namedtuple, defaultdict
from datetime import datetime

import pandas as pd
import numpy as np

from ib_insync import IB, util, MarketOrder, StopOrder
from ib_insync.contract import ContFuture, Future
from eventkit import Event
from eventkit.ops.op import Op
from logbook import Logger, StreamHandler, FileHandler, set_datetime_format

from indicators import (get_ATR, get_signals)
from trader import Consolidator, BaseStrategy


ib = IB()
ib.connect('127.0.0.1', 4002, clientId=0)


# volume = util.df(bars).volume.rolling(3).sum().mean().round()
# print(f'volume: {volume}')


class VolumeCandle(Consolidator):

    def __init__(self, bars, avg_periods=30, span=5500):
        self.span = span
        self.avg_periods = avg_periods
        self.bars = bars
        self.aggregator = 0
        self.reset_volume()
        # self.minTick = ib.reqContractDetails[0].minTick
        super().__init__(bars)

    @classmethod
    def create(cls, contract):
        cls.contract = contract
        return cls

    def reset_volume(self):
        """
        self.volume = util.df(self.bars).volume \
            .rolling(self.avg_periods).sum() \
            .ewm(span=self.span).mean().iloc[-1].round()
        """

        self.volume = util.df(self.bars).volume \
            .rolling(self.avg_periods).sum() \
            .mean().round()

        log.debug(f'volume: {self.volume} {self.contract.localSymbol}')

    def aggregate(self, bar):
        self.aggregator += bar.volume
        log.debug(
            f'{bar.date} current volume: {self.aggregator} {self.contract.localSymbol}')
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
        log.debug('newCandle emitted')


class Candle():

    periods = [5, 10, 20, 40, 80, ]  # 160, ]
    ema_fast = 120  # number of periods for moving average filter
    sl_atr = 1  # stop loss in ATRs
    atr_periods = 80  # number of periods to calculate ATR on
    time_int = 30  # interval in minutes to be used to define volume candle

    def __init__(self):
        log.debug('candle init')
        log.debug(f'contract: {self.contract}')
        self.candles = []
        self.counter = 0

    def freeze(self):
        if self.counter == 0:
            self.df.to_pickle('notebooks/freeze_df.pickle')
            log.debug('freezed data saved')
            self.counter += 1

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
        log.debug('>>>>>>>>>positions>>>>>>>>>')
        log.debug(ib.positions())
        log.debug('>>>>>>>>>orders>>>>>>>>>')
        log.debug(f'number of orders: {len(ib.orders())}')
        log.debug(f'trades: {ib.openTrades()}')
        log.debug('>>>>>>>>>last row<<<<<<<<<')
        log.debug(self.df.iloc[-1])
        # signal(1)
        self.process()

    def process(self):
        if self.df.backfill[-1]:
            return
        else:
            self.freeze()
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
            log.debug('entry signal emitted')
            log.debug(f'contract: {contract}, signal: {signal}, atr: {atr}')
            self.entrySignal.emit(contract, signal, atr)
        else:
            self.breakout(contract, signal)

    def breakout(self, contract, signal):
        positions = self.positions()
        if positions.get(contract):
            if positions.get(contract) * signal < 0:
                log.debug('close signal emitted')
                self.closeSignal.emit(contract, signal)


class Trader:

    def __init__(self):
        self.atr_dict = {}

    def onEntry(self, contract, signal, atr):
        log.debug(f'entry singnal handled for: {contract} {signal} {atr}')
        self.atr_dict[contract] = atr
        trade = self.trade(contract, signal)
        trade.filledEvent += self.attach_sl
        log.debug('entry order placed')

    def onClose(self, contract, signal):
        log.debug(f'close signal handled for: {contract} {signal}')
        positions = {p.contract: p.position for p in ib.positions()}
        if contract in positions:
            trade = self.trade(contract, signal)
            trade.filledEvent += self.remove_sl
            log.debug('closing order placed')

    def trade(self, contract, signal):
        if signal == 1:
            log.debug('entering buy order')
            order = MarketOrder('BUY', 1)
        elif signal == -1:
            log.debug('entering sell order')
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
        log.debug(f'TRADE PRICE: {price}')
        sl_points = self.atr_dict[contract]
        # TODO round to the nearest tick
        sl_price = round(price + sl_points * direction[action])
        log.debug(f'STOP LOSS PRICE: {sl_price}')
        order = StopOrder(side[action], amount, sl_price,
                          outsideRth=True, tif='GTC')
        ib.placeOrder(contract, order)
        log.debug('stop loss attached')

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
                log.debug('stop loss removed')


contracts = [
    ('NQ', 'GLOBEX'),
    ('ES', 'GLOBEX'),
    ('NKD', 'GLOBEX'),
    ('CL', 'NYMEX'),
    ('GC', 'NYMEX'),
]


def get_contract(*args):
    contract = ContFuture(*args)
    ib.qualifyContracts(contract)
    id = contract.conId
    contract = Future(conId=id)
    ib.qualifyContracts(contract)
    log.debug(f'Contract qualified: {contract}')
    return contract


# logging
set_datetime_format('local')
StreamHandler(sys.stdout, bubble=True).push_application()
FileHandler(
    f'logs/{__file__[:-3]}_{datetime.today().strftime("%Y-%m-%d_%H-%M")}',
    bubble=True, delay=True).push_application()
log = Logger(__name__)


# con = [get_contract(contract) for contract in contracts]
contract = get_contract('NQ', 'GLOBEX')

bars = ib.reqHistoricalData(
    contract,
    endDateTime='',
    durationStr='4 D',
    barSizeSetting='30 secs',
    whatToShow='TRADES',
    useRTH=False,
    formatDate=1,
    keepUpToDate=True)

candle = Candle.create(contract)
c = VolumeCandle.create(contract)(bars)
c += candle.append


# candles = [initialize_contract(c) for c in contracts]

signal = SignalProcessor()
t = Trader()

signal.entrySignal += t.onEntry
signal.closeSignal += t.onClose


class Strategy(BaseStrategy):
    pass


strategy = Strategy(ib)


util.patchAsyncio()
ib.run()
