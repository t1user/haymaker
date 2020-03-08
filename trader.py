from __future__ import annotations

from collections import defaultdict
from functools import partial
from typing import List, Type
from abc import ABC, abstractmethod


import pandas as pd
import numpy as np

from eventkit import Event
from ib_insync import util, Order, MarketOrder, StopOrder, IB
from ib_insync.objects import Fill, CommissionReport
from ib_insync.order import Trade
from ib_insync.contract import ContFuture, Future
from logbook import Logger

from indicators import get_ATR, get_signals
from objects import Params
from blotter import Blotter


log = Logger(__name__)


class BarStreamer:

    def __init__(self):
        log.debug(f'Requesting bars for {self.contract.localSymbol}')
        self.bars = self.get_bars(self.contract)
        log.debug(f'Bars received for {self.contract.localSymbol}')
        self.new_bars = []

    def get_bars(self, contract):
        return self.ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr='10 D',
            barSizeSetting='30 secs',
            whatToShow='TRADES',
            useRTH=False,
            formatDate=1,
            keepUpToDate=True)

    def subscribe(self):
        def onNewBar(bars, hasNewBar):
            if hasNewBar:
                self.aggregate(bars[-2])
        self.bars.updateEvent += onNewBar

    def aggregate(self):
        raise NotImplementedError


class StreamAggregator(BarStreamer, ABC):
    def __init__(self, ib, **kwargs):
        self.__dict__.update(kwargs)
        self.ib = ib
        self._createEvents()
        super().__init__()
        # self.process_back_data()
        self.subscribe()

    def _createEvents(self):
        self.newCandle = Event('newCandle')

    def process_back_data(self):
        self.backfill = True
        log.debug(f'generating startup data for {self.contract.localSymbol}')
        for counter, bar in enumerate(self.bars[:-2]):
            self.aggregate(bar)
            if counter % 10000 == 0:
                log.debug(
                    f'startup generator for {self.contract.localSymbol} giving up control')
                self.ib.sleep(0)
        log.debug(f'startup data generated for {self.contract.localSymbol}')
        self.backfill = False

    def create_candle(self):
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

    @abstractmethod
    def aggregate(self):
        raise NotImplementedError


class VolumeStreamer(StreamAggregator):

    def __init__(self, ib, **kwargs):
        super().__init__(ib, **kwargs)
        self.aggregator = 0
        if not self.volume:
            self.volume = self.reset_volume()

    def reset_volume(self):
        return util.df(self.bars).volume \
            .rolling(self.avg_periods).sum() \
            .mean().round()

    def aggregate(self, bar):
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

    def append(self, candle):
        raise NotImplementedError


class ResampledStreamer(StreamAggregator):

    def __init__(self, ib, **kwargs):
        super().__init__(ib, **kwargs)
        self.counter = 0

    def aggregate(self, bar):
        self.new_bars.append(bar)
        self.counter += 1
        if self.counter == self.avg_periods:
            self.create_candle()
            self.counter = 0


class Candle():

    def __init__(self, params: Params, streamer: Type[BarStreamer],
                 trader: Type[Trader], portfolio: Type[Portfolio],
                 ib: Type[IB], freeze_path: str, keep_ref: bool = True):
        self.__dict__.update(params.__dict__)
        self.params = params
        self.ib = ib
        self.portfolio = portfolio
        self.path = freeze_path
        log.debug(f'candle init for contract {self.contract.localSymbol}')
        self.candles = []
        self.params.details = self.get_details()
        trader.register(self.params, self.contract.symbol)
        self._createEvents()
        streamer = streamer(self.ib, **params.__dict__)
        streamer.newCandle.connect(
            self.append, keep_ref=keep_ref)
        self.entrySignal.connect(trader.onEntry, keep_ref=keep_ref)
        self.closeSignal.connect(trader.onClose, keep_ref=keep_ref)
        streamer.process_back_data()
        self.get_indicators()
        self.freeze()

    def _createEvents(self):
        self.entrySignal = Event('entrySignal')
        self.closeSignal = Event('closeSignal')

    def freeze(self):
        self.df.to_pickle(
            f'{self.path}/freeze_df_{self.contract.localSymbol}.pickle')
        log.debug(f'freezed data saved for {self.contract.localSymbol}')

    def get_details(self):
        return self.ib.reqContractDetails(self.contract)[0]

    """
    def get_trading_hours(self):
        return ib.reqContractDetails(self.contract)[0].tradingHours.split(';')
    """

    def append(self, candle):
        self.candles.append(candle)
        if not candle['backfill']:
            self.get_indicators()
            assert not self.df.iloc[-1].isna().any(), (
                f'Not enough data for indicators for instrument'
                f' {self.contract.localSymbol} '
                f'values: {self.df.iloc[-1].to_dict()}')
            self.process()

    def get_indicators(self):
        df = pd.DataFrame(self.candles)
        df.set_index('date', inplace=True)
        df['ema_fast'] = df.price.ewm(
            span=self.ema_fast, min_periods=int(self.ema_fast*.8)).mean()
        df['ema_slow'] = df.price.ewm(
            span=self.ema_slow, min_periods=int(self.ema_slow*.8)).mean()
        df['atr'] = get_ATR(df, self.atr_periods)
        df['signal'] = get_signals(df.price, self.periods)
        df['filter'] = np.sign(df['ema_fast'] - df['ema_slow'])
        df['filtered_signal'] = df['signal'] * \
            ((df['signal'] * df['filter']) == 1)
        self.df = df

    def process(self):
        message = (f'New candle for {self.contract.localSymbol} '
                   f'{self.df.iloc[-1].to_dict()}')
        log.debug(message)
        position = self.portfolio.positions.get(self.contract)
        if self.df.filtered_signal[-1] and not position:
            message = (f'entry signal emitted for {self.contract.localSymbol},'
                       f' signal: {self.df.filtered_signal[-1]},'
                       f' atr: {self.df.atr[-1]}')
            log.debug(message)
            number_of_contracts = self.portfolio.number_of_contracts(
                self.params, self.df.price[-1])
            if number_of_contracts:
                self.entrySignal.emit(
                    self.contract, self.df.signal[-1], self.df.atr[-1],
                    number_of_contracts)
            else:
                message = (f'Not enough equity to open position for: '
                           f'{self.contract.localSymbol}')
                log.warning(message)
        elif self.df.signal[-1] and position:
            if position * self.df.signal[-1] < 0:
                log.debug(
                    f'close signal emitted for {self.contract.localSymbol}')
                self.closeSignal.emit(self.contract, self.df.signal[-1])


class Manager:

    def __init__(self, ib: IB, contracts: List[Params],
                 streamer: BarStreamer, leverage: int,
                 blotter: Blotter = None, trailing: bool = True,
                 freeze_path: str = 'notebooks/freeze/live'):
        if blotter is None:
            blotter = Blotter()
        self.ib = ib
        self.contracts = contracts
        self.streamer = streamer
        self.portfolio = Portfolio(ib, leverage)
        self.path = freeze_path
        self.trader = Trader(ib, self.portfolio, blotter, trailing)
        alloc = round(sum([c.alloc for c in contracts]), 5)
        assert alloc == 1, "Portfolio allocations don't add-up to 1"
        log.debug(f'manager object initiated: {self}')

    def onConnected(self):
        self.trader.reconcile_stops()
        contracts = get_contracts(self.contracts, self.ib)
        log.debug(f'initializing candles')
        self.candles = [Candle(contract, self.streamer, self.trader,
                               self.portfolio, self.ib, self.path)
                        for contract in contracts]
        log.debug(f'onConnected run, candles: {self.candles}')

    def freeze(self):
        for candle in self.candles:
            candle.freeze()


class Portfolio:

    def __init__(self, ib, leverage):
        self.leverage = leverage
        self.ib = ib
        self.values = {}

    @property
    def account_value(self):
        self.update_value()
        return self.values['TotalCashBalance'] + min(
            self.values['UnrealizedPnL'], 0)

    @property
    def positions(self):
        positions = self.ib.positions()
        return {p.contract: p.position for p in positions}

    def update_value(self):
        tags = self.ib.accountValues()
        for item in tags:
            if item.currency == 'USD':
                try:
                    self.values[item.tag] = float(item.value)
                except ValueError:
                    pass

    def number_of_contracts(self, params, price):
        # self.account_value
        return int((1e+5 * self.leverage *
                    params.alloc) / (float(params.contract.multiplier) *
                                     price))


class Trader:

    def __init__(self, ib, portfolio, blotter, trailing=True):
        self.ib = ib
        self.portfolio = portfolio
        self.blotter = blotter
        self.trailing = trailing
        self.atr_dict = {}
        self.contracts = {}
        log.debug('Trader initialized')

    def onEntry(self, contract, signal, atr, amount) -> None:
        log.debug(
            f'entry signal handled for: {contract.localSymbol} {signal} {atr}')
        self.atr_dict[contract.symbol] = atr
        trade = self.trade(contract, signal, amount)
        trade.filledEvent += self.attach_sl
        self.attach_events(trade, 'ENTRY')

    def onClose(self, contract, signal) -> None:
        message = (f'close signal handled for: {contract.localSymbol}'
                   f' signal: {signal}')
        log.debug(message)
        if contract in self.portfolio.positions:
            self.remove_sl(contract)
            # make sure sl didn't close the position before being removed
            if contract in self.portfolio.positions:
                trade = self.trade(contract, signal,
                                   abs(self.portfolio.positions[contract]))
                self.attach_events(trade, 'CLOSE')

    def trade(self, contract, signal, amount) -> Trade:
        direction = {1: 'BUY', -1: 'SELL'}
        order = MarketOrder(direction[signal], amount)
        message = (f'entering {direction[signal]} order for {amount} '
                   f'{contract.localSymbol}')
        log.debug(message)
        return self.ib.placeOrder(contract, order)

    def attach_sl(self, trade: Trade) -> None:
        contract = trade.contract
        action = trade.order.action
        assert action in ('BUY', 'SELL')
        reverseAction = 'BUY' if action == 'SELL' else 'SELL'
        direction = 1 if reverseAction == 'BUY' else -1
        amount = trade.orderStatus.filled
        price = trade.orderStatus.avgFillPrice
        sl_points = self.atr_dict[contract.symbol]
        if not self.trailing:
            sl_price = self.round_tick(
                price + sl_points * direction *
                self.contracts[contract.symbol].sl_atr,
                self.contracts[contract.symbol].details.minTick)
            log.info(f'STOP LOSS PRICE: {sl_price}')
            order = StopOrder(reverseAction, amount, sl_price,
                              outsideRth=True, tif='GTC')
        else:
            distance = self.round_tick(
                sl_points * self.contracts[contract.symbol].sl_atr,
                self.contracts[contract.symbol].details.minTick)
            log.info(f'TRAILING STOP LOSS DISTANCE: {distance}')
            order = Order(orderType='TRAIL', action=reverseAction,
                          totalQuantity=amount, auxPrice=distance,
                          outsideRth=True, tif='GTC')
        trade = self.ib.placeOrder(contract, order)
        log.debug(f'stop loss attached for {trade.contract.localSymbol}')
        self.attach_events(trade, 'STOP-LOSS')

    def remove_sl(self, contract) -> None:
        open_trades = self.ib.openTrades()
        orders = defaultdict(list)
        for t in open_trades:
            orders[t.contract].append(t.order)
        for order in orders[contract]:
            if order.orderType in ('STP', 'TRAIL'):
                self.ib.cancelOrder(order)
                log.debug(f'stop loss removed for {contract.localSymbol}')

    def reconcile_stops(self) -> None:
        """
        To be executed on restart. For all existing stop-outs attach reporting
        events for the blotter.
        """
        # required to fill open trades list
        # self.ib.reqAllOpenOrders()
        trades = self.ib.openTrades()
        log.debug(f'open trades on re-connect: {trades}')
        for trade in trades:
            if trade.order.orderType in ('STP', 'TRAIL'
                                         ) and trade.orderStatus.remaining != 0:
                self.attach_events(trade, 'STOP-LOSS')

    @staticmethod
    def round_tick(price: float, tick_size: float) -> float:
        floor = price // tick_size
        remainder = price % tick_size
        if remainder > (tick_size / 2):
            floor += 1
        return round(floor * tick_size, 4)

    def register(self, params: Params, symbol: str):
        self.contracts[symbol] = params

    def attach_events(self, trade: Trade, reason: str) -> None:
        report_trade = partial(self.report_trade, reason)
        report_commission = partial(self.report_commission, reason)
        trade.filledEvent += report_trade
        trade.cancelledEvent += self.report_cancel
        trade.commissionReportEvent += report_commission
        log.debug(f'Reporting events attached for {trade.contract.localSymbol} '
                  f'{trade.order.action} {trade.order.totalQuantity} '
                  f'{trade.order.orderType}')

    def report_trade(self, reason: str, trade: Trade) -> None:
        message = (f'{reason} trade filled: {trade.contract.localSymbol} '
                   f'{trade.order.action} {trade.orderStatus.filled}'
                   f'@{trade.orderStatus.avgFillPrice}')
        log.info(message)

    def report_cancel(self, trade: Trade) -> None:
        message = (f'{trade.order.orderType} order {trade.order.action} '
                   f'{trade.orderStatus.remaining} (of '
                   f'{trade.order.totalQuantity}) for '
                   f'{trade.contract.localSymbol} cancelled')
        log.info(message)

    def report_commission(self, reason: str, trade: Trade, fill: Fill,
                          report: CommissionReport) -> None:
        log.info(f'sending commission report {report}')
        self.blotter.log_commission(trade, fill, report, reason)


def get_contracts(params: List[Params], ib: IB) -> List[Params]:

    def convert(contract_tuples: List[tuple]):
        cont_contracts = [ContFuture(*contract)
                          for contract in contract_tuples]
        ib.qualifyContracts(*cont_contracts)
        # log.debug(f'Contracts qualified: {cont_contracts}')

        # Converting to Futures potentially unnecessary
        # But removing below likely breaks the backtester
        # TODO
        ids = [contract.conId for contract in cont_contracts]
        contracts = [Future(conId=id) for id in ids]
        ib.qualifyContracts(*contracts)
        log.debug(f'Contracts qualified: {contracts}')
        return contracts

    if type(params[0].contract) not in (Future, ContFuture):
        contracts = [param.contract for param in params]
        for param, contract in zip(params, convert(contracts)):
            param.contract = contract

    return params
