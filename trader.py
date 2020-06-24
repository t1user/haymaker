from collections import defaultdict
from functools import partial
from typing import List, Dict, Any
from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd
from ib_insync import (Order, MarketOrder, StopOrder, IB, Event, Fill,
                       CommissionReport, Trade, ContFuture, Future, Contract)

from objects import Params
from blotter import Blotter
from logger import Logger, log_assert


log = Logger(__name__)


class Candle(ABC):

    def __init__(self, streamer, **kwargs):
        self.__dict__.update(kwargs)
        self.streamer = streamer
        self.streamer.newCandle.connect(self.append, keep_ref=True)
        self.df = None
        log.debug(f'candle init for contract: {kwargs}')
        self.candles = []
        self._createEvents()

    def _createEvents(self):
        self.signal = Event('signal')
        self.entrySignal = Event('entrySignal')
        self.closeSignal = Event('closeSignal')

    def __call__(self, ib: IB):
        log.debug(
            f'Candle {self.contract.localSymbol} initializing data stream...')
        self.details = ib.reqContractDetails(self.contract)[0]
        self.streamer(ib, self.contract)

    def append(self, candle: Dict[str, Any]):
        self.candles.append(candle)
        if not candle['backfill']:
            df = pd.DataFrame(self.candles)
            df.set_index('date', inplace=True)
            self.df = self.get_indicators(df)
            log_assert(not self.df.iloc[-1].isna().any(), (
                f'Not enough data for indicators for instrument'
                f' {self.contract.localSymbol} '
                f' index: {df.index[-1]}'
                f' values: {self.df.iloc[-1].to_dict()}'
                f'{self.df}'), __name__)
            self.process()

    def save(self, path):
        if self.df is not None:
            self.df.to_pickle(
                f'{path}/freeze_df_{self.contract.localSymbol}'
                f'.pickle')
        self.streamer.all_bars_df.to_pickle(
            f'{path}/all_bars_df_{self.contract.localSymbol}'
            f'.pickle')

    def set_now(self, now):
        self.streamer.now = now

    @abstractmethod
    def get_indicators(self, df):
        return df

    @abstractmethod
    def process(self):
        self.signal.emit(self)
        self.entrySignal.emit(self)
        self.closeSignal.emit(self)


class Portfolio(ABC):

    def __init__(self, ib: IB):
        self.ib = ib
        self.values = {}
        self._createEvents()
        # alloc = round(sum([c.alloc for c in contracts]), 5)
        # assert alloc == 1, "Portfolio allocations don't add-up to 1"

    def _createEvents(self):
        self.entrySignal = Event('entrySignal')
        self.closeSignal = Event('closeSignal')

    @property
    def account_value(self):
        self.update_value()
        return self.values['TotalCashBalance'] + min(
            self.values['UnrealizedPnL'], 0)

    @property
    def positions(self):
        positions = self.ib.positions()
        return {p.contract.symbol: p.position for p in positions}

    def update_value(self):
        tags = self.ib.accountValues()
        for item in tags:
            if item.currency == 'USD':
                try:
                    self.values[item.tag] = float(item.value)
                except ValueError:
                    pass

    def onEntry(self, signal):
        raise NotImplementedError

    def onClose(self, signal):
        raise NotImplementedError

    def onSignal(self, signal):
        raise NotImplementedError


class Manager:

    def __init__(self, ib: IB, candles: List[Candle],
                 portfolio_class: Portfolio,
                 blotter: Blotter = None, trailing: bool = True,
                 freeze_path: str = 'notebooks/freeze/live',
                 keep_ref: bool = True):

        log.debug(f'Manager args: ib: {ib}, candles: {candles}, '
                  f'portfolio: {portfolio_class}, blotter: {blotter}, '
                  f'trailing: {trailing}, freeze_path: {freeze_path} '
                  f'keep_ref: {keep_ref}')
        if blotter is None:
            blotter = Blotter()
        self.ib = ib
        self.candles = candles
        self.path = freeze_path
        self.trader = Trader(ib, blotter, trailing)
        self.keep_ref = keep_ref
        log.debug(f'manager object initiated: {self}')
        self.connect_portfolio(portfolio_class)

    def connect_portfolio(self, portfolio_class: Portfolio):
        self.portfolio = portfolio_class(self.ib)
        self.portfolio.entrySignal += self.trader.onEntry
        self.portfolio.closeSignal += self.trader.onClose

    def onStarted(self, *args, **kwargs):
        log.debug(f'manager onStarted')
        self.trader.reconcile_stops()
        self.candles = get_contracts(self.candles, self.ib)
        # allow backtester to convey simulation time
        now = kwargs.get('now') or datetime.now()
        self.connect_candles(now)

    def onScheduledUpdate(self):
        self.freeze()

    def freeze(self):
        """Function called periodically to keep record of system data"""
        for candle in self.candles:
            candle.save(self.path)
        log.debug('Freezed data saved')

    def connect_candles(self, now):
        for candle in self.candles:
            # register candles with Trader
            self.trader.register(candle)
            # make sure no previous events connected
            candle.entrySignal.clear()
            candle.closeSignal.clear()
            candle.signal.clear()
            # connect trade signals
            candle.entrySignal.connect(self.portfolio.onEntry,
                                       keep_ref=self.keep_ref)
            candle.closeSignal.connect(self.portfolio.onClose,
                                       keep_ref=self.keep_ref)
            candle.signal.connect(self.portfolio.onSignal,
                                  keep_ref=self.keep_ref)
            candle.set_now(now)
            # run candle logic
            candle(self.ib)


class Trader:

    def __init__(self, ib: IB, blotter: Blotter, trailing: bool = True):
        self.ib = ib
        self.blotter = blotter
        self.trailing = trailing
        self.contracts = {}
        log.debug('Trader initialized')

    def register(self, obj: Candle):
        self.contracts[obj.contract.symbol] = obj

    def onEntry(self, obj: Candle, signal: int,
                atr: float, amount: int) -> None:
        log.debug(
            f'entry signal handled for: {obj.contract.localSymbol} '
            f'{signal} {atr}')
        self.contracts[obj.contract.symbol].atr = atr
        trade = self.trade(obj.contract, signal, amount)
        trade.filledEvent += self.attach_sl
        self.attach_events(trade, 'ENTRY')

    def onClose(self, obj: Candle, signal: int, amount: int) -> None:
        message = (f'close signal handled for: {obj.contract.localSymbol}'
                   f' signal: {signal}')
        log.debug(message)
        self.remove_sl(obj.contract)
        # TODO can sl close position before being removed?
        # make sure sl didn't close the position before being removed
        trade = self.trade(obj.contract, signal, amount)
        self.attach_events(trade, 'CLOSE')

    def trade(self, contract: Contract, signal: int,
              amount: int) -> Trade:
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
        sl_points = self.contracts[contract.symbol].atr
        if not self.trailing:
            sl_price = self.round_tick(
                price + sl_points * direction *
                self.contracts[contract.symbol].sl_atr,
                self.contracts[contract].details.minTick)
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

    def remove_sl(self, contract: Contract) -> None:
        open_trades = self.ib.openTrades()
        orders = defaultdict(list)
        for t in open_trades:
            orders[t.contract.localSymbol].append(t.order)
        for order in orders[contract.localSymbol]:
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
        self.blotter.log_commission(trade, fill, report, reason)


def get_contracts(params: List[Params], ib: IB) -> List[Params]:

    def qualify(contract_tuples: List[tuple]):
        log.debug(f'converting contract tuples: {contract_tuples}')
        cont_contracts = [ContFuture(*contract)
                          for contract in contract_tuples]
        ib.qualifyContracts(*cont_contracts)
        log.debug(f'contracts qualified: {cont_contracts}')
        return cont_contracts

    log.debug(f'params received: {params}')
    if type(params[0].contract) not in (Future, ContFuture):
        log.debug(f'obtaining contract objects')
        contracts = [param.contract for param in params]
    else:
        contracts = [(param.contract.symbol, param.contract.exchange)
                     for param in params]

    for param, contract in zip(params, qualify(contracts)):
        param.contract = contract
    log.debug(f'get_contracts returning params: {params}')
    return params


"""
def get_contracts(params: List[Params], ib: IB) -> List[Params]:

    def convert(contract_tuples: List[tuple]):
        log.debug(f'converting contract tuples: {contract_tuples}')
        cont_contracts = [ContFuture(*contract)
                          for contract in contract_tuples]
        ib.qualifyContracts(*cont_contracts)
        log.debug(f'contracts qualified: {cont_contracts}')
        # log.debug(f'Contracts qualified: {cont_contracts}')

        # Converting to Futures potentially unnecessary
        # But removing below likely breaks the backtester
        # Not any more? TODO
        ids = [contract.conId for contract in cont_contracts]
        contracts = [Future(conId=id) for id in ids]
        ib.qualifyContracts(*contracts)
        log.debug(f'Contracts qualified: {contracts}')
        return contracts

    log.debug(f'params received: {params}')
    if type(params[0].contract) not in (Future, ContFuture):
        log.debug(f'obtaining contract objects')
        contracts = [param.contract for param in params]
    else:
        contracts = [(param.contract.symbol, param.contract.exchange)
                     for param in params]

    for param, contract in zip(params, convert(contracts)):
        param.contract = contract

    return params
"""
