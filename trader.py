from __future__ import annotations

from collections import defaultdict
from functools import partial
from typing import List, Dict, Any, Union, Type, Optional, Literal
from datetime import datetime

from ib_insync import (Order, MarketOrder, StopOrder, IB, Fill,
                       CommissionReport, Trade, Contract, TagValue)

from portfolio import Portfolio
from candle import Candle
from blotter import AbstractBaseBlotter, CsvBlotter

from saver import AbstractBaseSaver, PickleSaver
from logger import Logger


log = Logger(__name__)


class Manager:

    def __init__(self, ib: IB, candles: List[Candle],
                 portfolio: Portfolio,
                 trader: Optional[Trader] = None,
                 saver: AbstractBaseSaver = PickleSaver(),
                 contract_fields: Union[List[str], str] = 'contract',
                 keep_ref: bool = True):

        self.ib = ib
        self.candles = candles
        self.saver = saver
        self.trader = trader or Trader(ib)
        self.keep_ref = keep_ref
        self.portfolio = portfolio
        self.connect_portfolio(portfolio)
        log.debug(f'manager object initiated: {self}')

    def connect_portfolio(self, portfolio: Portfolio):
        self.portfolio.register(self.ib, self.candles)
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
            candle.save(self.saver)
        log.debug('Freezed data saved')

    def connect_candles(self, now):
        for candle in self.candles:
            # register candles with Trader
            for field in candle.contract_fields:
                self.trader.register(getattr(candle, field), candle)
            log.debug(
                f'contracts for candle {candle.contract.tradingClass}: '
                f'{[getattr(candle, field) for field in candle.contract_fields]}')
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

    def __str__(self):
        return (f'Manager: ib: {self.ib}, candles: {self.candles}, '
                f'portfolio: {self.portfolio}, '
                f'trader: {self.trader}, '
                f'saver: {self.saver}, '
                f'keep_ref: {self.keep_ref}')


class Trader:

    def __init__(self,
                 ib: IB, blotter: AbstractBaseBlotter = CsvBlotter(),
                 sl_type: Literal['fixed', 'trailing',
                                  'trailingFixed'] = 'trailing'):
        self.ib = ib
        self.blotter = blotter
        self.contracts = {}
        self.sl_type = sl_type
        self.ib.orderStatusEvent += self.verify_orders
        log.debug('Trader initialized')

    def register(self, contract: Contract, obj: Candle) -> None:
        log.debug(f'Registering: {contract.symbol}: {obj}')
        self.contracts[contract.symbol] = obj

    def verify_orders(self, trade: Trade) -> None:
        """
        Attempt to attach reporting events to the blotter for orders entered
        outside of the framework. This will not work if framework is not
        connected with clientId == 0.
        """
        if ((trade.order.orderId < 0)
                and (trade.orderStatus.status == 'Inactive')):
            self.attach_events(trade, 'MANUAL TRADE')

    def onEntry(self, contract: Contract, signal: int,
                atr: float, amount: int) -> None:
        log.debug(
            f'entry signal handled for: {contract.localSymbol} '
            f'signal: {signal} atr: {atr}')
        self.contracts[contract.symbol].atr = atr
        trade = self.trade(contract, signal, amount)
        trade.filledEvent += self.attach_sl
        self.attach_events(trade, 'ENTRY')

    def onClose(self, contract: Contract, signal: int, amount: int) -> None:
        message = (f'close signal handled for: {contract.localSymbol}'
                   f' signal: {signal}')
        log.debug(message)
        self.remove_sl(contract)
        # TODO can sl close position before being removed?
        # make sure sl didn't close the position before being removed
        trade = self.trade(contract, signal, amount)
        self.attach_events(trade, 'CLOSE')

    def trade(self, contract: Contract, signal: int,
              amount: int) -> Trade:
        direction = {1: 'BUY', -1: 'SELL'}
        order = MarketOrder(direction[signal], amount, algoStrategy='Adaptive',
                            algoParams=[
                                TagValue('adaptivePriority', 'Normal')],
                            tif='Day')
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
        if self.sl_type == 'fixed':
            sl_price = self.round_tick(
                price + sl_points * direction *
                self.contracts[contract.symbol].sl_atr,
                self.contracts[contract.symbol].details.minTick)
            log.info(f'STOP LOSS PRICE: {sl_price}')
            order = StopOrder(reverseAction, amount, sl_price,
                              outsideRth=True, tif='GTC')
        elif self.sl_type == 'trailing' or self.sl_type == 'trailingFixed':
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
        if self.sl_type == 'trailingFixed':
            sl = trade.order
            log.debug(sl)
            sl.adjustedOrderType = 'STP'
            sl.adjustedStopPrice = (
                price - direction * 2 * distance)
            # self.contracts[contract.symbol].details.minTick)
            log.debug(f'adjusted stop price: {sl.adjustedStopPrice}')
            log.debug(f'DISTANCE: {distance}')
            sl.triggerPrice = sl.adjustedStopPrice - direction * distance
            self.ib.placeOrder(contract, sl)
            log.debug(f'stop loss for {contract.localSymbol} will be '
                      f'fixed at {sl.triggerPrice}')

    def report_order_modification(self, trade):
        log.debug(f'Order modified: {trade.order}')

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
        log.info(f'open trades on re-connect: {len(trades)} '
                 f'{[t.contract.localSymbol for t in trades]}')
        log.debug(f'open orders on re-connect: {[o.order for o in trades]}')
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
        trade.modifyEvent += self.report_order_modification
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

    def __str__(self):
        return f'{self.__class__.__name__} with args: {self.__dict__}'


def get_contracts(candles: List[Candle], ib: IB,
                  ) -> List[Candle]:
    contract_list = []
    for candle in candles:
        for contract in candle.contract_fields:
            contract_list.append(getattr(candle, contract))
    ib.qualifyContracts(*contract_list)
    log.debug(f'contracts qualified: {contract_list}')
    return candles
