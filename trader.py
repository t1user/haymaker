from __future__ import annotations

from collections import defaultdict
from functools import partial
from typing import List, Dict, Any, Union, Type, Optional, Literal
from datetime import datetime

from ib_insync import (Order, MarketOrder, StopOrder, IB, Fill,
                       CommissionReport, Trade, Contract, TagValue)
import numpy as np

from portfolio import Portfolio
from candle import Candle
from blotter import AbstractBaseBlotter, CsvBlotter
from execution_models import BaseExecModel, EventDrivenExecModel
from saver import AbstractBaseSaver, PickleSaver
from logger import Logger


log = Logger(__name__)


class Manager:

    def __init__(self, ib: IB, candles: List[Candle],
                 portfolio: Portfolio,
                 exec_model: Optional[BaseExecModel] = None,
                 saver: AbstractBaseSaver = PickleSaver(),
                 contract_fields: Union[List[str], str] = 'contract',
                 keep_ref: bool = True):

        self.ib = ib
        self.candles = candles
        self.saver = saver
        self.exec_model = exec_model or EventDrivenExecModel()
        self.keep_ref = keep_ref
        self.portfolio = portfolio
        self.connect_portfolio(portfolio)
        log.debug(f'manager object initiated: {self}')

    def connect_portfolio(self, portfolio: Portfolio):
        self.portfolio.register(self.ib, self.candles)
        self.portfolio.entrySignal += self.exec_model.onEntry
        self.portfolio.closeSignal += self.exec_model.onClose

    def onStarted(self, *args, **kwargs):
        log.debug('manager onStarted')
        self.exec_model.reconcile_stops()
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
        Attempt to attach reporting events for orders entered
        outside of the framework. This will not work if framework is not
        connected with clientId == 0.
        """
        if ((trade.order.orderId <= 0)
            and (trade.order.orderType not in ('STP', 'TRAIL'))
                and (trade.orderStatus.status in ['PreSubmitted', 'Inactive'])):
            log.debug('manual trade reporting event attached')
            trade.filledEvent.clear()
            trade.cancelledEvent.clear()
            trade.commissionReportEvent.clear()
            trade.modifyEvent.clear()
            self.attach_events(trade, 'MANUAL TRADE')

    def trade(self, contract: Contract, order: Order, reason: str) -> Trade:
        trade = self.ib.placeOrder(contract, order)
        self.attach_events(trade, reason)
        log.debug(f'{contract.localSymbol} order placed: {order}')
        return trade

    def remove_sl(self, contract: Contract) -> None:
        open_trades = self.ib.openTrades()
        orders = defaultdict(list)
        for t in open_trades:
            orders[t.contract.localSymbol].append(t.order)
        for order in orders[contract.localSymbol]:
            if order.orderType in ('STP', 'TRAIL'):
                self.ib.cancelOrder(order)
                log.debug(f'stop loss removed for {contract.localSymbol}')

    def attach_events(self, trade: Trade, reason: str) -> None:
        report_trade = partial(self.report_trade, reason)
        report_commission = partial(self.report_commission, reason)
        # make sure events are not duplicated
        trade.filledEvent += report_trade
        trade.cancelledEvent += self.report_cancel
        trade.commissionReportEvent += report_commission
        trade.modifyEvent += self.report_modification
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

    def report_modification(self, trade):
        log.debug(f'Order modified: {trade.order}')

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
