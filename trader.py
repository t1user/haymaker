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
        Attempt to attach reporting events for orders entered
        outside of the framework. This will not work if framework is not
        connected with clientId == 0.
        """
        if ((trade.order.orderId <= 0)
            and (trade.order.orderType not in ('STP', 'TRAIL'))
                and (trade.orderStatus.status in ['PreSubmitted', 'Inactive'])):
            log.debug(f'manual trade reporting event attached')
            trade.filledEvent.clear()
            trade.cancelledEvent.clear()
            trade.commissionReportEvent.clear()
            trade.modifyEvent.clear()
            self.attach_events(trade, 'MANUAL TRADE')

    def onEntry(self, contract: Contract, signal: int,
                atr: float, amount: int) -> None:
        log.debug(
            f'{contract.localSymbol} entry signal: {signal} atr: {atr}')
        self.execModel.onEntry(contract, signal, atr, amount)

    def onClose(self, contract: Contract, signal: int, amount: int) -> None:
        log.debug(f'{contract.localSymbol} close signal: {signal}')
        self.execModel.onClose(contract, signal, amount)

    def emergencyClose(self, contract: Contract, signal: int,
                       amount: int) -> None:
        log.warning(f'Emergency close on restart: {contract.localSymbol} '
                    f'side: {signal} amount: {amount}')
        trade = self.trade(contract, signal, amount)
        self.attach_events(trade, 'EMERGENCY CLOSE')

    def trade(self, contract: Contract, order: Order, reason: str) -> Trade:
        trade = self.ib.placeOrder(contract, order)
        self.attach_events(trade, reason)
        log.debug(f'{contract.localSymbol} order placed: {order}')
        return trade

    def report_order_modification(self, trade):
        log.debug(f'Order modified: {trade.order}')

    def reconcile_stops(self) -> None:
        """
        To be executed on restart. Make sure all positions have corresponding
        stop-losses, if not send a closing order. For all existing
        stop-losses attach reporting events for the blotter.
        """

        trades = self.ib.openTrades()
        log.info(f'open trades on re-connect: {len(trades)} '
                 f'{[t.contract.localSymbol for t in trades]}')
        # attach reporting events
        for trade in trades:
            if trade.order.orderType in ('STP', 'TRAIL'
                                         ) and trade.orderStatus.remaining != 0:
                self.attach_events(trade, 'STOP-LOSS')

        # check for orphan positions
        positions = self.ib.positions()
        log.debug(f'positions on re-connect: {positions}')
        trade_contracts = set([t.contract for t in trades])
        position_contracts = set([p.contract for p in positions])
        orphan_contracts = position_contracts - trade_contracts
        orphan_positions = [position for position in positions
                            if position.contract in orphan_contracts]
        if orphan_positions:
            log.warning(f'orphan positions: {orphan_positions}')
            log.debug(f'len(orphan_positions): {len(orphan_positions)}')
            log.debug(f'orphan contracts: {orphan_contracts}')
            for p in orphan_positions:
                self.ib.qualifyContracts(p.contract)
                self.emergencyClose(
                    p.contract, -np.sign(p.position), int(np.abs(p.position)))
                log.error(f'emergencyClose position: {p.contract}, '
                          f'{-np.sign(p.position)}, {int(np.abs(p.position))}')
                t = self.ib.reqAllOpenOrders()
                log.debug(f'reqAllOpenOrders: {t}')
                log.debug(f'openTrades: {self.ib.openTrades()}')

    def attach_events(self, trade: Trade, reason: str) -> None:
        report_trade = partial(self.report_trade, reason)
        report_commission = partial(self.report_commission, reason)
        # make sure events are not duplicated
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
