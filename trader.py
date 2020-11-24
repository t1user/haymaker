from __future__ import annotations

from collections import defaultdict
from functools import partial

from ib_insync import Order, IB, Fill, CommissionReport, Trade, Contract

from candle import Candle
from blotter import AbstractBaseBlotter, CsvBlotter
from logger import Logger


log = Logger(__name__)


class Trader:

    def __init__(self, ib: IB,
                 blotter: AbstractBaseBlotter = CsvBlotter()) -> None:
        self.ib = ib
        self.blotter = blotter
        self.contracts = {}
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
                and (
                    trade.orderStatus.status in ['PreSubmitted', 'Inactive'])):
            log.debug('manual trade reporting event attached')
            trade.filledEvent.clear()
            trade.cancelledEvent.clear()
            trade.commissionReportEvent.clear()
            trade.modifyEvent.clear()
            self.attach_events(trade, 'MANUAL TRADE')

    def trades(self):
        return self.ib.openTrades()

    def positions(self):
        return self.ib.positions()

    def trade(self, contract: Contract, order: Order, reason: str) -> Trade:
        trade = self.ib.placeOrder(contract, order)
        self.attach_events(trade, reason)
        log.info(f'{contract.localSymbol} order placed: {order}')
        return trade

    def cancel(self, order: Order):
        self.ib.cancelOrder(order)
        log.info(f'Cancelled {order.orderType} for '
                 f'{order.contract.localSymbol}')

    def attach_events(self, trade: Trade, reason: str) -> None:
        report_trade = partial(self.report_trade, reason)
        report_commission = partial(self.report_commission, reason)
        # make sure events are not duplicated
        trade.filledEvent += report_trade
        trade.cancelledEvent += self.report_cancel
        trade.commissionReportEvent += report_commission
        trade.modifyEvent += self.report_modification
        log.debug(f'Reporting events attached for {trade.contract.localSymbol}'
                  f' {trade.order.action} {trade.order.totalQuantity}'
                  f' {trade.order.orderType}')

    def onStarted(self):
        """
        Attach reporting events to all open trades.
        """
        trades = self.trades()
        log.info(f'open trades on re-connect: {len(trades)} '
                 f'{[t.contract.localSymbol for t in trades]}')
        # attach reporting events
        for trade in trades:
            if trade.orderStatus.remaining != 0:
                if trade.order.orderType in ('STP', 'TRAIL'):
                    self.attach_events(trade, 'STOP-LOSS')
                else:
                    self.attach_events(trade, 'TAKE-PROFIT')

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
