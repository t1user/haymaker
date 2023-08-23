from __future__ import annotations

import logging
from collections import defaultdict
from functools import partial
from typing import Final

import ib_insync as ibi

from ib_tools.blotter import AbstractBaseBlotter, CsvBlotter
from ib_tools.state_machine import StateMachine

log = logging.getLogger(__name__)

csv_blotter: Final = CsvBlotter()


class Trader:
    def __init__(
        self, ib: ibi.IB, state_machine: StateMachine, trade_handler: "BaseTradeHandler"
    ) -> None:
        self.ib = ib
        self.state_machine = state_machine
        self.trade_handler = trade_handler
        self.ib.newOrderEvent += self.trace_manual_orders
        log.debug("Trader initialized")

    def trades(self) -> list[ibi.Trade]:
        return self.ib.openTrades()

    def positions(self) -> list[ibi.Position]:
        return self.ib.positions()

    def trade(
        self,
        contract: ibi.Contract,
        order: ibi.Order,
        reason: str = "",
        strategy_key: str = "unknown",
    ) -> ibi.Trade:
        trade = self.ib.placeOrder(contract, order)
        self.state_machine.register_order(strategy_key, reason, trade)

        if reason:
            self.trade_handler.attach_events(trade, reason)
            log.debug(f"{contract.localSymbol} order placed: {order}")
        else:
            log.debug(f"{contract.localSymbol} order updated: {order}")
        return trade

    def modify(self, contract: ibi.Contract, order: ibi.Order) -> ibi.Trade:
        trade = self.ib.placeOrder(contract, order)
        return trade

    def cancel(self, trade: ibi.Trade):
        order = trade.order
        cancelled_trade = self.ib.cancelOrder(order)
        log.info(f"Cancelled {order.orderType} for " f"{trade.contract.localSymbol}")
        return cancelled_trade

    def trace_manual_orders(self, trade: ibi.Trade) -> None:
        """
        Attempt to attach reporting events for orders entered
        outside of the framework. This will not work if framework is not
        connected with clientId == 0.
        """
        if trade.order.orderId <= 0:
            log.debug("manual trade reporting event attached")
            self.trade_handler.attach_events(trade, "MANUAL TRADE")

    def trades_for_contract(self, contract: ibi.Contract) -> list[ibi.Trade]:
        """Return open trades for a given contract."""

        open_trades = self.trades()
        trades = defaultdict(list)
        for t in open_trades:
            trades[t.contract.localSymbol].append(t)
        return trades[contract.localSymbol]

    def __repr__(self):
        items = (f"{k}={v}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({', '.join(items)})"


class BaseTradeHandler:
    def attach_events(self, trade: ibi.Trade, reason: str) -> None:
        report_trade = partial(self.onFilled, reason)
        report_commission = partial(self.onCommissionReport, reason)
        trade.statusEvent += self.onStatus
        trade.modifyEvent += self.onModify
        trade.fillEvent += self.onFill
        trade.commissionReportEvent += report_commission
        trade.filledEvent += report_trade
        trade.cancelEvent += self.onCancel
        trade.cancelledEvent += self.onCancelled

        log.debug(
            f"Reporting events attached for {trade.contract.localSymbol}"
            f" {trade.order.action} {trade.order.totalQuantity}"
            f" {trade.order.orderType}"
        )

    def onStatus(self, trade: ibi.Trade) -> None:
        pass

    def onModify(self, trade: ibi.Trade) -> None:
        pass

    def onFill(self, trade: ibi.Trade, fill: ibi.Fill) -> None:
        pass

    def onCommissionReport(
        self,
        reason: str,
        trade: ibi.Trade,
        fill: ibi.Fill,
        report: ibi.CommissionReport,
    ) -> None:
        pass

    def onFilled(self, reason: str, trade: ibi.Trade) -> None:
        pass

    def onCancel(self, trade: ibi.Trade) -> None:
        pass

    def onCancelled(self, trade: ibi.Trade) -> None:
        pass

    def __repr__(self):
        items = (f"{k}={v}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({', '.join(items)})"


class ReportTradeHandler(BaseTradeHandler):
    def __init__(self, blotter: AbstractBaseBlotter = csv_blotter) -> None:
        self.blotter = blotter

    def report_trade(self, reason: str, trade: ibi.Trade) -> None:
        message = (
            f"{reason} trade filled: {trade.contract.localSymbol} "
            f"{trade.order.action} {trade.orderStatus.filled}"
            f"@{trade.orderStatus.avgFillPrice}"
        )
        log.info(message)

    def report_cancel(self, trade: ibi.Trade) -> None:
        message = (
            f"{trade.order.orderType} order {trade.order.action} "
            f"{trade.orderStatus.remaining} (of "
            f"{trade.order.totalQuantity}) for "
            f"{trade.contract.localSymbol} cancelled"
        )
        log.info(message)

    def report_modification(self, trade):
        log.debug(f"Order modified: {trade.order}")

    def report_commission(
        self,
        reason: str,
        trade: ibi.Trade,
        fill: ibi.Fill,
        report: ibi.CommissionReport,
    ) -> None:
        self.blotter.log_commission(trade, fill, report, reason)
