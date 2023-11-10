from __future__ import annotations

import logging
from collections import defaultdict

import ib_insync as ibi

log = logging.getLogger(__name__)


class Trader:
    def __init__(
        self,
        ib: ibi.IB,
    ) -> None:
        self.ib = ib
        log.debug(f"Trader initialized: {self}")

    def trades(self) -> list[ibi.Trade]:
        return self.ib.openTrades()

    def positions(self) -> list[ibi.Position]:
        return self.ib.positions()

    def trade(
        self,
        contract: ibi.Contract,
        order: ibi.Order,
    ) -> ibi.Trade:
        trade = self.ib.placeOrder(contract, order)
        log.info(f"Placed {order.orderType} for {trade.contract.localSymbol}")
        return trade

    def modify(self, contract: ibi.Contract, order: ibi.Order) -> ibi.Trade:
        trade = self.ib.placeOrder(contract, order)
        return trade

    def cancel(self, trade: ibi.Trade):
        order = trade.order
        cancelled_trade = self.ib.cancelOrder(order)
        log.info(
            f"Sent cancel request for {order.orderType} {order.orderId} "
            f"for {trade.contract.localSymbol}"
        )
        return cancelled_trade

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
