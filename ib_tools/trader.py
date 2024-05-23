from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import ib_insync as ibi

log = logging.getLogger(__name__)


class Trader:
    def __init__(
        self,
        ib: ibi.IB,
    ) -> None:
        self.ib = ib
        log.debug(f"Trader initialized: {self}")

    def trade(
        self,
        contract: ibi.Contract,
        order: ibi.Order,
    ) -> ibi.Trade:
        """Execute trade. Place passed order with the broker."""
        trade = self.ib.placeOrder(contract, order)
        log.info(f"Placed {order.orderType} for {trade.contract.localSymbol}")
        return trade

    def modify(self, contract: ibi.Contract, order: ibi.Order) -> ibi.Trade:
        modified_trade = self.ib.placeOrder(contract, order)
        log.info(f"Trade modified: {modified_trade}")
        return modified_trade

    def cancel(self, trade: ibi.Trade) -> Optional[ibi.Trade]:
        order = trade.order
        cancelled_trade = self.ib.cancelOrder(order)
        log.info(
            f"Sent cancel request for {order.orderType} {order.orderId} "
            f"for {trade.contract.localSymbol}"
        )
        return cancelled_trade

    def trades_for_contract(self, contract: ibi.Contract) -> list[ibi.Trade]:
        """Return open trades for a given contract."""
        return [t for t in self.ib.openTrades() if t.contract == contract]

    def position_for_contract(self, contract: ibi.Contract) -> float:
        return next(
            (v.position for v in self.ib.positions() if v.contract == contract), 0.0
        )

    def positions(self) -> dict[ibi.Contract, float]:
        return {p.contract: p.position for p in self.ib.positions()}

    def __repr__(self):
        items = (f"{k}={v}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({', '.join(items)})"


class FakeTrader(Trader):
    """
    Object to replace real :class:`Trader` to prevent system from any
    subsequent trading.
    """

    def trade(self, contract, order):
        log.debug(f"Not trading: {contract} {order}")
        return ibi.Trade(
            contract=contract,
            order=order,
            log=[
                ibi.TradeLogEntry(
                    time=datetime.now(tz=timezone.utc),
                    message="Ignored order in self nuke mode.",
                )
            ],
        )
