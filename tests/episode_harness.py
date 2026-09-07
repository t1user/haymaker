"""Independent broker boundary for full one-to-one component pipelines.

The broker owns its own fill ledger. Tests drive IB events, never Book's
projection methods, and use the real Trader and Controller.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import ib_insync as ibi
import pandas as pd

from haymaker.components import PandasSignalModel


async def settle_events() -> None:
    """Let eventkit coroutines and deferred convergence finish locally."""
    for _ in range(8):
        await asyncio.sleep(0)


class EpisodeBroker(ibi.IB):
    """Simulate submission, fills and OCA at the IB boundary, without sockets."""

    def __init__(self) -> None:
        super().__init__()
        self.submitted: list[ibi.Trade] = []
        self.quantities: dict[ibi.Contract, float] = {}
        self._execution_number = 0

    def placeOrder(self, contract: ibi.Contract, order: ibi.Order) -> ibi.Trade:
        """Accept a broker order and assign real-shaped IB identifiers."""
        order.orderId = len(self.submitted) + 1
        order.permId = 1000 + order.orderId
        trade = ibi.Trade(
            contract=contract,
            order=order,
            orderStatus=ibi.OrderStatus(
                orderId=order.orderId,
                permId=order.permId,
                status=ibi.OrderStatus.Submitted,
                remaining=order.totalQuantity,
            ),
        )
        self.submitted.append(trade)
        self.newOrderEvent.emit(trade)
        return trade

    def cancelOrder(
        self, order: ibi.Order, manualCancelOrderTime: str = ""
    ) -> ibi.Trade | None:
        """Confirm cancellation and notify both broker and Trade listeners."""
        trade = next(t for t in self.submitted if t.order is order)
        if trade.isActive():
            trade.orderStatus.status = ibi.OrderStatus.Cancelled
            self.orderStatusEvent.emit(trade)
            trade.cancelledEvent.emit(trade)
        return trade

    def openTrades(self) -> list[ibi.Trade]:
        """Return only currently working broker trades."""
        return [trade for trade in self.submitted if trade.isActive()]

    def positions(self, account: str = "") -> list[ibi.Position]:
        """Return the independently accumulated broker position snapshot."""
        return [
            ibi.Position("test", contract, quantity, 100)
            for contract, quantity in self.quantities.items()
            if quantity
        ]

    async def fill(
        self,
        trade: ibi.Trade,
        quantity: float | None = None,
        *,
        notify_filled: bool = True,
    ) -> ibi.Fill:
        """Execute shares, apply broker OCA, and emit normal IB fill events."""
        quantity = trade.remaining() if quantity is None else quantity
        if not trade.isActive() or not 0 < quantity <= trade.remaining():
            raise ValueError("Fill must belong to an active order's remainder")
        self._execution_number += 1
        now = datetime.now(timezone.utc)
        side = 1 if trade.order.action == "BUY" else -1
        self.quantities[trade.contract] = (
            self.quantities.get(trade.contract, 0) + side * quantity
        )
        execution = ibi.Execution(
            execId=f"episode-{self._execution_number}",
            orderId=trade.order.orderId,
            permId=trade.order.permId,
            time=now,
            side="BOT" if side > 0 else "SLD",
            shares=quantity,
            price=100,
        )
        fill = ibi.Fill(trade.contract, execution, ibi.CommissionReport(), now)
        trade.fills.append(fill)
        trade.orderStatus.filled += quantity
        trade.orderStatus.remaining -= quantity
        complete = trade.remaining() == 0
        if complete:
            trade.orderStatus.status = ibi.OrderStatus.Filled
        self.execDetailsEvent.emit(trade, fill)
        self.orderStatusEvent.emit(trade)
        trade.fillEvent.emit(trade, fill)
        await settle_events()
        if complete:
            if trade.order.ocaGroup:
                for other in self.openTrades():
                    if other.order.ocaGroup == trade.order.ocaGroup:
                        self.cancelOrder(other.order)
            if notify_filled:
                trade.filledEvent.emit(trade)
        await settle_events()
        return fill

    async def commission(self, trade: ibi.Trade, fill: ibi.Fill) -> None:
        """Deliver a delayed commission without changing broker quantity."""
        report = ibi.CommissionReport(
            execId=fill.execution.execId,
            commission=1.25,
            currency="USD",
            realizedPNL=0,
        )
        fill.commissionReport.execId = report.execId
        fill.commissionReport.commission = report.commission
        fill.commissionReport.currency = report.currency
        fill.commissionReport.realizedPNL = report.realizedPNL
        self.commissionReportEvent.emit(trade, fill, report)
        await settle_events()


class EpisodeSignalModel(PandasSignalModel):
    """Emit the supplied calculation dataframe through the standard template."""

    def df(self, data: pd.DataFrame) -> pd.DataFrame:
        """Treat the test's calculated rows as the user's calculation output."""
        return data


def observation(value: int, atr: float = 5) -> pd.DataFrame:
    """Build a single timestamped binary observation with bracket metadata."""
    return pd.DataFrame(
        {"signal": [value], "atr": [atr]},
        index=pd.DatetimeIndex([datetime.now(timezone.utc)]),
    )
