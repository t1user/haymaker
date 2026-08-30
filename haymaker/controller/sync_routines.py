"""Order rebinding and aggregate position checks used during Controller sync."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Self

import ib_insync as ibi

from haymaker.book import Book

log = logging.getLogger(__name__)


class OrderSync:
    """Classify broker and persisted Book order state for one sync pass."""

    def __init__(self, ib: ibi.IB, book: Book) -> None:
        self.ib = ib
        self.book = book
        self.unknown: list[ibi.Trade] = []
        self.inactive: list[ibi.Trade] = []
        self.done: list[ibi.Trade] = []
        self.errors: list[ibi.Trade] = []
        self.update_trades().review_trades().handle_inactive_trades().report()

    @property
    def lists(self) -> tuple[list[ibi.Trade], list[ibi.Trade], list[ibi.Trade]]:
        return self.unknown, self.done, self.errors

    def update_trades(self) -> Self:
        """Rebind live open Trades and collect unknown broker orders."""

        for trade in self.ib.openTrades():
            if unknown := self.book.rebind_trade(trade):
                self.unknown.append(unknown)
        return self

    def review_trades(self) -> Self:
        """Find Book-active trades no longer present in broker openTrades."""

        broker_ids = {trade.order.orderId for trade in self.ib.openTrades()}
        for info in self.book.active_orders():
            if info.orderId not in broker_ids:
                self.inactive.append(info.trade)
        return self

    def handle_inactive_trades(self) -> Self:
        """Match inactive local Trades to session history or execution fills."""

        known = {
            trade.order.permId: trade
            for trade in self.ib.trades()
            if trade.order.permId
        }
        for old_trade in self.inactive:
            current = known.get(old_trade.order.permId)
            if current is not None:
                current.order.orderId = old_trade.order.orderId
            else:
                current = self._reconstruct_from_fills(old_trade)
            if current is None:
                self.errors.append(old_trade)
            else:
                self.done.append(current)
                self.book.rebind_trade(current)
        return self

    def _reconstruct_from_fills(self, trade: ibi.Trade) -> ibi.Trade | None:
        fills = [
            fill
            for fill in self.ib.fills()
            if fill.execution.orderId == trade.order.orderId
            or (trade.order.permId and fill.execution.permId == trade.order.permId)
        ]
        if not fills:
            return None
        trade.fills = fills
        filled = sum(fill.execution.shares for fill in fills)
        remaining = max(trade.order.totalQuantity - filled, 0)
        trade.orderStatus = ibi.OrderStatus(
            orderId=trade.order.orderId,
            status=(
                ibi.OrderStatus.Filled if remaining == 0 else ibi.OrderStatus.Submitted
            ),
            filled=filled,
            remaining=remaining,
        )
        return trade

    def report(self) -> Self:
        if any(self.lists):
            log.debug(
                "Order sync: unknown=%s done=%s unmatched=%s",
                len(self.unknown),
                len(self.done),
                len(self.errors),
            )
        return self

    @property
    def is_ok(self) -> bool:
        return not any(self.lists)

    @property
    def is_error(self) -> bool:
        return bool(self.errors or self.unknown)


class PositionSync:
    """Compare one fresh broker snapshot with fill-accounted Book quantity."""

    def __init__(self, positions: Iterable[ibi.Position], book: Book) -> None:
        self.positions = tuple(positions)
        self.book = book
        self.broker_positions: dict[ibi.Contract, float] = {}
        self.errors: dict[ibi.Contract, float] = {}
        self.verify_positions().report()

    def verify_positions(self) -> Self:
        self.broker_positions = {
            position.contract: position.position for position in self.positions
        }
        logical = self.book.logical_positions()
        self.errors = {
            contract: logical.get(contract, 0.0)
            - self.broker_positions.get(contract, 0.0)
            for contract in set(self.broker_positions) | set(logical)
            if logical.get(contract, 0.0) != self.broker_positions.get(contract, 0.0)
        }
        return self

    def report(self) -> Self:
        if self.errors:
            log.critical(
                "Failed to match Book positions to broker: %s",
                {
                    contract.localSymbol or contract.symbol: difference
                    for contract, difference in self.errors.items()
                },
            )
        else:
            log.debug("Positions sync OK.")
        return self

    @property
    def is_ok(self) -> bool:
        return not self.errors

    @property
    def is_error(self) -> bool:
        return bool(self.errors)
