"""Order rebinding and aggregate position checks used during Controller sync."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Self

import ib_insync as ibi

from haymaker.book import Book, FillRecord
from haymaker.components.messages import StandardOrderRole

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
        self.unresolved: list[ibi.Trade] = []
        self.recovered_fills: list[tuple[ibi.Trade, ibi.Fill]] = []
        self._fills_by_perm_id: dict[int, list[ibi.Fill]] = {}
        self._fills_by_order_id: dict[tuple[int, int], list[ibi.Fill]] = {}
        for fill in ib.fills():
            self._fills_by_perm_id.setdefault(fill.execution.permId, []).append(fill)
            self._fills_by_order_id.setdefault(
                (fill.execution.clientId, fill.execution.orderId), []
            ).append(fill)

    def run(self) -> Self:
        """Rebind, recover and classify one pass explicitly."""
        return self.update_trades().review_trades().handle_inactive_trades().report()

    @property
    def lists(self) -> tuple[list[ibi.Trade], list[ibi.Trade], list[ibi.Trade]]:
        return self.unknown, self.done, self.errors

    def update_trades(self) -> Self:
        """Rebind live open Trades and collect unknown broker orders."""

        for trade in self.ib.openTrades():
            if unknown := self.book.rebind_trade(trade):
                self.unknown.append(unknown)
            else:
                self.recovered_fills.extend(
                    (trade, fill) for fill in self._unseen_fills(trade)
                )
                info = self.book.orders.by_id(trade.order.orderId)
                if info is not None and info.role in {
                    StandardOrderRole.UNKNOWN,
                    StandardOrderRole.MANUAL,
                }:
                    self.unknown.append(trade)
        return self

    def _unseen_fills(self, trade: ibi.Trade) -> tuple[ibi.Fill, ...]:
        """Return validated broker executions not yet normalized by Book."""

        info = self.book.orders.by_id(
            trade.order.orderId
        ) or self.book.orders.by_perm_id(trade.order.permId)
        if info is None:
            raise ValueError("Rebound broker Trade has no Book order record")
        merged = info.execution_trade(
            (*trade.fills, *self._matching_broker_fills(trade))
        )
        received = {record.deduplication_key for record in info.fills}
        return tuple(
            fill
            for fill in merged.fills
            if FillRecord.from_fill(trade, fill).deduplication_key not in received
        )

    def _matching_broker_fills(self, trade: ibi.Trade) -> tuple[ibi.Fill, ...]:
        """Return execution-history fills attributable to one broker Trade."""

        return tuple(
            self._fills_by_perm_id.get(trade.order.permId, [])
            if trade.order.permId
            else self._fills_by_order_id.get(
                (trade.order.clientId, trade.order.orderId), []
            )
        )

    def review_trades(self) -> Self:
        """Find Book-active trades no longer present in broker openTrades."""

        broker_ids = {trade.order.orderId for trade in self.ib.openTrades()}
        for info in self.book.orders.active():
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
                info = self.book.orders.by_id(old_trade.order.orderId)
                if info is not None:
                    current.fills = info.execution_trade(
                        (*current.fills, *self._matching_broker_fills(current))
                    ).fills
            else:
                current = self._reconstruct_from_fills(old_trade)
            if current is None:
                self.errors.append(old_trade)
            else:
                self.book.rebind_trade(current)
                if current.isDone():
                    self.done.append(current)
                else:
                    self.recovered_fills.extend(
                        (current, fill) for fill in self._unseen_fills(current)
                    )
                    self.unresolved.append(current)
        return self

    def _reconstruct_from_fills(self, trade: ibi.Trade) -> ibi.Trade | None:
        """Merge persisted and broker execution evidence, including fill price."""
        info = self.book.orders.by_id(trade.order.orderId)
        if info is None:
            return None
        fills = list(self._matching_broker_fills(trade))
        if not fills and not info.fills:
            return None
        return info.execution_trade(fills)

    def resolve_completed_trades(self, completed: list[ibi.Trade]) -> None:
        """Resolve missing terminal status using broker completed-order evidence."""
        known = {
            trade.order.permId: trade
            for trade in completed
            if trade.order.permId and trade.isDone()
        }
        for trade in tuple(self.unresolved):
            terminal = known.get(trade.order.permId)
            if terminal is None:
                continue
            terminal.order.orderId = trade.order.orderId
            info = self.book.orders.by_id(trade.order.orderId)
            if info is None:
                raise ValueError("Unresolved trade lost its Book record")
            terminal.fills = info.execution_trade((*trade.fills, *terminal.fills)).fills
            self.book.rebind_trade(terminal)
            self.recovered_fills = [
                (terminal if recovered is trade else recovered, fill)
                for recovered, fill in self.recovered_fills
            ]
            self.unresolved.remove(trade)
            self.done.append(terminal)

    def report(self) -> Self:
        if self.recovered_fills or any(self.lists):
            log.debug(
                "Order sync: unknown=%s recovered_fills=%s done=%s unmatched=%s",
                len(self.unknown),
                len(self.recovered_fills),
                len(self.done),
                len(self.errors),
            )
        return self

    @property
    def is_ok(self) -> bool:
        return not (any(self.lists) or self.unresolved)

    @property
    def is_error(self) -> bool:
        return bool(self.errors or self.unknown or self.unresolved)


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
        logical = self.book.positions.by_contract()
        self.errors = {
            contract: logical.get(contract, 0.0)
            - self.broker_positions.get(contract, 0.0)
            for contract in set(self.broker_positions) | set(logical)
            if logical.get(contract, 0.0) != self.broker_positions.get(contract, 0.0)
        }
        return self

    def report(self) -> Self:
        if self.errors:
            log.debug(
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
