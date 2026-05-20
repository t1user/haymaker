from __future__ import annotations

import datetime as dt
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field

import ib_insync as ibi

from haymaker.state_machine import OrderInfo, Strategy


@dataclass(frozen=True)
class SyncResult:
    """Outcome of broker/local synchronization."""

    ok: bool
    reason: str = "ok"


@dataclass(frozen=True)
class BrokerSnapshot:
    """Broker state captured once for a sync cycle."""

    positions: tuple[ibi.Position, ...]
    open_trades: tuple[ibi.Trade, ...]
    trades: tuple[ibi.Trade, ...]
    fills: tuple[ibi.Fill, ...]
    captured_at: dt.datetime

    @property
    def positions_by_contract(self) -> dict[ibi.Contract, float]:
        """Return broker positions keyed by contract."""
        return {position.contract: position.position for position in self.positions}

    @property
    def open_trades_by_order_id(self) -> dict[int, ibi.Trade]:
        """Return active broker trades keyed by order id."""
        return {trade.order.orderId: trade for trade in self.open_trades}

    @property
    def trades_by_order_id_or_perm_id(self) -> dict[int, ibi.Trade]:
        """Return session trades keyed by order id, falling back to perm id."""
        return {
            trade.order.orderId or trade.order.permId: trade for trade in self.trades
        }

    @property
    def fills_by_order_id(self) -> defaultdict[int, list[ibi.Fill]]:
        """Return broker fills grouped by order id."""
        fills: defaultdict[int, list[ibi.Fill]] = defaultdict(list)
        for fill in self.fills:
            fills[fill.execution.orderId].append(fill)
        return fills

    def position_for_contract(self, contract: ibi.Contract) -> float:
        """Return broker position for ``contract`` from this snapshot."""
        return self.positions_by_contract.get(contract, 0.0)

    def open_trades_for_contract(self, contract: ibi.Contract) -> list[ibi.Trade]:
        """Return active broker trades for ``contract`` from this snapshot."""
        return [trade for trade in self.open_trades if trade.contract == contract]

    def position_is_protected(
        self, contract: ibi.Contract, broker_position: float
    ) -> bool:
        """Return True if snapshot contains enough opposite-side stop protection."""
        protective_action = "SELL" if broker_position > 0 else "BUY"
        protective_order_types = {
            "STP",
            "STP LMT",
            "TRAIL",
            "TRAIL LIMIT",
            "TRAILLIT",
            "TRAIL LIT",
            "TRAILLMT",
            "TRAIL LMT",
        }
        protective_quantity = sum(
            trade.order.totalQuantity
            for trade in self.open_trades_for_contract(contract)
            if (
                trade.isActive()
                and trade.order.action == protective_action
                and trade.order.orderType in protective_order_types
            )
        )
        return protective_quantity >= abs(broker_position)


@dataclass(frozen=True)
class LocalSnapshot:
    """State-machine data captured once for a sync phase."""

    orders_by_id: Mapping[int, OrderInfo]
    strategies: Mapping[str, Strategy]
    positions_by_contract: Mapping[ibi.Contract, float]
    strategies_by_contract: Mapping[ibi.Contract, list[str]]
    orders_by_strategy: Mapping[str, tuple[OrderInfo, ...]]

    def orders_for_strategy(self, strategy: str) -> tuple[OrderInfo, ...]:
        """Return active order records for ``strategy``."""
        return self.orders_by_strategy.get(strategy, ())


@dataclass(frozen=True)
class OrderFindings:
    """Read-only order differences between local and broker state."""

    unknown_broker_trades: tuple[ibi.Trade, ...] = ()
    broker_trade_updates: tuple[tuple[OrderInfo, ibi.Trade], ...] = ()
    inactive_local_orders: tuple[OrderInfo, ...] = ()
    missing_active_orders: tuple[OrderInfo, ...] = ()

    @property
    def has_findings(self) -> bool:
        """Return True when any order discrepancy was found."""
        return any(
            (
                self.unknown_broker_trades,
                self.broker_trade_updates,
                self.inactive_local_orders,
                self.missing_active_orders,
            )
        )


@dataclass
class OrderRecoveryResult:
    """Actions taken while applying order reconciliation findings."""

    unknown_broker_trades: list[ibi.Trade] = field(default_factory=list)
    updated_orders: list[OrderInfo] = field(default_factory=list)
    pruned_orders: list[OrderInfo] = field(default_factory=list)
    done_trades: list[ibi.Trade] = field(default_factory=list)
    faulty_orders: list[OrderInfo] = field(default_factory=list)

    @property
    def recovery_happened(self) -> bool:
        """Return True when order sync should suppress correction trades."""
        return any(
            (
                self.unknown_broker_trades,
                self.done_trades,
                self.faulty_orders,
            )
        )


@dataclass(frozen=True)
class PositionFindings:
    """Read-only position differences between local and broker state."""

    errors: Mapping[ibi.Contract, float] = field(default_factory=dict)


@dataclass(frozen=True)
class BracketIssue:
    """Missing bracket information for one strategy."""

    strategy: Strategy
    existing_orders: tuple[OrderInfo, ...]
    expected_bracket_count: int


@dataclass(frozen=True)
class BracketFindings:
    """Read-only order/position bracket findings."""

    obsolete_orders: tuple[tuple[str, OrderInfo], ...] = ()
    missing_brackets: tuple[BracketIssue, ...] = ()

    @property
    def has_findings(self) -> bool:
        """Return True when any bracket discrepancy was found."""
        return bool(self.obsolete_orders or self.missing_brackets)
