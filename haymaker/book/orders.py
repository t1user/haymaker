"""Order attribution, broker execution evidence and order serialization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Literal

import ib_insync as ibi

from ..misc import action_to_signal, decode_tree, tree
from ..validators import (
    aware_datetime,
    finite_number,
    ib_contract,
    non_empty_string,
    readonly_mapping,
)


def fill_direction(record: FillRecord) -> Literal[-1, 1]:
    """Resolve the broker side of a normalized execution, rejecting ambiguity."""
    if record.execution.side == "BOT":
        return 1
    if record.execution.side == "SLD":
        return -1
    raise ValueError(f"Ambiguous fill side: {record.execution.side}")


def _execution_key(trade: ibi.Trade, fill: ibi.Fill) -> str:
    """Return a stable broker execution identity."""

    if fill.execution.execId:
        return fill.execution.execId
    execution = fill.execution
    timestamp = execution.time.isoformat() if execution.time else ""
    return (
        f"{trade.order.permId}:{trade.order.orderId}:{timestamp}:"
        f"{execution.side}:{execution.shares}:{execution.price}"
    )


@dataclass(frozen=True, kw_only=True)
class FillRecord:
    """Persist complete broker execution evidence for one fill."""

    execution: ibi.Execution
    time: datetime
    contract: ibi.Contract
    commission_report: ibi.CommissionReport | None = None
    deduplication_key: str

    def __post_init__(self) -> None:
        if not isinstance(self.execution, ibi.Execution):
            raise TypeError("execution must be an ib_insync.Execution")
        ib_contract(self.contract)
        aware_datetime(self.time, "time")
        if self.commission_report is not None and not isinstance(
            self.commission_report, ibi.CommissionReport
        ):
            raise TypeError(
                "commission_report must be None or an ib_insync.CommissionReport"
            )
        object.__setattr__(
            self,
            "deduplication_key",
            non_empty_string(self.deduplication_key, "deduplication_key"),
        )

    @classmethod
    def from_fill(cls, trade: ibi.Trade, fill: ibi.Fill) -> FillRecord:
        """Create a normalized record from one IB fill callback."""

        return cls(
            execution=fill.execution,
            time=aware_datetime(fill.time, "fill.time"),
            contract=fill.contract,
            commission_report=(
                fill.commissionReport
                if fill.commissionReport.execId
                or fill.commissionReport.commission
                or fill.commissionReport.realizedPNL
                else None
            ),
            deduplication_key=_execution_key(trade, fill),
        )

    def to_fill(self) -> ibi.Fill:
        """Return IB-shaped execution evidence, without requiring a commission."""
        return ibi.Fill(
            self.contract,
            self.execution,
            self.commission_report or ibi.CommissionReport(),
            self.time,
        )

    def encode(self) -> dict[str, Any]:
        """Return a persistence-ready normalized fill document."""

        return {
            "execution": tree(self.execution),
            "time": self.time,
            "contract": tree(self.contract),
            "commission_report": (
                tree(self.commission_report)
                if self.commission_report is not None
                else None
            ),
            "deduplication_key": self.deduplication_key,
        }

    @classmethod
    def decode(cls, data: Mapping[str, Any]) -> FillRecord:
        """Restore a FillRecord from its persisted representation."""

        return cls(
            execution=decode_tree(data["execution"]),
            time=decode_tree(data["time"]),
            contract=decode_tree(data["contract"]),
            commission_report=decode_tree(data.get("commission_report")),
            deduplication_key=str(data["deduplication_key"]),
        )


@dataclass(kw_only=True)
class OrderInfo:
    """Attribute one live or historical IB Trade to framework execution.

    Args:
        trade: Complete IB Trade object.
        role: Standard or custom order role.
        submitted_at: Time the controller accepted the submission.
        execution_model_name: Stable configured model identity.
        source_key: Optional one-to-one input identity.
        position_id: Optional independently managed position episode.
        params: Diagnostic inputs captured at submission.
        fills: Normalized execution evidence already observed.
    """

    trade: ibi.Trade
    role: str
    submitted_at: datetime
    execution_model_name: str
    source_key: str | None = None
    position_id: str | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    fills: Sequence[FillRecord] = field(default_factory=tuple)
    _applied_fill_keys: set[str] = field(default_factory=set, repr=False)
    previous_order_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.trade, ibi.Trade):
            raise TypeError("trade must be an ib_insync.Trade")
        self.role = non_empty_string(self.role, "role")
        aware_datetime(self.submitted_at, "submitted_at")
        self.execution_model_name = non_empty_string(
            self.execution_model_name, "execution_model_name"
        )
        if self.source_key is not None:
            self.source_key = non_empty_string(self.source_key, "source_key")
        self.params = readonly_mapping(self.params, "params")
        self.fills = tuple(self.fills)
        self._applied_fill_keys.update(
            record.deduplication_key for record in self.fills
        )

    @property
    def orderId(self) -> int:
        """Return the broker client-local order id."""

        return self.trade.order.orderId

    @property
    def permId(self) -> int:
        """Return the broker permanent order id."""

        return self.trade.order.permId

    @property
    def active(self) -> bool:
        """Return whether IB reports this Trade as working."""

        return self.trade.isActive()

    @property
    def priority(self) -> int:
        """Return a monotonically useful order-status persistence priority."""

        if not self.trade.log:
            return int(self.submitted_at.timestamp() * 1000)
        return int(max(entry.time.timestamp() for entry in self.trade.log) * 1000)

    @property
    def signed_total_quantity(self) -> float:
        """Return the signed submitted order quantity."""

        return self.trade.order.totalQuantity * action_to_signal(
            self.trade.order.action
        )

    @property
    def signed_working_quantity(self) -> float:
        """Return the unfilled signed quantity still working at IB."""

        if not self.active:
            return 0.0
        filled = sum(record.execution.shares for record in self.fills)
        remaining = max(float(self.trade.order.totalQuantity) - filled, 0.0)
        return remaining * action_to_signal(self.trade.order.action)

    def add_fill(self, record: FillRecord) -> bool:
        """Append unseen execution evidence and report whether it was new."""

        if record.deduplication_key in self._applied_fill_keys:
            return False
        self.fills = (*self.fills, record)
        self._applied_fill_keys.add(record.deduplication_key)
        return True

    def execution_trade(self, extra_fills: Sequence[ibi.Fill] = ()) -> ibi.Trade:
        """Reconstruct fill quantity and price from normalized execution evidence.

        Saved fills and new broker fills are merged by execution identity.
        This does not account or persist new fills; Controller owns that step.
        Combo-leg fills do not count as additional BAG fills.

        Args:
            extra_fills: Broker fills belonging to this order.

        Raises:
            ValueError: If repeated executions disagree or quantities are invalid.
        """
        records = {record.deduplication_key: record for record in self.fills}
        for fill in extra_fills:
            record = FillRecord.from_fill(self.trade, fill)
            previous = records.get(record.deduplication_key)
            if previous is not None and (
                previous.execution.shares != record.execution.shares
                or previous.execution.price != record.execution.price
                or previous.execution.side != record.execution.side
                or previous.contract != record.contract
            ):
                raise ValueError(f"Conflicting execution {record.deduplication_key!r}")
            if previous is None:
                records[record.deduplication_key] = record
        fills = [record.to_fill() for record in records.values()]
        quantity = 0.0
        cost = 0.0
        for fill in fills:
            if (
                isinstance(self.trade.contract, ibi.Bag)
                and fill.contract.secType != "BAG"
            ):
                continue
            shares = finite_number(fill.execution.shares, "Execution shares")
            price = finite_number(fill.execution.price, "Execution price")
            if shares <= 0:
                raise ValueError("Execution shares must be positive")
            quantity += shares
            cost += shares * price
        if quantity > self.trade.order.totalQuantity:
            raise ValueError("Execution quantity exceeds submitted order quantity")
        status = replace(
            self.trade.orderStatus,
            filled=quantity,
            remaining=self.trade.order.totalQuantity - quantity,
            avgFillPrice=cost / quantity if quantity else 0.0,
        )
        if quantity and status.remaining == 0:
            status.status = ibi.OrderStatus.Filled
        return ibi.Trade(
            contract=self.trade.contract,
            order=self.trade.order,
            orderStatus=status,
            fills=fills,
            log=list(self.trade.log),
        )

    def encode(self) -> dict[str, Any]:
        """Return the complete order document stored by Book."""

        return {
            "orderId": self.orderId,
            "clientId": self.trade.order.clientId,
            "permId": self.permId,
            "trade": tree(self.trade),
            "role": self.role,
            "submitted_at": self.submitted_at,
            "execution_model_name": self.execution_model_name,
            "source_key": self.source_key,
            "position_id": self.position_id,
            "params": tree(dict(self.params)),
            "fills": [record.encode() for record in self.fills],
            "applied_fill_keys": sorted(self._applied_fill_keys),
            "active": self.active,
            "priority": self.priority,
            "previous_order_ids": list(self.previous_order_ids),
        }

    @classmethod
    def decode(cls, data: Mapping[str, Any]) -> OrderInfo:
        """Restore an OrderInfo from the current Book schema."""

        if "target_key" in data:
            raise ValueError("Old keyed order schema requires standalone conversion")
        return cls(
            trade=decode_tree(data["trade"]),
            role=str(data["role"]),
            submitted_at=decode_tree(data["submitted_at"]),
            execution_model_name=str(data["execution_model_name"]),
            source_key=data.get("source_key"),
            position_id=data.get("position_id"),
            params=decode_tree(data.get("params", {})),
            fills=tuple(FillRecord.decode(fill) for fill in data.get("fills", ())),
            _applied_fill_keys=set(data.get("applied_fill_keys", ())),
            previous_order_ids=tuple(data.get("previous_order_ids", ())),
        )
