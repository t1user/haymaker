"""Order attribution, broker execution evidence and order serialization."""

from __future__ import annotations

from collections import defaultdict
from ..saver import AbstractBaseSaver
from .persistence import PersistenceWriter

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
    qualified_contract,
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
    _persistence_priority: int = field(default=0, repr=False)

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

        timestamp = max(
            (entry.time for entry in self.trade.log), default=self.submitted_at
        )
        return max(self._persistence_priority, int(timestamp.timestamp() * 1000))

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
            _persistence_priority=int(data.get("priority", 0)),
        )


class OrderStore:
    """Own broker order records, their queries and evidence persistence.

    Book coordinates mutations that also affect positions. The underscored
    mutation methods are internal; public methods query the owned records.
    """

    def __init__(self, saver: AbstractBaseSaver, writer: PersistenceWriter) -> None:
        """Use Book's shared writer; never construct an independent queue."""
        self._saver = saver
        self._writer = writer
        self._items: dict[int, OrderInfo] = {}
        self._received_keys: defaultdict[str, set[str]] = defaultdict(set)
        self._rejections: defaultdict[str, int] = defaultdict(int)

    def _put(self, info: OrderInfo) -> None:
        """Persist order evidence before the coordinator updates projections."""
        if not info.orderId:
            raise ValueError("Cannot persist an order with orderId 0")
        info._persistence_priority = info.priority
        self._writer.save(self._saver, info.encode())
        self._items[info.orderId] = info
        self._index_fills(info)

    def _index_fills(self, info: OrderInfo) -> None:
        """Maintain source checkpoints without scanning historical orders on update."""
        if info.source_key is not None and not isinstance(info.trade.contract, ibi.Bag):
            self._received_keys[info.source_key].update(
                record.deduplication_key for record in info.fills
            )

    def received_fill_keys(self, source_key: str) -> frozenset[str]:
        """Return evidence already received for one source."""
        return frozenset(self._received_keys.get(source_key, ()))

    def _restore(self, order_documents: Sequence[Mapping[str, Any]]) -> None:
        """Restore canonical records, excluding documents superseded by rebinding."""
        self._items = {}
        for document in order_documents:
            document = dict(document)
            document.pop("_id", None)
            info = OrderInfo.decode(document)
            if not info.orderId:
                raise ValueError("Persisted order must have a non-zero orderId")
            self._items[info.orderId] = info
        superseded = {
            order_id
            for info in self._items.values()
            for order_id in info.previous_order_ids
        }
        for info in tuple(self._items.values()):
            for order_id in info.previous_order_ids:
                previous = self._items.get(order_id)
                if previous is not None and (
                    not info.permId or previous.permId != info.permId
                ):
                    raise ValueError("Rebound order history has conflicting permId")
        current = {
            key: info for key, info in self._items.items() if key not in superseded
        }
        owners = {info.permId for info in current.values()}
        if any(info.permId not in owners for info in self._items.values()):
            raise ValueError("Rebound order history has no unambiguous current record")
        self._items = current

        self._received_keys.clear()
        for info in self._items.values():
            self._index_fills(info)

    def _rebind(self, trade: ibi.Trade) -> tuple[OrderInfo | None, int]:
        """Bind a known live Trade and return its record and previous broker ID."""
        info = self.by_id(trade.order.orderId)
        if info is None:
            info = self.by_perm_id(trade.order.permId)
        elif info.permId and trade.order.permId and info.permId != trade.order.permId:
            raise ValueError("Broker orderId and permId identify different orders")
        if info is None:
            return None, 0
        old_order_id = info.orderId
        if self._items.get(old_order_id) is not info:
            # IB may mutate the same Trade in place before delivering its event.
            old_order_id = next(
                key for key, saved in self._items.items() if saved is info
            )
        if not trade.order.orderId:
            trade.order.orderId = old_order_id
        if trade.order.orderId in info.previous_order_ids:
            raise ValueError(
                "Cannot reuse a superseded broker orderId during rebinding"
            )
        if old_order_id != trade.order.orderId:
            info.previous_order_ids = tuple(
                key
                for key in dict.fromkeys((*info.previous_order_ids, old_order_id))
                if key != trade.order.orderId
            )
        # Recovered Trades can have shorter logs. Their newer evidence must not
        # lose MongoSaver's priority comparison against the prior document.
        info._persistence_priority = info.priority
        info.trade = trade
        if old_order_id != info.orderId:
            self._items.pop(old_order_id, None)
        return info, old_order_id

    def by_id(self, order_id: int, *, active_only: bool = False) -> OrderInfo | None:
        """Look up an order by actual broker orderId."""

        info = self._items.get(order_id)
        if info is None or (active_only and not info.active):
            return None
        return info

    def by_perm_id(self, perm_id: int) -> OrderInfo | None:
        """Fall back to broker permanent-id order lookup."""

        if not perm_id:
            return None
        return next(
            (info for info in self._items.values() if info.permId == perm_id),
            None,
        )

    def query(
        self,
        *,
        source_key: str | None = None,
        contract: ibi.Contract | None = None,
        role: str | None = None,
        execution_model_name: str | None = None,
        active_only: bool = False,
    ) -> tuple[OrderInfo, ...]:
        """Query persisted orders using any supported attribution fields."""

        con_id = contract.conId if contract is not None else None
        return tuple(
            info
            for info in self._items.values()
            if (not active_only or info.active)
            and (source_key is None or info.source_key == source_key)
            and (con_id is None or info.trade.contract.conId == con_id)
            and (role is None or info.role == role)
            and (
                execution_model_name is None
                or info.execution_model_name == execution_model_name
            )
        )

    def active(
        self,
        *,
        source_key: str | None = None,
        contract: ibi.Contract | None = None,
        role: str | None = None,
        execution_model_name: str | None = None,
    ) -> tuple[OrderInfo, ...]:
        """Query active orders using any supported attribution fields."""

        return self.query(
            source_key=source_key,
            contract=contract,
            role=role,
            execution_model_name=execution_model_name,
            active_only=True,
        )

    def owner_for_contract(
        self,
        contract: ibi.Contract,
        *,
        role: str | None = None,
    ) -> str | None:
        """Return the unambiguous active-order owner for one Contract.

        Args:
            contract: Concrete qualified Contract used to find active orders.
            role: Optional exact order-role filter.

        Returns:
            The execution-model name, or ``None`` when no order matches.

        Raises:
            RuntimeError: More than one model owns matching active orders.
        """

        con_id = qualified_contract(contract).conId
        candidates = {
            info.execution_model_name
            for info in self.active(contract=contract, role=role)
        }
        return self._one_active_order_model(candidates, f"conId={con_id}")

    @staticmethod
    def _one_active_order_model(candidates: set[str], identity: str) -> str | None:
        """Require unambiguous ownership among relevant working orders."""

        if len(candidates) > 1:
            raise RuntimeError(
                f"Ambiguous active-order ownership for {identity}: "
                f"{sorted(candidates)}"
            )
        return next(iter(candidates), None)

    def register_rejection(self, model_name: str) -> None:
        """Count one rejection for the current process, not durable recovery."""
        self._rejections[model_name] += 1

    def rejection_count(self, model_name: str) -> int:
        """Return this process's rejection count for a configured model."""
        return self._rejections.get(model_name, 0)
