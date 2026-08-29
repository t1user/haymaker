"""Typed persistent accounting and recovery state for live execution."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Literal
from uuid import uuid4

import ib_insync as ibi

from .async_wrappers import QueueShutdownPolicy, SyncQueueRunner
from .blotter import Blotter
from .misc import action_to_signal, decode_tree, sign, tree
from .saver import AbstractBaseSaver, MongoSaver
from .validators import (
    aware_datetime,
    finite_number,
    ib_contract,
    non_empty_string,
    qualified_contract,
    readonly_mapping,
)

log = logging.getLogger(__name__)

DEFAULT_ORDER_COLLECTION_NAME = "orders"
DEFAULT_STATE_COLLECTION_NAME = "state"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _contract_key(contract: ibi.Contract) -> int:
    return qualified_contract(contract).conId


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

    def __post_init__(self) -> None:
        if not isinstance(self.trade, ibi.Trade):
            raise TypeError("trade must be an ib_insync.Trade")
        self.role = non_empty_string(self.role, "role")
        aware_datetime(self.submitted_at, "submitted_at")
        self.execution_model_name = non_empty_string(
            self.execution_model_name, "execution_model_name"
        )
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
        }

    @classmethod
    def decode(cls, data: Mapping[str, Any]) -> OrderInfo:
        """Restore an OrderInfo from the current Book schema."""

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
        )


@dataclass(frozen=True, kw_only=True)
class PositionState:
    """Recover one independently managed one-to-one position episode."""

    source_key: str
    execution_model_name: str
    contract: ibi.Contract | None = None
    quantity: float = 0.0
    target_quantity: float | None = None
    target_created_at: datetime | None = None
    position_id: str | None = None
    blocked_direction: Literal[-1, 1] | None = None
    bracket_inputs: Mapping[str, Any] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_key", non_empty_string(self.source_key, "source_key")
        )
        object.__setattr__(
            self,
            "execution_model_name",
            non_empty_string(self.execution_model_name, "execution_model_name"),
        )
        if self.contract is not None:
            ib_contract(self.contract)
        if isinstance(self.blocked_direction, bool) or (
            self.blocked_direction not in (None, -1, 1)
        ):
            raise ValueError("blocked_direction must be None, -1, or 1")
        object.__setattr__(self, "quantity", finite_number(self.quantity, "quantity"))
        if self.target_quantity is not None:
            object.__setattr__(
                self,
                "target_quantity",
                finite_number(self.target_quantity, "target_quantity"),
            )
        if self.target_created_at is not None:
            aware_datetime(self.target_created_at, "target_created_at")
        aware_datetime(self.updated_at, "updated_at")
        object.__setattr__(
            self,
            "bracket_inputs",
            readonly_mapping(self.bracket_inputs, "bracket_inputs"),
        )


@dataclass(frozen=True, kw_only=True)
class TargetState:
    """Recover the latest shared direct-execution target."""

    execution_model_name: str
    contract: ibi.Contract
    target_quantity: float
    target_created_at: datetime
    updated_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "execution_model_name",
            non_empty_string(self.execution_model_name, "execution_model_name"),
        )
        _contract_key(self.contract)
        object.__setattr__(
            self,
            "target_quantity",
            finite_number(self.target_quantity, "target_quantity"),
        )
        aware_datetime(self.target_created_at, "target_created_at")
        aware_datetime(self.updated_at, "updated_at")


class Book:
    """Own typed accounting, order recovery, persistence, and blotter access.

    Book performs no allocation and no broker API calls. All mutations use one
    ordered critical queue when asynchronous saving is enabled, so order
    evidence and the resulting projections retain deterministic persistence
    order.
    """

    def __init__(
        self,
        *,
        order_saver: AbstractBaseSaver | None = None,
        state_saver: AbstractBaseSaver | None = None,
        blotter: Blotter | None = None,
        save_async: bool = True,
        max_rejected_orders: int = 3,
        restore: bool = False,
    ) -> None:
        self._order_saver = order_saver or MongoSaver(
            DEFAULT_ORDER_COLLECTION_NAME, query_key="orderId"
        )
        self._state_saver = state_saver or MongoSaver(
            DEFAULT_STATE_COLLECTION_NAME, query_key="state_key"
        )
        self.blotter = blotter
        self.max_rejected_orders = max_rejected_orders
        self._orders: dict[int, OrderInfo] = {}
        self._positions: dict[str, PositionState] = {}
        self._targets: dict[tuple[str, int], TargetState] = {}
        self._portfolio_states: dict[str, Mapping[str, Any]] = {}
        self._rejected_orders: defaultdict[str, int] = defaultdict(int)
        self._save_async = save_async
        if restore:
            self._restore_documents(*self._read_documents())
        self._mutation_queue = (
            SyncQueueRunner(
                "Book",
                shutdown_policy=QueueShutdownPolicy.DRAIN,
            )
            if save_async
            else None
        )

    def _save(self, saver: AbstractBaseSaver, document: dict[str, Any]) -> None:
        """Persist one mutation in Book's deterministic write order."""

        if self._mutation_queue is None:
            saver.save(document)
        else:
            self._mutation_queue.enqueue(saver.save, document)

    async def close(self) -> None:
        """Drain critical pending mutations before shutdown."""

        if self._mutation_queue is not None:
            await self._mutation_queue.close()

    def clear_state(self) -> None:
        """Durably flatten and clear typed recovery projections.

        This is reserved for explicit Controller reset/zero startup actions;
        historical order evidence remains intact. Flat records are persisted
        before in-memory removal so stale targets cannot return after restart.
        """

        cleared_at = _utc_now()
        for position in self._positions.values():
            cleared_position = replace(
                position,
                quantity=0.0,
                target_quantity=0.0,
                target_created_at=cleared_at,
                position_id=None,
                blocked_direction=None,
                bracket_inputs={},
                updated_at=cleared_at,
            )
            self._save(
                self._state_saver,
                self._encode_position(cleared_position),
            )
        for target in self._targets.values():
            cleared_target = replace(
                target,
                target_quantity=0.0,
                target_created_at=cleared_at,
                updated_at=cleared_at,
            )
            self._save(
                self._state_saver,
                self._encode_target(cleared_target),
            )
        for portfolio_key in self._portfolio_states:
            self._save(
                self._state_saver,
                {
                    "state_key": f"portfolio:{portfolio_key}",
                    "state_type": "portfolio",
                    "portfolio_key": portfolio_key,
                    "state": {},
                    "updated_at": cleared_at,
                },
            )
        self._positions.clear()
        self._targets.clear()
        self._portfolio_states.clear()

    def _restore_documents(
        self,
        order_documents: Sequence[Mapping[str, Any]],
        state_documents: Sequence[Mapping[str, Any]],
    ) -> None:
        """Replace in-memory state with decoded persistence documents."""

        self._orders = {}
        for document in order_documents:
            document = dict(document)
            document.pop("_id", None)
            info = OrderInfo.decode(document)
            if not info.orderId:
                raise ValueError("Persisted order must have a non-zero orderId")
            self._orders[info.orderId] = info
        self._positions = {}
        self._targets = {}
        self._portfolio_states = {}
        for raw in state_documents:
            document = dict(raw)
            document.pop("_id", None)
            state_type = document.get("state_type")
            if state_type == "position":
                position_state = self._decode_position(document)
                self._positions[position_state.source_key] = position_state
            elif state_type == "target":
                target_state = self._decode_target(document)
                self._targets[
                    (
                        target_state.execution_model_name,
                        target_state.contract.conId,
                    )
                ] = target_state
            elif state_type == "portfolio":
                self._portfolio_states[str(document["portfolio_key"])] = (
                    MappingProxyType(decode_tree(document.get("state", {})))
                )
            else:
                raise ValueError(f"Unknown Book state_type: {state_type!r}")

    def _read_documents(
        self,
    ) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
        """Read both physical recovery collections in one worker call."""

        return self._order_saver.read({}), self._state_saver.read({})

    def save_order(self, info: OrderInfo) -> OrderInfo:
        """Register or update one complete order record."""

        if not info.orderId:
            raise ValueError("Cannot persist an order with orderId 0")
        self._orders[info.orderId] = info
        self._save(self._order_saver, info.encode())
        return info

    def order_by_id(
        self, order_id: int, *, active_only: bool = False
    ) -> OrderInfo | None:
        """Look up an order by actual broker orderId."""

        info = self._orders.get(order_id)
        if info is None or (active_only and not info.active):
            return None
        return info

    def order_by_perm_id(self, perm_id: int) -> OrderInfo | None:
        """Fall back to broker permanent-id order lookup."""

        if not perm_id:
            return None
        return next(
            (info for info in self._orders.values() if info.permId == perm_id),
            None,
        )

    def active_orders(
        self,
        *,
        source_key: str | None = None,
        contract: ibi.Contract | None = None,
        role: str | None = None,
        execution_model_name: str | None = None,
    ) -> tuple[OrderInfo, ...]:
        """Query active orders using any supported attribution fields."""

        con_id = contract.conId if contract is not None else None
        return tuple(
            info
            for info in self._orders.values()
            if info.active
            and (source_key is None or info.source_key == source_key)
            and (con_id is None or info.trade.contract.conId == con_id)
            and (role is None or info.role == role)
            and (
                execution_model_name is None
                or info.execution_model_name == execution_model_name
            )
        )

    def prune_order(self, order_id: int) -> None:
        """Mark an unrecoverable order inactive while retaining evidence."""

        info = self._orders.get(order_id)
        if info is None:
            return
        info.trade.orderStatus.status = ibi.OrderStatus.Cancelled
        self.save_order(info)

    def rebind_trade(self, trade: ibi.Trade) -> ibi.Trade | None:
        """Replace a recovered stored Trade with the current live IB object.

        Returns:
            The unknown trade when no orderId or permId match exists, otherwise
            ``None`` after any necessary rebind.
        """

        info = self.order_by_id(trade.order.orderId) or self.order_by_perm_id(
            trade.order.permId
        )
        if info is None:
            return trade
        if info.trade is not trade:
            old_order_id = info.orderId
            if not trade.order.orderId:
                trade.order.orderId = old_order_id
            info.trade = trade
            if old_order_id != info.orderId:
                self._orders.pop(old_order_id, None)
            self.save_order(info)
        return None

    def position_state(self, source_key: str) -> PositionState | None:
        """Return one recovered one-to-one state without creating it."""

        return self._positions.get(source_key)

    def position_states(self) -> Mapping[str, PositionState]:
        """Return a read-only view of all one-to-one states."""

        return MappingProxyType(self._positions)

    def update_position(self, state: PositionState) -> PositionState:
        """Replace and persist one one-to-one state."""

        self._positions[state.source_key] = state
        self._save(self._state_saver, self._encode_position(state))
        return state

    def target_state(
        self, execution_model_name: str, contract: ibi.Contract
    ) -> TargetState | None:
        """Return the latest direct target for a model and concrete Contract."""

        return self._targets.get((execution_model_name, _contract_key(contract)))

    def update_target(self, state: TargetState) -> TargetState:
        """Replace and persist one latest direct target."""

        key = (state.execution_model_name, _contract_key(state.contract))
        current = self._targets.get(key)
        if current is not None and (
            state.target_created_at < current.target_created_at
        ):
            return current
        self._targets[key] = state
        self._save(self._state_saver, self._encode_target(state))
        return state

    def latest_target(
        self,
        execution_model_name: str,
        contract: ibi.Contract | None = None,
    ) -> TargetState | tuple[TargetState, ...] | None:
        """Recover globally current targets owned by one execution model."""

        if contract is not None:
            state = self.latest_target_for_contract(contract)
            if state is not None and state.execution_model_name == execution_model_name:
                return state
            return None
        return tuple(
            state
            for state in self.latest_targets()
            if state.execution_model_name == execution_model_name
        )

    def latest_targets(self) -> tuple[TargetState, ...]:
        """Return one globally current direct target per concrete Contract."""

        latest: dict[int, TargetState] = {}
        for state in self._targets.values():
            con_id = _contract_key(state.contract)
            current = latest.get(con_id)
            if current is None or (
                state.target_created_at,
                state.updated_at,
            ) > (
                current.target_created_at,
                current.updated_at,
            ):
                latest[con_id] = state
        return tuple(latest.values())

    def latest_target_for_contract(self, contract: ibi.Contract) -> TargetState | None:
        """Return the globally current direct target for one Contract."""

        con_id = _contract_key(contract)
        return next(
            (
                state
                for state in self.latest_targets()
                if state.contract.conId == con_id
            ),
            None,
        )

    def effective_quantity(self, source_key: str) -> float:
        """Return fill-accounted quantity plus relevant working episode orders."""

        state = self._positions.get(source_key)
        quantity = state.quantity if state is not None else 0.0
        working = sum(
            info.signed_working_quantity
            for info in self.active_orders(source_key=source_key)
            if info.role in {"OPEN", "CLOSE", "TARGET_ADJUSTMENT", "ROLL"}
        )
        return quantity + working

    def aggregate_quantity(self, contract: ibi.Contract) -> float:
        """Return aggregate fill-accounted logical quantity for a Contract."""

        con_id = _contract_key(contract)
        one_to_one = sum(
            state.quantity
            for state in self._positions.values()
            if state.contract is not None and state.contract.conId == con_id
        )
        direct = 0.0
        for info in self._orders.values():
            if info.source_key is not None or info.trade.contract.conId != con_id:
                continue
            direct += sum(
                record.execution.shares * (1 if record.execution.side == "BOT" else -1)
                for record in info.fills
            )
        return one_to_one + direct

    def logical_positions(self) -> dict[ibi.Contract, float]:
        """Return non-zero aggregate logical quantities by concrete Contract."""

        contracts: dict[int, ibi.Contract] = {}
        for state in self._positions.values():
            if state.contract is not None and state.contract.conId:
                contracts[state.contract.conId] = state.contract
        for info in self._orders.values():
            contract = info.trade.contract
            if contract.conId:
                contracts[contract.conId] = contract
        return {
            contract: quantity
            for contract in contracts.values()
            if (quantity := self.aggregate_quantity(contract))
        }

    def aggregate_with_working(
        self, contract: ibi.Contract, execution_model_name: str | None = None
    ) -> float:
        """Return aggregate logical quantity including matching working orders."""

        return self.aggregate_quantity(contract) + sum(
            info.signed_working_quantity
            for info in self.active_orders(
                contract=contract,
                execution_model_name=execution_model_name,
            )
            if info.role in {"OPEN", "CLOSE", "TARGET_ADJUSTMENT", "ROLL"}
        )

    def positions_for_contract(
        self, contract: ibi.Contract
    ) -> tuple[PositionState, ...]:
        """Return non-flat one-to-one states attributed to a Contract."""

        con_id = _contract_key(contract)
        return tuple(
            state
            for state in self._positions.values()
            if state.contract is not None
            and state.contract.conId == con_id
            and state.quantity
        )

    def create_position_episode(
        self,
        source_key: str,
        execution_model_name: str,
        contract: ibi.Contract,
        *,
        target_quantity: float,
        target_created_at: datetime,
        bracket_inputs: Mapping[str, Any] = MappingProxyType({}),
    ) -> PositionState:
        """Create and persist a fresh independently managed episode."""

        state = PositionState(
            source_key=source_key,
            execution_model_name=execution_model_name,
            contract=contract,
            target_quantity=target_quantity,
            target_created_at=target_created_at,
            position_id=str(uuid4()),
            bracket_inputs=bracket_inputs,
        )
        return self.update_position(state)

    def close_position_episode(self, source_key: str) -> PositionState:
        """Mark one source flat while preserving its recovery attribution."""

        state = self._positions[source_key]
        return self.update_position(
            replace(
                state,
                quantity=0.0,
                target_quantity=0.0,
                position_id=None,
                bracket_inputs={},
                updated_at=_utc_now(),
            )
        )

    def blocked_direction(self, source_key: str) -> Literal[-1, 1] | None:
        """Return the persisted stopped direction for a one-to-one source."""

        state = self._positions.get(source_key)
        return state.blocked_direction if state is not None else None

    def update_blocked_direction(
        self, source_key: str, direction: Literal[-1, 1] | None
    ) -> PositionState:
        """Persist or clear a one-to-one stop-out direction."""

        if isinstance(direction, bool) or direction not in (None, -1, 1):
            raise ValueError("direction must be None, -1, or 1")
        state = self._positions[source_key]
        return self.update_position(
            replace(state, blocked_direction=direction, updated_at=_utc_now())
        )

    def apply_fill(self, trade: ibi.Trade, fill: ibi.Fill) -> bool:
        """Conditionally apply one unseen Fill and persist evidence first.

        Returns:
            ``True`` when the fill was new and projections changed, otherwise
            ``False`` for a replayed execution.
        """

        info = self.order_by_id(trade.order.orderId) or self.order_by_perm_id(
            trade.order.permId
        )
        if info is None:
            raise KeyError(
                f"No order record for orderId={trade.order.orderId} "
                f"permId={trade.order.permId}"
            )
        record = FillRecord.from_fill(trade, fill)
        if not info.add_fill(record):
            return False

        # Queue order evidence before the projection based on it.
        self.save_order(info)
        if isinstance(trade.contract, ibi.Bag):
            return True
        if info.source_key is not None:
            state = self._positions.get(info.source_key)
            if state is None:
                state = PositionState(
                    source_key=info.source_key,
                    execution_model_name=info.execution_model_name,
                    contract=trade.contract,
                )
            old_quantity = state.quantity
            if fill.execution.side == "BOT":
                quantity = old_quantity + fill.execution.shares
            elif fill.execution.side == "SLD":
                quantity = old_quantity - fill.execution.shares
            else:
                raise ValueError(f"Ambiguous fill side: {fill.execution.side}")
            blocked = state.blocked_direction
            if (
                info.role in {"STOP_LOSS", "TAKE_PROFIT"}
                and old_quantity
                and quantity == 0
            ):
                blocked = 1 if old_quantity > 0 else -1
            elif info.role == "OPEN" and quantity != 0:
                blocked = None
            episode_closed = quantity == 0 and info.role in {
                "CLOSE",
                "STOP_LOSS",
                "TAKE_PROFIT",
            }
            protective_exit = quantity == 0 and info.role in {
                "STOP_LOSS",
                "TAKE_PROFIT",
            }
            target_quantity = 0.0 if protective_exit else state.target_quantity
            state = replace(
                state,
                contract=trade.contract,
                quantity=quantity,
                target_quantity=target_quantity,
                blocked_direction=blocked,
                position_id=None if episode_closed else state.position_id,
                bracket_inputs=(
                    {}
                    if episode_closed and not target_quantity
                    else state.bracket_inputs
                ),
                updated_at=_utc_now(),
            )
            self.update_position(state)
        return True

    def update_commission(
        self,
        trade: ibi.Trade,
        fill: ibi.Fill,
        report: ibi.CommissionReport,
    ) -> bool:
        """Attach a late CommissionReport to normalized Fill evidence.

        Returns:
            ``True`` when matching fill evidence was updated, otherwise
            ``False`` when the order or execution is unknown.
        """

        if not isinstance(report, ibi.CommissionReport):
            raise TypeError("report must be an ib_insync.CommissionReport")
        info = self.order_by_id(trade.order.orderId) or self.order_by_perm_id(
            trade.order.permId
        )
        if info is None:
            return False
        key = _execution_key(trade, fill)
        records = list(info.fills)
        for index, record in enumerate(records):
            if record.deduplication_key == key:
                records[index] = replace(
                    record,
                    commission_report=report,
                )
                info.trade = trade
                info.fills = tuple(records)
                self.save_order(info)
                return True
        return False

    def save_portfolio_state(
        self, portfolio_key: str, state: Mapping[str, Any]
    ) -> None:
        """Persist one Portfolio's normalized recovery mapping."""

        portfolio_key = non_empty_string(portfolio_key, "portfolio_key")
        copied = readonly_mapping(state, "state")
        self._portfolio_states[portfolio_key] = copied
        self._save(
            self._state_saver,
            {
                "state_key": f"portfolio:{portfolio_key}",
                "state_type": "portfolio",
                "portfolio_key": portfolio_key,
                "state": tree(dict(copied)),
                "updated_at": _utc_now(),
            },
        )

    def load_portfolio_state(self, portfolio_key: str) -> Mapping[str, Any] | None:
        """Load one Portfolio recovery mapping."""

        return self._portfolio_states.get(portfolio_key)

    def blotter_records(
        self, source_key: str, position_id: str | None = None
    ) -> tuple[Mapping[str, Any], ...]:
        """Query available blotter rows by source and optional episode."""

        if self.blotter is None:
            return ()
        return tuple(
            row
            for row in self.blotter.records()
            if row.get("source_key") == source_key
            and (position_id is None or row.get("position_id") == position_id)
        )

    def register_rejected_order(self, execution_model_name: str) -> None:
        """Count one broker rejection against a configured model."""

        self._rejected_orders[execution_model_name] += 1

    def verify_for_rejections(self, execution_model_name: str) -> bool:
        """Return whether a model remains below the rejection threshold."""

        return (
            self._rejected_orders.get(execution_model_name, 0)
            < self.max_rejected_orders
        )

    def execution_model_names(self) -> set[str]:
        """Return all persisted model names referenced by live state."""

        return {
            *(state.execution_model_name for state in self._positions.values()),
            *(state.execution_model_name for state in self._targets.values()),
            *(info.execution_model_name for info in self._orders.values()),
        }

    def active_order_model_for_contract(
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

        con_id = _contract_key(contract)
        candidates = {
            info.execution_model_name
            for info in self.active_orders(contract=contract, role=role)
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

    @staticmethod
    def _encode_position(state: PositionState) -> dict[str, Any]:
        return {
            "state_key": f"position:{state.source_key}",
            "state_type": "position",
            "source_key": state.source_key,
            "execution_model_name": state.execution_model_name,
            "contract": tree(state.contract),
            "quantity": state.quantity,
            "target_quantity": state.target_quantity,
            "target_created_at": state.target_created_at,
            "position_id": state.position_id,
            "blocked_direction": state.blocked_direction,
            "bracket_inputs": tree(dict(state.bracket_inputs)),
            "updated_at": state.updated_at,
        }

    @staticmethod
    def _decode_position(data: Mapping[str, Any]) -> PositionState:
        return PositionState(
            source_key=str(data["source_key"]),
            execution_model_name=str(data["execution_model_name"]),
            contract=decode_tree(data.get("contract")),
            quantity=float(data.get("quantity", 0.0)),
            target_quantity=data.get("target_quantity"),
            target_created_at=decode_tree(data.get("target_created_at")),
            position_id=data.get("position_id"),
            blocked_direction=data.get("blocked_direction"),
            bracket_inputs=decode_tree(data.get("bracket_inputs", {})),
            updated_at=decode_tree(data["updated_at"]),
        )

    @staticmethod
    def _encode_target(state: TargetState) -> dict[str, Any]:
        return {
            "state_key": (
                f"target:{state.execution_model_name}:{state.contract.conId}"
            ),
            "state_type": "target",
            "execution_model_name": state.execution_model_name,
            "conId": state.contract.conId,
            "contract": tree(state.contract),
            "target_quantity": state.target_quantity,
            "target_created_at": state.target_created_at,
            "updated_at": state.updated_at,
        }

    @staticmethod
    def _decode_target(data: Mapping[str, Any]) -> TargetState:
        return TargetState(
            execution_model_name=str(data["execution_model_name"]),
            contract=decode_tree(data["contract"]),
            target_quantity=float(data["target_quantity"]),
            target_created_at=decode_tree(data["target_created_at"]),
            updated_at=decode_tree(data["updated_at"]),
        )


__all__ = [
    "Book",
    "FillRecord",
    "OrderInfo",
    "PositionState",
    "TargetState",
]
