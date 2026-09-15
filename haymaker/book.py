"""Typed persistent accounting and recovery state for live execution."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Literal
from uuid import uuid4

import ib_insync as ibi

from .async_wrappers import QueueProcessingError, QueueShutdownPolicy, SyncQueueRunner
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


@dataclass(frozen=True, kw_only=True)
class PositionState:
    """Recover an episode separately from its latest accepted target.

    ``contract`` and ``bracket_inputs`` belong to the holding or submitted
    entry. ``target_contract`` and ``target_bracket_inputs`` belong to the
    pending opening destination; accepting a reversal must not change the
    Contract or protection inputs of the episode being closed.
    """

    source_key: str
    execution_model_name: str
    contract: ibi.Contract | None = None
    quantity: float = 0.0
    target_quantity: float | None = None
    target_created_at: datetime | None = None
    target_contract: ibi.Contract | None = None
    target_bracket_inputs: Mapping[str, Any] = field(default_factory=dict)
    position_id: str | None = None
    blocked_direction: Literal[-1, 1] | None = None
    bracket_inputs: Mapping[str, Any] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=_utc_now)
    _applied_fill_keys: frozenset[str] = field(default_factory=frozenset, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_applied_fill_keys", frozenset(self._applied_fill_keys)
        )
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
        if self.target_contract is not None:
            ib_contract(self.target_contract)
        object.__setattr__(
            self,
            "target_bracket_inputs",
            readonly_mapping(self.target_bracket_inputs, "target_bracket_inputs"),
        )
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
    """Recover the latest shared direct-execution target.

    ``fill_evidence_start_at`` is Book-owned reset state. It preserves order
    and Fill history while excluding evidence that predates an explicit
    account-state clear from the rebuilt direct position.
    """

    execution_model_name: str
    contract: ibi.Contract
    target_quantity: float
    target_created_at: datetime
    fill_evidence_start_at: datetime | None = None
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
        if self.fill_evidence_start_at is not None:
            aware_datetime(
                self.fill_evidence_start_at,
                "fill_evidence_start_at",
            )
        aware_datetime(self.updated_at, "updated_at")


@dataclass(frozen=True, kw_only=True)
class ContractPosition:
    """Persist an accounted net balance, independently of execution mode.

    Book updates this balance with the underlying episode, order or roll
    mutation. Recovery verifies it against those records before trading starts.
    It is neither a desired target nor a broker-position snapshot.
    """

    contract: ibi.Contract
    quantity: float
    updated_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        """Validate a concrete Contract and finite signed balance."""
        qualified_contract(self.contract)
        object.__setattr__(self, "quantity", finite_number(self.quantity, "quantity"))
        aware_datetime(self.updated_at, "updated_at")


class FutureRollMode(StrEnum):
    """Identify the process's mutually exclusive futures-roll accounting mode.

    ``DIRECT`` rolls account-wide concrete Contract targets.
    ``BRACKET`` rolls one-to-one episodes attributed by ``source_key`` and
    reinstalls their protective orders.
    """

    BRACKET = "BRACKET"
    DIRECT = "DIRECT"


class FutureRollStage(StrEnum):
    """Record durable progress through one recoverable futures-series roll.

    Values describe broker work as well as bracket-protection replacement.
    ``BLOCKED`` is intentionally durable and requires operator review;
    ``COMPLETE`` is the only state excluded from active recovery.
    """

    PLANNED = "PLANNED"
    WAITING_FOR_ACTIVE_WORK = "WAITING_FOR_ACTIVE_WORK"
    ROLL_ORDER_ACTIVE = "ROLL_ORDER_ACTIVE"
    ROLL_FILLED = "ROLL_FILLED"
    LOADING_REFERENCE_PRICE = "LOADING_REFERENCE_PRICE"
    CANCELLING_PROTECTION = "CANCELLING_PROTECTION"
    INSTALLING_STOP = "INSTALLING_STOP"
    INSTALLING_TAKE_PROFIT = "INSTALLING_TAKE_PROFIT"
    COMPLETE = "COMPLETE"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, kw_only=True)
class RollParticipant:
    """Persist one logical holding participating in a futures-series roll.

    Args:
        execution_model_name: Stable target-model name that owns the holding.
        quantity: Signed quantity attributed to this participant. A zero
            non-trading source records an episode that closed while waiting.
        source_key: One-to-one identity in bracket mode.
        position_id: Optional one-to-one position episode.
        requires_trade: Whether this participant supplies the physical BAG
            order. Offset logical sources may require only price adjustment and
            protection replacement.

    A source identifies a one-to-one episode; absence denotes direct mode.
    """

    execution_model_name: str
    quantity: float
    source_key: str | None = None
    position_id: str | None = None
    requires_trade: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "execution_model_name",
            non_empty_string(self.execution_model_name, "execution_model_name"),
        )
        if self.source_key is not None:
            object.__setattr__(
                self, "source_key", non_empty_string(self.source_key, "source_key")
            )
        object.__setattr__(self, "quantity", finite_number(self.quantity, "quantity"))
        if not self.quantity and (self.source_key is None or self.requires_trade):
            raise ValueError("Zero roll quantity is only valid for a skipped source")
        if not isinstance(self.requires_trade, bool):
            raise TypeError("requires_trade must be a bool")


@dataclass(frozen=True, kw_only=True)
class RollState:
    """Recover the current roll operation for one registered futures series.

    Args:
        series_key: ContractRegistry identity shared by all qualified expiries
            of one registered blueprint.
        mode: Exclusive direct or bracket accounting mode.
        executor_name: Stable FutureRollExecutor recovery identity.
        old_contract: Concrete held Future being left.
        new_contract: Concrete policy-selected Future being entered.
        participants: Frozen logical holdings included in this operation.
        target_transfers: Idempotent concrete target snapshots for direct rolls.
        occurrence_keys: Schedule markers to record upon successful completion.
        completed_occurrences: Previously completed schedule markers retained
            when the next operation replaces this series' current record.
        participant_index: Durable cursor for serial participant processing.
        stage: Last durably accepted roll stage.
        roll_order_id: Current BAG order id, when one is active or filled.
        old_protection_order_ids: Bracket orders being replaced.
        replacement_stop_order_id: New critical stop order id.
        replacement_take_profit_order_id: New optional take-profit order id.
        reference_price: Filled or observed calendar-spread price used to shift
            protective prices.
        failure_reason: Actionable explanation when ``stage`` is ``BLOCKED``.
        created_at: Start time of this roll generation.
        updated_at: Time of the latest durable transition.

    Book stores one current record per ``series_key``. Framework executors
    normally construct and advance it; custom executors must persist each stage
    before performing its broker side effect.
    """

    series_key: str
    mode: FutureRollMode
    executor_name: str
    old_contract: ibi.Future
    new_contract: ibi.Future
    participants: Sequence[RollParticipant]
    target_transfers: Sequence[TargetState] = ()
    occurrence_keys: Sequence[str] = ()
    completed_occurrences: Sequence[str] = ()
    participant_index: int = 0
    stage: FutureRollStage = FutureRollStage.PLANNED
    roll_order_id: int | None = None
    old_protection_order_ids: Sequence[int] = ()
    replacement_stop_order_id: int | None = None
    replacement_take_profit_order_id: int | None = None
    reference_price: float | None = None
    failure_reason: str | None = None
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "series_key", non_empty_string(self.series_key, "series_key")
        )
        object.__setattr__(
            self,
            "executor_name",
            non_empty_string(self.executor_name, "executor_name"),
        )
        if not isinstance(self.mode, FutureRollMode):
            raise TypeError("mode must be a FutureRollMode")
        if not isinstance(self.stage, FutureRollStage):
            raise TypeError("stage must be a FutureRollStage")
        if not isinstance(self.old_contract, ibi.Future) or not isinstance(
            self.new_contract, ibi.Future
        ):
            raise TypeError("old_contract and new_contract must be Futures")
        qualified_contract(self.old_contract, "old_contract")
        qualified_contract(self.new_contract, "new_contract")
        if self.old_contract.conId == self.new_contract.conId:
            raise ValueError("old_contract and new_contract must differ")
        participants = tuple(self.participants)
        if not participants or not all(
            isinstance(participant, RollParticipant) for participant in participants
        ):
            raise ValueError("participants must contain RollParticipant values")
        object.__setattr__(self, "participants", participants)
        if any(
            (p.source_key is None) != (self.mode is FutureRollMode.DIRECT)
            for p in participants
        ):
            raise ValueError("Roll participant attribution does not match mode")
        transfers = tuple(self.target_transfers)
        if not all(isinstance(target, TargetState) for target in transfers):
            raise TypeError("target_transfers must contain TargetState values")
        object.__setattr__(self, "target_transfers", transfers)
        for name in ("occurrence_keys", "completed_occurrences"):
            keys = tuple(non_empty_string(key, name) for key in getattr(self, name))
            object.__setattr__(self, name, tuple(dict.fromkeys(keys)))
        if not isinstance(self.participant_index, int) or isinstance(
            self.participant_index, bool
        ):
            raise TypeError("participant_index must be an integer")
        if not 0 <= self.participant_index <= len(participants):
            raise ValueError("participant_index is outside participants")
        old_order_ids = tuple(self.old_protection_order_ids)
        if not all(
            isinstance(order_id, int) and order_id > 0 for order_id in old_order_ids
        ):
            raise ValueError("old_protection_order_ids must be positive integers")
        object.__setattr__(self, "old_protection_order_ids", old_order_ids)
        for name in (
            "roll_order_id",
            "replacement_stop_order_id",
            "replacement_take_profit_order_id",
        ):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
            ):
                raise ValueError(f"{name} must be None or a positive integer")
        if self.reference_price is not None:
            object.__setattr__(
                self,
                "reference_price",
                finite_number(self.reference_price, "reference_price"),
            )
        if self.failure_reason is not None:
            object.__setattr__(
                self,
                "failure_reason",
                non_empty_string(self.failure_reason, "failure_reason"),
            )
        aware_datetime(self.created_at, "created_at")
        aware_datetime(self.updated_at, "updated_at")

    @property
    def current_participant(self) -> RollParticipant | None:
        """Return the participant at the durable cursor, if any."""

        if self.participant_index == len(self.participants):
            return None
        return self.participants[self.participant_index]

    @property
    def terminal(self) -> bool:
        """Return whether automatic work must no longer advance."""

        return self.stage in {FutureRollStage.COMPLETE, FutureRollStage.BLOCKED}


class Book:
    """Own typed accounting, order recovery, persistence, and blotter access.

    Book performs no allocation and no broker API calls. All mutations use one
    ordered critical queue when asynchronous saving is enabled, so order
    evidence and the resulting projections retain deterministic persistence
    order. Both execution modes query maintained concrete-Contract balances;
    historical reconstruction is confined to startup verification and reset.
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
        self._targets: dict[int, TargetState] = {}
        self._target_fill_cutoffs: dict[int, datetime] = {}
        self._rolls: dict[str, RollState] = {}
        self._portfolio_states: dict[str, Mapping[str, Any]] = {}
        self._balances: dict[int, ContractPosition] = {}
        self._contributions: dict[tuple[str, str | int], dict[ibi.Contract, float]] = {}
        self._rejected_orders: defaultdict[str, int] = defaultdict(int)
        self._save_async = save_async
        self._mutation_queue: SyncQueueRunner | None = None
        self._write_failure: Exception | None = None
        if restore:
            self._restore_documents(*self._read_documents())
            self._recover_position_fills()
            # Runtime constructs Book before a running loop exists. Finish
            # startup repairs synchronously, just like the recovery reads.
            for balance in self._rebuild_balances():
                self._save_balance(balance)
        self._mutation_queue = (
            SyncQueueRunner(
                "Book",
                shutdown_policy=QueueShutdownPolicy.DRAIN,
                max_failures=1,
            )
            if save_async
            else None
        )

    def _save(self, saver: AbstractBaseSaver, document: dict[str, Any]) -> None:
        """Persist one mutation in Book's deterministic write order."""

        self.check_writable()
        try:
            if self._mutation_queue is None:
                saver.save(document)
            else:
                self._mutation_queue.enqueue(saver.save, document)
        except Exception as exc:
            self._write_failure = exc
            raise

    def check_writable(self) -> None:
        """Reject new broker work after critical persistence has halted."""
        if self._write_failure is not None:
            raise QueueProcessingError(
                "Book persistence has halted"
            ) from self._write_failure
        if self._mutation_queue is not None:
            self._mutation_queue.check_accepting_work()

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
                target_contract=None,
                target_bracket_inputs={},
                position_id=None,
                blocked_direction=None,
                bracket_inputs={},
                updated_at=cleared_at,
            )
            self._save(
                self._state_saver,
                self._encode_position(cleared_position),
            )
        targets_to_clear = dict(self._targets)
        # Residual liquidation/manual evidence can exist without a Portfolio
        # target. It needs the same durable reset cutoff as attributed targets.
        for contract in self.logical_positions():
            targets_to_clear.setdefault(
                contract.conId,
                TargetState(
                    execution_model_name="reset",
                    contract=contract,
                    target_quantity=0,
                    target_created_at=cleared_at,
                ),
            )
        for target in targets_to_clear.values():
            cleared_target = replace(
                target,
                target_quantity=0.0,
                target_created_at=cleared_at,
                fill_evidence_start_at=cleared_at,
                updated_at=cleared_at,
            )
            document = self._encode_target(cleared_target)
            document["cleared"] = True
            self._save(self._state_saver, document)
            self._target_fill_cutoffs[target.contract.conId] = cleared_at
        for roll in self._rolls.values():
            self._save(
                self._state_saver,
                self._encode_roll(
                    replace(
                        roll,
                        participant_index=len(roll.participants),
                        stage=FutureRollStage.COMPLETE,
                        roll_order_id=None,
                        old_protection_order_ids=(),
                        replacement_stop_order_id=None,
                        replacement_take_profit_order_id=None,
                        failure_reason=None,
                        updated_at=cleared_at,
                    )
                ),
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
        self._rolls.clear()
        self._portfolio_states.clear()
        for balance in self._rebuild_balances():
            self._save_balance(balance)

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
        superseded = {
            order_id
            for info in self._orders.values()
            for order_id in info.previous_order_ids
        }
        for info in tuple(self._orders.values()):
            for order_id in info.previous_order_ids:
                previous = self._orders.get(order_id)
                if previous is not None and (
                    not info.permId or previous.permId != info.permId
                ):
                    raise ValueError("Rebound order history has conflicting permId")
        self._orders = {
            key: info for key, info in self._orders.items() if key not in superseded
        }
        self._positions = {}
        self._targets = {}
        self._target_fill_cutoffs = {}
        self._rolls = {}
        self._portfolio_states = {}
        self._balances = {}
        self._contributions = {}
        for raw in state_documents:
            document = dict(raw)
            document.pop("_id", None)
            state_type = document.get("state_type")
            if state_type == "position":
                position_state = self._decode_position(document)
                self._positions[position_state.source_key] = position_state
            elif state_type == "target":
                target_state = self._decode_target(document)
                if target_state.fill_evidence_start_at is not None:
                    self._target_fill_cutoffs[target_state.contract.conId] = (
                        target_state.fill_evidence_start_at
                    )
                if not document.get("cleared", False):
                    self._targets[target_state.contract.conId] = target_state
            elif state_type == "roll":
                roll_state = self._decode_roll(document)
                self._rolls[roll_state.series_key] = roll_state
            elif state_type == "portfolio":
                self._portfolio_states[str(document["portfolio_key"])] = (
                    MappingProxyType(decode_tree(document.get("state", {})))
                )
            elif state_type == "balance":
                balance = self._decode_balance(document)
                self._balances[balance.contract.conId] = balance
            else:
                raise ValueError(f"Unknown Book state_type: {state_type!r}")

    def _read_documents(
        self,
    ) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
        """Read both physical recovery collections in one worker call."""

        return self._order_saver.read({}), self._state_saver.read({})

    def save_order(self, info: OrderInfo) -> OrderInfo:
        """Persist one order, then account for changes to its recorded fills.

        Re-saving status, commissions or a rebound Trade does not apply the
        same quantity twice. Only this order's contribution is recalculated.
        """

        if not info.orderId:
            raise ValueError("Cannot persist an order with orderId 0")
        self._orders[info.orderId] = info
        self._save(self._order_saver, info.encode())
        changes: dict[tuple[str, str | int], dict[ibi.Contract, float]] = {
            ("order", info.orderId): self._order_positions(info)
        }
        for roll in self._active_bracket_rolls():
            if info.params.get("roll_state_key") == roll.series_key:
                changes[("roll", roll.series_key)] = dict(
                    self._roll_physical_quantities(roll)
                )
        self._update_balances(changes)
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

    def orders(
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
            for info in self._orders.values()
            if (not active_only or info.active)
            and (source_key is None or info.source_key == source_key)
            and (con_id is None or info.trade.contract.conId == con_id)
            and (role is None or info.role == role)
            and (
                execution_model_name is None
                or info.execution_model_name == execution_model_name
            )
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

        return self.orders(
            source_key=source_key,
            contract=contract,
            role=role,
            execution_model_name=execution_model_name,
            active_only=True,
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
            existing = self._orders.get(trade.order.orderId)
            if existing is not None and existing is not info:
                raise ValueError("Rebound orderId already belongs to another order")
            if old_order_id != trade.order.orderId:
                info.previous_order_ids = tuple(
                    key
                    for key in dict.fromkeys((*info.previous_order_ids, old_order_id))
                    if key != trade.order.orderId
                )
            info.trade = trade
            if old_order_id != info.orderId:
                self._orders.pop(old_order_id, None)
                contribution = self._contributions.pop(("order", old_order_id), {})
                self._contributions[("order", info.orderId)] = contribution
            self.save_order(info)
        return None

    def position_state(self, source_key: str) -> PositionState | None:
        """Return one recovered one-to-one state without creating it."""

        return self._positions.get(source_key)

    def position_states(self) -> Mapping[str, PositionState]:
        """Return a read-only view of all one-to-one states."""

        return MappingProxyType(self._positions)

    def update_position(self, state: PositionState) -> PositionState:
        """Record an authoritative source state including all received fills.

        Reconciliation and episode transitions checkpoint the evidence already
        known to Book. Fill application uses its own individual checkpoint so
        startup can finish an interrupted evidence-to-position transition.
        """
        previous = self._positions.get(state.source_key)
        keys = {
            record.deduplication_key
            for info in self.orders(source_key=state.source_key)
            if not isinstance(info.trade.contract, ibi.Bag)
            for record in info.fills
        }
        state = replace(
            state,
            _applied_fill_keys=state._applied_fill_keys
            | (previous._applied_fill_keys if previous else frozenset())
            | keys,
        )
        return self._save_position(state)

    def _save_position(self, state: PositionState) -> PositionState:
        """Persist an already checkpointed source state and its net contribution."""

        self._positions[state.source_key] = state
        self._save(self._state_saver, self._encode_position(state))
        self._update_balances(
            {("source", state.source_key): self._episode_positions(state)}
        )
        return state

    def target_state(self, contract: ibi.Contract) -> TargetState | None:
        """Return the latest direct setpoint for this exact qualified Contract."""

        return self._targets.get(_contract_key(contract))

    def update_target(self, state: TargetState) -> TargetState:
        """Replace and persist one latest direct target."""

        cutoff = self._target_fill_cutoffs.get(state.contract.conId)
        if cutoff is not None and state.fill_evidence_start_at != cutoff:
            state = replace(state, fill_evidence_start_at=cutoff)
        elif state.fill_evidence_start_at is not None:
            self._target_fill_cutoffs[state.contract.conId] = (
                state.fill_evidence_start_at
            )
        current = self._targets.get(state.contract.conId)
        if current is not None and (
            state.target_created_at < current.target_created_at
        ):
            return current
        self._targets[state.contract.conId] = state
        self._save(self._state_saver, self._encode_target(state))
        if cutoff is None and state.fill_evidence_start_at is not None:
            for balance in self._rebuild_balances():
                self._save_balance(balance)
        return state

    def target_states(
        self, execution_model_name: str | None = None
    ) -> tuple[TargetState, ...]:
        """Return recovered direct targets, optionally restricted by owner."""

        return tuple(
            state
            for state in self._targets.values()
            if execution_model_name is None
            or state.execution_model_name == execution_model_name
        )

    def latest_targets(self) -> tuple[TargetState, ...]:
        """Return every latest direct target, one per concrete conId."""

        return tuple(self._targets.values())

    def roll_state(self, series_key: str) -> RollState | None:
        """Return the current persisted roll for one futures series."""

        return self._rolls.get(series_key)

    def roll_states(self, *, active_only: bool = False) -> tuple[RollState, ...]:
        """Return all current series rolls, optionally excluding completed ones."""

        return tuple(
            state
            for state in self._rolls.values()
            if not active_only or state.stage is not FutureRollStage.COMPLETE
        )

    def update_roll(self, state: RollState) -> RollState:
        """Replace and persist one series roll recovery state."""

        previous = self._rolls.get(state.series_key)
        self._rolls[state.series_key] = state
        self._save(self._state_saver, self._encode_roll(state))
        changes: dict[tuple[str, str | int], dict[ibi.Contract, float]] = {
            ("roll", state.series_key): (
                dict(self._roll_physical_quantities(state))
                if state.mode is FutureRollMode.BRACKET
                and state.stage is not FutureRollStage.COMPLETE
                else {}
            )
        }
        # Replace the roll and its episodes together: a logical Contract move
        # during a roll must not temporarily count both the old and new holding.
        sources = {
            p.source_key
            for roll in (previous, state)
            if roll is not None
            for p in roll.participants
            if p.source_key is not None
        }
        for source in sorted(sources):
            position = self._positions.get(source)
            changes[("source", source)] = (
                self._episode_positions(position) if position is not None else {}
            )
        self._update_balances(changes)
        return state

    def roll_state_for_source(self, source_key: str) -> RollState | None:
        """Return a non-complete roll containing one one-to-one source."""

        return next(
            (
                state
                for state in self._rolls.values()
                if state.stage is not FutureRollStage.COMPLETE
                and any(
                    participant.source_key == source_key
                    for participant in state.participants
                )
            ),
            None,
        )

    def roll_state_for_contract(self, contract: ibi.Contract) -> RollState | None:
        """Return an active direct roll reserving either concrete endpoint."""
        con_id = _contract_key(contract)
        return next(
            (
                state
                for state in self._rolls.values()
                if not state.terminal
                and state.mode is FutureRollMode.DIRECT
                and con_id in (state.old_contract.conId, state.new_contract.conId)
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
            if info.role in {"OPEN", "CLOSE", "TARGET_ADJUSTMENT"}
        )
        return quantity + working

    def _order_positions(self, info: OrderInfo) -> dict[ibi.Contract, float]:
        """Calculate one source-less order's contribution to account balances.

        Episodes already account for source-attributed orders. Roll leg evidence
        replaces the BAG fallback, rather than adding the same movement twice.
        This runs on mutation/recovery, never on a position query.
        """
        if info.source_key is not None:
            return {}
        quantities: defaultdict[int, float] = defaultdict(float)
        contracts: dict[int, ibi.Contract] = {}
        endpoints: tuple[ibi.Contract, ...] = ()
        records = info.fills
        if isinstance(info.trade.contract, ibi.Bag):
            if info.role != "ROLL":
                return {}
            old = info.params.get("old_contract")
            new = info.params.get("new_contract")
            if not isinstance(old, ibi.Contract) or not isinstance(new, ibi.Contract):
                raise ValueError(
                    f"Direct ROLL orderId={info.orderId} lacks old/new Contracts"
                )
            endpoints = (old, new)
            legs = tuple(
                r for r in records if r.contract.conId in (old.conId, new.conId)
            )
            records = legs or records
        for record in records:
            direction = self._fill_direction(record)
            movements = (
                ((endpoints[0], -direction), (endpoints[1], direction))
                if endpoints
                and record.contract.conId not in {c.conId for c in endpoints}
                else ((record.contract, direction),)
            )
            for contract, signed in movements:
                if not contract.conId:
                    continue
                cutoff = self._target_fill_cutoffs.get(contract.conId)
                if cutoff is not None and record.time < cutoff:
                    continue
                contracts[contract.conId] = contract
                quantities[contract.conId] += record.execution.shares * signed
        return {
            contracts[con_id]: quantity
            for con_id, quantity in quantities.items()
            if quantity
        }

    @staticmethod
    def _fill_direction(record: FillRecord) -> int:
        if record.execution.side == "BOT":
            return 1
        if record.execution.side == "SLD":
            return -1
        raise ValueError(f"Ambiguous fill side: {record.execution.side}")

    def aggregate_quantity(self, contract: ibi.Contract) -> float:
        """Read one maintained net balance by conId, in either execution mode.

        This excludes desired targets and unfilled working quantities. It does
        not query the broker or scan historical fills.
        """
        balance = self._balances.get(_contract_key(contract))
        return balance.quantity if balance is not None else 0.0

    def logical_positions(self) -> dict[ibi.Contract, float]:
        """Read non-flat accounted balances, without reconstructing history.

        Both execution modes use this account-level view. One-to-one sources
        are netted per concrete Contract; episode ownership remains available
        separately through position_states(). The returned mapping is a copy.
        """
        return {
            balance.contract: balance.quantity
            for balance in self._balances.values()
            if balance.quantity
        }

    def _episode_positions(self, state: PositionState) -> dict[ibi.Contract, float]:
        """Return an episode's contribution when no roll owns its movement."""
        if (
            state.contract is None
            or not state.contract.conId
            or not state.quantity
            or self.roll_state_for_source(state.source_key) is not None
        ):
            return {}
        return {state.contract: state.quantity}

    def _update_balances(
        self,
        changes: Mapping[tuple[str, str | int], dict[ibi.Contract, float]],
        *,
        persist: bool = True,
    ) -> None:
        """Apply changed accounting contributions through one balance path.

        Replacing the prior contribution makes order re-saves and repeated roll
        stages harmless. Only changed Contracts are persisted, after evidence.
        """
        deltas: defaultdict[int, float] = defaultdict(float)
        contracts: dict[int, ibi.Contract] = {}
        for owner, positions in changes.items():
            previous = self._contributions.get(owner, {})
            for multiplier, mapping in ((-1, previous), (1, positions)):
                for contract, quantity in mapping.items():
                    contracts[contract.conId] = contract
                    deltas[contract.conId] += multiplier * quantity
            if positions:
                self._contributions[owner] = positions
            else:
                self._contributions.pop(owner, None)
        for con_id, delta in deltas.items():
            if not delta:
                continue
            current = self._balances.get(con_id)
            balance = ContractPosition(
                contract=contracts[con_id],
                quantity=(current.quantity if current is not None else 0.0) + delta,
            )
            self._balances[con_id] = balance
            if persist:
                self._save_balance(balance)

    def _rebuild_balances(self) -> tuple[ContractPosition, ...]:
        """Verify saved balances once at restore/reset, repairing partial writes.

        Episode state remains authoritative for one-to-one corrections; fills
        recover source-less orders, and active rolls recover physical movement.
        No second applied-execution ledger or multi-document transaction is
        needed: order evidence precedes balances in the critical save queue.
        """
        saved = self._balances
        self._balances = {}
        self._contributions = {}
        changes: dict[tuple[str, str | int], dict[ibi.Contract, float]] = {
            ("order", info.orderId): self._order_positions(info)
            for info in self._orders.values()
        }
        changes.update(
            {
                ("source", source): self._episode_positions(state)
                for source, state in self._positions.items()
            }
        )
        changes.update(
            {
                ("roll", state.series_key): dict(self._roll_physical_quantities(state))
                for state in self._active_bracket_rolls()
            }
        )
        self._update_balances(changes, persist=False)
        repaired = []
        for con_id in sorted(saved.keys() | self._balances.keys()):
            previous = saved.get(con_id)
            current = self._balances.get(con_id)
            if current is None and previous is not None:
                current = replace(previous, quantity=0.0, updated_at=_utc_now())
            if current is None:
                continue
            if previous is not None and previous.quantity == current.quantity:
                self._balances[con_id] = previous
            else:
                self._balances[con_id] = current
                repaired.append(current)
        return tuple(repaired)

    def _save_balance(self, balance: ContractPosition) -> None:
        """Persist a shared Contract balance under its natural identity."""
        self._save(
            self._state_saver,
            {
                "state_key": f"balance:{balance.contract.conId}",
                "state_type": "balance",
                "conId": balance.contract.conId,
                "contract": tree(balance.contract),
                "quantity": balance.quantity,
                "updated_at": balance.updated_at,
            },
        )

    @staticmethod
    def _decode_balance(document: Mapping[str, Any]) -> ContractPosition:
        """Validate the concrete identity of a persisted balance."""
        balance = ContractPosition(
            contract=decode_tree(document["contract"]),
            quantity=document["quantity"],
            updated_at=decode_tree(document["updated_at"]),
        )
        if (
            document.get("state_key") != f"balance:{balance.contract.conId}"
            or document.get("conId") != balance.contract.conId
        ):
            raise ValueError("Persisted balance identity does not match its Contract")
        return balance

    def _active_bracket_rolls(self) -> tuple[RollState, ...]:
        """Return incomplete one-to-one rolls that own broker projections."""

        return tuple(
            state
            for state in self._rolls.values()
            if state.mode is FutureRollMode.BRACKET
            and state.stage is not FutureRollStage.COMPLETE
        )

    def _roll_physical_quantities(
        self, state: RollState
    ) -> Mapping[ibi.Contract, float]:
        """Project broker-net old/new quantities from persisted roll Fills."""

        old_quantity = sum(participant.quantity for participant in state.participants)
        moved = sum(
            record.execution.shares * self._fill_direction(record)
            for info in self.orders(role="ROLL")
            if info.submitted_at >= state.created_at
            if info.params.get("roll_state_key") == state.series_key
            and info.params.get("old_contract") == state.old_contract
            and info.params.get("new_contract") == state.new_contract
            for record in info.fills
        )
        quantities = {
            state.old_contract: old_quantity - moved,
            state.new_contract: moved,
        }
        return MappingProxyType(
            {
                contract: quantity
                for contract, quantity in quantities.items()
                if quantity
            }
        )

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
            target_contract=contract,
            target_quantity=target_quantity,
            target_created_at=target_created_at,
            position_id=str(uuid4()),
            bracket_inputs=bracket_inputs,
            target_bracket_inputs=bracket_inputs,
            blocked_direction=self.blocked_direction(source_key),
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
                target_contract=None,
                target_bracket_inputs={},
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
        if not isinstance(trade.contract, ibi.Bag):
            self._fill_direction(record)
        new = info.add_fill(record)
        if new:
            # Evidence must precede the source checkpoint and net balance.
            self.save_order(info)
        state = self._position_after_fill(info, record)
        if state is not None:
            self._save_position(state)
            return True
        return new

    def _position_after_fill(
        self, info: OrderInfo, record: FillRecord
    ) -> PositionState | None:
        """Calculate a source transition only when its checkpoint lacks the fill."""
        if info.source_key is not None and not isinstance(info.trade.contract, ibi.Bag):
            state = self._positions.get(info.source_key) or PositionState(
                source_key=info.source_key,
                execution_model_name=info.execution_model_name,
                contract=info.trade.contract,
            )
            if record.deduplication_key in state._applied_fill_keys:
                return None
            old_quantity = state.quantity
            quantity = old_quantity + record.execution.shares * self._fill_direction(
                record
            )
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
            return replace(
                state,
                contract=info.trade.contract,
                quantity=quantity,
                target_quantity=target_quantity,
                target_contract=None if protective_exit else state.target_contract,
                target_bracket_inputs=(
                    {} if protective_exit else state.target_bracket_inputs
                ),
                blocked_direction=blocked,
                position_id=None if episode_closed else state.position_id,
                bracket_inputs=(
                    {}
                    if episode_closed and not target_quantity
                    else state.bracket_inputs
                ),
                updated_at=_utc_now(),
                _applied_fill_keys=state._applied_fill_keys
                | {record.deduplication_key},
            )
        return None

    def _recover_position_fills(self) -> None:
        """Finish missing source projections without replaying checkpointed fills."""
        evidence = sorted(
            (
                (record.time, info.orderId, index, info, record)
                for info in self._orders.values()
                for index, record in enumerate(info.fills)
            ),
            key=lambda item: item[:3],
        )
        for _, _, _, info, record in evidence:
            state = self._position_after_fill(info, record)
            if state is not None:
                self._positions[state.source_key] = state
                self._save(self._state_saver, self._encode_position(state))

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

        return self._portfolio_states.get(
            non_empty_string(portfolio_key, "portfolio_key")
        )

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
            "target_contract": tree(state.target_contract),
            "target_bracket_inputs": tree(dict(state.target_bracket_inputs)),
            "position_id": state.position_id,
            "blocked_direction": state.blocked_direction,
            "bracket_inputs": tree(dict(state.bracket_inputs)),
            "updated_at": state.updated_at,
            "applied_fill_keys": sorted(state._applied_fill_keys),
        }

    @staticmethod
    def _decode_position(data: Mapping[str, Any]) -> PositionState:
        if "applied_fill_keys" not in data:
            raise ValueError(
                "Position state lacks a fill checkpoint; standalone conversion required"
            )
        return PositionState(
            source_key=str(data["source_key"]),
            execution_model_name=str(data["execution_model_name"]),
            contract=decode_tree(data.get("contract")),
            quantity=float(data.get("quantity", 0.0)),
            target_quantity=data.get("target_quantity"),
            target_created_at=decode_tree(data.get("target_created_at")),
            target_contract=decode_tree(data["target_contract"]),
            target_bracket_inputs=decode_tree(data["target_bracket_inputs"]),
            position_id=data.get("position_id"),
            blocked_direction=data.get("blocked_direction"),
            bracket_inputs=decode_tree(data.get("bracket_inputs", {})),
            updated_at=decode_tree(data["updated_at"]),
            _applied_fill_keys=frozenset(data["applied_fill_keys"]),
        )

    @staticmethod
    def _encode_roll_participant(participant: RollParticipant) -> dict[str, Any]:
        return {
            "execution_model_name": participant.execution_model_name,
            "quantity": participant.quantity,
            "source_key": participant.source_key,
            "position_id": participant.position_id,
            "requires_trade": participant.requires_trade,
        }

    @staticmethod
    def _decode_roll_participant(data: Mapping[str, Any]) -> RollParticipant:
        if "target_key" in data:
            raise ValueError("Old keyed roll schema requires standalone conversion")
        return RollParticipant(
            execution_model_name=str(data["execution_model_name"]),
            quantity=float(data["quantity"]),
            source_key=data.get("source_key"),
            position_id=data.get("position_id"),
            requires_trade=bool(data.get("requires_trade", True)),
        )

    @classmethod
    def _encode_roll(cls, state: RollState) -> dict[str, Any]:
        return {
            "state_key": f"roll:{state.series_key}",
            "state_type": "roll",
            "series_key": state.series_key,
            "mode": state.mode.value,
            "executor_name": state.executor_name,
            "old_contract": tree(state.old_contract),
            "new_contract": tree(state.new_contract),
            "participants": [
                cls._encode_roll_participant(participant)
                for participant in state.participants
            ],
            "target_transfers": [
                Book._encode_target(t) for t in state.target_transfers
            ],
            "participant_index": state.participant_index,
            "occurrence_keys": list(state.occurrence_keys),
            "completed_occurrences": list(state.completed_occurrences),
            "stage": state.stage.value,
            "roll_order_id": state.roll_order_id,
            "old_protection_order_ids": list(state.old_protection_order_ids),
            "replacement_stop_order_id": state.replacement_stop_order_id,
            "replacement_take_profit_order_id": (
                state.replacement_take_profit_order_id
            ),
            "reference_price": state.reference_price,
            "failure_reason": state.failure_reason,
            "created_at": state.created_at,
            "updated_at": state.updated_at,
        }

    @classmethod
    def _decode_roll(cls, data: Mapping[str, Any]) -> RollState:
        return RollState(
            series_key=str(data["series_key"]),
            mode=FutureRollMode(str(data["mode"])),
            executor_name=str(data["executor_name"]),
            old_contract=decode_tree(data["old_contract"]),
            new_contract=decode_tree(data["new_contract"]),
            participants=tuple(
                cls._decode_roll_participant(participant)
                for participant in data["participants"]
            ),
            target_transfers=tuple(
                Book._decode_target(t) for t in data.get("target_transfers", ())
            ),
            participant_index=int(data.get("participant_index", 0)),
            occurrence_keys=tuple(data.get("occurrence_keys", ())),
            completed_occurrences=tuple(data.get("completed_occurrences", ())),
            stage=FutureRollStage(str(data["stage"])),
            roll_order_id=data.get("roll_order_id"),
            old_protection_order_ids=tuple(data.get("old_protection_order_ids", ())),
            replacement_stop_order_id=data.get("replacement_stop_order_id"),
            replacement_take_profit_order_id=data.get(
                "replacement_take_profit_order_id"
            ),
            reference_price=data.get("reference_price"),
            failure_reason=data.get("failure_reason"),
            created_at=decode_tree(data["created_at"]),
            updated_at=decode_tree(data["updated_at"]),
        )

    @staticmethod
    def _encode_target(state: TargetState) -> dict[str, Any]:
        return {
            "state_key": f"target:{state.contract.conId}",
            "state_type": "target",
            "execution_model_name": state.execution_model_name,
            "conId": state.contract.conId,
            "contract": tree(state.contract),
            "target_quantity": state.target_quantity,
            "target_created_at": state.target_created_at,
            "fill_evidence_start_at": state.fill_evidence_start_at,
            "updated_at": state.updated_at,
        }

    @staticmethod
    def _decode_target(data: Mapping[str, Any]) -> TargetState:
        if "target_key" in data or data.get("state_key") != f"target:{data['conId']}":
            raise ValueError("Old keyed target schema requires standalone conversion")
        return TargetState(
            execution_model_name=str(data["execution_model_name"]),
            contract=decode_tree(data["contract"]),
            target_quantity=float(data["target_quantity"]),
            target_created_at=decode_tree(data["target_created_at"]),
            fill_evidence_start_at=decode_tree(data.get("fill_evidence_start_at")),
            updated_at=decode_tree(data["updated_at"]),
        )


__all__ = [
    "Book",
    "FillRecord",
    "FutureRollMode",
    "FutureRollStage",
    "OrderInfo",
    "RollParticipant",
    "RollState",
    "PositionState",
    "TargetState",
]
