"""Signal positions and shared concrete-contract position accounting."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Literal

import ib_insync as ibi

from ..enums import ExecutionMode
from ..misc import decode_tree, tree
from ..validators import (
    aware_datetime,
    finite_number,
    ib_contract,
    non_empty_string,
    qualified_contract,
    readonly_mapping,
)
from types import MappingProxyType
from uuid import uuid4
from ..saver import AbstractBaseSaver
from .persistence import utc_now, PersistenceWriter
from .rolls import RollState, FutureRollStage
from .schema import require_schema
from .orders import FillRecord, OrderInfo, fill_direction


@dataclass(frozen=True, kw_only=True)
class SignalPosition:
    """Filled exposure attributed to one stable signal stream.

    A pending or flat position can have no Contract or position ID. The ID is
    retained through rolling and replaced on the next independently managed opening.

    Args:
        signal_key: Stable named stream, preserved across openings.
        execution_model_name: Execution owner used for recovery.
        contract: Held or pending-entry contract, independent of the next target.
        filled_quantity: Signed remaining filled exposure, excluding working orders.
        position_id: Identity of this opening, retained through rolling.
        updated_at: Time of the last accounting state update.
    """

    signal_key: str
    execution_model_name: str
    contract: ibi.Contract | None = None
    filled_quantity: float = 0.0
    position_id: str | None = None
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        non_empty_string(self.signal_key, "signal_key")
        non_empty_string(self.execution_model_name, "execution_model_name")
        if self.contract is not None:
            ib_contract(self.contract)
        object.__setattr__(
            self,
            "filled_quantity",
            finite_number(self.filled_quantity, "filled_quantity"),
        )
        aware_datetime(self.updated_at, "updated_at")


@dataclass(frozen=True, kw_only=True)
class SignalTarget:
    """Accepted desired exposure and inputs for the next signal opening.

    ``contract`` is the opening destination. Closing uses SignalPosition.contract;
    a reversal must leave that holding and its ProtectionState unchanged.

    Args:
        signal_key: Stable signal whose desired exposure is accepted.
        execution_model_name: Owner responsible for converging this target.
        target_quantity: Absolute signed desired quantity.
        target_created_at: Accepted message time; absent for untimed recovered state.
        contract: Destination for the next opening; may be absent when cleared.
        protection_inputs: Saved inputs for that opening, separate from held protection.
    """

    signal_key: str
    execution_model_name: str
    target_quantity: float
    target_created_at: datetime | None = None
    contract: ibi.Contract | None = None
    protection_inputs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        non_empty_string(self.signal_key, "signal_key")
        non_empty_string(self.execution_model_name, "execution_model_name")
        object.__setattr__(
            self,
            "target_quantity",
            finite_number(self.target_quantity, "target_quantity"),
        )
        if self.target_created_at is not None:
            aware_datetime(self.target_created_at, "target_created_at")
        if self.contract is not None:
            ib_contract(self.contract)
        object.__setattr__(
            self,
            "protection_inputs",
            readonly_mapping(self.protection_inputs, "protection_inputs"),
        )


@dataclass(frozen=True, kw_only=True)
class ProtectionState:
    """Saved protection inputs for the current signal position.

    Protective order status and fills remain authoritative in Book.orders.

    Args:
        signal_key: Signal owning the current position.
        position_id: Position identity to which these inputs belong.
        protection_inputs: Saved inputs used to build that position's protection.
    """

    signal_key: str
    position_id: str | None
    protection_inputs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        non_empty_string(self.signal_key, "signal_key")
        object.__setattr__(
            self,
            "protection_inputs",
            readonly_mapping(self.protection_inputs, "protection_inputs"),
        )


@dataclass(frozen=True, kw_only=True)
class SignalExecutionState:
    """Coordinate immutable records, direction blocking and a shared fill checkpoint.

    This recovery envelope is internal. All records are replaced together and
    serialized into one flat document; public queries expose only their own scope.
    """

    position: SignalPosition
    protection: ProtectionState
    target: SignalTarget | None = None
    blocked_direction: Literal[-1, 1] | None = None
    _applied_fill_keys: frozenset[str] = field(default_factory=frozenset, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_applied_fill_keys", frozenset(self._applied_fill_keys)
        )
        if isinstance(self.blocked_direction, bool) or self.blocked_direction not in (
            None,
            -1,
            1,
        ):
            raise ValueError("blocked_direction must be None, -1, or 1")
        if (
            self.protection.signal_key != self.position.signal_key
            or self.protection.position_id != self.position.position_id
        ):
            raise ValueError("Protection must identify the current signal position")
        if self.target is not None and (
            self.target.signal_key != self.position.signal_key
            or self.target.execution_model_name != self.position.execution_model_name
        ):
            raise ValueError("Signal position and target must share attribution")

    @property
    def signal_key(self) -> str:
        return self.position.signal_key

    @property
    def execution_model_name(self) -> str:
        return self.position.execution_model_name

    @property
    def contract(self) -> ibi.Contract | None:
        return self.position.contract

    @property
    def filled_quantity(self) -> float:
        return self.position.filled_quantity

    @property
    def position_id(self) -> str | None:
        return self.position.position_id

    @property
    def updated_at(self) -> datetime:
        return self.position.updated_at

    @property
    def protection_inputs(self) -> Mapping[str, Any]:
        return self.protection.protection_inputs

    @property
    def target_quantity(self) -> float | None:
        return self.target.target_quantity if self.target is not None else None

    @property
    def target_created_at(self) -> datetime | None:
        return self.target.target_created_at if self.target is not None else None

    @property
    def target_contract(self) -> ibi.Contract | None:
        return self.target.contract if self.target is not None else None

    @property
    def target_protection_inputs(self) -> Mapping[str, Any]:
        return (
            self.target.protection_inputs
            if self.target is not None
            else MappingProxyType({})
        )

    def with_fill(
        self, info: OrderInfo, record: FillRecord
    ) -> SignalExecutionState | None:
        """Apply one fill and all its dependent effects within the same checkpoint."""
        if record.deduplication_key in self._applied_fill_keys:
            return None
        old_quantity = self.filled_quantity
        quantity = old_quantity + record.execution.shares * fill_direction(record)
        blocked = self.blocked_direction
        protective_exit = quantity == 0 and info.role in {"STOP_LOSS", "TAKE_PROFIT"}
        if protective_exit and old_quantity:
            blocked = 1 if old_quantity > 0 else -1
        elif info.role == "OPEN" and quantity != 0:
            blocked = None
        closed = quantity == 0 and info.role in {"CLOSE", "STOP_LOSS", "TAKE_PROFIT"}
        target = self.target
        if protective_exit:
            target = SignalTarget(
                signal_key=self.signal_key,
                execution_model_name=self.execution_model_name,
                target_quantity=0,
                target_created_at=self.target_created_at,
            )
        position = replace(
            self.position,
            contract=info.trade.contract,
            filled_quantity=quantity,
            position_id=None if closed else self.position_id,
            updated_at=utc_now(),
        )
        protection = replace(
            self.protection,
            position_id=position.position_id,
            protection_inputs=(
                {}
                if closed and (target is None or not target.target_quantity)
                else self.protection_inputs
            ),
        )
        return replace(
            self,
            position=position,
            target=target,
            protection=protection,
            blocked_direction=blocked,
            _applied_fill_keys=self._applied_fill_keys | {record.deduplication_key},
        )

    def corrected(
        self, filled_quantity: float, corrected_at: datetime
    ) -> SignalExecutionState:
        """Align position and target without discarding direction or fill checkpoints."""
        flat = filled_quantity == 0
        position = replace(
            self.position,
            filled_quantity=filled_quantity,
            position_id=None if flat else self.position_id,
            updated_at=corrected_at,
        )
        protection = replace(
            self.protection,
            position_id=position.position_id,
            protection_inputs={} if flat else self.protection_inputs,
        )
        target = SignalTarget(
            signal_key=self.signal_key,
            execution_model_name=self.execution_model_name,
            target_quantity=filled_quantity,
            target_created_at=corrected_at,
            contract=None if flat else self.contract,
            protection_inputs=protection.protection_inputs,
        )
        return replace(self, position=position, target=target, protection=protection)

    def encode(self) -> dict[str, Any]:
        """Serialize the position and its applied-fill checkpoint together."""
        return {
            "schema_version": 1,
            "state_key": f"signal_execution:{self.signal_key}",
            "state_type": "signal_execution",
            "signal_key": self.signal_key,
            "execution_model_name": self.execution_model_name,
            "contract": tree(self.contract),
            "filled_quantity": self.filled_quantity,
            "target_quantity": self.target_quantity,
            "target_created_at": self.target_created_at,
            "target_contract": tree(self.target_contract),
            "target_protection_inputs": tree(dict(self.target_protection_inputs)),
            "position_id": self.position_id,
            "blocked_direction": self.blocked_direction,
            "protection_inputs": tree(dict(self.protection_inputs)),
            "updated_at": self.updated_at,
            "applied_fill_keys": sorted(self._applied_fill_keys),
        }

    @classmethod
    def decode(cls, data: Mapping[str, Any]) -> SignalExecutionState:
        """Restore checkpointed state; uncheckpointed schemas need conversion."""
        require_schema(data)
        if "applied_fill_keys" not in data:
            raise ValueError(
                "Position state lacks a fill checkpoint; standalone conversion required"
            )
        position = SignalPosition(
            signal_key=str(data["signal_key"]),
            execution_model_name=str(data["execution_model_name"]),
            contract=decode_tree(data.get("contract")),
            filled_quantity=float(data.get("filled_quantity", 0)),
            position_id=data.get("position_id"),
            updated_at=decode_tree(data["updated_at"]),
        )
        if data.get("state_key") != f"signal_execution:{position.signal_key}":
            raise ValueError("Signal execution identity does not match its signal")
        target = (
            SignalTarget(
                signal_key=position.signal_key,
                execution_model_name=position.execution_model_name,
                target_quantity=data["target_quantity"],
                target_created_at=decode_tree(data.get("target_created_at")),
                contract=decode_tree(data["target_contract"]),
                protection_inputs=decode_tree(data["target_protection_inputs"]),
            )
            if data.get("target_quantity") is not None
            else None
        )
        return cls(
            position=position,
            target=target,
            protection=ProtectionState(
                signal_key=position.signal_key,
                position_id=position.position_id,
                protection_inputs=decode_tree(data.get("protection_inputs", {})),
            ),
            blocked_direction=data.get("blocked_direction"),
            _applied_fill_keys=frozenset(data["applied_fill_keys"]),
        )


@dataclass(frozen=True, kw_only=True)
class ContractPosition:
    """Persist an accounted net filled position, independently of execution mode.

    Book updates this contract_position with the underlying position, order or roll
    mutation. Recovery verifies it against those records before trading starts.
    It is neither a desired target nor a broker-position snapshot.
    """

    contract: ibi.Contract
    filled_quantity: float
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        """Validate a concrete Contract and finite signed filled quantity."""
        qualified_contract(self.contract)
        object.__setattr__(
            self,
            "filled_quantity",
            finite_number(self.filled_quantity, "filled_quantity"),
        )
        aware_datetime(self.updated_at, "updated_at")

    def encode(self) -> dict[str, Any]:
        """Serialize one maintained contract position."""
        return {
            "schema_version": 1,
            "state_key": f"contract_position:{self.contract.conId}",
            "state_type": "contract_position",
            "conId": self.contract.conId,
            "contract": tree(self.contract),
            "filled_quantity": self.filled_quantity,
            "updated_at": self.updated_at,
        }

    @classmethod
    def decode(cls, document: Mapping[str, Any]) -> ContractPosition:
        """Validate the concrete identity of a persisted contract position."""
        require_schema(document)
        contract_position = cls(
            contract=decode_tree(document["contract"]),
            filled_quantity=document["filled_quantity"],
            updated_at=decode_tree(document["updated_at"]),
        )
        if (
            document.get("state_key")
            != f"contract_position:{contract_position.contract.conId}"
            or document.get("conId") != contract_position.contract.conId
        ):
            raise ValueError(
                "Persisted contract position identity does not match its Contract"
            )
        return contract_position


class PositionStore:
    """Own signal positions and the shared net contract positions derived from accounting.

    Read signal attribution separately from concrete-contract totals. Mutations
    are coordinated by Book so order evidence and roll checkpoints precede the
    contract positions based on them; this owner never queries the broker or peer stores.
    """

    def __init__(self, saver: AbstractBaseSaver, writer: PersistenceWriter) -> None:
        """Share the state collection and the single accounting writer."""
        self._saver = saver
        self._writer = writer
        self._signals: dict[str, SignalExecutionState] = {}
        self._contract_positions: dict[int, ContractPosition] = {}
        self._contributions: dict[tuple[str, str | int], dict[ibi.Contract, float]] = {}

    def _restore_signal(self, document: Mapping[str, Any]) -> None:
        """Restore a signal checkpoint without applying live mutation hooks."""
        state = SignalExecutionState.decode(document)
        self._signals[state.signal_key] = state

    def _restore_contract_position(self, document: Mapping[str, Any]) -> None:
        """Load a contract position for subsequent dependency-aware startup verification."""
        contract_position = ContractPosition.decode(document)
        self._contract_positions[contract_position.contract.conId] = contract_position

    def _put_signal(self, state: SignalExecutionState) -> SignalExecutionState:
        """Persist a checkpointed signal; Book then updates its net contribution."""
        self._writer.save(self._saver, state.encode())
        self._signals[state.signal_key] = state
        return state

    def _checkpoint(
        self, state: SignalExecutionState, received_keys: frozenset[str]
    ) -> SignalExecutionState:
        """Keep explicit state updates authoritative over previously received fills."""
        previous = self._signals.get(state.signal_key)
        return replace(
            state,
            _applied_fill_keys=state._applied_fill_keys
            | received_keys
            | (previous._applied_fill_keys if previous else frozenset()),
        )

    def _clear(self, cleared_at: datetime) -> None:
        """Save flat signal tombstones while retaining their fill checkpoints."""
        for state in self._signals.values():
            self._put_signal(
                replace(state.corrected(0, cleared_at), blocked_direction=None)
            )
        self._signals.clear()

    def _rekey_order(self, old_id: int, new_id: int) -> None:
        """Move a rebound order's contribution without applying its fills again."""
        contribution = self._contributions.pop(("order", old_id), {})
        self._contributions[("order", new_id)] = contribution

    def _for_signal(self, signal_key: str) -> SignalExecutionState | None:
        """Return one recovered signal execution state without creating it."""

        return self._signals.get(signal_key)

    def _signal_states(self) -> Mapping[str, SignalExecutionState]:
        """Return a read-only view of all signal execution states."""

        return MappingProxyType(self._signals)

    def _states_for_contract(
        self, contract: ibi.Contract
    ) -> tuple[SignalExecutionState, ...]:
        """Return non-flat signal execution states attributed to a Contract."""

        con_id = qualified_contract(contract).conId
        return tuple(
            state
            for state in self._signals.values()
            if state.contract is not None
            and state.contract.conId == con_id
            and state.filled_quantity
        )

    def for_signal(self, signal_key: str) -> SignalPosition | None:
        """Return filled exposure for a signal, including valid pending/flat state."""
        state = self._for_signal(signal_key)
        return state.position if state is not None else None

    def for_contract(self, contract: ibi.Contract) -> ContractPosition | None:
        """Return the maintained filled position for one concrete conId."""
        return self._contract_positions.get(qualified_contract(contract).conId)

    def by_signal(self) -> Mapping[str, SignalPosition]:
        """Return an immutable snapshot of positions keyed by signal."""
        return MappingProxyType(
            {key: state.position for key, state in self._signals.items()}
        )

    def signal_positions_for_contract(
        self, contract: ibi.Contract
    ) -> tuple[SignalPosition, ...]:
        """Return non-flat attributed positions for one concrete Contract."""
        return tuple(state.position for state in self._states_for_contract(contract))

    def filled_quantity(self, contract: ibi.Contract) -> float:
        """Read one maintained net filled position by conId, in either execution mode.

        This excludes desired targets and unfilled working quantities. It does
        not query the broker or scan historical fills.
        """
        contract_position = self._contract_positions.get(
            qualified_contract(contract).conId
        )
        return (
            contract_position.filled_quantity if contract_position is not None else 0.0
        )

    def by_contract(self) -> dict[ibi.Contract, float]:
        """Read non-flat accounted contract positions, without reconstructing history.

        Both execution modes use this account-level view. Signal positions
        are netted per concrete Contract; position ownership remains available
        separately through by_signal(). The returned mapping is a copy.
        """
        return {
            contract_position.contract: contract_position.filled_quantity
            for contract_position in self._contract_positions.values()
            if contract_position.filled_quantity
        }

    def blocked_direction(self, signal_key: str) -> Literal[-1, 1] | None:
        """Return the persisted stopped direction for a signal."""

        state = self._signals.get(signal_key)
        return state.blocked_direction if state is not None else None

    def _order_contribution(
        self, info: OrderInfo, cutoffs: Mapping[int, datetime]
    ) -> dict[ibi.Contract, float]:
        """Calculate one unattributed order's contribution to account contract positions.

        Positions already account for signal-attributed orders. Roll leg evidence
        replaces the BAG fallback, rather than adding the same movement twice.
        This runs on mutation/recovery, never on a position query.
        """
        if info.signal_key is not None:
            return {}
        return self._fill_contribution(info, cutoffs)

    def _fill_contribution(
        self, info: OrderInfo, cutoffs: Mapping[int, datetime]
    ) -> dict[ibi.Contract, float]:
        """Project concrete fill movement, sharing BAG/leg rules across both modes."""
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
                raise ValueError(f"ROLL orderId={info.orderId} lacks old/new Contracts")
            endpoints = (old, new)
            legs = tuple(
                r for r in records if r.contract.conId in (old.conId, new.conId)
            )
            records = legs or records
        for record in records:
            direction = fill_direction(record)
            movements = (
                ((endpoints[0], -direction), (endpoints[1], direction))
                if endpoints
                and record.contract.conId not in {c.conId for c in endpoints}
                else ((record.contract, direction),)
            )
            for contract, signed in movements:
                if not contract.conId:
                    continue
                cutoff = cutoffs.get(contract.conId)
                if cutoff is not None and record.time < cutoff:
                    continue
                contracts[contract.conId] = contract
                quantities[contract.conId] += record.execution.shares * signed
        return {
            contracts[con_id]: filled_quantity
            for con_id, filled_quantity in quantities.items()
            if filled_quantity
        }

    def _position_contribution(
        self, state: SignalExecutionState, rolls: Sequence[RollState]
    ) -> dict[ibi.Contract, float]:
        """Count a position unless an unfinished roll owns its physical movement."""
        if (
            state.contract is None
            or not state.contract.conId
            or not state.filled_quantity
        ):
            return {}
        if any(
            roll.stage is not FutureRollStage.COMPLETE
            and any(p.signal_key == state.signal_key for p in roll.participants)
            for roll in rolls
        ):
            return {}
        return {state.contract: state.filled_quantity}

    def _replace_contributions(
        self,
        changes: Mapping[tuple[str, str | int], dict[ibi.Contract, float]],
        *,
        persist: bool = True,
    ) -> None:
        """Apply changed accounting contributions through one contract-position path.

        Replacing the prior contribution makes order re-saves and repeated roll
        stages harmless. Only changed Contracts are persisted, after evidence.
        """
        deltas: defaultdict[int, float] = defaultdict(float)
        contracts: dict[int, ibi.Contract] = {}
        for owner, positions in changes.items():
            previous = self._contributions.get(owner, {})
            for multiplier, mapping in ((-1, previous), (1, positions)):
                for contract, filled_quantity in mapping.items():
                    contracts[contract.conId] = contract
                    deltas[contract.conId] += multiplier * filled_quantity
            if positions:
                self._contributions[owner] = positions
            else:
                self._contributions.pop(owner, None)
        for con_id, delta in deltas.items():
            if not delta:
                continue
            current = self._contract_positions.get(con_id)
            contract_position = ContractPosition(
                contract=contracts[con_id],
                filled_quantity=(
                    current.filled_quantity if current is not None else 0.0
                )
                + delta,
            )
            self._contract_positions[con_id] = contract_position
            if persist:
                self._put_contract_position(contract_position)

    def _rebuild(
        self,
        orders: Sequence[OrderInfo],
        rolls: Sequence[RollState],
        cutoffs: Mapping[int, datetime],
    ) -> tuple[ContractPosition, ...]:
        """Verify saved contract positions once at restore/reset, repairing partial writes.

        Position state remains authoritative for signal execution corrections; fills
        recover unattributed orders, and active rolls recover physical movement.
        No second applied-execution ledger or multi-document transaction is
        needed: order evidence precedes contract positions in the critical save queue.
        """
        saved = self._contract_positions
        self._contract_positions = {}
        self._contributions = {}
        changes: dict[tuple[str, str | int], dict[ibi.Contract, float]] = {
            ("order", info.orderId): self._order_contribution(info, cutoffs)
            for info in orders
        }
        changes.update(
            {
                ("signal", signal): self._position_contribution(state, rolls)
                for signal, state in self._signals.items()
            }
        )
        changes.update(
            {
                ("roll", state.series_key): dict(self._roll_contribution(state, orders))
                for state in rolls
                if state.mode is ExecutionMode.SIGNAL
                and state.stage is not FutureRollStage.COMPLETE
            }
        )
        self._replace_contributions(changes, persist=False)
        repaired = []
        for con_id in sorted(saved.keys() | self._contract_positions.keys()):
            previous = saved.get(con_id)
            current = self._contract_positions.get(con_id)
            if current is None and previous is not None:
                current = replace(previous, filled_quantity=0.0, updated_at=utc_now())
            if current is None:
                continue
            if (
                previous is not None
                and previous.filled_quantity == current.filled_quantity
            ):
                self._contract_positions[con_id] = previous
            else:
                self._contract_positions[con_id] = current
                repaired.append(current)
        return tuple(repaired)

    def _put_contract_position(self, contract_position: ContractPosition) -> None:
        """Persist a shared Contract position under its natural identity."""
        self._writer.save(self._saver, contract_position.encode())

    def _roll_contribution(
        self, state: RollState, orders: Sequence[OrderInfo]
    ) -> Mapping[ibi.Contract, float]:
        """Project broker-net old/new quantities from persisted roll Fills."""

        quantities: defaultdict[ibi.Contract, float] = defaultdict(float)
        quantities[state.old_contract] = sum(
            p.filled_quantity for p in state.participants
        )
        for info in orders:
            if (
                info.role == "ROLL"
                and info.submitted_at >= state.created_at
                and info.params.get("roll_state_key") == state.series_key
                and info.params.get("old_contract") == state.old_contract
                and info.params.get("new_contract") == state.new_contract
            ):
                for contract, movement in self._fill_contribution(info, {}).items():
                    quantities[contract] += movement
        return MappingProxyType(
            {
                contract: filled_quantity
                for contract, filled_quantity in quantities.items()
                if filled_quantity
            }
        )

    def _after_fill(
        self,
        info: OrderInfo,
        record: FillRecord,
        *,
        initial_checkpoint: frozenset[str] = frozenset(),
    ) -> SignalExecutionState | None:
        """Calculate a signal transition only when its checkpoint lacks the fill."""
        if info.signal_key is not None and not isinstance(info.trade.contract, ibi.Bag):
            state = self._signals.get(info.signal_key) or SignalExecutionState(
                position=SignalPosition(
                    signal_key=info.signal_key,
                    execution_model_name=info.execution_model_name,
                    contract=info.trade.contract,
                ),
                protection=ProtectionState(
                    signal_key=info.signal_key, position_id=None
                ),
                _applied_fill_keys=initial_checkpoint,
            )
            return state.with_fill(info, record)
        return None

    def _new_position(
        self,
        signal_key: str,
        execution_model_name: str,
        contract: ibi.Contract,
        *,
        target_quantity: float,
        target_created_at: datetime,
        protection_inputs: Mapping[str, Any] = MappingProxyType({}),
    ) -> SignalExecutionState:
        """Construct a new position for Book to persist with an evidence checkpoint."""

        position = SignalPosition(
            signal_key=signal_key,
            execution_model_name=execution_model_name,
            contract=contract,
            position_id=str(uuid4()),
        )
        return SignalExecutionState(
            position=position,
            target=SignalTarget(
                signal_key=signal_key,
                execution_model_name=execution_model_name,
                contract=contract,
                target_quantity=target_quantity,
                target_created_at=target_created_at,
                protection_inputs=protection_inputs,
            ),
            protection=ProtectionState(
                signal_key=signal_key,
                position_id=position.position_id,
                protection_inputs=protection_inputs,
            ),
            blocked_direction=self.blocked_direction(signal_key),
        )

    def _closed_position(self, signal_key: str) -> SignalExecutionState:
        state = self._signals[signal_key]
        # Closing keeps the original acceptance time, matching fill-driven closure.
        closed = state.corrected(0, utc_now())
        assert closed.target is not None
        return replace(
            closed,
            target=replace(closed.target, target_created_at=state.target_created_at),
        )


class ProtectionStore:
    """Read current protection from the authoritative signal-state owner."""

    def __init__(self, positions: PositionStore) -> None:
        self._positions = positions

    def for_signal(self, signal_key: str) -> ProtectionState | None:
        """Return saved inputs for the current position, if the signal is known."""
        state = self._positions._for_signal(signal_key)
        return state.protection if state is not None else None
