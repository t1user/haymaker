"""Source episodes and shared concrete-contract position accounting."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Literal

import ib_insync as ibi

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
from .rolls import RollState, FutureRollMode, FutureRollStage
from .orders import FillRecord, OrderInfo, fill_direction


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
    updated_at: datetime = field(default_factory=utc_now)
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

    def with_fill(self, info: OrderInfo, record: FillRecord) -> PositionState | None:
        """Return the episode after one unapplied fill, or None for a duplicate."""
        if record.deduplication_key in self._applied_fill_keys:
            return None
        old_quantity = self.quantity
        quantity = old_quantity + record.execution.shares * fill_direction(record)
        blocked = self.blocked_direction
        if info.role in {"STOP_LOSS", "TAKE_PROFIT"} and old_quantity and quantity == 0:
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
        target_quantity = 0.0 if protective_exit else self.target_quantity
        return replace(
            self,
            contract=info.trade.contract,
            quantity=quantity,
            target_quantity=target_quantity,
            target_contract=None if protective_exit else self.target_contract,
            target_bracket_inputs=(
                {} if protective_exit else self.target_bracket_inputs
            ),
            blocked_direction=blocked,
            position_id=None if episode_closed else self.position_id,
            bracket_inputs=(
                {} if episode_closed and not target_quantity else self.bracket_inputs
            ),
            updated_at=utc_now(),
            _applied_fill_keys=self._applied_fill_keys | {record.deduplication_key},
        )

    def corrected(self, quantity: float, corrected_at: datetime) -> PositionState:
        """Align an episode and its pending target to an accepted broker correction.

        Choosing the source and quantity belongs to reconciliation. Flat
        corrections clear episode inputs, but preserve the direction block and
        the fill checkpoint so historical evidence cannot undo the correction.
        """
        flat = quantity == 0
        return replace(
            self,
            quantity=quantity,
            target_quantity=quantity,
            target_contract=None if flat else self.contract,
            target_bracket_inputs={} if flat else self.bracket_inputs,
            target_created_at=corrected_at,
            position_id=None if flat else self.position_id,
            bracket_inputs={} if flat else self.bracket_inputs,
            updated_at=corrected_at,
        )

    def encode(self) -> dict[str, Any]:
        """Serialize the episode and its applied-fill checkpoint together."""
        return {
            "state_key": f"position:{self.source_key}",
            "state_type": "position",
            "source_key": self.source_key,
            "execution_model_name": self.execution_model_name,
            "contract": tree(self.contract),
            "quantity": self.quantity,
            "target_quantity": self.target_quantity,
            "target_created_at": self.target_created_at,
            "target_contract": tree(self.target_contract),
            "target_bracket_inputs": tree(dict(self.target_bracket_inputs)),
            "position_id": self.position_id,
            "blocked_direction": self.blocked_direction,
            "bracket_inputs": tree(dict(self.bracket_inputs)),
            "updated_at": self.updated_at,
            "applied_fill_keys": sorted(self._applied_fill_keys),
        }

    @classmethod
    def decode(cls, data: Mapping[str, Any]) -> PositionState:
        """Restore checkpointed state; uncheckpointed schemas need conversion."""
        if "applied_fill_keys" not in data:
            raise ValueError(
                "Position state lacks a fill checkpoint; standalone conversion required"
            )
        return cls(
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


@dataclass(frozen=True, kw_only=True)
class ContractPosition:
    """Persist an accounted net balance, independently of execution mode.

    Book updates this balance with the underlying episode, order or roll
    mutation. Recovery verifies it against those records before trading starts.
    It is neither a desired target nor a broker-position snapshot.
    """

    contract: ibi.Contract
    quantity: float
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        """Validate a concrete Contract and finite signed balance."""
        qualified_contract(self.contract)
        object.__setattr__(self, "quantity", finite_number(self.quantity, "quantity"))
        aware_datetime(self.updated_at, "updated_at")

    def encode(self) -> dict[str, Any]:
        """Serialize one maintained contract balance."""
        return {
            "state_key": f"balance:{self.contract.conId}",
            "state_type": "balance",
            "conId": self.contract.conId,
            "contract": tree(self.contract),
            "quantity": self.quantity,
            "updated_at": self.updated_at,
        }

    @classmethod
    def decode(cls, document: Mapping[str, Any]) -> ContractPosition:
        """Validate the concrete identity of a persisted balance."""
        balance = cls(
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


class PositionStore:
    """Own source episodes and the shared net balances derived from accounting.

    Read source attribution separately from concrete-contract totals. Mutations
    are coordinated by Book so order evidence and roll checkpoints precede the
    balances based on them; this owner never queries the broker or peer stores.
    """

    def __init__(self, saver: AbstractBaseSaver, writer: PersistenceWriter) -> None:
        """Share the state collection and the single accounting writer."""
        self._saver = saver
        self._writer = writer
        self._sources: dict[str, PositionState] = {}
        self._balances: dict[int, ContractPosition] = {}
        self._contributions: dict[tuple[str, str | int], dict[ibi.Contract, float]] = {}

    def _restore_source(self, document: Mapping[str, Any]) -> None:
        """Restore a source checkpoint without applying live mutation hooks."""
        state = PositionState.decode(document)
        self._sources[state.source_key] = state

    def _restore_balance(self, document: Mapping[str, Any]) -> None:
        """Load a balance for subsequent dependency-aware startup verification."""
        balance = ContractPosition.decode(document)
        self._balances[balance.contract.conId] = balance

    def _put_source(self, state: PositionState) -> PositionState:
        """Persist a checkpointed source; Book then updates its net contribution."""
        self._writer.save(self._saver, state.encode())
        self._sources[state.source_key] = state
        return state

    def _checkpoint(
        self, state: PositionState, received_keys: frozenset[str]
    ) -> PositionState:
        """Keep explicit state updates authoritative over previously received fills."""
        previous = self._sources.get(state.source_key)
        return replace(
            state,
            _applied_fill_keys=state._applied_fill_keys
            | received_keys
            | (previous._applied_fill_keys if previous else frozenset()),
        )

    def _clear(self, cleared_at: datetime) -> None:
        """Save flat source tombstones while retaining their applied-fill checkpoints."""
        for state in self._sources.values():
            self._put_source(
                replace(
                    state,
                    quantity=0,
                    target_quantity=0,
                    target_created_at=cleared_at,
                    target_contract=None,
                    target_bracket_inputs={},
                    position_id=None,
                    blocked_direction=None,
                    bracket_inputs={},
                    updated_at=cleared_at,
                )
            )
        self._sources.clear()

    def _rekey_order(self, old_id: int, new_id: int) -> None:
        """Move a rebound order's contribution without applying its fills again."""
        contribution = self._contributions.pop(("order", old_id), {})
        self._contributions[("order", new_id)] = contribution

    def for_source(self, source_key: str) -> PositionState | None:
        """Return one recovered one-to-one state without creating it."""

        return self._sources.get(source_key)

    def source_states(self) -> Mapping[str, PositionState]:
        """Return a read-only view of all one-to-one states."""

        return MappingProxyType(self._sources)

    def source_states_for_contract(
        self, contract: ibi.Contract
    ) -> tuple[PositionState, ...]:
        """Return non-flat one-to-one states attributed to a Contract."""

        con_id = qualified_contract(contract).conId
        return tuple(
            state
            for state in self._sources.values()
            if state.contract is not None
            and state.contract.conId == con_id
            and state.quantity
        )

    def quantity(self, contract: ibi.Contract) -> float:
        """Read one maintained net balance by conId, in either execution mode.

        This excludes desired targets and unfilled working quantities. It does
        not query the broker or scan historical fills.
        """
        balance = self._balances.get(qualified_contract(contract).conId)
        return balance.quantity if balance is not None else 0.0

    def by_contract(self) -> dict[ibi.Contract, float]:
        """Read non-flat accounted balances, without reconstructing history.

        Both execution modes use this account-level view. One-to-one sources
        are netted per concrete Contract; episode ownership remains available
        separately through source_states(). The returned mapping is a copy.
        """
        return {
            balance.contract: balance.quantity
            for balance in self._balances.values()
            if balance.quantity
        }

    def blocked_direction(self, source_key: str) -> Literal[-1, 1] | None:
        """Return the persisted stopped direction for a one-to-one source."""

        state = self._sources.get(source_key)
        return state.blocked_direction if state is not None else None

    def _order_contribution(
        self, info: OrderInfo, cutoffs: Mapping[int, datetime]
    ) -> dict[ibi.Contract, float]:
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
            contracts[con_id]: quantity
            for con_id, quantity in quantities.items()
            if quantity
        }

    def _episode_contribution(
        self, state: PositionState, rolls: Sequence[RollState]
    ) -> dict[ibi.Contract, float]:
        """Count an episode unless an unfinished roll owns its physical movement."""
        if state.contract is None or not state.contract.conId or not state.quantity:
            return {}
        if any(
            roll.stage is not FutureRollStage.COMPLETE
            and any(p.source_key == state.source_key for p in roll.participants)
            for roll in rolls
        ):
            return {}
        return {state.contract: state.quantity}

    def _replace_contributions(
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
                self._put_balance(balance)

    def _rebuild(
        self,
        orders: Sequence[OrderInfo],
        rolls: Sequence[RollState],
        cutoffs: Mapping[int, datetime],
    ) -> tuple[ContractPosition, ...]:
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
            ("order", info.orderId): self._order_contribution(info, cutoffs)
            for info in orders
        }
        changes.update(
            {
                ("source", source): self._episode_contribution(state, rolls)
                for source, state in self._sources.items()
            }
        )
        changes.update(
            {
                ("roll", state.series_key): dict(self._roll_contribution(state, orders))
                for state in rolls
                if state.mode is FutureRollMode.BRACKET
                and state.stage is not FutureRollStage.COMPLETE
            }
        )
        self._replace_contributions(changes, persist=False)
        repaired = []
        for con_id in sorted(saved.keys() | self._balances.keys()):
            previous = saved.get(con_id)
            current = self._balances.get(con_id)
            if current is None and previous is not None:
                current = replace(previous, quantity=0.0, updated_at=utc_now())
            if current is None:
                continue
            if previous is not None and previous.quantity == current.quantity:
                self._balances[con_id] = previous
            else:
                self._balances[con_id] = current
                repaired.append(current)
        return tuple(repaired)

    def _put_balance(self, balance: ContractPosition) -> None:
        """Persist a shared Contract balance under its natural identity."""
        self._writer.save(self._saver, balance.encode())

    def _roll_contribution(
        self, state: RollState, orders: Sequence[OrderInfo]
    ) -> Mapping[ibi.Contract, float]:
        """Project broker-net old/new quantities from persisted roll Fills."""

        old_quantity = sum(participant.quantity for participant in state.participants)
        moved = sum(
            record.execution.shares * fill_direction(record)
            for info in orders
            if info.role == "ROLL"
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

    def _after_fill(self, info: OrderInfo, record: FillRecord) -> PositionState | None:
        """Calculate a source transition only when its checkpoint lacks the fill."""
        if info.source_key is not None and not isinstance(info.trade.contract, ibi.Bag):
            state = self._sources.get(info.source_key) or PositionState(
                source_key=info.source_key,
                execution_model_name=info.execution_model_name,
                contract=info.trade.contract,
            )
            return state.with_fill(info, record)
        return None

    def _new_episode(
        self,
        source_key: str,
        execution_model_name: str,
        contract: ibi.Contract,
        *,
        target_quantity: float,
        target_created_at: datetime,
        bracket_inputs: Mapping[str, Any] = MappingProxyType({}),
    ) -> PositionState:
        """Construct a new episode for Book to persist with an evidence checkpoint."""

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
        return state

    def _closed_episode(self, source_key: str) -> PositionState:
        """Mark one source flat while preserving its recovery attribution."""

        state = self._sources[source_key]
        return replace(
            state,
            quantity=0.0,
            target_quantity=0.0,
            target_contract=None,
            target_bracket_inputs={},
            position_id=None,
            bracket_inputs={},
            updated_at=utc_now(),
        )
