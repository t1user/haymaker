"""Coordinate local accounting across state owners; never call the broker."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from types import MappingProxyType
from typing import Any, Literal
from uuid import uuid4

import ib_insync as ibi

from ..blotter import Blotter
from ..misc import decode_tree, tree
from ..saver import AbstractBaseSaver, MongoSaver
from ..validators import non_empty_string, qualified_contract, readonly_mapping
from .orders import OrderInfo, FillRecord, _execution_key
from .positions import PositionState, ContractPosition
from .targets import TargetState
from .rolls import RollState, FutureRollMode, FutureRollStage
from .persistence import (
    DEFAULT_ORDER_COLLECTION_NAME,
    DEFAULT_STATE_COLLECTION_NAME,
    utc_now,
    PersistenceWriter,
)

log = logging.getLogger(__name__)


def _contract_key(contract: ibi.Contract) -> int:
    return qualified_contract(contract).conId


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
        self._writer = PersistenceWriter()
        if restore:
            self._restore_documents(*self._read_documents())
            self._recover_position_fills()
            # Runtime constructs Book before a running loop exists. Finish
            # startup repairs synchronously, just like the recovery reads.
            for balance in self._rebuild_balances():
                self._save_balance(balance)
        if save_async:
            self._writer.enable_async()

    def _save(self, saver: AbstractBaseSaver, document: dict[str, Any]) -> None:
        """Submit a serialized state change to the shared writer."""
        self._writer.save(saver, document)

    def check_writable(self) -> None:
        """Reject new broker work after critical persistence has halted."""
        self._writer.check_writable()

    async def close(self) -> None:
        """Drain critical pending mutations before shutdown."""
        await self._writer.close()

    def clear_state(self) -> None:
        """Durably flatten and clear typed recovery projections.

        This is reserved for explicit Controller reset/zero startup actions;
        historical order evidence remains intact. Flat records are persisted
        before in-memory removal so stale targets cannot return after restart.
        """

        cleared_at = utc_now()
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
                cleared_position.encode(),
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
            document = cleared_target.encode()
            document["cleared"] = True
            self._save(self._state_saver, document)
            self._target_fill_cutoffs[target.contract.conId] = cleared_at
        for roll in self._rolls.values():
            self._save(
                self._state_saver,
                (
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
                    ).encode()
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
                position_state = PositionState.decode(document)
                self._positions[position_state.source_key] = position_state
            elif state_type == "target":
                target_state = TargetState.decode(document)
                if target_state.fill_evidence_start_at is not None:
                    self._target_fill_cutoffs[target_state.contract.conId] = (
                        target_state.fill_evidence_start_at
                    )
                if not document.get("cleared", False):
                    self._targets[target_state.contract.conId] = target_state
            elif state_type == "roll":
                roll_state = RollState.decode(document)
                self._rolls[roll_state.series_key] = roll_state
            elif state_type == "portfolio":
                self._portfolio_states[str(document["portfolio_key"])] = (
                    MappingProxyType(decode_tree(document.get("state", {})))
                )
            elif state_type == "balance":
                balance = ContractPosition.decode(document)
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
        self._save(self._state_saver, state.encode())
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
        self._save(self._state_saver, state.encode())
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
        self._save(self._state_saver, state.encode())
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
                current = replace(previous, quantity=0.0, updated_at=utc_now())
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
        self._save(self._state_saver, balance.encode())

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
                updated_at=utc_now(),
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
            replace(state, blocked_direction=direction, updated_at=utc_now())
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
            return state.with_fill(info, record)
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
                self._save(self._state_saver, state.encode())

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
                "updated_at": utc_now(),
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
