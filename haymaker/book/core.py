"""Coordinate local accounting across state owners; never call the broker.

Collection owners implement data-specific transitions and persistence. Book
sequences changes across them using one fail-stop writer, and completes recovery
before any broker work is permitted. Mutation methods are synchronous; when
queued saving is enabled their return means acceptance, not database commit.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from types import MappingProxyType
from typing import Any, Literal

import ib_insync as ibi

from ..enums import ExecutionMode
from ..blotter import Blotter
from ..saver import AbstractBaseSaver, MongoSaver
from .orders import OrderInfo, FillRecord, OrderStore, _execution_key, fill_direction
from .positions import (
    SignalExecutionState,
    SignalPosition,
    SignalTarget,
    PositionStore,
    ProtectionStore,
    ProtectionState,
)
from .schema import require_schema
from .targets import ContractTarget, TargetStore
from .rolls import RollState, RollStore, FutureRollStage
from .portfolio import PortfolioStateStore
from .persistence import (
    DEFAULT_ORDER_COLLECTION_NAME,
    DEFAULT_STATE_COLLECTION_NAME,
    utc_now,
    PersistenceWriter,
)


class Book:
    """Coordinate typed accounting through state-owning collections.

    Query orders, positions, targets, rolls and portfolios through the named
    owners. Use Book's mutation methods when a change affects accounting across
    owners; do not call their private mutation hooks directly. Book performs no
    allocation or broker API calls.

    Args:
        order_saver: Synchronous keyed order backend; defaults to Mongo orders.
            Injected backends must restore aware datetimes and raise on failure.
        state_saver: Synchronous keyed state backend with the same timestamp and
            failure contract; defaults to Mongo state.
        blotter: Optional reporting service, never an accounting dependency.
        save_async: Queue normal writes on one critical writer; defaults to True.
        max_rejected_orders: Per-model process rejection threshold; defaults to 3.
        restore: Load and repair saved accounting before returning; defaults to False.

    Attributes:
        orders: Order queries and execution attribution.
        positions: Signal and concrete-contract filled positions.
        targets: Accepted signal and contract targets.
        protection: Saved inputs for each current signal position.
        rolls: Durable futures-roll checkpoints.
        portfolios: Opaque custom Portfolio recovery mappings.
        blotter: Optional trade reporting service.
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
        """Assemble owners and finish any requested recovery before enabling queues."""
        order_saver = order_saver or MongoSaver(
            DEFAULT_ORDER_COLLECTION_NAME, query_key="orderId", tz_aware=True
        )
        state_saver = state_saver or MongoSaver(
            DEFAULT_STATE_COLLECTION_NAME, query_key="state_key", tz_aware=True
        )
        self._writer = PersistenceWriter()
        self.orders = OrderStore(order_saver, self._writer)
        self.positions = PositionStore(state_saver, self._writer)
        self.targets = TargetStore(state_saver, self._writer, self.positions)
        self.protection = ProtectionStore(self.positions)
        self.rolls = RollStore(state_saver, self._writer)
        self.portfolios = PortfolioStateStore(state_saver, self._writer)
        self.blotter = blotter
        self.max_rejected_orders = max_rejected_orders
        if restore:
            self._restore_documents(order_saver.read({}), state_saver.read({}))
            self._recover_position_fills()
            self._repair_contract_positions()
        if save_async:
            self._writer.enable_async()

    def check_writable(self) -> None:
        """Reject new broker work after critical persistence has halted."""
        self._writer.check_writable()

    async def close(self) -> None:
        """Drain accepted critical writes before process shutdown."""
        await self._writer.close()

    def _restore_documents(
        self,
        order_documents: Sequence[Mapping[str, Any]],
        state_documents: Sequence[Mapping[str, Any]],
    ) -> None:
        """Dispatch each physical document once, before cross-state verification."""
        for document in (*order_documents, *state_documents):
            require_schema(document)
        self.orders._restore(order_documents)
        loaders: dict[str, Callable[[Mapping[str, Any]], None]] = {
            "signal_execution": self.positions._restore_signal,
            "contract_position": self.positions._restore_contract_position,
            "contract_execution": self.targets._restore,
            "roll": self.rolls._restore,
            "portfolio": self.portfolios._restore,
        }
        for document in state_documents:
            kind = document.get("state_type")
            if kind not in loaders:
                raise ValueError(f"Unknown Book state_type: {kind!r}")
            loaders[kind](document)

    def clear_state(self) -> None:
        """Persist reset tombstones without deleting order or execution evidence."""
        self.check_writable()
        cleared_at = utc_now()
        contracts = tuple(self.positions.by_contract())
        self.positions._clear(cleared_at)
        self.targets._clear(contracts, cleared_at)
        self.rolls._clear(cleared_at)
        self.portfolios._clear(cleared_at)
        self._repair_contract_positions()

    def _repair_contract_positions(self) -> None:
        """Verify shared contract positions after loading/reset, never during a query."""
        for contract_position in self.positions._rebuild(
            self.orders.query(), self.rolls.all(), self.targets.cutoffs
        ):
            self.positions._put_contract_position(contract_position)

    def save_order(self, info: OrderInfo) -> OrderInfo:
        """Save order evidence before replacing its physical position contribution."""
        self.orders._put(info)
        changes: dict[tuple[str, str | int], dict[ibi.Contract, float]] = {
            ("order", info.orderId): self.positions._order_contribution(
                info, self.targets.cutoffs
            )
        }
        for roll in self.rolls.active_signal():
            if info.params.get("roll_state_key") == roll.series_key:
                changes[("roll", roll.series_key)] = dict(
                    self.positions._roll_contribution(
                        roll, self.orders.query(role="ROLL")
                    )
                )
        self.positions._replace_contributions(changes)
        return info

    def rebind_trade(self, trade: ibi.Trade) -> ibi.Trade | None:
        """Bind a known live Trade, preserving the identity of its accounted fills.

        Return an unmatched Trade unchanged; return None after a known rebind.
        """
        info = self._bind_trade(trade)
        if info is None:
            return trade
        self.save_order(info)
        return None

    def _bind_trade(self, trade: ibi.Trade) -> OrderInfo | None:
        """Resolve a broker callback and move, rather than duplicate, its contribution."""
        self.check_writable()
        info, previous_id = self.orders._rebind(trade)
        if info is not None and previous_id != info.orderId:
            self.positions._rekey_order(previous_id, info.orderId)
        return info

    def prune_order(self, order_id: int) -> None:
        """Retire working status while preserving evidence and accounted quantity."""
        info = self.orders.by_id(order_id)
        if info is not None:
            info.trade.orderStatus.status = ibi.OrderStatus.Cancelled
            self.save_order(info)

    def update_position(self, position: SignalPosition) -> SignalPosition:
        """Update filled exposure while preserving its target and protection."""
        current = self.positions._for_signal(position.signal_key)
        if current is None:
            state = SignalExecutionState(
                position=position,
                protection=ProtectionState(
                    signal_key=position.signal_key, position_id=position.position_id
                ),
            )
        else:
            state = replace(
                current,
                position=position,
                protection=replace(
                    current.protection, position_id=position.position_id
                ),
            )
        return self._update_signal_state(state).position

    def update_signal_target(self, target: SignalTarget) -> SignalTarget:
        """Accept a signal setpoint without changing the held position inputs."""
        current = self.positions._for_signal(target.signal_key)
        if current is not None:
            if (
                current.target_created_at is not None
                and target.target_created_at is not None
                and target.target_created_at < current.target_created_at
            ):
                assert current.target is not None
                return current.target
            state = replace(
                current,
                target=target,
                position=replace(
                    current.position,
                    execution_model_name=target.execution_model_name,
                    updated_at=utc_now(),
                ),
            )
        else:
            state = SignalExecutionState(
                target=target,
                position=SignalPosition(
                    signal_key=target.signal_key,
                    execution_model_name=target.execution_model_name,
                ),
                protection=ProtectionState(
                    signal_key=target.signal_key, position_id=None
                ),
            )
        self._update_signal_state(state)
        return target

    def correct_signal_position(
        self, signal_key: str, filled_quantity: float, corrected_at: datetime
    ) -> SignalPosition:
        """Align filled and desired exposure, preserving blocks and checkpoints."""
        state = self.positions._for_signal(signal_key)
        if state is None:
            raise KeyError(signal_key)
        return self._update_signal_state(
            state.corrected(filled_quantity, corrected_at)
        ).position

    def _update_signal_state(self, state: SignalExecutionState) -> SignalExecutionState:
        """Record authoritative signal state and update its shared net contribution."""
        state = self.positions._checkpoint(
            state, self.orders.received_fill_keys(state.signal_key)
        )
        return self._save_position(state)

    def _save_position(self, state: SignalExecutionState) -> SignalExecutionState:
        """Commit a checkpoint before updating the aggregate position derived from it."""
        self.positions._put_signal(state)
        self.positions._replace_contributions(
            {
                ("signal", state.signal_key): self.positions._position_contribution(
                    state, self.rolls.all(active_only=True)
                )
            }
        )
        return state

    def update_target(self, state: ContractTarget) -> ContractTarget:
        """Accept a latest target, retaining any established accounting reset cutoff."""
        return self.targets._put(state)

    def update_roll(self, state: RollState) -> RollState:
        """Persist a roll checkpoint and replace all affected contributions together."""
        previous = self.rolls.for_series(state.series_key)
        self.rolls._put(state)
        changes: dict[tuple[str, str | int], dict[ibi.Contract, float]] = {
            ("roll", state.series_key): (
                dict(
                    self.positions._roll_contribution(
                        state, self.orders.query(role="ROLL")
                    )
                )
                if state.mode is ExecutionMode.SIGNAL
                and state.stage is not FutureRollStage.COMPLETE
                else {}
            )
        }
        signals = {
            p.signal_key
            for roll in (previous, state)
            if roll is not None
            for p in roll.participants
            if p.signal_key is not None
        }
        for signal in sorted(signals):
            position = self.positions._for_signal(signal)
            changes[("signal", signal)] = (
                self.positions._position_contribution(
                    position, self.rolls.all(active_only=True)
                )
                if position is not None
                else {}
            )
        self.positions._replace_contributions(changes)
        return state

    def effective_quantity(self, signal_key: str) -> float:
        """Return fill-accounted quantity plus relevant working position orders."""

        state = self.positions._for_signal(signal_key)
        quantity = state.filled_quantity if state is not None else 0.0
        working = sum(
            info.signed_working_quantity
            for info in self.orders.active(signal_key=signal_key)
            if info.role in {"OPEN", "CLOSE", "TARGET_ADJUSTMENT"}
        )
        return quantity + working

    def aggregate_with_working(
        self, contract: ibi.Contract, execution_model_name: str | None = None
    ) -> float:
        """Return aggregate logical quantity including matching working orders."""

        return self.positions.filled_quantity(contract) + sum(
            info.signed_working_quantity
            for info in self.orders.active(
                contract=contract,
                execution_model_name=execution_model_name,
            )
            if info.role in {"OPEN", "CLOSE", "TARGET_ADJUSTMENT", "ROLL"}
        )

    def create_signal_position(
        self,
        signal_key: str,
        execution_model_name: str,
        contract: ibi.Contract,
        *,
        target_quantity: float,
        target_created_at: datetime,
        protection_inputs: Mapping[str, Any] = MappingProxyType({}),
    ) -> SignalPosition:
        """Create and persist a fresh independently managed position."""

        return self._update_signal_state(
            self.positions._new_position(
                signal_key,
                execution_model_name,
                contract,
                target_quantity=target_quantity,
                target_created_at=target_created_at,
                protection_inputs=protection_inputs,
            )
        ).position

    def close_signal_position(self, signal_key: str) -> SignalPosition:
        """Close one signal position through the shared accounting path."""
        return self._update_signal_state(
            self.positions._closed_position(signal_key)
        ).position

    def update_blocked_direction(
        self, signal_key: str, direction: Literal[-1, 1] | None
    ) -> SignalPosition:
        """Persist or clear a signal execution stop-out direction."""

        if isinstance(direction, bool) or direction not in (None, -1, 1):
            raise ValueError("direction must be None, -1, or 1")
        state = self.positions._for_signal(signal_key)
        if state is None:
            raise KeyError(signal_key)
        return self._update_signal_state(
            replace(
                state,
                blocked_direction=direction,
                position=replace(state.position, updated_at=utc_now()),
            )
        ).position

    def apply_fill(self, trade: ibi.Trade, fill: ibi.Fill) -> bool:
        """Conditionally apply one unseen Fill and persist evidence first.

        Returns:
            ``True`` when the fill was new and projections changed, otherwise
            ``False`` for a replayed execution.
        """

        info = self._bind_trade(trade)
        if info is None:
            raise KeyError(
                f"No order record for orderId={trade.order.orderId} "
                f"permId={trade.order.permId}"
            )
        record = FillRecord.from_fill(trade, fill)
        if not isinstance(trade.contract, ibi.Bag):
            fill_direction(record)
        received = (
            self.orders.received_fill_keys(info.signal_key)
            if info.signal_key and self.positions._for_signal(info.signal_key) is None
            else frozenset()
        )
        new = info.add_fill(record)
        # Persist the live Trade even on a replay: its ID/status may have changed.
        # Evidence must precede the signal checkpoint and net contract position.
        self.save_order(info)
        # A signal first seen after an explicit clear starts beyond previously
        # accounted evidence. Startup repairs intentionally omit this baseline.
        state = self.positions._after_fill(info, record, initial_checkpoint=received)
        if state is not None:
            self._save_position(state)
            return True
        return new

    def _recover_position_fills(self) -> None:
        """Finish missing signal projections without replaying checkpointed fills."""
        evidence = sorted(
            (
                (record.time, info.orderId, index, info, record)
                for info in self.orders.query()
                if info.signal_key is not None
                and not isinstance(info.trade.contract, ibi.Bag)
                for index, record in enumerate(info.fills)
            ),
            key=lambda item: item[:3],
        )
        for _, _, _, info, record in evidence:
            state = self.positions._after_fill(info, record)
            if state is not None:
                self.positions._put_signal(state)

    def update_commission(
        self,
        trade: ibi.Trade,
        fill: ibi.Fill,
        report: ibi.CommissionReport,
    ) -> bool:
        """Persist the live Trade and attach a late report to matching Fill evidence.

        Returns:
            ``True`` when matching fill evidence was updated, otherwise
            ``False`` when the order or execution is unknown.
        """

        if not isinstance(report, ibi.CommissionReport):
            raise TypeError("report must be an ib_insync.CommissionReport")
        info = self._bind_trade(trade)
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
                info.fills = tuple(records)
                # Historical reports need to update Trade diagnostics too;
                # the callback's Fill can be a different broker-history object.
                for trade_fill in trade.fills:
                    if _execution_key(trade, trade_fill) == key:
                        ibi.util.dataclassUpdate(trade_fill.commissionReport, report)
                self.save_order(info)
                return True
        # Complete Trade diagnostics still matter when no normalized fill matches.
        self.save_order(info)
        return False

    def blotter_records(
        self, signal_key: str, position_id: str | None = None
    ) -> tuple[Mapping[str, Any], ...]:
        """Query available blotter rows by signal and optional position."""

        if self.blotter is None:
            return ()
        return tuple(
            row
            for row in self.blotter.records()
            if row.get("signal_key") == signal_key
            and (position_id is None or row.get("position_id") == position_id)
        )

    def verify_for_rejections(self, execution_model_name: str) -> bool:
        """Return whether a model is below the configured process rejection limit."""
        return (
            self.orders.rejection_count(execution_model_name) < self.max_rejected_orders
        )

    def execution_model_names(self) -> set[str]:
        """Return all persisted model names referenced by live state."""

        return {
            *(
                state.execution_model_name
                for state in self.positions._signal_states().values()
            ),
            *(state.execution_model_name for state in self.targets.all()),
            *(info.execution_model_name for info in self.orders.query()),
        }
