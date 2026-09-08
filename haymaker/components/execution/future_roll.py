"""Recoverable mode-specific futures-roll execution."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from functools import partial
from itertools import combinations
from math import isclose, isnan
from typing import TYPE_CHECKING, ClassVar
from uuid import uuid4

import eventkit as ev  # type: ignore
import ib_insync as ibi

from ... import misc
from ...async_wrappers import create_background_task
from ...book import (
    FutureRollMode,
    FutureRollStage,
    OrderInfo,
    RollParticipant,
    RollState,
    TargetState,
)
from ...validators import finite_number, non_empty_string
from ..messages import StandardOrderRole

if TYPE_CHECKING:
    from ...book import Book
    from ...controller import Controller

log = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class RollHolding:
    """Describe one fill-accounted holding offered to FutureRoller.

    Args:
        contract: Concrete held Future.
        quantity: Signed logical quantity held in that expiry.
        execution_model_name: Target model that owns the logical state.
        source_key: One-to-one source identity in bracket mode.
        position_id: Optional one-to-one position episode.
    """

    contract: ibi.Future
    quantity: float
    execution_model_name: str
    source_key: str | None = None
    position_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.contract, ibi.Future) or not self.contract.conId:
            raise ValueError("contract must be a qualified Future")
        object.__setattr__(self, "quantity", finite_number(self.quantity, "quantity"))
        if not self.quantity:
            raise ValueError("quantity must not be zero")
        object.__setattr__(
            self,
            "execution_model_name",
            non_empty_string(self.execution_model_name, "execution_model_name"),
        )
        if self.source_key is not None:
            non_empty_string(self.source_key, "source_key")


class FutureRollExecutor(ABC):
    """Execute persisted futures-roll plans for one exclusive runtime mode.

    Args:
        name: Stable recovery identity. The concrete class name is used when
            omitted.

    Executors are public customization points but are not Atoms and do not
    consume PositionTarget. A target execution model binds one executor to the
    process-owned FutureRoller. Subclasses must derive work from Book and make
    every advancement idempotent.
    """

    mode: ClassVar[FutureRollMode]

    def __init__(self, *, name: str | None = None) -> None:
        self.name = non_empty_string(
            type(self).__name__ if name is None else name,
            "FutureRollExecutor name",
        )
        self.completedEvent = ev.Event("futureRollCompleted")
        self._controller: Controller | None = None
        self._bound_trades: dict[int, ibi.Trade] = {}

    def __str__(self) -> str:
        """Return stable executor name and implementation class."""

        return f"{self.name}[{type(self).__name__}]"

    @property
    def controller(self) -> Controller:
        """Return the Controller installed by FutureRoller."""

        if self._controller is None:
            raise RuntimeError("FutureRollExecutor has not been bound")
        return self._controller

    @property
    def book(self) -> Book:
        """Return the Controller-owned Book."""

        return self.controller.book

    def bind(self, controller: Controller) -> None:
        """Bind this executor once to its process Controller."""

        if self._controller is not None and self._controller is not controller:
            raise ValueError(
                "FutureRollExecutor is already bound to another Controller"
            )
        self._controller = controller

    @abstractmethod
    def holdings(self) -> tuple[RollHolding, ...]:
        """Return current non-flat futures holdings for this mode."""

    @abstractmethod
    def create_state(
        self,
        series_key: str,
        old_contract: ibi.Future,
        new_contract: ibi.Future,
        holdings: Sequence[RollHolding],
    ) -> RollState:
        """Build a persisted plan before any broker side effect."""

    @abstractmethod
    def advance(self, state: RollState) -> None:
        """Advance one persisted plan as far as current evidence permits."""

    def recover(self, state: RollState) -> None:
        """Rebind or advance one persisted plan after synchronization."""

        if state.executor_name != self.name or state.mode is not self.mode:
            raise RuntimeError(
                f"RollState for {state.series_key!r} requires "
                f"{state.executor_name}[{state.mode.value}]"
            )
        self.advance(state)

    @staticmethod
    def make_combo(old: ibi.Future, new: ibi.Future) -> ibi.Bag:
        """Construct the two-leg spread used to move one futures holding."""

        return ibi.Bag(
            symbol=new.symbol,
            exchange=new.exchange,
            currency=new.currency,
            multiplier=new.multiplier,
            comboLegs=[
                ibi.ComboLeg(
                    conId=old.conId,
                    ratio=1,
                    action="SELL",
                    exchange=old.exchange,
                ),
                ibi.ComboLeg(
                    conId=new.conId,
                    ratio=1,
                    action="BUY",
                    exchange=new.exchange,
                ),
            ],
        )

    def _submit_roll(self, state: RollState, participant: RollParticipant) -> None:
        combo = self.make_combo(state.old_contract, state.new_contract)
        identity = participant.source_key or str(state.old_contract.conId)
        params = {
            "roll_state_key": state.series_key,
            "old_contract": state.old_contract,
            "new_contract": state.new_contract,
        }
        trade = self.controller.trade(
            combo,
            ibi.MarketOrder(
                "BUY" if participant.quantity > 0 else "SELL",
                abs(participant.quantity),
            ),
            role=StandardOrderRole.ROLL,
            execution_model_name=self.name,
            source_key=participant.source_key,
            position_id=participant.position_id,
            params=params,
        )
        if trade is None:
            self._block(state, f"Roll submission failed for {identity!r}")
            return
        active = replace(
            state,
            stage=FutureRollStage.ROLL_ORDER_ACTIVE,
            roll_order_id=trade.order.orderId,
            updated_at=datetime.now(timezone.utc),
        )
        self.book.update_roll(active)
        self._bind_roll(trade, state.series_key)
        if self._fully_filled(trade):
            self.onRollFilledEvent(trade, state.series_key)

    def _bind_roll(self, trade: ibi.Trade, series_key: str) -> None:
        if self._bound_trades.get(trade.order.orderId) is trade:
            return
        self._bound_trades[trade.order.orderId] = trade
        trade.filledEvent += partial(self.onRollFilledEvent, series_key=series_key)
        trade.cancelledEvent += partial(
            self.onRollCancelledEvent, series_key=series_key
        )

    def _roll_info(self, state: RollState) -> OrderInfo | None:
        if state.roll_order_id is not None:
            info = self.book.order_by_id(state.roll_order_id)
            if info is not None:
                return info
        return next(
            (
                info
                for info in self.book.orders(
                    role=StandardOrderRole.ROLL,
                    execution_model_name=self.name,
                )
                if info.submitted_at >= state.created_at
                if info.params.get("roll_state_key") == state.series_key
                and info.params.get("old_contract") == state.old_contract
                and info.params.get("new_contract") == state.new_contract
            ),
            None,
        )

    @staticmethod
    def _fully_filled(trade: ibi.Trade) -> bool:
        return (
            trade.orderStatus.status == ibi.OrderStatus.Filled
            or trade.filled() >= trade.order.totalQuantity
        )

    def _recover_roll_order(self, state: RollState) -> bool:
        info = self._roll_info(state)
        if info is None:
            self._block(state, "Persisted roll order cannot be recovered")
            return False
        if state.roll_order_id != info.orderId:
            state = self.book.update_roll(
                replace(
                    state,
                    roll_order_id=info.orderId,
                    updated_at=datetime.now(timezone.utc),
                )
            )
        if self._fully_filled(info.trade):
            self.onRollFilledEvent(info.trade, state.series_key)
        elif info.active:
            self._bind_roll(info.trade, state.series_key)
        else:
            self._block(state, "Roll order became terminal before complete Fill")
        return True

    def onRollFilledEvent(self, trade: ibi.Trade, series_key: str) -> None:
        """Advance after IB reports a complete roll Fill."""

        if not self._fully_filled(trade):
            return
        state = self.book.roll_state(series_key)
        if state is None or state.stage is not FutureRollStage.ROLL_ORDER_ACTIVE:
            return
        filled = self.book.update_roll(
            replace(
                state,
                stage=FutureRollStage.ROLL_FILLED,
                reference_price=trade.orderStatus.avgFillPrice,
                updated_at=datetime.now(timezone.utc),
            )
        )
        self._defer(self._after_roll_fill, filled, trade)

    def _defer(self, callback: Callable[..., object], *args: object) -> None:
        """Run state advancement after current broker callbacks settle."""

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            callback(*args)
            return

        def enqueue_after_broker_callbacks() -> None:
            loop.call_soon(callback, *args)

        loop.call_soon(enqueue_after_broker_callbacks)

    def onRollCancelledEvent(self, trade: ibi.Trade, series_key: str) -> None:
        """Block automatic repair when a roll terminates before full Fill."""

        state = self.book.roll_state(series_key)
        if state is None or state.stage is not FutureRollStage.ROLL_ORDER_ACTIVE:
            return
        if self._fully_filled(trade):
            self.onRollFilledEvent(trade, series_key)
        else:
            self._block(state, "Roll order cancelled before complete Fill")

    @abstractmethod
    def _after_roll_fill(self, state: RollState, trade: ibi.Trade) -> None:
        """Apply mode-specific accounting after a complete combo Fill."""

    def _block(self, state: RollState, reason: str) -> None:
        blocked = replace(
            state,
            stage=FutureRollStage.BLOCKED,
            failure_reason=reason,
            updated_at=datetime.now(timezone.utc),
        )
        self.book.update_roll(blocked)
        log.critical(
            "Futures roll blocked for series %s: %s",
            state.series_key,
            reason,
        )

    def _complete(self, state: RollState) -> None:
        completed = replace(
            state,
            participant_index=len(state.participants),
            completed_occurrences=(
                *state.completed_occurrences,
                *state.occurrence_keys,
            ),
            stage=FutureRollStage.COMPLETE,
            roll_order_id=None,
            old_protection_order_ids=(),
            replacement_stop_order_id=None,
            replacement_take_profit_order_id=None,
            failure_reason=None,
            updated_at=datetime.now(timezone.utc),
        )
        self.book.update_roll(completed)
        self.completedEvent.emit(completed)


class DirectFutureRollExecutor(FutureRollExecutor):
    """Roll a concrete direct holding and durably transfer its absolute target.

    Existing endpoint adjustments finish first. A saved post-roll target
    snapshot transfers the old target additively to the destination exactly
    once; newer explicit Portfolio targets supersede that snapshot. Different
    expiries in the same series may coexist. Override target_transfers() to
    customize the target policy without replacing broker execution.
    """

    mode = FutureRollMode.DIRECT

    def holdings(self) -> tuple[RollHolding, ...]:
        """Return each non-flat concrete Future with its current target owner."""
        result = []
        for contract, quantity in self.book.direct_positions().items():
            state = self.book.target_state(contract)
            if isinstance(contract, ibi.Future) and state is not None and quantity:
                result.append(
                    RollHolding(
                        contract=contract,
                        quantity=quantity,
                        execution_model_name=state.execution_model_name,
                    )
                )
        return tuple(result)

    def create_state(
        self,
        series_key: str,
        old_contract: ibi.Future,
        new_contract: ibi.Future,
        holdings: Sequence[RollHolding],
    ) -> RollState:
        """Capture one concrete holding; other expiries are independent."""
        if len(holdings) != 1 or holdings[0].contract.conId != old_contract.conId:
            raise ValueError("A direct roll requires exactly one old-Contract holding")
        holding = holdings[0]
        return RollState(
            series_key=series_key,
            mode=self.mode,
            executor_name=self.name,
            old_contract=old_contract,
            new_contract=new_contract,
            participants=(
                RollParticipant(
                    execution_model_name=holding.execution_model_name,
                    quantity=holding.quantity,
                ),
            ),
        )

    def target_transfers(self, state: RollState) -> tuple[TargetState, ...]:
        """Capture default post-roll targets before submission.

        The old target becomes zero and is added to the destination's target.
        A custom executor may override this allocation policy. Return absolute
        TargetStates; the executor saves them before any roll submission.
        """
        old = self.book.target_state(state.old_contract)
        if old is None:
            raise ValueError("Direct roll requires the old Contract's TargetState")
        new = self.book.target_state(state.new_contract)
        now = datetime.now(timezone.utc)
        return (
            replace(old, target_quantity=0, target_created_at=now, updated_at=now),
            TargetState(
                execution_model_name=(
                    new.execution_model_name if new else old.execution_model_name
                ),
                contract=state.new_contract,
                target_quantity=(new.target_quantity if new else 0)
                + old.target_quantity,
                target_created_at=now,
            ),
        )

    def advance(self, state: RollState) -> None:
        """Wait for endpoint adjustments, execute the roll, or resume completion."""
        if state.terminal:
            return
        if state.stage is FutureRollStage.ROLL_ORDER_ACTIVE:
            self._recover_roll_order(state)
            return
        if state.stage is FutureRollStage.ROLL_FILLED:
            info = self._roll_info(state)
            if info is None:
                self._block(state, "Filled direct roll evidence is missing")
            else:
                self._after_roll_fill(state, info.trade)
            return
        if state.stage not in {
            FutureRollStage.PLANNED,
            FutureRollStage.WAITING_FOR_ACTIVE_WORK,
        }:
            self._block(state, f"Unsupported direct roll stage {state.stage.value}")
            return
        active = tuple(
            info
            for contract in (state.old_contract, state.new_contract)
            for info in self.book.active_orders(
                contract=contract, role=StandardOrderRole.TARGET_ADJUSTMENT
            )
        )
        if active:
            self.book.update_roll(
                replace(state, stage=FutureRollStage.WAITING_FOR_ACTIVE_WORK)
            )
            for info in active:
                self._bind_adjustment(info.trade, state.series_key)
            return
        quantity = self.book.direct_quantity(state.old_contract)
        if not quantity:
            self._complete(state)
            return
        participant = replace(state.participants[0], quantity=quantity)
        planned = self.book.update_roll(
            replace(
                state,
                participants=(participant,),
                participant_index=0,
                target_transfers=state.target_transfers or self.target_transfers(state),
                stage=FutureRollStage.PLANNED,
                updated_at=datetime.now(timezone.utc),
            )
        )
        self._submit_roll(planned, participant)

    def _bind_adjustment(self, trade: ibi.Trade, series_key: str) -> None:
        key = -trade.order.orderId
        if self._bound_trades.get(key) is trade:
            return
        self._bound_trades[key] = trade
        trade.filledEvent += partial(
            self.onAdjustmentFilledEvent, series_key=series_key
        )
        trade.cancelledEvent += partial(
            self.onAdjustmentCancelledEvent, series_key=series_key
        )

    def onAdjustmentFilledEvent(self, trade: ibi.Trade, series_key: str) -> None:
        """Continue a waiting roll after accounting for the adjustment."""
        self._defer_resume(series_key)

    def onAdjustmentCancelledEvent(self, trade: ibi.Trade, series_key: str) -> None:
        """Continue a waiting roll after endpoint cancellation."""
        self._defer_resume(series_key)

    def _defer_resume(self, series_key: str) -> None:
        """Let asynchronous accounting settle; support synchronous drivers too."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._resume_series(series_key)
        else:
            loop.call_soon(self._resume_series, series_key)

    def _resume_series(self, series_key: str) -> None:
        state = self.book.roll_state(series_key)
        if state is not None:
            self.advance(state)

    def _after_roll_fill(self, state: RollState, trade: ibi.Trade) -> None:
        current = self.book.roll_state(state.series_key)
        if current is None or current.stage is not FutureRollStage.ROLL_FILLED:
            return
        if self.book.direct_quantity(current.old_contract):
            self._block(current, "Direct roll did not flatten its old Contract")
            return
        if not current.target_transfers:
            self._block(current, "Direct roll lacks durable target transfers")
            return
        for target in current.target_transfers:
            # Book ignores older snapshots. Re-applying an identical absolute
            # target is idempotent across crashes between the two writes.
            self.book.update_target(target)
        self._complete(current)


class BracketFutureRollExecutor(FutureRollExecutor):
    """Roll one-to-one episodes and reinstall critical protection serially.

    Args:
        name: Stable recovery identity. The class name is used when omitted.

    Logical sources may offset inside the broker's net position. The executor
    submits only the deterministic broker-net BAG work, applies a market spread
    price to offset sources, and preserves each source and ``position_id``.
    A participant is not complete until its replacement stop is confirmed
    active. Take-profit replacement is best effort and never substitutes for
    stop-loss protection.
    """

    mode = FutureRollMode.BRACKET

    def __init__(self, *, name: str | None = None) -> None:
        super().__init__(name=name)
        self._reference_price_tasks: set[str] = set()

    def holdings(self) -> tuple[RollHolding, ...]:
        """Return non-flat, roll-enabled one-to-one futures holdings."""

        holdings: list[RollHolding] = []
        policies = self.controller.future_roll_policies
        for state in self.book.position_states().values():
            if (
                not state.quantity
                or not isinstance(state.contract, ibi.Future)
                or not policies.get(state.source_key, True)
            ):
                continue
            holdings.append(
                RollHolding(
                    contract=state.contract,
                    quantity=state.quantity,
                    execution_model_name=state.execution_model_name,
                    source_key=state.source_key,
                    position_id=state.position_id,
                )
            )
        return tuple(holdings)

    def create_state(
        self,
        series_key: str,
        old_contract: ibi.Future,
        new_contract: ibi.Future,
        holdings: Sequence[RollHolding],
    ) -> RollState:
        """Create a source queue with only the broker-net subset traded."""

        physical_sources = self._physical_sources(holdings)
        participants = tuple(
            RollParticipant(
                source_key=holding.source_key,
                execution_model_name=holding.execution_model_name,
                quantity=holding.quantity,
                position_id=holding.position_id,
                requires_trade=holding.source_key in physical_sources,
            )
            for holding in holdings
        )
        return RollState(
            series_key=series_key,
            mode=self.mode,
            executor_name=self.name,
            old_contract=old_contract,
            new_contract=new_contract,
            participants=participants,
        )

    @staticmethod
    def _physical_sources(
        holdings: Sequence[RollHolding], *, total: float | None = None
    ) -> set[str | None]:
        """Choose the smallest deterministic source subset equal to broker net."""

        if total is None:
            total = sum(holding.quantity for holding in holdings)
        if not total:
            return set()
        for size in range(1, len(holdings) + 1):
            for selected in combinations(holdings, size):
                if isclose(
                    sum(holding.quantity for holding in selected),
                    total,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    return {holding.source_key for holding in selected}
        raise ValueError("Logical sources cannot be attributed to broker net")

    def advance(self, state: RollState) -> None:
        """Advance the current source through trade and protection stages."""

        if state.terminal:
            return
        if state.stage is FutureRollStage.ROLL_ORDER_ACTIVE:
            self._recover_roll_order(state)
            return
        if state.stage is FutureRollStage.ROLL_FILLED:
            info = self._roll_info(state)
            self._move_position_and_cancel_protection(
                state,
                info.trade if info is not None else None,
            )
            return
        if state.stage is FutureRollStage.LOADING_REFERENCE_PRICE:
            self._request_reference_price(state)
            return
        if state.stage is FutureRollStage.CANCELLING_PROTECTION:
            self._continue_cancellation(state)
            return
        if state.stage is FutureRollStage.INSTALLING_STOP:
            self._continue_stop_installation(state)
            return
        if state.stage is FutureRollStage.INSTALLING_TAKE_PROFIT:
            self._continue_take_profit_installation(state)
            return
        if state.stage not in {
            FutureRollStage.PLANNED,
            FutureRollStage.WAITING_FOR_ACTIVE_WORK,
        }:
            self._block(state, f"Unsupported bracket roll stage {state.stage.value}")
            return
        participant = state.current_participant
        if participant is None:
            self._complete(state)
            return
        if participant.source_key is None:
            self._block(state, "Bracket roll participant is missing source_key")
            return
        refreshed = self._refresh_pending(state)
        if refreshed is None:
            return
        state = refreshed
        participant = state.current_participant
        if participant is None:
            self._complete(state)
        elif not participant.quantity:
            self._finish_participant(state)
        elif participant.requires_trade:
            self._submit_roll(state, participant)
        else:
            self._request_reference_price(state)

    def _refresh_pending(self, state: RollState) -> RollState | None:
        """Reconcile all unprocessed episodes before planning any broker work.

        Completed non-trading participants may offset remaining physical work.
        Preserve that offset when rebuilding the subset, not just the current
        participant's quantity. Zero non-trading entries retain notification
        attribution for episodes closed while waiting.
        """
        remaining = state.participants[state.participant_index :]
        active_work = tuple(
            info
            for participant in remaining
            for info in self.book.active_orders(source_key=participant.source_key)
            if info.role in {StandardOrderRole.OPEN, StandardOrderRole.CLOSE}
        )
        if active_work:
            waiting = self.book.update_roll(
                replace(
                    state,
                    stage=FutureRollStage.WAITING_FOR_ACTIVE_WORK,
                    updated_at=datetime.now(timezone.utc),
                )
            )
            for info in active_work:
                self._bind_adjustment(info.trade, waiting.series_key)
            return None
        pending = []
        for participant in remaining:
            position = self.book.position_state(participant.source_key or "")
            if position is None or (
                position.quantity
                and (
                    position.position_id != participant.position_id
                    or position.contract != state.old_contract
                    or position.execution_model_name != participant.execution_model_name
                )
            ):
                self._block(
                    state, "Unprocessed source episode changed before roll submission"
                )
                return None
            pending.append(
                replace(participant, quantity=position.quantity, requires_trade=False)
            )
        completed = tuple(state.participants[: state.participant_index])
        holdings = tuple(
            RollHolding(
                contract=state.old_contract,
                quantity=p.quantity,
                execution_model_name=p.execution_model_name,
                source_key=p.source_key,
                position_id=p.position_id,
            )
            for p in pending
            if p.quantity
        )
        net = sum(p.quantity for p in pending) + sum(
            p.quantity for p in completed if not p.requires_trade
        )
        try:
            physical = self._physical_sources(holdings, total=net)
        except ValueError:
            self._block(
                state,
                "Changed source quantities cannot preserve completed roll offsets",
            )
            return None
        participants = completed + tuple(
            replace(p, requires_trade=p.source_key in physical) for p in pending
        )
        if (
            participants == state.participants
            and state.stage is FutureRollStage.PLANNED
        ):
            return state
        return self.book.update_roll(
            replace(
                state,
                participants=participants,
                stage=FutureRollStage.PLANNED,
                updated_at=datetime.now(timezone.utc),
            )
        )

    def _bind_adjustment(self, trade: ibi.Trade, series_key: str) -> None:
        key = -trade.order.orderId
        if self._bound_trades.get(key) is trade:
            return
        self._bound_trades[key] = trade
        trade.filledEvent += partial(
            self.onAdjustmentFilledEvent, series_key=series_key
        )
        trade.cancelledEvent += partial(
            self.onAdjustmentCancelledEvent, series_key=series_key
        )

    def onAdjustmentFilledEvent(self, trade: ibi.Trade, series_key: str) -> None:
        """Resume a waiting bracket roll after OPEN/CLOSE work fills."""
        self._defer(self._resume_pending, series_key)

    def onAdjustmentCancelledEvent(self, trade: ibi.Trade, series_key: str) -> None:
        """Resume a waiting bracket roll after OPEN/CLOSE work cancels."""
        self._defer(self._resume_pending, series_key)

    def _resume_pending(self, series_key: str) -> None:
        """Load current durable work after broker accounting callbacks settle."""
        state = self.book.roll_state(series_key)
        if state is not None:
            self.advance(state)

    def _request_reference_price(self, state: RollState) -> None:
        if state.series_key in self._reference_price_tasks:
            return
        if state.stage is FutureRollStage.PLANNED:
            state = self.book.update_roll(
                replace(
                    state,
                    stage=FutureRollStage.LOADING_REFERENCE_PRICE,
                    updated_at=datetime.now(timezone.utc),
                )
            )
        self._reference_price_tasks.add(state.series_key)
        create_background_task(
            self._load_reference_price(state.series_key),
            name=f"future-roll-price-{state.series_key}",
        )

    async def _load_reference_price(self, series_key: str) -> None:
        state = self.book.roll_state(series_key)
        if state is None or state.stage is not FutureRollStage.LOADING_REFERENCE_PRICE:
            self._reference_price_tasks.discard(series_key)
            return
        combo = self.make_combo(state.old_contract, state.new_contract)
        ticker = self.controller.ib.reqMktData(combo, "221")
        try:
            price = float("nan")
            for _ in range(500):
                price = ticker.marketPrice()
                if not isnan(price):
                    break
                await asyncio.sleep(0.01)
            if isnan(price):
                self._block(state, "No combo price for logical roll adjustment")
                return
            moved = self.book.update_roll(
                replace(
                    state,
                    stage=FutureRollStage.ROLL_FILLED,
                    reference_price=price,
                    updated_at=datetime.now(timezone.utc),
                )
            )
            self._move_position_and_cancel_protection(moved, None)
        finally:
            self.controller.ib.cancelMktData(combo)
            self._reference_price_tasks.discard(series_key)

    def _after_roll_fill(self, state: RollState, trade: ibi.Trade) -> None:
        self._move_position_and_cancel_protection(state, trade)

    def _move_position_and_cancel_protection(
        self, state: RollState, trade: ibi.Trade | None
    ) -> None:
        current = self.book.roll_state(state.series_key)
        if current is None or current.stage is not FutureRollStage.ROLL_FILLED:
            return
        state = current
        participant = state.current_participant
        if participant is None or participant.source_key is None:
            self._block(state, "Filled bracket roll has no source participant")
            return
        if participant.requires_trade:
            info = self._roll_info(state)
            if info is None or not isclose(
                self._signed_filled_quantity(info),
                participant.quantity,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                self._block(
                    state,
                    "Bracket roll lacks complete normalized Fill evidence",
                )
                return
        position = self.book.position_state(participant.source_key)
        if (
            position is None
            or position.position_id != participant.position_id
            or position.quantity != participant.quantity
        ):
            self._block(state, "Source episode changed while roll was in flight")
            return
        if position.contract is None:
            self._block(state, "Source position has no held Contract")
            return
        if position.contract.conId == state.old_contract.conId:
            self.book.update_position(
                replace(
                    position,
                    contract=state.new_contract,
                    updated_at=datetime.now(timezone.utc),
                )
            )
        elif position.contract.conId != state.new_contract.conId:
            self._block(state, "Source position moved to an unexpected Contract")
            return
        protections = tuple(
            info
            for info in self.book.active_orders(source_key=participant.source_key)
            if info.role in {StandardOrderRole.STOP_LOSS, StandardOrderRole.TAKE_PROFIT}
        )
        if not any(info.role == StandardOrderRole.STOP_LOSS for info in protections):
            self._block(state, "Rolled bracket position has no critical stop")
            return
        cancelling = self.book.update_roll(
            replace(
                state,
                stage=FutureRollStage.CANCELLING_PROTECTION,
                old_protection_order_ids=tuple(info.orderId for info in protections),
                reference_price=(
                    trade.orderStatus.avgFillPrice
                    if trade is not None
                    else state.reference_price
                ),
                updated_at=datetime.now(timezone.utc),
            )
        )
        for info in protections:
            self._cancel_protection(info, state.series_key)
        current = self.book.roll_state(cancelling.series_key)
        if (
            current is not None
            and current.stage is FutureRollStage.CANCELLING_PROTECTION
        ):
            self._continue_cancellation(current)

    def _cancel_protection(self, info: OrderInfo, series_key: str) -> None:
        """Bind recovery callback and request cancellation of old protection."""

        if not info.active:
            return
        if self._bound_trades.get(info.orderId) is not info.trade:
            self._bound_trades[info.orderId] = info.trade
            info.trade.cancelledEvent += partial(
                self.onProtectionCancelledEvent,
                series_key=series_key,
            )
        self.controller.cancel(info.trade)

    def onProtectionCancelledEvent(self, trade: ibi.Trade, series_key: str) -> None:
        """Continue once an old protective order is terminal."""

        state = self.book.roll_state(series_key)
        if state is not None:
            self._continue_cancellation(state)

    def _continue_cancellation(self, state: RollState) -> None:
        current = self.book.roll_state(state.series_key)
        if (
            current is None
            or current.stage is not FutureRollStage.CANCELLING_PROTECTION
        ):
            return
        state = current
        active = tuple(
            info
            for order_id in state.old_protection_order_ids
            if (info := self.book.order_by_id(order_id)) is not None and info.active
        )
        if active:
            for info in active:
                self._cancel_protection(info, state.series_key)
            return
        self._install_stop(state)

    def _old_protection(
        self, state: RollState, role: StandardOrderRole
    ) -> OrderInfo | None:
        return next(
            (
                info
                for order_id in state.old_protection_order_ids
                if (info := self.book.order_by_id(order_id)) is not None
                and info.role == role
            ),
            None,
        )

    def _install_stop(self, state: RollState) -> None:
        stop_info = self._old_protection(state, StandardOrderRole.STOP_LOSS)
        participant = state.current_participant
        if stop_info is None or participant is None:
            self._block(state, "Critical stop evidence is unavailable")
            return
        pending = self.book.update_roll(
            replace(
                state,
                stage=FutureRollStage.INSTALLING_STOP,
                updated_at=datetime.now(timezone.utc),
            )
        )
        oca_group = str(uuid4())
        order = self._replacement_order(
            stop_info,
            pending.reference_price or 0.0,
            oca_group,
        )
        params = {
            **dict(stop_info.params),
            "roll_state_key": pending.series_key,
        }
        trade = self.controller.trade(
            pending.new_contract,
            order,
            role=StandardOrderRole.STOP_LOSS,
            execution_model_name=participant.execution_model_name,
            source_key=participant.source_key,
            position_id=participant.position_id,
            params=params,
        )
        if trade is None:
            self._block(pending, "Replacement stop submission failed")
            return
        installed = self.book.update_roll(
            replace(
                pending,
                replacement_stop_order_id=trade.order.orderId,
                updated_at=datetime.now(timezone.utc),
            )
        )
        trade.statusEvent += partial(
            self.onReplacementStopStatusEvent,
            series_key=state.series_key,
        )
        trade.filledEvent += partial(
            self.onReplacementStopStatusEvent,
            series_key=state.series_key,
        )
        trade.cancelledEvent += partial(
            self.onReplacementStopStatusEvent,
            series_key=state.series_key,
        )
        self._continue_stop_installation(installed)

    def _replacement_info(
        self, state: RollState, role: StandardOrderRole
    ) -> OrderInfo | None:
        order_id = (
            state.replacement_stop_order_id
            if role is StandardOrderRole.STOP_LOSS
            else state.replacement_take_profit_order_id
        )
        if order_id is not None:
            info = self.book.order_by_id(order_id)
            if info is not None:
                return info
        participant = state.current_participant
        if participant is None:
            return None
        return next(
            (
                info
                for info in self.book.orders(
                    source_key=participant.source_key,
                    contract=state.new_contract,
                    role=role,
                )
                if info.submitted_at >= state.created_at
                if info.params.get("roll_state_key") == state.series_key
            ),
            None,
        )

    def onReplacementStopStatusEvent(self, trade: ibi.Trade, series_key: str) -> None:
        """Advance only after the replacement stop is active or filled."""

        state = self.book.roll_state(series_key)
        if state is not None:
            self._continue_stop_installation(state)

    def _continue_stop_installation(self, state: RollState) -> None:
        current = self.book.roll_state(state.series_key)
        if current is None or current.stage is not FutureRollStage.INSTALLING_STOP:
            return
        state = current
        info = self._replacement_info(state, StandardOrderRole.STOP_LOSS)
        if info is None:
            self._install_stop(state)
            return
        if state.replacement_stop_order_id != info.orderId:
            state = self.book.update_roll(
                replace(
                    state,
                    replacement_stop_order_id=info.orderId,
                    updated_at=datetime.now(timezone.utc),
                )
            )
        status = info.trade.orderStatus.status
        if status == ibi.OrderStatus.Filled:
            self._finish_participant(state)
        elif info.active and status in {
            ibi.OrderStatus.PreSubmitted,
            ibi.OrderStatus.Submitted,
        }:
            self._install_take_profit(state, info)
        elif info.trade.isDone():
            self._block(state, "Replacement stop became terminal before activation")

    def _install_take_profit(self, state: RollState, stop_info: OrderInfo) -> None:
        take_profit = self._old_protection(state, StandardOrderRole.TAKE_PROFIT)
        if take_profit is None:
            self._finish_participant(state)
            return
        pending = self.book.update_roll(
            replace(
                state,
                stage=FutureRollStage.INSTALLING_TAKE_PROFIT,
                updated_at=datetime.now(timezone.utc),
            )
        )
        order = self._replacement_order(
            take_profit,
            pending.reference_price or 0.0,
            stop_info.trade.order.ocaGroup,
        )
        participant = pending.current_participant
        if participant is None:
            self._block(pending, "Take-profit participant is missing")
            return
        params = {
            **dict(take_profit.params),
            "roll_state_key": pending.series_key,
        }
        trade = self.controller.trade(
            pending.new_contract,
            order,
            role=StandardOrderRole.TAKE_PROFIT,
            execution_model_name=participant.execution_model_name,
            source_key=participant.source_key,
            position_id=participant.position_id,
            params=params,
        )
        if trade is None:
            log.warning(
                "Optional take-profit replacement failed for source %s",
                participant.source_key,
            )
            self._finish_participant(pending)
            return
        installed = self.book.update_roll(
            replace(
                pending,
                replacement_take_profit_order_id=trade.order.orderId,
                updated_at=datetime.now(timezone.utc),
            )
        )
        self._finish_participant(installed)

    def _continue_take_profit_installation(self, state: RollState) -> None:
        current = self.book.roll_state(state.series_key)
        if (
            current is None
            or current.stage is not FutureRollStage.INSTALLING_TAKE_PROFIT
        ):
            return
        state = current
        info = self._replacement_info(state, StandardOrderRole.TAKE_PROFIT)
        if info is not None:
            self._finish_participant(
                replace(
                    state,
                    replacement_take_profit_order_id=info.orderId,
                )
            )
            return
        stop_info = self._replacement_info(state, StandardOrderRole.STOP_LOSS)
        if stop_info is None:
            self._block(state, "Replacement stop disappeared before take-profit")
            return
        self._install_take_profit(state, stop_info)

    def _finish_participant(self, state: RollState) -> None:
        current = self.book.roll_state(state.series_key)
        if (
            current is None
            or current.participant_index != state.participant_index
            or current.stage is FutureRollStage.COMPLETE
        ):
            return
        state = current
        index = state.participant_index + 1
        if index >= len(state.participants):
            self._complete(state)
            return
        next_state = self.book.update_roll(
            replace(
                state,
                participant_index=index,
                stage=FutureRollStage.PLANNED,
                roll_order_id=None,
                old_protection_order_ids=(),
                replacement_stop_order_id=None,
                replacement_take_profit_order_id=None,
                reference_price=None,
                failure_reason=None,
                updated_at=datetime.now(timezone.utc),
            )
        )
        self.advance(next_state)

    @staticmethod
    def _signed_filled_quantity(info: OrderInfo) -> float:
        """Return normalized signed Fill evidence for one BAG order."""

        if info.trade.order.action == "BUY":
            direction = 1
        elif info.trade.order.action == "SELL":
            direction = -1
        else:
            raise ValueError(
                f"Ambiguous roll order action: {info.trade.order.action!r}"
            )
        return direction * sum(record.execution.shares for record in info.fills)

    @staticmethod
    def _replacement_order(
        info: OrderInfo,
        roll_price: float,
        oca_group: str,
    ) -> ibi.Order:
        options = ibi.util.dataclassNonDefaults(info.trade.order)
        for key in ("orderId", "permId", "softDollarTier", "clientId"):
            options.pop(key, None)
        options["ocaGroup"] = oca_group
        if info.trade.order.ocaType:
            options["ocaType"] = info.trade.order.ocaType
        if options.get("orderType") == "FIX PEGGED":
            options["orderType"] = "TRAIL"
            options["auxPrice"] = misc.round_tick(
                (
                    info.params.get("trail_multiple")
                    or info.params.get("adjusted_multiple")
                )
                * info.params["sl_points"],
                info.params["min_tick"],
            )
        for field_name in ("lmtPrice", "trailStopPrice", "adjustedStopPrice"):
            if options.get(field_name):
                options[field_name] += roll_price
        return ibi.Order(**options)


__all__ = [
    "BracketFutureRollExecutor",
    "DirectFutureRollExecutor",
    "FutureRollExecutor",
    "FutureRollMode",
    "FutureRollStage",
    "RollHolding",
    "RollParticipant",
    "RollState",
]
