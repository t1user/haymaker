"""Stateful absolute-target execution models."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import ib_insync as ibi

from ...base import Atom
from ...book import FutureRollMode, RollState, TargetState
from ...misc import action, sign
from ...validators import non_empty_string, order_field_validator, qualified_contract
from ..messages import PositionTarget, StandardOrderRole
from .future_roll import FutureRollExecutor


def _order_options(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    """Copy and validate a set of IB Order keyword arguments."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result = dict(value)
    order_field_validator(result)
    return result


class ExecutionModel(Atom, ABC):
    """Abstract stateful consumer of absolute PositionTargets.

    Args:
        name: Stable persistence and routing identity. The class name is used
            when omitted.

    Models validate and persist a latest target before submitting work, retain
    only target state and broker evidence in Book, and emit accepted targets
    to Controller for delayed verification. Invalid targets raise before any
    state change or broker submission.
    """

    def __init__(self, *, name: str | None = None) -> None:
        super().__init__()
        self.name = non_empty_string(
            type(self).__name__ if name is None else name,
            "ExecutionModel name",
        )
        runtime = getattr(self, "runtime", None)
        if runtime is None or getattr(runtime, "controller", None) is None:
            raise RuntimeError(f"{type(self).__name__} requires a ready RuntimeContext")
        self.controller = runtime.controller
        self._started_generation = -1
        self._bound_trades: dict[int, ibi.Trade] = {}
        self.connect(self.controller)

    def __str__(self) -> str:
        """Return stable model name and implementation class."""

        return f"{self.name}[{type(self).__name__}]"

    def onStart(self, data: Any, source: Atom | None = None) -> None:
        """Recover outstanding work once per supervised workload generation."""

        generation = self.runtime.workload_generation
        if generation != self._started_generation:
            self._started_generation = generation
            self.recover()
        super().onStart(data, source)

    def onData(self, target: PositionTarget, *args: object) -> None:
        """Validate and accept one target."""

        self._validate_target(target)
        if self.accept(target):
            self.dataEvent.emit(target, self.name)

    def _validate_target(self, target: PositionTarget) -> None:
        if not isinstance(target, PositionTarget):
            raise TypeError(f"{type(self).__name__} accepts only PositionTarget")
        qualified_contract(target.contract, "PositionTarget contract")

    @abstractmethod
    def accept(self, target: PositionTarget) -> bool:
        """Persist and begin converging to one valid target.

        Returns:
            ``True`` when the target became the latest accepted setpoint, or
            ``False`` when an older target was ignored.
        """

    @abstractmethod
    def recover(self) -> None:
        """Resume persisted work after startup."""

    def _defer(self, callback: Any, *args: Any) -> None:
        """Run convergence after current broker-event callbacks settle."""

        try:
            asyncio.get_running_loop().call_soon(callback, *args)
        except RuntimeError:
            callback(*args)

    def _bind_once(
        self,
        trade: ibi.Trade,
        *,
        filled: Any,
        cancelled: Any,
    ) -> None:
        """Bind callbacks once for the current live Trade object."""

        if self._bound_trades.get(trade.order.orderId) is trade:
            return
        self._bound_trades[trade.order.orderId] = trade
        trade.filledEvent += filled
        trade.cancelledEvent += cancelled


class SerialTargetExecutionModel(ExecutionModel):
    """Converge each Contract serially to its newest absolute target.

    Args:
        name: Stable configured model name.
        order: Model-specific IB order fields. Precedence is built-in market
            fallback, global opening defaults, then this mapping.
        future_roll_executor: Optional process-shared direct futures-roll
            executor. Omit it to use the built-in
            :class:`DirectFutureRollExecutor`.

    The model supports arbitrary quantities and same-side resizing, ignores
    optional intent, and keeps at most one active ``TARGET_ADJUSTMENT`` order
    per stable direct ``target_key``.
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        order: Mapping[str, Any] = {},
        future_roll_executor: FutureRollExecutor | None = None,
    ) -> None:
        self.order_options = {
            "orderType": "MKT",
            **self.runtime.order_defaults.open,
            **_order_options(order, "order"),
        }
        super().__init__(name=name)
        self.future_roll_executor = self.controller.future_roller.register_executor(
            FutureRollMode.DIRECT,
            future_roll_executor,
        )
        self.controller.future_roller.completedEvent += self.onFutureRollCompletedEvent

    def accept(self, target: PositionTarget) -> bool:
        """Persist a newer target and submit only when no adjustment is active."""

        if target.target_key is None:
            raise ValueError("SerialTargetExecutionModel requires target_key")
        if target.source_key is not None:
            raise ValueError("SerialTargetExecutionModel does not accept source_key")
        current = self.book.target_state(target.target_key)
        if current is not None and target.created_at < current.target_created_at:
            return False
        self._validate_target_identity(target, current)
        self.book.update_target(
            TargetState(
                target_key=target.target_key,
                execution_model_name=self.name,
                contract=target.contract,
                target_quantity=target.target_quantity,
                target_created_at=target.created_at,
            )
        )
        self._converge(target.target_key)
        return True

    def recover(self) -> None:
        """Resume every persisted target owned by this model."""

        for state in self.book.target_states(self.name):
            for info in self.book.active_orders(
                target_key=state.target_key,
                role=StandardOrderRole.TARGET_ADJUSTMENT,
                execution_model_name=self.name,
            ):
                self._bind_adjustment(info.trade, state.target_key)
            self._converge(state.target_key)

    def _bind_adjustment(self, trade: ibi.Trade, target_key: str) -> None:
        """Resume convergence after one adjustment completes."""

        self._bind_once(
            trade,
            filled=lambda _trade: self._defer(self._converge, target_key),
            cancelled=lambda _trade: self._defer(self._converge, target_key),
        )

    def _converge(self, target_key: str) -> None:
        if self.book.roll_state_for_target(target_key) is not None:
            return
        active = self.book.active_orders(
            target_key=target_key,
            role=StandardOrderRole.TARGET_ADJUSTMENT,
            execution_model_name=self.name,
        )
        if active:
            return
        state = self.book.target_state(target_key)
        if state is None or state.execution_model_name != self.name:
            return
        contract, current = self._execution_contract(state)
        delta = state.target_quantity - current
        if delta == 0:
            return
        order = ibi.Order(
            **self.order_options,
            action=action(sign(delta)),
            totalQuantity=abs(delta),
        )
        trade = self.controller.trade(
            contract,
            order,
            role=StandardOrderRole.TARGET_ADJUSTMENT,
            execution_model_name=self.name,
            target_key=target_key,
        )
        if trade is not None:
            self._bind_adjustment(trade, target_key)

    def _validate_target_identity(
        self,
        target: PositionTarget,
        current: TargetState | None,
    ) -> None:
        """Reject target keys that ambiguously identify live instruments."""

        if current is not None and not self._same_instrument(
            current.contract, target.contract
        ):
            raise ValueError(
                f"target_key {target.target_key!r} cannot change instrument"
            )
        for state in self.book.target_states():
            if state.target_key == target.target_key:
                continue
            if not self._target_is_live(state):
                continue
            if self._same_instrument(state.contract, target.contract):
                raise ValueError(
                    f"Futures series or Contract for target_key "
                    f"{target.target_key!r} is already owned by "
                    f"{state.target_key!r}"
                )

    def _target_is_live(self, state: TargetState) -> bool:
        return bool(
            state.target_quantity
            or self.book.direct_quantity(state.target_key)
            or self.book.active_orders(target_key=state.target_key)
            or self.book.roll_state_for_target(state.target_key)
        )

    def _same_instrument(self, first: ibi.Contract, second: ibi.Contract) -> bool:
        if first.conId == second.conId:
            return True
        if not isinstance(first, ibi.Future) or not isinstance(second, ibi.Future):
            return False
        return self.contract_registry.series_key(
            first
        ) == self.contract_registry.series_key(second)

    def _execution_contract(self, state: TargetState) -> tuple[ibi.Contract, float]:
        positions = self.book.direct_positions(state.target_key)
        if len(positions) > 1:
            raise RuntimeError(
                f"Direct target {state.target_key!r} has split physical holdings"
            )
        if not positions:
            return state.contract, 0.0
        held_contract, quantity = next(iter(positions.items()))
        if not self._same_instrument(held_contract, state.contract):
            raise RuntimeError(
                f"Direct target {state.target_key!r} holds a different instrument"
            )
        if (
            isinstance(held_contract, ibi.Future)
            and held_contract.conId != state.contract.conId
        ):
            series_key = self.contract_registry.series_key(held_contract)
            current = self.contract_registry.current_for_series(series_key)
            if all(held_contract.conId != contract.conId for contract in current):
                return held_contract, state.target_quantity
        return held_contract, quantity

    def onFutureRollCompletedEvent(self, state: RollState) -> None:
        """Resume convergence after a completed roll containing this model."""

        for participant in state.participants:
            if (
                participant.target_key is not None
                and participant.execution_model_name == self.name
            ):
                self._converge(participant.target_key)


__all__ = [
    "ExecutionModel",
    "SerialTargetExecutionModel",
]
