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
    per concrete Contract. Different expiries have independent setpoints; a
    Portfolio, not this model, chooses how to allocate among them.
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
        """Persist the newest concrete target without changing its Contract."""
        if target.source_key is not None:
            raise ValueError("SerialTargetExecutionModel does not accept source_key")
        current = self.book.target_state(target.contract)
        if current is not None and target.created_at < current.target_created_at:
            return False
        owner = self.book.active_order_model_for_contract(
            target.contract, role=StandardOrderRole.TARGET_ADJUSTMENT
        )
        if owner is not None and owner != self.name:
            raise ValueError(
                f"Active adjustment for conId={target.contract.conId} belongs to {owner!r}"
            )
        self.book.update_target(
            TargetState(
                execution_model_name=self.name,
                contract=target.contract,
                target_quantity=target.target_quantity,
                target_created_at=target.created_at,
            )
        )
        self._converge(target.contract)
        return True

    def recover(self) -> None:
        """Resume persisted concrete targets, rebinding active broker work."""
        for state in self.book.target_states(self.name):
            for info in self.book.active_orders(
                contract=state.contract, role=StandardOrderRole.TARGET_ADJUSTMENT
            ):
                self._bind_adjustment(info.trade)
            self._converge(state.contract)

    def _bind_adjustment(self, trade: ibi.Trade) -> None:
        """Resume convergence after the broker's current adjustment terminates."""
        self._bind_once(
            trade,
            filled=lambda _trade: self._defer(self._converge, trade.contract),
            cancelled=lambda _trade: self._defer(self._converge, trade.contract),
        )

    def _converge(self, contract: ibi.Contract) -> None:
        if self.book.roll_state_for_contract(contract) is not None:
            return
        if self.book.active_orders(
            contract=contract, role=StandardOrderRole.TARGET_ADJUSTMENT
        ):
            return
        state = self.book.target_state(contract)
        if state is None or state.execution_model_name != self.name:
            return
        delta = state.target_quantity - self.book.direct_quantity(contract)
        if not delta:
            return
        order = ibi.Order(
            **self.order_options, action=action(sign(delta)), totalQuantity=abs(delta)
        )
        trade = self.controller.trade(
            contract,
            order,
            role=StandardOrderRole.TARGET_ADJUSTMENT,
            execution_model_name=self.name,
        )
        if trade is not None:
            self._bind_adjustment(trade)

    def onFutureRollCompletedEvent(self, state: RollState) -> None:
        """Resume each concrete endpoint after roll accounting is complete."""
        for contract in (state.old_contract, state.new_contract):
            self._converge(contract)


__all__ = [
    "ExecutionModel",
    "SerialTargetExecutionModel",
]
