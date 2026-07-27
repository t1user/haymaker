"""Stateful absolute-target execution models."""

from __future__ import annotations

import asyncio
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
from numbers import Real
from typing import Any, ClassVar
from uuid import uuid4

import ib_insync as ibi

from ..base import Atom
from ..book import PositionState, TargetState
from ..misc import action, sign
from ..validators import order_field_validator
from .bracket_legs import AbstractBracketLeg
from .messages import (
    PositionIntent,
    PositionTarget,
    StandardOrderRole,
)


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

    input_type: ClassVar[type] = PositionTarget
    output_type: ClassVar[type] = PositionTarget

    def __init__(self, *, name: str | None = None) -> None:
        super().__init__()
        self.name = name or type(self).__name__
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("ExecutionModel name must be a non-empty string")
        runtime = getattr(self, "runtime", None)
        if runtime is None or getattr(runtime, "controller", None) is None:
            raise RuntimeError(
                f"{type(self).__name__} requires a ready RuntimeContext"
            )
        self.controller = runtime.controller
        self._started_generation = -1
        self._bound_trades: dict[int, ibi.Trade] = {}
        self.connect(self.controller)

    def __str__(self) -> str:
        """Return stable model name and implementation class."""

        return f"{self.name}[{type(self).__name__}]"

    def validate_source(self, source: Atom) -> None:
        """Require an upstream Atom declaring PositionTarget output."""

        if getattr(source, "output_type", None) is not PositionTarget:
            raise TypeError(
                f"{type(self).__name__} requires a source declaring "
                "output_type=PositionTarget"
            )

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
        if not target.contract.conId:
            raise ValueError("PositionTarget contract must have a non-zero conId")

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

    The model supports arbitrary quantities and same-side resizing, ignores
    optional intent, and keeps at most one active ``TARGET_ADJUSTMENT`` order
    per concrete Contract.
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        order: Mapping[str, Any] = {},
    ) -> None:
        self.order_options = {
            "orderType": "MKT",
            **self.runtime.order_defaults.open,
            **_order_options(order, "order"),
        }
        super().__init__(name=name)

    def accept(self, target: PositionTarget) -> bool:
        """Persist a newer target and submit only when no adjustment is active."""

        current = self.book.target_state(self.name, target.contract)
        if current is not None and target.created_at < current.target_created_at:
            return False
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
        """Resume every persisted target owned by this model."""

        states = self.book.latest_target(self.name)
        if isinstance(states, tuple):
            for state in states:
                for info in self.book.active_orders(
                    contract=state.contract,
                    role=StandardOrderRole.TARGET_ADJUSTMENT,
                    execution_model_name=self.name,
                ):
                    self._bind_adjustment(info.trade, state.contract)
                self._converge(state.contract)

    def _bind_adjustment(
        self, trade: ibi.Trade, contract: ibi.Contract
    ) -> None:
        """Resume convergence after one adjustment completes."""

        self._bind_once(
            trade,
            filled=lambda _trade: self._defer(self._converge, contract),
            cancelled=lambda _trade: self._defer(self._converge, contract),
        )

    def _converge(self, contract: ibi.Contract) -> None:
        active = self.book.active_orders(
            contract=contract,
            role=StandardOrderRole.TARGET_ADJUSTMENT,
            execution_model_name=self.name,
        )
        if active:
            return
        state = self.book.target_state(self.name, contract)
        if state is None:
            return
        current = self.book.aggregate_quantity(contract)
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
        )
        if trade is not None:
            self._bind_adjustment(trade, contract)


class BracketExecutionModel(ExecutionModel):
    """Execute one source as independently managed bracketed episodes.

    Args:
        source_key: Required stable source identity.
        stop: Required protective stop leg.
        take_profit: Optional take-profit leg. Stop-loss protection remains
            critical even when no take-profit is configured.
        name: Stable configured model name.
        open_order: Model-specific entry Order fields.
        close_order: Model-specific close Order fields.
        stop_order: Model-specific stop Order fields.
        take_profit_order: Model-specific take-profit Order fields.
        oca_type: IB OCA behavior; defaults to global order configuration.

    Newly received targets require an initially consistent PositionIntent.
    After acceptance, numeric target is authoritative and recovery derives work
    from Book instead of replaying intent. Regular closes join the protective
    orders' OCA group so the filled close cancels remaining brackets. Non-zero
    same-side resizing is not supported; use
    :class:`SerialTargetExecutionModel` for that policy.
    """

    def __init__(
        self,
        source_key: str,
        *,
        stop: AbstractBracketLeg,
        take_profit: AbstractBracketLeg | None = None,
        name: str | None = None,
        open_order: Mapping[str, Any] = {},
        close_order: Mapping[str, Any] = {},
        stop_order: Mapping[str, Any] = {},
        take_profit_order: Mapping[str, Any] = {},
        oca_type: int | None = None,
    ) -> None:
        if not source_key:
            raise ValueError("source_key must not be empty")
        if not isinstance(stop, AbstractBracketLeg):
            raise TypeError("stop must be an AbstractBracketLeg")
        if take_profit is not None and not isinstance(
            take_profit, AbstractBracketLeg
        ):
            raise TypeError("take_profit must be an AbstractBracketLeg or None")
        self.source_key = source_key
        self.stop = stop
        self.take_profit = take_profit
        defaults = self.runtime.order_defaults
        self.open_options = {
            "orderType": "MKT",
            **defaults.open,
            **_order_options(open_order, "open_order"),
        }
        self.close_options = {
            "orderType": "MKT",
            **defaults.close,
            **_order_options(close_order, "close_order"),
        }
        self.stop_options = {
            "orderType": "STP",
            **defaults.stop,
            **_order_options(stop_order, "stop_order"),
        }
        self.take_profit_options = {
            "orderType": "LMT",
            **defaults.take_profit,
            **_order_options(take_profit_order, "take_profit_order"),
        }
        self.oca_type = defaults.oca_type if oca_type is None else oca_type
        if self.oca_type not in {1, 2, 3}:
            raise ValueError("oca_type must be 1, 2, or 3")
        super().__init__(name=name)

    def accept(self, target: PositionTarget) -> bool:
        """Validate intent, persist target state, and converge the episode."""

        if target.source_key != self.source_key:
            raise ValueError(
                f"Expected source_key {self.source_key!r}, "
                f"got {target.source_key!r}"
            )
        if target.intent is None:
            raise ValueError("BracketExecutionModel requires PositionIntent")
        effective = self.book.effective_quantity(self.source_key)
        expected = self._expected_intent(effective, target.target_quantity)
        if target.intent is not expected:
            raise ValueError(
                f"Intent {target.intent.value} is inconsistent with effective "
                f"quantity {effective} and target {target.target_quantity}; "
                f"expected {expected.value}"
            )
        state = self.book.position_state(self.source_key)
        if (
            state is not None
            and state.target_created_at is not None
            and target.created_at < state.target_created_at
        ):
            return False
        if state is None:
            state = PositionState(
                source_key=self.source_key,
                execution_model_name=self.name,
                contract=target.contract,
            )
        elif state.execution_model_name != self.name and (
            state.quantity or self.book.active_orders(source_key=self.source_key)
        ):
            raise ValueError(
                f"Source {self.source_key!r} is owned by "
                f"{state.execution_model_name!r}"
            )
        bracket_inputs = (
            self._bracket_inputs(target.metadata)
            if target.target_quantity
            else state.bracket_inputs if state is not None else {}
        )
        state = replace(
            state,
            execution_model_name=self.name,
            contract=target.contract,
            target_quantity=target.target_quantity,
            target_created_at=target.created_at,
            bracket_inputs=bracket_inputs,
            updated_at=datetime.now(timezone.utc),
        )
        self.book.update_position(state)
        self._converge()
        return True

    def recover(self) -> None:
        """Resume the persisted source target without replaying old intent."""

        state = self.book.position_state(self.source_key)
        if state is not None:
            if state.execution_model_name != self.name:
                raise RuntimeError(
                    f"Persisted source {self.source_key!r} requires missing "
                    f"model {state.execution_model_name!r}"
                )
            for info in self.book.active_orders(source_key=self.source_key):
                if info.execution_model_name != self.name:
                    continue
                if info.role == StandardOrderRole.OPEN:
                    self._bind_entry(info.trade)
                elif info.role == StandardOrderRole.CLOSE:
                    self._bind_convergence(info.trade)
            self._converge()

    @staticmethod
    def _expected_intent(
        effective: float, target: float
    ) -> PositionIntent:
        if effective == 0:
            return PositionIntent.OPEN if target != 0 else PositionIntent.CLOSE
        if target == 0:
            return PositionIntent.CLOSE
        if sign(effective) != sign(target):
            return PositionIntent.REVERSE
        if effective == target:
            return PositionIntent.OPEN
        raise ValueError(
            "BracketExecutionModel does not support non-zero same-side resizing"
        )

    def _bracket_inputs(
        self, metadata: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Retain only fields needed to reconstruct configured brackets."""

        required = {
            leg.vol_field
            for leg in (self.stop, self.take_profit)
            if leg is not None
        }
        missing = required - metadata.keys()
        if missing:
            names = ", ".join(sorted(missing))
            raise KeyError(f"Missing bracket input(s): {names}")
        values: dict[str, float] = {}
        for name in required:
            value = metadata[name]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"Bracket input {name!r} must be a real number")
            normalized = float(value)
            if not math.isfinite(normalized):
                raise ValueError(f"Bracket input {name!r} must be finite")
            values[name] = normalized
        return values

    def _converge(self) -> None:
        state = self.book.position_state(self.source_key)
        if state is None or state.target_quantity is None or state.contract is None:
            return
        active_adjustments = tuple(
            info
            for info in self.book.active_orders(source_key=self.source_key)
            if info.role in {
                StandardOrderRole.OPEN,
                StandardOrderRole.CLOSE,
            }
        )
        if active_adjustments:
            return
        current = state.quantity
        target = state.target_quantity
        if current == target:
            return
        if current and target and sign(current) == sign(target):
            raise RuntimeError(
                "Recovered bracket target requests unsupported same-side resizing"
            )
        if current:
            self._submit_close(state)
        else:
            self._submit_open(state)

    def _submit_open(self, state: PositionState) -> None:
        target = state.target_quantity or 0.0
        if target == 0 or state.contract is None:
            return
        if state.position_id is None:
            state = self.book.create_position_episode(
                self.source_key,
                self.name,
                state.contract,
                target_quantity=target,
                target_created_at=state.target_created_at
                or datetime.now(timezone.utc),
                bracket_inputs=state.bracket_inputs,
            )
        order = ibi.Order(
            **self.open_options,
            action=action(sign(target)),
            totalQuantity=abs(target),
        )
        trade = self.controller.trade(
            state.contract,
            order,
            role=StandardOrderRole.OPEN,
            execution_model_name=self.name,
            source_key=self.source_key,
            position_id=state.position_id,
            params=state.bracket_inputs,
        )
        if trade is not None:
            self._bind_entry(trade)

    def _submit_close(self, state: PositionState) -> None:
        assert state.contract is not None
        options = {
            **self.close_options,
            **self._active_bracket_oca_options(),
        }
        order = ibi.Order(
            **options,
            action=action(-sign(state.quantity)),
            totalQuantity=abs(state.quantity),
        )
        trade = self.controller.trade(
            state.contract,
            order,
            role=StandardOrderRole.CLOSE,
            execution_model_name=self.name,
            source_key=self.source_key,
            position_id=state.position_id,
            params=state.bracket_inputs,
        )
        if trade is not None:
            self._bind_convergence(trade)

    def _bind_entry(self, trade: ibi.Trade) -> None:
        """Restore entry-fill bracket creation and cancellation handling."""

        self._bind_once(
            trade,
            filled=self._on_entry_filled,
            cancelled=lambda _trade: self._defer(self._converge),
        )

    def _bind_convergence(self, trade: ibi.Trade) -> None:
        """Restore convergence after a close completes or is cancelled."""

        self._bind_once(
            trade,
            filled=lambda _trade: self._defer(self._converge),
            cancelled=lambda _trade: self._defer(self._converge),
        )

    def _active_bracket_oca_options(self) -> dict[str, Any]:
        """Return the shared OCA identity of this episode's active brackets."""

        groups = {
            info.trade.order.ocaGroup
            for info in self.book.active_orders(source_key=self.source_key)
            if info.role in {
                StandardOrderRole.STOP_LOSS,
                StandardOrderRole.TAKE_PROFIT,
            }
            and info.trade.order.ocaGroup
        }
        if not groups:
            return {}
        if len(groups) != 1:
            raise RuntimeError(
                f"Active brackets for {self.source_key!r} have different OCA groups"
            )
        return {"ocaGroup": groups.pop(), "ocaType": self.oca_type}

    def _on_entry_filled(self, trade: ibi.Trade) -> None:
        """Attach protection only after a complete entry fill."""

        if trade.filled() < trade.order.totalQuantity:
            return
        state = self.book.position_state(self.source_key)
        if state is None or state.position_id is None:
            return
        oca_group = str(uuid4())
        dynamic = {"ocaGroup": oca_group, "ocaType": self.oca_type}
        params = dict(state.bracket_inputs)
        for leg, role, base_options in (
            (self.stop, StandardOrderRole.STOP_LOSS, self.stop_options),
            (
                self.take_profit,
                StandardOrderRole.TAKE_PROFIT,
                self.take_profit_options,
            ),
        ):
            if leg is None:
                continue
            memo: dict[str, Any] = {}
            leg_options = leg(
                params,
                trade,
                memo,
                self.contract_registry.details,
            )
            order = ibi.Order(**{**base_options, **leg_options, **dynamic})
            self.controller.trade(
                trade.contract,
                order,
                role=role,
                execution_model_name=self.name,
                source_key=self.source_key,
                position_id=state.position_id,
                params=memo,
            )
        self._defer(self._converge)


__all__ = [
    "BracketExecutionModel",
    "ExecutionModel",
    "SerialTargetExecutionModel",
]
