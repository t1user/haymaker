"""Bracket-specific episode execution and protective-order builders."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

import ib_insync as ibi

from ...book import FutureRollMode, OrderInfo, PositionState, RollState
from ...contract_registry import DetailsContainer
from ...misc import action, round_tick, sign
from ...validators import finite_number, non_empty_string
from ..messages import PositionIntent, PositionTarget, StandardOrderRole
from .future_roll import FutureRollExecutor
from .roll_policies import FutureRollPolicy
from .models import ExecutionModel, _order_options

log = logging.getLogger(__name__)

# ====================================================================================
# Parameters for various types of orders packaged into objects required by exec models
# ====================================================================================


class AbstractBracketLeg(ABC):
    """Build one protective-order leg after a complete entry fill.

    BracketExecutionModel calls the leg with validated PositionTarget metadata,
    the completely filled entry Trade, and Contract details. Subclasses return
    IB Order keyword arguments for a stop or take-profit order.

    Args:
        stop_multiple: Multiple applied to the configured volatility field.
        vol_field: Metadata field containing the distance basis. Defaults to
            ``"atr"``.

    Raises:
        KeyError: If the configured volatility field is absent.
    """

    vol_field: str = "atr"

    def __init__(self, stop_multiple: float, vol_field: Optional[str] = None) -> None:
        self.stop_multiple = stop_multiple
        if vol_field:
            self.vol_field = vol_field

    def __call__(
        self,
        params: dict,
        trade: ibi.Trade,
        memo: Optional[dict] = None,
        details: DetailsContainer | None = None,
    ) -> dict[str, Any]:
        # trade params are params extracted from trade
        # params are passed by the caller
        trade_params = self._extract_trade(trade)
        trade_params["min_tick"] = self.min_tick(trade_params["contract"], details)
        trade_params["vol_field_name"] = self.vol_field
        trade_params["vol_field_value"] = params[self.vol_field]
        trade_params["sl_points"] = self.stop_multiple * trade_params["vol_field_value"]
        order = self._order(trade_params)
        # any notes made on this object will be accessible for logging by caller
        # sub-classes can add keys to trade_params thus logging their parameters
        if memo is not None:
            memo.update(trade_params)
            memo.update(self.__dict__)
        return order

    def min_tick(self, contract, details: DetailsContainer | None = None):
        details = details or DetailsContainer()
        try:
            minTick = details[contract].minTick
        except KeyError:
            log.critical(
                f"No details for contract {contract}. "
                f"Will attempt to send bracket order with minTick 0.25 ",
                exc_info=True,
            )
            log.debug(f"Details: {details}")
            minTick = 0.25
        return minTick

    @staticmethod
    def _extract_trade(trade: ibi.Trade) -> dict[str, Any]:
        trade_params = {
            "contract": trade.contract,
            "action": trade.order.action,
            "amount": trade.orderStatus.filled,
            "price": trade.orderStatus.avgFillPrice,
        }
        trade_params["reverseAction"] = (
            "BUY" if trade_params["action"] == "SELL" else "SELL"
        )
        trade_params["direction"] = 1 if trade_params["reverseAction"] == "BUY" else -1
        return trade_params

    @abstractmethod
    def _order(self, params: dict[str, Any]) -> dict[str, Any]: ...

    def __repr__(self):
        attrs = ", ".join((f"{i}={j}" for i, j in self.__dict__.items()))

        return f"{self.__class__.__name__}({attrs})"


class FixedStop(AbstractBracketLeg):
    """Create a fixed-price stop from entry price and volatility distance.

    ``stop_multiple * metadata[vol_field]`` determines the distance, rounded
    to Contract minimum tick. The generated GTC stop closes the completely
    filled entry quantity and is permitted outside regular trading hours.
    """

    def _order(self, params: dict[str, Any]) -> dict[str, Any]:
        params["sl_price"] = round_tick(
            params["price"] + params["sl_points"] * params["direction"],
            params["min_tick"],
        )
        log.info(f"STOP LOSS PRICE: {params['sl_price']}")
        return {
            "orderType": "STP",
            "action": params["reverseAction"],
            "totalQuantity": params["amount"],
            "auxPrice": params["sl_price"],
            "outsideRth": True,
            "tif": "GTC",
        }


class TrailingStop(AbstractBracketLeg):
    """Create a fixed-distance trailing stop for the filled entry quantity.

    The trailing distance is ``stop_multiple * metadata[vol_field]`` rounded to
    Contract minimum tick. The generated trailing order is GTC and active
    outside regular trading hours.
    """

    def _order(self, params: dict[str, Any]) -> dict[str, Any]:
        params["distance"] = round_tick(params["sl_points"], params["min_tick"])
        log.info(f"TRAILING STOP LOSS DISTANCE: {params['distance']}")
        return {
            "orderType": "TRAIL",
            "action": params["reverseAction"],
            "totalQuantity": params["amount"],
            "auxPrice": params["distance"],
            "outsideRth": True,
            "tif": "GTC",
        }


class AdjustableTrailingFixedStop(TrailingStop):
    """Create a trailing stop that later becomes a fixed stop.

    Args:
        stop_multiple: Initial trailing-distance multiple of ``vol_field``.
        trigger_multiple: Trailing-distance multiple from entry at which IB
            changes order type.
        fixed_stop_multiple: Trailing-distance multiple used to place the
            adjusted fixed stop relative to its trigger.
        **kwargs: Optional ``vol_field`` accepted by AbstractBracketLeg.
    """

    def __init__(
        self,
        stop_multiple: float,
        trigger_multiple: float,
        fixed_stop_multiple: float,
        **kwargs,
    ) -> None:
        super().__init__(stop_multiple, **kwargs)
        self.trigger_multiple = trigger_multiple
        self.fixed_stop_multiple = fixed_stop_multiple

    def _order(self, params: dict[str, Any]) -> dict[str, Any]:
        k = super()._order(params)
        # log.debug(f"super order: {k}")
        # k is from super order, params is from Trade object
        # k['auxPrice] is: stop_multiple * vol_field (a.k.a. self.sl_points)

        # when trigger price is penetrated
        k["triggerPrice"] = (
            params["price"]
            - params["direction"] * k["auxPrice"] * self.trigger_multiple
        )
        # the parent order will be turned into s STP order
        k["adjustedOrderType"] = "STP"
        # with the given STP price
        k["adjustedStopPrice"] = (
            k["triggerPrice"]
            + params["direction"] * self.fixed_stop_multiple * k["auxPrice"]
        )
        log.debug(
            f"{params['contract'].localSymbol} TRAIL of: {k['auxPrice']} with trigger:"
            f"{k['triggerPrice']} will be fixed to {k['adjustedStopPrice']}"
        )
        return k


class AdjustableFixedTrailingStop(FixedStop):
    """Create a fixed stop that later becomes a trailing stop.

    Args:
        stop_multiple: Initial stop-distance multiple of ``vol_field``.
        trigger_multiple: Stop-distance multiple from entry at which IB
            changes order type.
        trail_multiple: Stop-distance multiple used as the adjusted trailing
            amount.
        **kwargs: Optional ``vol_field`` accepted by AbstractBracketLeg.
    """

    def __init__(
        self,
        stop_multiple: float,
        trigger_multiple: float,
        trail_multiple: float,
        **kwargs,
    ) -> None:
        super().__init__(stop_multiple, **kwargs)
        self.trigger_multiple = trigger_multiple
        self.trail_multiple = trail_multiple

    def _order(self, params: dict[str, Any]) -> dict[str, Any]:
        k = super()._order(params)

        # k is from super order, params is from Trade object
        # k['auxPrice] is: stop_multiple * vol_field (a.k.a. self.sl_points)

        # when trigger price is penetrated
        k["triggerPrice"] = round_tick(
            params["price"]
            - params["sl_points"] * self.trigger_multiple * params["direction"],
            params["min_tick"],
        )
        # the parent order will be turned int a TRAIL order
        k["adjustedOrderType"] = "TRAIL"
        # trailing by an amount (0) or a percent (100)...
        k["adjustableTrailingUnit"] = 0
        # of ...
        k["adjustedTrailingAmount"] = round_tick(
            self.trail_multiple * params["sl_points"], params["min_tick"]
        )
        # with a stop price
        k["adjustedStopPrice"] = (
            k["triggerPrice"] + k["adjustedTrailingAmount"] * params["direction"]
        )

        log.debug(
            f"{params['contract'].localSymbol} STP at {k['auxPrice']} "
            f"with trigger: {k['triggerPrice']} will TRAIL at: "
            f"{k['adjustedTrailingAmount']}"
        )
        return k


class AdjustableTrailingStop(TrailingStop):
    """Create a trailing stop whose distance widens after a trigger.

    Args:
        stop_multiple: Initial trailing-distance multiple of ``vol_field``.
        trigger_multiple: Initial-distance multiple from entry at which IB
            adjusts the order.
        adjusted_multiple: Initial-distance multiple used as the new trailing
            amount.
        **kwargs: Optional ``vol_field`` accepted by AbstractBracketLeg.
    """

    def __init__(
        self,
        stop_multiple: float,
        trigger_multiple: float,
        adjusted_multiple: float,
        **kwargs,
    ) -> None:
        super().__init__(stop_multiple, **kwargs)
        self.trigger_multiple = trigger_multiple
        self.adjusted_multiple = adjusted_multiple

    def _order(self, params: dict[str, Any]) -> dict[str, Any]:
        k = super()._order(params)

        # when trigger is penetrated
        k["triggerPrice"] = (
            params["price"]
            - params["direction"] * self.trigger_multiple * k["auxPrice"]
        )
        # sl order will remain trailing order
        k["adjustedOrderType"] = "TRAIL"
        # with a stop price of
        k["adjustedStopPrice"] = (
            k["triggerPrice"]
            + params["direction"] * k["auxPrice"] * self.adjusted_multiple
        )
        # being trailed by fixed amount
        k["adjustableTrailingUnit"] = 0
        # of:
        k["adjustedTrailingAmount"] = round_tick(
            k["auxPrice"] * self.adjusted_multiple, params["min_tick"]
        )
        return k


class TakeProfitAsStopMultiple(AbstractBracketLeg):
    """Create a take-profit limit as a multiple of stop distance.

    Args:
        stop_multiple: Multiple converting ``vol_field`` to stop distance.
        tp_multiple: Multiple converting stop distance to take-profit distance.
        **kwargs: Optional ``vol_field`` accepted by AbstractBracketLeg.

    The generated GTC limit closes the filled entry quantity and is permitted
    outside regular trading hours.
    """

    def __init__(self, stop_multiple: float, tp_multiple: float, **kwargs) -> None:
        super().__init__(stop_multiple, **kwargs)
        self.tp_multiple = tp_multiple

    def _order(self, params: dict[str, Any]) -> dict[str, Any]:
        tp_price = round_tick(
            params["price"]
            - params["sl_points"] * params["direction"] * self.tp_multiple,
            params["min_tick"],
        )
        log.info(f"TAKE PROFIT PRICE: {tp_price}")
        return {
            "orderType": "LMT",
            "action": params["reverseAction"],
            "totalQuantity": params["amount"],
            "lmtPrice": tp_price,
            "outsideRth": True,
            "tif": "GTC",
        }


class FlexibleTakeProfitAsStopMultiple(AbstractBracketLeg):
    """Create a GTC take-profit limit without forcing ``outsideRth``.

    Args:
        stop_multiple: Multiple converting ``vol_field`` to stop distance.
        tp_multiple: Multiple converting stop distance to take-profit distance.
        **kwargs: Optional ``vol_field`` accepted by AbstractBracketLeg.

    Use this variant when order defaults or model options should decide
    outside-regular-hours behavior.
    """

    def __init__(self, stop_multiple: float, tp_multiple: float, **kwargs) -> None:
        super().__init__(stop_multiple, **kwargs)
        self.tp_multiple = tp_multiple

    def _order(self, params: dict[str, Any]) -> dict[str, Any]:
        tp_price = round_tick(
            params["price"]
            - params["sl_points"] * params["direction"] * self.tp_multiple,
            params["min_tick"],
        )
        log.info(f"TAKE PROFIT PRICE: {tp_price}")
        return {
            "orderType": "LMT",
            "action": params["reverseAction"],
            "totalQuantity": params["amount"],
            "lmtPrice": tp_price,
            "tif": "GTC",
        }


class BracketExecutionModel(ExecutionModel):
    """Execute one source as independently managed bracketed episodes.

    Args:
        source_key: Required stable source identity.
        stop: Required protective stop leg.
        take_profit: Optional take-profit leg. Stop-loss protection remains
            critical even when no take-profit is configured.
        auto_roll_futures: Whether Controller-owned futures rolling should
            automatically roll this source's open position. Defaults to
            ``True``; disable only when the strategy intentionally manages its
            own one-to-one futures roll.
        future_roll_executor: Optional process-shared bracket futures-roll
            executor. Omit it to use the built-in
            :class:`BracketFutureRollExecutor`.
        roll_policy: Optional FutureRollPolicy for this source. The default
            rolls past contracts into ACTIVE; auto_roll_futures=False wins.
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
    :class:`~haymaker.components.SerialTargetExecutionModel` for that policy.

    OPEN uses the incoming target Contract. CLOSE uses the held or pending-entry
    Contract recorded for this source, regardless of the target Contract.
    REVERSE closes that episode completely before opening the incoming
    Contract as a new episode. Book persists the held and pending-target
    Contracts and their bracket inputs separately, so restarts and changes to
    ACTIVE/NEXT cannot redirect a close or lose a reversal destination.

    If entry completion was missed while offline, recovery installs the initial
    brackets from saved episode inputs and the quantity-weighted entry fills.
    Keep the model configuration recovery-compatible. Existing stops are not
    recreated, and missing optional take-profits alone do not require repair.
    Missing execution evidence or Contract ticks prevents automatic installation.
    """

    def __init__(
        self,
        source_key: str,
        *,
        stop: AbstractBracketLeg,
        take_profit: AbstractBracketLeg | None = None,
        auto_roll_futures: bool = True,
        future_roll_executor: FutureRollExecutor | None = None,
        roll_policy: FutureRollPolicy | None = None,
        name: str | None = None,
        open_order: Mapping[str, Any] = {},
        close_order: Mapping[str, Any] = {},
        stop_order: Mapping[str, Any] = {},
        take_profit_order: Mapping[str, Any] = {},
        oca_type: int | None = None,
    ) -> None:
        source_key = non_empty_string(source_key, "source_key")
        if not isinstance(stop, AbstractBracketLeg):
            raise TypeError("stop must be an AbstractBracketLeg")
        if take_profit is not None and not isinstance(take_profit, AbstractBracketLeg):
            raise TypeError("take_profit must be an AbstractBracketLeg or None")
        if not isinstance(auto_roll_futures, bool):
            raise TypeError("auto_roll_futures must be a bool")
        configured_roll_policy = self.runtime.future_roll_policies.get(source_key)
        if (
            configured_roll_policy is not None
            and configured_roll_policy != auto_roll_futures
        ):
            raise ValueError(
                f"Conflicting futures-roll policy for source {source_key!r}"
            )
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
        self.future_roll_executor = self.controller.future_roller.register_executor(
            FutureRollMode.BRACKET,
            future_roll_executor,
        )
        self.controller.future_roller.completedEvent += self.onFutureRollCompletedEvent
        if roll_policy is not None:
            self.controller.future_roller.register_policy(
                roll_policy, source_key=source_key
            )
        self.runtime.future_roll_policies[source_key] = auto_roll_futures
        self.controller.register_protection_recovery(
            source_key, self.recover_protection
        )

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
            )
        elif state.execution_model_name != self.name and (
            state.quantity or self.book.active_orders(source_key=self.source_key)
        ):
            raise ValueError(
                f"Source {self.source_key!r} is owned by "
                f"{state.execution_model_name!r}"
            )
        bracket_inputs = (
            self._bracket_inputs(target.metadata) if target.target_quantity else {}
        )
        state = replace(
            state,
            execution_model_name=self.name,
            target_contract=target.contract,
            target_quantity=target.target_quantity,
            target_created_at=target.created_at,
            target_bracket_inputs=bracket_inputs,
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
            self.recover_protection()
            self._converge()

    def recover_protection(self) -> None:
        """Restore missed initial brackets from the held episode's entry fills.

        Controller invokes this after reconciliation and before its configured
        missing-bracket policy. Existing stops, including terminal historical
        stops, are not recreated: recovering broker-maintained trailing state
        or intentionally removed protection is outside initial installation.
        Partial entries, active exits and rolls retain their existing sequencing.

        Raises:
            RuntimeError: If the episode or complete entry evidence is ambiguous.
        """
        pending = self._initial_protection()
        if pending is None:
            return
        state, orders = pending
        entries = [info for info in orders if info.role == StandardOrderRole.OPEN]
        if len(entries) != 1:
            raise RuntimeError(
                "Missing or ambiguous entry evidence for initial brackets"
            )
        # Unknown broker orders must be resolved before deciding a leg is absent.
        if any(
            trade.contract == state.contract
            and self.book.order_by_id(trade.order.orderId) is None
            and self.book.order_by_perm_id(trade.order.permId) is None
            for trade in self.ib.openTrades()
        ):
            raise RuntimeError("Unattributed broker orders prevent bracket recovery")
        details = self.contract_registry.get_details(state.contract)
        if (
            details is None
            or finite_number(details.minTick, "Contract minimum tick") <= 0
        ):
            raise RuntimeError(
                "Contract minimum tick is unavailable for bracket recovery"
            )
        self.ensure_entry_brackets(entries[0].trade)

    def _initial_protection(self) -> tuple[PositionState, tuple[OrderInfo, ...]] | None:
        """Select an unprotected episode without interfering with other stages."""
        state = self.book.position_state(self.source_key)
        if state is None or not state.quantity:
            return None
        if state.execution_model_name != self.name:
            raise RuntimeError(f"Source {self.source_key!r} belongs to another model")
        orders = self._episode_orders(state)
        if self.book.roll_state_for_source(self.source_key) is not None or any(
            info.role == StandardOrderRole.STOP_LOSS
            or (
                info.active
                and info.role in {StandardOrderRole.OPEN, StandardOrderRole.CLOSE}
            )
            for info in orders
        ):
            return None
        return state, orders

    def _episode_orders(self, state: PositionState) -> tuple[OrderInfo, ...]:
        """Restrict evidence to the current independently managed episode."""
        if state.position_id is None:
            raise RuntimeError("Held position has no episode identity")
        return tuple(
            info
            for info in self.book.orders(source_key=self.source_key)
            if info.position_id == state.position_id
        )

    @staticmethod
    def _expected_intent(effective: float, target: float) -> PositionIntent:
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

    def _bracket_inputs(self, metadata: Mapping[str, Any]) -> Mapping[str, Any]:
        """Retain only fields needed to reconstruct configured brackets."""

        required = {
            leg.vol_field for leg in (self.stop, self.take_profit) if leg is not None
        }
        missing = required - metadata.keys()
        if missing:
            names = ", ".join(sorted(missing))
            raise KeyError(f"Missing bracket input(s): {names}")
        values: dict[str, float] = {}
        for name in required:
            values[name] = finite_number(metadata[name], f"Bracket input {name!r}")
        return values

    def _converge(self) -> None:
        if self.book.roll_state_for_source(self.source_key) is not None:
            return
        state = self.book.position_state(self.source_key)
        if state is None or state.target_quantity is None:
            return
        active_adjustments = tuple(
            info
            for info in self.book.active_orders(source_key=self.source_key)
            if info.role
            in {
                StandardOrderRole.OPEN,
                StandardOrderRole.CLOSE,
            }
        )
        if active_adjustments:
            return
        current = state.quantity
        target = state.target_quantity
        if current == target:
            if (
                state.target_contract is not None
                and state.target_created_at is not None
            ):
                self._notify_target_reached(
                    PositionTarget(
                        contract=state.target_contract,
                        target_quantity=target,
                        source_key=self.source_key,
                        created_at=state.target_created_at,
                    )
                )
            return
        if current and target and sign(current) == sign(target):
            raise RuntimeError(
                "Recovered bracket target requests unsupported same-side resizing"
            )
        if current:
            self._submit_close(state)
        else:
            self._submit_open(state)

    def onFutureRollCompletedEvent(self, state: RollState) -> None:
        """Resume source convergence after its roll and protection complete."""

        if any(
            participant.source_key == self.source_key
            for participant in state.participants
        ):
            self._converge()

    def _submit_open(self, state: PositionState) -> None:
        target = state.target_quantity or 0.0
        if target == 0:
            return
        contract = state.target_contract
        if contract is None:
            raise RuntimeError("Pending entry has no target Contract in Book")
        if state.position_id is None or state.contract != state.target_contract:
            state = self.book.create_position_episode(
                self.source_key,
                self.name,
                contract,
                target_quantity=target,
                target_created_at=state.target_created_at or datetime.now(timezone.utc),
                bracket_inputs=state.target_bracket_inputs,
            )
        order = ibi.Order(
            **self.open_options,
            action=action(sign(target)),
            totalQuantity=abs(target),
        )
        trade = self.controller.trade(
            contract,
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
        if state.contract is None:
            raise RuntimeError("Cannot close an episode without its held Contract")
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
            if info.role
            in {
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
        """Run initial installation after Controller's fill accounting settles."""

        if trade.filled() < trade.order.totalQuantity:
            return
        self._defer(self._complete_entry, trade)

    def _complete_entry(self, trade: ibi.Trade) -> None:
        """Install protection before resuming a newer close or reversal target."""
        try:
            self.ensure_entry_brackets(trade)
        except Exception as exc:
            self.controller.disable_trading(
                f"Initial bracket installation failed for {self.source_key!r}: {exc}"
            )
            return
        self._defer(self._converge)

    def ensure_entry_brackets(self, entry_trade: ibi.Trade) -> None:
        """Install missing initial protection using accounted entry executions.

        Both live completion and recovery use the configured bracket legs, the
        held episode's saved inputs, and its quantity-weighted execution price.
        Repeated calls leave existing stops untouched. A surviving take-profit
        supplies the OCA group; otherwise a new group is created for both legs.
        A missing optional take-profit alone is not repaired.

        Args:
            entry_trade: The current episode's fully filled opening Trade.

        Raises:
            RuntimeError: If evidence is incomplete, inconsistent or submission
                of the required stop is suppressed.
        """
        pending = self._initial_protection()
        if pending is None:
            return
        state, orders = pending
        info = self.book.order_by_id(
            entry_trade.order.orderId
        ) or self.book.order_by_perm_id(entry_trade.order.permId)
        if info is None:
            raise RuntimeError("Entry order has no persisted evidence")
        if info.position_id != state.position_id:
            return  # A delayed callback must never protect a subsequent episode.
        trade = self._entry_for_brackets(info, state)
        dynamic, has_take_profit = self._initial_oca_options(orders, state)
        self._install_entry_brackets(trade, state, dynamic, has_take_profit)

    def _entry_for_brackets(self, info: OrderInfo, state: PositionState) -> ibi.Trade:
        """Require a complete, fill-accounted entry for this exact held episode."""
        trade = info.execution_trade()
        if (
            info.role != StandardOrderRole.OPEN
            or info.source_key != self.source_key
            or info.execution_model_name != self.name
            or state.execution_model_name != self.name
            or trade.contract != state.contract
            or trade.orderStatus.filled != trade.order.totalQuantity
            or trade.orderStatus.filled != abs(state.quantity)
            or trade.order.action != action(sign(state.quantity))
            or any(
                fill.contract != state.contract
                or fill.execution.side != ("BOT" if state.quantity > 0 else "SLD")
                for fill in trade.fills
            )
        ):
            raise RuntimeError("Incomplete or inconsistent entry evidence for brackets")
        return trade

    def _initial_oca_options(
        self, orders: tuple[OrderInfo, ...], state: PositionState
    ) -> tuple[dict[str, Any], bool]:
        """Reuse a surviving take-profit's OCA identity, or allocate one group."""
        take_profits = [
            order for order in orders if order.role == StandardOrderRole.TAKE_PROFIT
        ]
        if take_profits and (
            len(take_profits) != 1
            or not take_profits[0].active
            or take_profits[0].trade.contract != state.contract
            or take_profits[0].trade.remaining() != abs(state.quantity)
            or take_profits[0].trade.order.action != action(-sign(state.quantity))
            or not take_profits[0].trade.order.ocaGroup
            or take_profits[0].trade.order.ocaType not in {1, 2, 3}
        ):
            raise RuntimeError(
                "Existing take-profit is incompatible with initial brackets"
            )
        oca_group = (
            take_profits[0].trade.order.ocaGroup if take_profits else str(uuid4())
        )
        oca_type = (
            take_profits[0].trade.order.ocaType if take_profits else self.oca_type
        )
        return {"ocaGroup": oca_group, "ocaType": oca_type}, bool(take_profits)

    def _install_entry_brackets(
        self,
        trade: ibi.Trade,
        state: PositionState,
        dynamic: Mapping[str, Any],
        has_take_profit: bool,
    ) -> None:
        """Use the configured leg builders for both live and recovered entries."""
        params = dict(self._bracket_inputs(state.bracket_inputs))
        for leg, role, base_options in (
            (self.stop, StandardOrderRole.STOP_LOSS, self.stop_options),
            (
                self.take_profit,
                StandardOrderRole.TAKE_PROFIT,
                self.take_profit_options,
            ),
        ):
            if leg is None or (
                role == StandardOrderRole.TAKE_PROFIT and has_take_profit
            ):
                continue
            memo: dict[str, Any] = {}
            leg_options = leg(
                params,
                trade,
                memo,
                self.contract_registry.details,
            )
            order = ibi.Order(**{**base_options, **leg_options, **dynamic})
            submitted = self.controller.trade(
                trade.contract,
                order,
                role=role,
                execution_model_name=self.name,
                source_key=self.source_key,
                position_id=state.position_id,
                params=memo,
            )
            if submitted is None and role == StandardOrderRole.STOP_LOSS:
                raise RuntimeError("Critical initial stop submission was suppressed")
            if role == StandardOrderRole.STOP_LOSS and submitted is not None:
                if submitted.orderStatus.status == ibi.OrderStatus.Filled:
                    return  # Do not install another exit while this fill is accounted.
                if not submitted.isActive():
                    raise RuntimeError(
                        "Critical initial stop was rejected or cancelled"
                    )


__all__ = [
    "AbstractBracketLeg",
    "AdjustableFixedTrailingStop",
    "AdjustableTrailingFixedStop",
    "AdjustableTrailingStop",
    "BracketExecutionModel",
    "FixedStop",
    "FlexibleTakeProfitAsStopMultiple",
    "TakeProfitAsStopMultiple",
    "TrailingStop",
]
