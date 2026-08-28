"""First-match routing for preconstructed stateful ExecutionModels."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import ib_insync as ibi

from ...base import Atom
from ...book import OrderInfo, TargetState
from ...validators import qualified_contract
from ..messages import PositionTarget, StandardOrderRole
from .models import ExecutionModel

TargetPredicate = Callable[[PositionTarget], bool]
log = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class ExecutionRule:
    """Pair one target predicate with a preconfigured ExecutionModel.

    Args:
        predicate: Callable returning whether this rule matches a target.
        model: Stateful model directly invoked on the first match.
    """

    predicate: TargetPredicate
    model: ExecutionModel

    def __post_init__(self) -> None:
        if not callable(self.predicate):
            raise TypeError("predicate must be callable")
        if not isinstance(self.model, ExecutionModel):
            raise TypeError("model must be an ExecutionModel")


@dataclass(eq=False)
class ExecutionRouter(Atom):
    """Route each target to exactly one stateful execution model.

    Args:
        rules: Fixed, ordered first-match rules.
        default_model: Optional fallback. Without it unmatched targets fail
            closed.

    Model names must be unique. Current rules always select the model. An
    active direct adjustment order may continue only when those rules still
    select its persisted owner; a disagreement blocks routed execution for the
    workload rather than silently overriding the rules.
    """

    rules: Sequence[ExecutionRule]
    default_model: ExecutionModel | None = None
    models_by_name: dict[str, ExecutionModel] = field(default_factory=dict, repr=False)
    _started_generation: int = field(init=False, repr=False)
    _blocked_reason: str | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize Atom state and validate the fixed routing table."""

        Atom.__init__(self)
        self.rules = tuple(self.rules)
        if not all(isinstance(rule, ExecutionRule) for rule in self.rules):
            raise TypeError("rules must contain ExecutionRule instances")
        if self.default_model is not None and not isinstance(
            self.default_model, ExecutionModel
        ):
            raise TypeError("default_model must be an ExecutionModel or None")
        models = [rule.model for rule in self.rules]
        if self.default_model is not None:
            models.append(self.default_model)
        for model in models:
            existing = self.models_by_name.get(model.name)
            if existing is not None and existing is not model:
                raise ValueError(f"Duplicate ExecutionModel name: {model.name!r}")
            self.models_by_name[model.name] = model
        self._started_generation = -1
        self._blocked_reason = None

    def onStart(self, data: object, source: Atom | None = None) -> None:
        """Validate recovery and start models once per workload generation."""

        generation = self.runtime.workload_generation
        if generation != self._started_generation:
            self._started_generation = generation
            self._blocked_reason = self._working_order_block_reason()
            assignments: tuple[TargetState, ...] = ()
            if self._blocked_reason is None:
                try:
                    assignments = self._idle_target_assignments()
                except Exception as exc:
                    self._blocked_reason = (
                        "idle target recovery could not be routed: "
                        f"{type(exc).__name__}: {exc}"
                    )
            if self._blocked_reason is not None:
                log.critical("ExecutionRouter blocked: %s", self._blocked_reason)
            else:
                for state in assignments:
                    self.book.update_target(state)
                for model in self.models_by_name.values():
                    model.onStart(data, self)
        super().onStart(data, source)

    def onData(self, data: PositionTarget, *args: object) -> None:
        """Invoke the selected model when active ownership remains consistent."""

        if not isinstance(data, PositionTarget):
            raise TypeError("ExecutionRouter accepts only PositionTarget")
        if self._blocked_reason is not None:
            log.critical(
                "PositionTarget suppressed while ExecutionRouter is blocked: %s",
                self._blocked_reason,
            )
            return
        model = self._model_for_rules(data)
        try:
            owner = self.book.active_order_model_for_contract(
                data.contract,
                role=StandardOrderRole.TARGET_ADJUSTMENT,
            )
        except RuntimeError as exc:
            log.critical("PositionTarget suppressed: %s", exc)
            return
        if owner is not None and owner != model.name:
            log.critical(
                "PositionTarget for conId=%s selected model %r while active "
                "TARGET_ADJUSTMENT belongs to %r; target suppressed",
                data.contract.conId,
                model.name,
                owner,
            )
            return
        model.onData(data)

    def _model_for_rules(self, target: PositionTarget) -> ExecutionModel:
        model = next(
            (rule.model for rule in self.rules if rule.predicate(target)),
            self.default_model,
        )
        if model is None:
            raise LookupError(f"No ExecutionModel matched target {target!r}")
        return model

    def _working_order_block_reason(self) -> str | None:
        """Return why active direct work cannot be recovered under current rules."""

        grouped: dict[int, list[OrderInfo]] = {}
        for info in self.book.active_orders(role=StandardOrderRole.TARGET_ADJUSTMENT):
            grouped.setdefault(info.trade.contract.conId, []).append(info)

        for con_id, orders in grouped.items():
            owners = {info.execution_model_name for info in orders}
            if len(owners) != 1:
                return (
                    f"conId={con_id} has active TARGET_ADJUSTMENT orders owned "
                    f"by multiple models: {sorted(owners)}"
                )
            owner = next(iter(owners))
            state = self.book.target_state(owner, orders[0].trade.contract)
            if state is None:
                return (
                    f"active TARGET_ADJUSTMENT for conId={con_id}, model "
                    f"{owner!r} has no recoverable TargetState"
                )
            try:
                selected = self._model_for_rules(self._target_from_state(state))
            except Exception as exc:
                return (
                    f"active TARGET_ADJUSTMENT for conId={con_id}, model "
                    f"{owner!r} cannot be routed: {type(exc).__name__}: {exc}"
                )
            if selected.name != owner:
                return (
                    f"active TARGET_ADJUSTMENT for conId={con_id} belongs to "
                    f"{owner!r}, but current rules select {selected.name!r}"
                )
        return None

    def _idle_target_assignments(self) -> tuple[TargetState, ...]:
        """Build current-rule ownership updates for idle recovered targets."""

        assignments: list[TargetState] = []
        for state in self.book.latest_targets():
            if self.book.active_orders(
                contract=state.contract,
                role=StandardOrderRole.TARGET_ADJUSTMENT,
            ):
                continue
            model = self._model_for_rules(self._target_from_state(state))
            if model.name == state.execution_model_name:
                continue
            assignments.append(
                TargetState(
                    execution_model_name=model.name,
                    contract=state.contract,
                    target_quantity=state.target_quantity,
                    target_created_at=state.target_created_at,
                )
            )
        return tuple(assignments)

    @staticmethod
    def _target_from_state(state: TargetState) -> PositionTarget:
        """Reconstruct the PositionTarget fields persisted for direct recovery."""

        return PositionTarget(
            contract=state.contract,
            target_quantity=state.target_quantity,
            created_at=state.target_created_at,
        )


def contract_is(contract: ibi.Contract) -> TargetPredicate:
    """Build a predicate matching one concrete Contract by conId.

    Args:
        contract: Qualified Contract with non-zero ``conId``.

    Returns:
        PositionTarget predicate suitable for ExecutionRule.
    """

    contract = qualified_contract(contract)
    return lambda target: target.contract.conId == contract.conId


def symbol_is(symbol: str) -> TargetPredicate:
    """Build a predicate matching a target Contract root symbol.

    Args:
        symbol: Exact IB root symbol.

    Returns:
        PositionTarget predicate suitable for ExecutionRule.
    """

    return lambda target: target.contract.symbol == symbol


def security_type_is(security_type: str) -> TargetPredicate:
    """Build a predicate matching an IB security type.

    Args:
        security_type: Exact ``Contract.secType`` value.

    Returns:
        PositionTarget predicate suitable for ExecutionRule.
    """

    return lambda target: target.contract.secType == security_type


def exchange_is(exchange: str) -> TargetPredicate:
    """Build a predicate matching a Contract exchange.

    Args:
        exchange: Exact ``Contract.exchange`` value.

    Returns:
        PositionTarget predicate suitable for ExecutionRule.
    """

    return lambda target: target.contract.exchange == exchange


def where(predicate: TargetPredicate) -> TargetPredicate:
    """Validate and return an arbitrary target predicate.

    Args:
        predicate: Callable receiving PositionTarget and returning truthiness.

    Returns:
        The same callable for use in ExecutionRule.

    Note:
        Predicates used for recoverable direct execution must be deterministic
        from the persisted Contract, target quantity, and creation time.
        Signal metadata is not part of TargetState recovery.
    """

    if not callable(predicate):
        raise TypeError("predicate must be callable")
    return predicate


__all__ = [
    "ExecutionRouter",
    "ExecutionRule",
    "contract_is",
    "exchange_is",
    "security_type_is",
    "symbol_is",
    "where",
]
