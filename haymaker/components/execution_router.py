"""First-match routing for preconstructed stateful ExecutionModels."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar

import ib_insync as ibi

from ..base import Atom
from ..book import TargetState
from .execution_models import ExecutionModel
from .messages import PositionTarget


TargetPredicate = Callable[[PositionTarget], bool]


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


class ExecutionRouter(Atom):
    """Route each target to exactly one stateful execution model.

    Args:
        rules: Fixed, ordered first-match rules.
        default_model: Optional fallback. Without it unmatched targets fail
            closed.

    Model names must be unique. Working-order affinity wins until those orders
    become terminal; otherwise current rules own both held quantity and target
    recovery.
    """

    input_type: ClassVar[type] = PositionTarget

    def __init__(
        self,
        rules: Sequence[ExecutionRule],
        default_model: ExecutionModel | None = None,
    ) -> None:
        super().__init__()
        self.rules = tuple(rules)
        if not all(isinstance(rule, ExecutionRule) for rule in self.rules):
            raise TypeError("rules must contain ExecutionRule instances")
        if default_model is not None and not isinstance(
            default_model, ExecutionModel
        ):
            raise TypeError("default_model must be an ExecutionModel or None")
        self.default_model = default_model
        models = [rule.model for rule in self.rules]
        if default_model is not None:
            models.append(default_model)
        self.models_by_name: dict[str, ExecutionModel] = {}
        for model in models:
            existing = self.models_by_name.get(model.name)
            if existing is not None and existing is not model:
                raise ValueError(f"Duplicate ExecutionModel name: {model.name!r}")
            self.models_by_name[model.name] = model
        self._started_generation = -1

    def validate_source(self, source: Atom) -> None:
        """Require an upstream Atom declaring PositionTarget output."""

        if getattr(source, "output_type", None) is not PositionTarget:
            raise TypeError(
                "ExecutionRouter requires a source declaring "
                "output_type=PositionTarget"
            )

    def onStart(self, data: object, source: Atom | None = None) -> None:
        """Start every configured model once per workload generation."""

        missing = (
            self.book.routing_affinity_names() - self.models_by_name.keys()
        )
        if missing:
            raise RuntimeError(
                "Persisted execution-model affinity is unavailable: "
                f"{sorted(missing)}"
            )
        generation = self.runtime.workload_generation
        if generation != self._started_generation:
            self._started_generation = generation
            self._handoff_recoverable_targets()
            for model in self.models_by_name.values():
                model.onStart(data, self)
        super().onStart(data, source)

    def onData(self, target: PositionTarget, *args: object) -> None:
        """Invoke only the model selected by affinity or first matching rule."""

        if not isinstance(target, PositionTarget):
            raise TypeError("ExecutionRouter accepts only PositionTarget")
        affinity = (
            self.book.affinity_for_source(target.source_key)
            if target.source_key is not None
            else self.book.affinity_for_contract(target.contract)
        )
        model: ExecutionModel | None
        if affinity is not None:
            model = self.models_by_name.get(affinity)
            if model is None:
                raise RuntimeError(
                    f"Persisted ExecutionModel {affinity!r} is unavailable"
                )
        else:
            model = self._model_for_rules(target)
        model.onData(target)

    def _model_for_rules(self, target: PositionTarget) -> ExecutionModel:
        model = next(
            (rule.model for rule in self.rules if rule.predicate(target)),
            self.default_model,
        )
        if model is None:
            raise LookupError(f"No ExecutionModel matched target {target!r}")
        return model

    def _handoff_recoverable_targets(self) -> None:
        """Assign idle direct targets to the models selected by current rules."""

        for state in self.book.latest_targets():
            if self.book.affinity_for_contract(state.contract) is not None:
                continue
            target = PositionTarget(
                contract=state.contract,
                target_quantity=state.target_quantity,
                created_at=state.target_created_at,
            )
            model = self._model_for_rules(target)
            if model.name == state.execution_model_name:
                continue
            self.book.update_target(
                TargetState(
                    execution_model_name=model.name,
                    contract=state.contract,
                    target_quantity=state.target_quantity,
                    target_created_at=state.target_created_at,
                )
            )


def contract_is(contract: ibi.Contract) -> TargetPredicate:
    """Build a predicate matching one concrete Contract by conId.

    Args:
        contract: Qualified Contract with non-zero ``conId``.

    Returns:
        PositionTarget predicate suitable for ExecutionRule.
    """

    if not isinstance(contract, ibi.Contract) or not contract.conId:
        raise ValueError("contract must be an IB Contract with non-zero conId")
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
