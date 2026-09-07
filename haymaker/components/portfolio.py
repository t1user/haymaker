"""Portfolio boundaries for direct and one-to-one target allocation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Any, Protocol

import ib_insync as ibi

from ..base import Atom
from ..validators import finite_number
from .messages import PositionProposal, PositionTarget, Signal, SignalType


class PortfolioStateMixin(Atom):
    """Opt into explicit persistence of a custom Portfolio's normalized state.

    Declare a stable ``portfolio_key`` on your subclass and combine this mixin
    with :class:`Portfolio`. Call :meth:`load_state` before consuming inputs and
    :meth:`save_state` when your allocation state changes. Neither loading nor
    saving is automatic. The default uses Book's ordered state persistence;
    override these two methods for an independent backend and manage that
    backend's lifecycle yourself. There is no cross-backend transaction with
    execution targets.

    Example::

        class MyPortfolio(PortfolioStateMixin, Portfolio):
            portfolio_key = "account_allocation"

            def onStart(self, data, source=None):
                self.allocations = dict(self.load_state() or {})
                super().onStart(data, source)

    Implement ``process`` and save a normalized recovery mapping appropriate
    to your policy, not raw received Signals.
    """

    portfolio_key: str

    def load_state(self) -> Mapping[str, Any] | None:
        """Load the saved mapping, or return None for a first run."""
        return self.book.load_portfolio_state(self.portfolio_key)

    def save_state(self, state: Mapping[str, Any]) -> None:
        """Save normalized state using Book's configured ordered write policy."""
        self.book.save_portfolio_state(self.portfolio_key, state)


class PositionAllocator(Protocol):
    """Allocate zero or one absolute target for a one-to-one proposal.

    Implement this protocol when :class:`FixedSizeAllocator` does not express
    the required one-to-one sizing policy. :class:`PortfolioWrapper` passes
    each immutable PositionProposal to
    :meth:`PositionAllocator.target_for`; implementations return one absolute
    PositionTarget or ``None`` to suppress execution. They must preserve
    proposal Contract and source identity. PortfolioWrapper supplies the
    proposal's mandatory intent on the emitted target.
    """

    def target_for(self, proposal: PositionProposal) -> PositionTarget | None:
        """Return the proposal's absolute signed target, or suppress it."""


Sizing = float | Mapping[str, float] | Callable[[PositionProposal], float]


class FixedSizeAllocator:
    """Allocate proposal direction using fixed or source-specific size.

    Args:
        sizing: Positive size, mapping keyed by ``source_key``, or callable
            evaluated for each proposal.

    Missing mapping entries raise ``KeyError``. The signed result is proposal
    direction multiplied by the allocated size; source, Contract, and Signal
    metadata are preserved. PortfolioWrapper adds proposal intent when the
    target enters the one-to-one execution path.
    """

    def __init__(self, sizing: Sizing = 1.0) -> None:
        self.sizing = sizing

    def target_for(self, proposal: PositionProposal) -> PositionTarget | None:
        """Allocate one immutable absolute target."""

        if not isinstance(proposal, PositionProposal):
            raise TypeError("FixedSizeAllocator accepts only PositionProposal")
        if callable(self.sizing):
            size = self.sizing(proposal)
        elif isinstance(self.sizing, Mapping):
            size = self.sizing[proposal.signal.source_key]
        else:
            size = self.sizing
        size = finite_number(size, "allocated size")
        if size < 0:
            raise ValueError("allocated size must not be negative")
        return PositionTarget(
            contract=proposal.signal.contract,
            target_quantity=proposal.target_direction * size,
            source_key=proposal.signal.source_key,
            metadata=proposal.signal.metadata,
        )


class PortfolioWrapper(Atom):
    """Adapt one-to-one PositionProposals to a narrow allocator.

    Args:
        allocator: Object implementing :class:`PositionAllocator`.

    The wrapper accepts only proposals with mandatory intent and emits zero or
    one absolute PositionTarget. It is the supported adapter for dedicated
    one-to-one execution paths; account-wide recomputation uses
    :class:`Portfolio` directly.

    The wrapper preserves the proposal Contract, including on CLOSE. It does
    not look up holdings: BracketExecutionModel resolves the source's held
    Contract and orchestrates closing or reversing its episode.
    """

    def __init__(self, allocator: PositionAllocator) -> None:
        super().__init__()
        if not callable(getattr(allocator, "target_for", None)):
            raise TypeError("allocator must implement target_for()")
        self.allocator = allocator

    def onData(self, proposal: PositionProposal, *args: object) -> None:
        """Allocate and emit at most one target."""

        if not isinstance(proposal, PositionProposal):
            raise TypeError("PortfolioWrapper accepts only PositionProposal")
        target = self.allocator.target_for(proposal)
        if target is None:
            return
        if not isinstance(target, PositionTarget):
            if isinstance(target, Iterable):
                raise TypeError("PositionAllocator must not return multiple targets")
            raise TypeError("PositionAllocator must return PositionTarget or None")
        if target.contract != proposal.signal.contract:
            raise ValueError("PositionAllocator changed proposal Contract")
        if target.source_key != proposal.signal.source_key:
            raise ValueError("PositionAllocator changed proposal source_key")
        self.dataEvent.emit(replace(target, intent=proposal.intent))


class Portfolio(Atom, ABC):
    """Base account-wide Signal-to-target decision boundary.

    Args:
        sources: Optional complete expected source universe. ``None`` enables
            dynamic membership without a completeness policy.
        supported_signal_types: Signal semantics accepted by this Portfolio.

    Concrete implementations own input state, synchronization, duplicate,
    lateness, timeout, and recomputation policy. Base Portfolio immediately
    passes each valid Signal to :meth:`process` and emits every returned
    absolute target. Each target addresses its exact qualified Contract; the
    Portfolio chooses how to allocate among expiries and must aggregate all
    source allocations for a Contract before emitting its setpoint.
    """

    def __init__(
        self,
        sources: Collection[str] | None = None,
        *,
        supported_signal_types: Collection[SignalType] = (
            SignalType.STATE,
            SignalType.EVENT,
        ),
    ) -> None:
        super().__init__()
        self.sources = frozenset(sources) if sources is not None else None
        if self.sources is not None and (
            not all(isinstance(source, str) and source for source in self.sources)
        ):
            raise ValueError("sources must contain non-empty strings")
        self.supported_signal_types = frozenset(supported_signal_types)
        if not self.supported_signal_types or not all(
            isinstance(signal_type, SignalType)
            for signal_type in self.supported_signal_types
        ):
            raise ValueError("supported_signal_types must contain SignalType values")

    @property
    def expected_sources(self) -> frozenset[str] | None:
        """Return the declared source universe, if registration is enabled."""

        return self.sources

    def positions_for_blueprint(
        self, contract: ibi.Contract
    ) -> Mapping[ibi.Contract, float]:
        """Query filled holdings across a registered declaration's members.

        Args:
            contract: Registered blueprint or any qualified member of it.

        Returns:
            Read-only mapping of non-flat concrete Contracts to signed filled
            quantities. Working orders and desired allocations are not fills;
            query Book's active orders and targets separately. For one exact
            Contract use ``self.book.aggregate_quantity(contract)``.

        Raises:
            KeyError: If the Contract has no registered blueprint membership.

        Source-level allocations in direct mode belong to the custom Portfolio:
        net broker fills cannot identify each contributing Signal source.
        """
        members = {
            member.conId for member in self.contract_registry.contracts_for(contract)
        }
        return MappingProxyType(
            {
                held: quantity
                for held, quantity in self.book.logical_positions().items()
                if held.conId in members
            }
        )

    def onData(self, signal: Signal, *args: object) -> None:
        """Validate one Signal and emit all recomputed targets."""

        if not isinstance(signal, Signal):
            raise TypeError("Portfolio accepts only Signal")
        if self.sources is not None and signal.source_key not in self.sources:
            raise KeyError(f"Unknown Portfolio source_key: {signal.source_key}")
        if signal.signal_type not in self.supported_signal_types:
            raise ValueError(
                f"{type(self).__name__} does not support "
                f"{signal.signal_type.value} Signals"
            )
        targets = self.process(signal)
        if targets is None:
            raise TypeError("Portfolio.process() must return an iterable")
        for target in targets:
            if not isinstance(target, PositionTarget):
                raise TypeError(
                    "Portfolio.process() must yield PositionTarget instances"
                )
            if target.source_key is not None:
                raise ValueError("Portfolio targets must not contain source_key")
            if target.intent is not None:
                raise ValueError("Portfolio targets must not contain PositionIntent")
            self.dataEvent.emit(target)

    @abstractmethod
    def process(self, signal: Signal) -> Iterable[PositionTarget]:
        """Update Portfolio policy and return zero or more absolute targets."""


__all__ = [
    "FixedSizeAllocator",
    "Portfolio",
    "PortfolioWrapper",
    "PortfolioStateMixin",
    "PositionAllocator",
]
