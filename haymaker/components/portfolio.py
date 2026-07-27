"""Portfolio boundaries for direct and one-to-one target allocation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import replace
import math
from typing import ClassVar, Protocol

from ..base import Atom
from .messages import PositionProposal, PositionTarget, Signal, SignalType


class PositionAllocator(Protocol):
    """Allocate zero or one absolute target for a one-to-one proposal.

    Implement this protocol when :class:`FixedSizeAllocator` does not express
    the required one-to-one sizing policy. :class:`PortfolioWrapper` passes
    each immutable PositionProposal to
    :meth:`PositionAllocator.target_for`; implementations return one absolute
    PositionTarget or ``None`` to suppress execution. They must preserve
    proposal Contract, source identity, and intent.
    """

    def target_for(
        self, proposal: PositionProposal
    ) -> PositionTarget | None:
        """Return the proposal's absolute signed target, or suppress it."""


Sizing = (
    float
    | Mapping[str, float]
    | Callable[[PositionProposal], float]
)


class FixedSizeAllocator:
    """Allocate proposal direction using fixed or source-specific size.

    Args:
        sizing: Positive size, mapping keyed by ``source_key``, or callable
            evaluated for each proposal.

    Missing mapping entries raise ``KeyError``. The signed result is proposal
    direction multiplied by the allocated size; source, Contract, intent, and
    Signal metadata are preserved.
    """

    def __init__(self, sizing: Sizing = 1.0) -> None:
        self.sizing = sizing

    def target_for(
        self, proposal: PositionProposal
    ) -> PositionTarget | None:
        """Allocate one immutable absolute target."""

        if not isinstance(proposal, PositionProposal):
            raise TypeError("FixedSizeAllocator accepts only PositionProposal")
        if callable(self.sizing):
            size = self.sizing(proposal)
        elif isinstance(self.sizing, Mapping):
            size = self.sizing[proposal.signal.source_key]
        else:
            size = self.sizing
        if isinstance(size, bool) or not isinstance(size, (int, float)):
            raise TypeError("allocated size must be a real number")
        if not math.isfinite(size):
            raise ValueError("allocated size must be finite")
        if size < 0:
            raise ValueError("allocated size must not be negative")
        return PositionTarget(
            contract=proposal.signal.contract,
            target_quantity=proposal.target_direction * float(size),
            source_key=proposal.signal.source_key,
            intent=proposal.intent,
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
    """

    input_type: ClassVar[type] = PositionProposal
    output_type: ClassVar[type] = PositionTarget

    def __init__(self, allocator: PositionAllocator) -> None:
        super().__init__()
        if not callable(getattr(allocator, "target_for", None)):
            raise TypeError("allocator must implement target_for()")
        self.allocator = allocator

    def validate_source(self, source: Atom) -> None:
        """Require an upstream Atom declaring PositionProposal output."""

        if getattr(source, "output_type", None) is not PositionProposal:
            raise TypeError(
                "PortfolioWrapper requires a source declaring "
                "output_type=PositionProposal"
            )

    def onData(self, proposal: PositionProposal, *args: object) -> None:
        """Allocate and emit at most one target."""

        if not isinstance(proposal, PositionProposal):
            raise TypeError("PortfolioWrapper accepts only PositionProposal")
        if proposal.intent is None:
            raise ValueError("PositionProposal intent is mandatory")
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
    absolute target.
    """

    input_type: ClassVar[type] = Signal
    output_type: ClassVar[type] = PositionTarget

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
            raise ValueError(
                "supported_signal_types must contain SignalType values"
            )

    @property
    def expected_sources(self) -> frozenset[str] | None:
        """Return the declared source universe, if registration is enabled."""

        return self.sources

    def validate_source(self, source: Atom) -> None:
        """Require an upstream Atom declaring Signal output."""

        if getattr(source, "output_type", None) is not Signal:
            raise TypeError(
                f"{type(self).__name__} requires a source declaring "
                "output_type=Signal"
            )

    def onData(self, signal: Signal, *args: object) -> None:
        """Validate one Signal and emit all recomputed targets."""

        if not isinstance(signal, Signal):
            raise TypeError("Portfolio accepts only Signal")
        if (
            self.sources is not None
            and signal.source_key not in self.sources
        ):
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
            self.dataEvent.emit(target)

    @abstractmethod
    def process(self, signal: Signal) -> Iterable[PositionTarget]:
        """Update Portfolio policy and return zero or more absolute targets."""


__all__ = [
    "FixedSizeAllocator",
    "Portfolio",
    "PortfolioWrapper",
    "PositionAllocator",
]
