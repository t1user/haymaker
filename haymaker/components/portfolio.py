"""Portfolio boundaries for contract and signal execution target allocation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Any, Protocol

import ib_insync as ibi

from ..base import Atom
from ..validators import finite_number
from .messages import PositionRequest, PositionTarget, Signal, SignalType


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
        return self.book.portfolios.load(self.portfolio_key)

    def save_state(self, state: Mapping[str, Any]) -> None:
        """Save normalized state using Book's configured ordered write policy."""
        self.book.portfolios.save(self.portfolio_key, state)


class PositionAllocator(Protocol):
    """Allocate zero or one absolute target for a signal execution request.

    Implement this protocol when :class:`FixedSizeAllocator` does not express
    the required signal execution sizing policy. :class:`PortfolioWrapper` passes
    each immutable PositionRequest to
    :meth:`~haymaker.components.PositionAllocator.target_for`; implementations return one absolute
    PositionTarget or ``None`` to suppress execution. They must preserve
    request Contract and signal identity. PortfolioWrapper supplies the
    request's mandatory intent on the emitted target.
    """

    def target_for(self, request: PositionRequest) -> PositionTarget | None:
        """Return the request's absolute signed target, or suppress it."""


Sizing = float | Mapping[str, float] | Callable[[PositionRequest], float]


class FixedSizeAllocator:
    """Allocate request direction using fixed or signal-specific size.

    Args:
        sizing: Positive size, mapping keyed by ``signal_key``, or callable
            evaluated for each request.

    Missing mapping entries raise ``KeyError``. The signed result is request
    direction multiplied by the allocated size; signal, Contract, and Signal
    metadata are preserved. PortfolioWrapper adds request intent when the
    target enters the signal execution execution path.

    Zero or negative sizes raise ValueError. To suppress a request, provide
    a custom PositionAllocator returning None rather than a zero OPEN target.
    """

    def __init__(self, sizing: Sizing = 1.0) -> None:
        self.sizing = sizing

    def target_for(self, request: PositionRequest) -> PositionTarget | None:
        """Allocate one immutable absolute target."""

        if not isinstance(request, PositionRequest):
            raise TypeError("FixedSizeAllocator accepts only PositionRequest")
        if callable(self.sizing):
            size = self.sizing(request)
        elif isinstance(self.sizing, Mapping):
            size = self.sizing[request.signal.signal_key]
        else:
            size = self.sizing
        size = finite_number(size, "allocated size")
        if size <= 0:
            raise ValueError(
                "allocated size must be positive; use a custom allocator returning None to suppress execution"
            )
        return PositionTarget(
            contract=request.signal.contract,
            target_quantity=request.target_direction * size,
            signal_key=request.signal.signal_key,
            metadata=request.signal.metadata,
        )


class PortfolioWrapper(Atom):
    """Adapt signal execution PositionRequests to a narrow allocator.

    Args:
        allocator: Object implementing :class:`PositionAllocator`.

    The wrapper accepts only requests with mandatory intent and emits zero or
    one absolute PositionTarget. It is the supported adapter for dedicated
    signal execution execution paths; account-wide recomputation uses
    :class:`Portfolio` directly.

    The wrapper preserves the request Contract, including on CLOSE. It does
    not look up holdings: BracketExecutionModel resolves the signal's held
    Contract and orchestrates closing or reversing its position.
    """

    def __init__(self, allocator: PositionAllocator) -> None:
        super().__init__()
        if not callable(getattr(allocator, "target_for", None)):
            raise TypeError("allocator must implement target_for()")
        self.allocator = allocator

    def onData(self, data: PositionRequest, *args: object) -> None:
        """Allocate and emit at most one target."""

        request = data
        if not isinstance(request, PositionRequest):
            raise TypeError("PortfolioWrapper accepts only PositionRequest")
        target = self.allocator.target_for(request)
        if target is None:
            return
        if not isinstance(target, PositionTarget):
            if isinstance(target, Iterable):
                raise TypeError("PositionAllocator must not return multiple targets")
            raise TypeError("PositionAllocator must return PositionTarget or None")
        if target.contract != request.signal.contract:
            raise ValueError("PositionAllocator changed request Contract")
        if target.signal_key != request.signal.signal_key:
            raise ValueError("PositionAllocator changed request signal_key")
        self.dataEvent.emit(replace(target, intent=request.intent))


class Portfolio(Atom, ABC):
    """Base account-wide Signal-to-target decision boundary.

    Args:
        signal_keys: Optional complete expected signal universe. ``None`` enables
            dynamic membership without a completeness policy. For one signal,
            pass ``{"alpha"}``, not the bare string ``"alpha"``.
        supported_signal_types: Signal semantics accepted by this Portfolio.

    Concrete implementations own input state, synchronization, duplicate,
    lateness, timeout, and recomputation policy. Base Portfolio immediately
    passes each valid Signal to :meth:`process` and emits every returned
    absolute target. Each target addresses its exact qualified Contract; the
    Portfolio chooses how to allocate among expiries and must aggregate all
    signal allocations for a Contract before emitting its setpoint.
    """

    def __init__(
        self,
        signal_keys: Collection[str] | None = None,
        *,
        supported_signal_types: Collection[SignalType] = (
            SignalType.STATE,
            SignalType.EVENT,
        ),
    ) -> None:
        super().__init__()
        if isinstance(signal_keys, str):
            raise TypeError(
                "signal_keys must be a collection of keys, for example {'alpha'}, not a string"
            )
        self.signal_keys = frozenset(signal_keys) if signal_keys is not None else None
        if self.signal_keys is not None and (
            not all(isinstance(key, str) and key for key in self.signal_keys)
        ):
            raise ValueError("signal_keys must contain non-empty strings")
        self.supported_signal_types = frozenset(supported_signal_types)
        if not self.supported_signal_types or not all(
            isinstance(signal_type, SignalType)
            for signal_type in self.supported_signal_types
        ):
            raise ValueError("supported_signal_types must contain SignalType values")

    @property
    def expected_signal_keys(self) -> frozenset[str] | None:
        """Return the declared signal universe, if registration is enabled."""

        return self.signal_keys

    def positions_for_contract_spec(
        self, contract: ibi.Contract
    ) -> Mapping[ibi.Contract, float]:
        """Query filled holdings across a registered declaration's members.

        Args:
            contract: Registered contract_spec or any qualified member of it.

        Returns:
            Read-only mapping of non-flat concrete Contracts to signed filled
            quantities. Working orders and desired allocations are not fills;
            query Book's active orders and targets separately. For one exact
            Contract use ``self.book.positions.filled_quantity(contract)``.

        Raises:
            KeyError: If the Contract has no registered contract_spec membership.

        Signal-level allocations in contract mode belong to the custom Portfolio:
        net broker fills cannot identify each contributing signal stream.
        """
        members = {
            member.conId for member in self.contract_registry.contracts_for(contract)
        }
        return MappingProxyType(
            {
                held: quantity
                for held, quantity in self.book.positions.by_contract().items()
                if held.conId in members
            }
        )

    def onData(self, data: Signal, *args: object) -> None:
        """Validate one Signal and emit all recomputed targets."""

        signal = data
        if not isinstance(signal, Signal):
            raise TypeError("Portfolio accepts only Signal")
        if self.signal_keys is not None and signal.signal_key not in self.signal_keys:
            raise KeyError(f"Unknown Portfolio signal_key: {signal.signal_key}")
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
            if target.signal_key is not None:
                raise ValueError("Portfolio targets must not contain signal_key")
            if target.intent is not None:
                raise ValueError("Portfolio targets must not contain TargetIntent")
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
