from __future__ import annotations

import logging
from copy import deepcopy
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    ClassVar,
    NamedTuple,
    Self,
    Sequence,
    overload,
)

import ib_insync as ibi

from .contract_registry import ContractRegistry
from .details_processor import Details
from .enums import ActiveNext

if TYPE_CHECKING:
    from .book import Book
    from .contract_selector import AbstractBaseContractSelector
    from .runtime import RuntimeContext

log = logging.getLogger(__name__)


class MissingContractError(Exception):
    """Indicate that a registered contract has no current resolution."""


class ContractManagingDescriptor:
    # DON'T CHANGE THIS TO PROPERTY or it will screw up dataclasses
    # that inherit from Atom
    """Resolve an Atom's assigned contract through the runtime registry.

    Assignment registers an unqualified contract blueprint. Instance access
    returns the registry's currently selected concrete contract for the
    configured ACTIVE or NEXT futures role. PREVIOUS remains available for
    direct selector and registry queries, but is not an operational Atom role.
    """

    def __set_name__(self, obj: type[Atom], name: str) -> None:
        self.name = f"_{name}_blueprint"

    def __set__(self, obj: Atom, value: ibi.Contract) -> None:
        if not isinstance(value, ibi.Contract):
            raise TypeError(f"attr contract must be ibi.Contract, not: {type(value)}")
        obj.__dict__[self.name] = deepcopy(value)
        obj.contract_registry.register_blueprint(value)

    @overload
    def __get__(self, obj: None, owner: type[Atom] | None = None) -> Self: ...

    @overload
    def __get__(
        self, obj: Atom, owner: type[Atom] | None = None
    ) -> ibi.Contract | None: ...

    def __get__(
        self, obj: Atom | None, owner: type[Atom] | None = None
    ) -> Self | ibi.Contract | None:
        if obj is None:
            return self
        contract_blueprint = obj.__dict__.get(self.name)
        if contract_blueprint is None:
            return None
        which_contract = obj.which_contract
        if which_contract not in (ActiveNext.ACTIVE, ActiveNext.NEXT):
            raise ValueError(
                "Atom.which_contract supports only ACTIVE or NEXT, "
                f"not {which_contract}"
            )
        try:
            return obj.contract_registry.get_contract(
                contract_blueprint, which_contract
            )
        except KeyError:
            raise MissingContractError(
                f"Unknown contract: {contract_blueprint} on "
                f"{obj.__class__.__name__}"
            )


class ContractRollData(NamedTuple):
    old_contract: ibi.Contract
    new_contract: ibi.Contract


class Atom:
    """Compose an event-driven processing step around arbitrary Python messages.

    ``Atom`` is Haymaker's general composition primitive. Subclasses implement
    one focused operation and are connected to form a graph: startup and data
    travel downstream, while feedback travels toward the preceding Atom.
    ``Atom`` does not impose a trading-message schema, create strategy state, or
    emit output automatically.

    Override :meth:`onData` to process one input and explicitly emit any output
    with ``self.dataEvent.emit(...)``. Override :meth:`onStart` for per-workload
    initialization and normally call ``super().onStart(data, source)`` last to
    continue startup. Override :meth:`onFeedback` only when the default reverse
    forwarding is insufficient.

    Connections use event references rather than copying messages. Every branch
    of a fan-out therefore receives the same object; a branch that mutates its
    input must copy it first. Use :meth:`connect` for fan-out and :meth:`pipe` or
    :class:`Pipe` for a linear chain.

    Haymaker installs runtime services before importing the live strategy
    module, so strategy code can use the runtime-backed properties directly.
    Runtime-context installation itself is framework plumbing rather than a
    user extension point.

    Attributes:
        events (Sequence[str]): Names of the standard lifecycle events:
            ``startEvent``, ``dataEvent``, and ``feedbackEvent``.
        startEvent (eventkit.event.Event): Downstream startup event. The base
            :meth:`onStart` emits the unchanged startup payload and this Atom as
            its source.
        dataEvent (eventkit.event.Event): Downstream data event. Subclasses emit it
            explicitly after producing output.
        feedbackEvent (eventkit.event.Event): Reverse-direction feedback event. The
            base :meth:`onFeedback` emits the supplied payload unchanged.
        contract (ib_insync.contract.Contract | None): Optional contract associated with
            this component. Assignment registers the unqualified blueprint;
            access asks :attr:`contract_registry` for its current resolution.
            Before startup this may still be the blueprint; after qualification
            it is the selected concrete Contract. Components unrelated to a
            single instrument should leave it unset.
        which_contract (haymaker.enums.ActiveNext): Futures role returned by :attr:`contract`.
            Supported roles are ``ActiveNext.ACTIVE``, the
            default, and ``ActiveNext.NEXT`` for components
            that intentionally operate on the early-entry contract. PREVIOUS is
            reserved for direct selector and registry queries.
        ib (ib_insync.ib.IB): Runtime broker client.
        book (Book): Runtime accounting and recovery service.
        contract_registry (haymaker.contract_registry.ContractRegistry): Runtime contract qualification
            and selection registry.
        request_restart (Callable | None): Current supervisor restart callback,
            or ``None`` before one has been installed.
        contract_details (haymaker.details_processor.Details): Details for the resolved :attr:`contract`.
            These normally become available during startup. Missing details
            produce an empty ``Details`` value and an error log.
        contract_selector (haymaker.contract_selector.AbstractBaseContractSelector): Selector registered
            for the contract blueprint. Access raises until a contract has been
            assigned and its selector initialized by the runtime.

    Note:
        The public methods and operations are:

        * ``Atom()`` initializes the standard events. Subclass constructors must
          call ``super().__init__()``.
        * ``onStart(data, source=None)`` performs startup and forwards its
          arbitrary mutable payload downstream.
        * ``onData(data, *args)`` processes one input. The base implementation
          always raises ``NotImplementedError``.
        * ``onFeedback(data, *args)`` forwards feedback toward the preceding
          Atom.
        * ``onContractChanged(old_contract, new_contract)`` reacts when the
          resolved contract changes; Controller remains responsible for rolling
          held positions.
        * ``contract_blueprint`` returns a copy of the assigned declaration,
          independent of current ACTIVE/NEXT selection. ``contract_selector``
          exposes its initialized chain and Contract-selection operations.
        * ``validate_source(source)`` returns normally for a compatible
          prospective upstream Atom and raises for structural incompatibility.
          The default accepts every Atom.
        * ``connect(*targets)`` connects this Atom directly to one or more
          downstream targets after all targets validate the source and returns
          this Atom.
        * ``disconnect(*targets)`` removes direct startup, data, and
          reverse-feedback connections and returns this Atom.
        * ``clear()`` removes all outgoing startup and data connections,
          including the reverse-feedback links created by :meth:`connect`.
        * ``pipe(*targets)`` builds a linear :class:`Pipe` beginning with this
          Atom.
        * ``repr(atom)`` returns a concise representation of its non-default
          instance state and resolved contract.
        * Dataclass-based Atom subclasses must use ``@dataclass(eq=False)`` at
          every dataclass-decorated inheritance level. Atoms are stateful graph
          nodes, so identically configured instances retain distinct identity
          equality and remain hashable.

        ``source += target`` is shorthand for ``source.connect(target)`` and
        ``source -= target`` is shorthand for ``source.disconnect(target)``.

    Example:
        A custom Atom can process any Python value::

            class Scale(Atom):
                def __init__(self, factor: float) -> None:
                    self.factor = factor
                    super().__init__()

                def onData(self, value: float, *args: object) -> None:
                    self.dataEvent.emit(value * self.factor)

            calculation = Scale(2).pipe(Scale(3))
            results = []
            calculation.dataEvent += results.append
            calculation.onData(4)
            assert results == [24]
    """

    runtime: ClassVar[RuntimeContext]
    events: ClassVar[Sequence[str]] = (
        "startEvent",
        "dataEvent",
        "feedbackEvent",
    )
    contract = ContractManagingDescriptor()
    # these should be overriden by instances if neccessary to change
    which_contract: ActiveNext = ActiveNext.ACTIVE
    _contract_blueprint: ibi.Contract | None = None

    @classmethod
    def set_runtime_context(cls, runtime: RuntimeContext) -> None:
        """Install process runtime services on all Atoms."""

        cls.runtime = runtime

    def __init__(self) -> None:
        self._downstream_targets: list[Atom] = []
        self._createEvents()
        self._log = logging.getLogger(f"strategy.{self.__class__.__name__}")
        self._contract_memo: ibi.Contract | None = None
        self._roll_contract_data: ContractRollData | None = None

    @property
    def ib(self) -> ibi.IB:
        """Return the runtime IB client."""

        return self.runtime.ib

    @property
    def book(self) -> Book:
        """Return the runtime accounting book."""

        return self.runtime.book

    @property
    def contract_registry(self) -> ContractRegistry:
        """Return the runtime contract registry."""

        return self.runtime.contract_registry

    @property
    def request_restart(self):
        """Return the current runtime restart callback if it is available."""

        runtime = getattr(type(self), "runtime", None)
        if runtime is None:
            return None
        return runtime.request_restart

    @property
    def contract_details(self) -> Details:
        """Return broker details for the resolved contract.

        Returns:
            Contract details registered for :attr:`contract`. If details are
            unavailable, returns an empty :class:`Details` object and logs the
            missing contract.
        """
        details = self.contract_registry.get_details(self.contract)
        if details is None:
            log.error(f"Missing contract details for: {self.contract}")
            # empty details
            details = Details(ibi.ContractDetails())
        return details

    @property
    def contract_blueprint(self) -> ibi.Contract:
        """Return the assigned Contract declaration, independently of ACTIVE/NEXT.

        A copy prevents callers or broker qualification from mutating the
        registered identity. Pass this declaration to a custom direct
        Portfolio when it should select a concrete Contract itself.

        Raises:
            KeyError: If this Atom has no assigned Contract.
        """
        if self._contract_blueprint is None:
            raise KeyError(f"Contract not set on {type(self).__name__}")
        return deepcopy(self._contract_blueprint)

    @property
    def contract_selector(self) -> AbstractBaseContractSelector:
        """Return the selector registered for this Atom's contract blueprint.

        Raises:
            KeyError: If no contract has been assigned to this Atom.
            RuntimeError: If the runtime has not initialized the assigned
                contract's selector.
        """
        if self._contract_blueprint is None:
            raise KeyError(
                f"contract_selector not available because contract not set on {self}"
            )
        selector = self.contract_registry.get_selector(self._contract_blueprint)
        if selector is None:
            raise RuntimeError(
                "Contract selector not initialized for "
                f"{self._contract_blueprint} on {type(self).__name__}"
            )
        return selector

    def _createEvents(self) -> None:
        self.startEvent = ibi.Event("startEvent")
        self.dataEvent = ibi.Event("dataEvent")
        self.feedbackEvent = ibi.Event("feedbackEvent")
        # not chained, for internal use only
        self._contractChangedEvent = ibi.Event("contractChangedEvent")
        self._contractChangedEvent += self.onContractChanged

    def _log_event_error(self, event: ibi.Event, exception: Exception) -> None:
        self._log.error(f"Event error {event.name()}: {exception}", exc_info=True)

    def onStart(self, data: Any, source: Atom | None = None) -> Awaitable[None] | None:
        """Run synchronous startup work and forward arbitrary mutable data.

        Args:
            data: Shared user-controlled startup payload. Atom reserves no keys.
            source: Immediate upstream Atom, when available.

        Subclasses normally perform their initialization first and call
        ``super().onStart(data, source)`` last so downstream startup continues.
        """
        self._process_contract_change()
        self.startEvent.emit(data, self)

    def _process_contract_change(self) -> None:
        if (self._contract_memo is not None) and (self._contract_memo != self.contract):
            self._contractChangedEvent.emit(self._contract_memo, self.contract)
        self._contract_memo = self.contract

    def onData(self, data: Any, *args: Any) -> Awaitable[None] | None:
        """Process one input.

        Subclasses must implement processing and explicitly emit output.

        Raises:
            NotImplementedError: Always, unless overridden.
        """

        raise NotImplementedError(f"{type(self).__name__}.onData must be implemented")

    def onFeedback(self, data: Any, *args: Any) -> Awaitable[None] | None:
        """Forward optional feedback toward the preceding Atom.

        Args:
            data: Arbitrary feedback payload.
            *args: Additional event values supplied by the downstream source.
        """
        self.feedbackEvent.emit(data)

    def onContractChanged(
        self, old_contract: ibi.Contract, new_contract: ibi.Contract
    ) -> Awaitable[None] | None:
        """Record a change in the concrete contract resolved for this Atom.

        Override this hook when component-local data must be adjusted after a
        futures contract change. Call ``super().onContractChanged(...)`` to
        preserve the recorded roll information. Controller owns rolling any
        held broker position.

        Args:
            old_contract: Previously resolved concrete contract.
            new_contract: Newly resolved concrete contract.
        """
        log.info(
            f"{self!s} {self.which_contract!s} contract changed: {old_contract.localSymbol} "
            f"--> {new_contract.localSymbol}"
        )
        self._roll_contract_data = ContractRollData(old_contract, new_contract)

    def validate_source(self, source: Atom) -> None:
        """Validate one prospective upstream connection.

        The default accepts every source. Components override this only when
        they require a concrete upstream capability, such as a Streamer API.
        Message types are validated by ``onData`` when values arrive rather
        than advertised through connection metadata. An override must return
        normally when ``source`` is compatible and raise ``TypeError`` or a
        more specific domain exception otherwise; returning a boolean has no
        effect.

        Args:
            source: Atom that would emit into this Atom.

        Raises:
            TypeError: If an override determines that ``source`` is
                structurally incompatible.
        """

    def connect(self, *targets: Atom) -> Self:
        """Connect this Atom directly to one or more targets.

        Every target validates this source before any connection changes.
        ``+=`` is an alias. Use ``pipe`` for a linear multi-stage chain.

        Args:
            targets: One or more Atoms connected in one-to-many fan-out.

        Returns:
            This source Atom.

        Raises:
            TypeError: If a target is not an Atom.
            Exception: Propagates any incompatibility raised by a target's
                :meth:`validate_source`. No connection is changed unless every
                target accepts this source.
        """
        for target in targets:
            if not isinstance(target, Atom):
                raise TypeError("targets must be Atom instances")
            target.validate_source(self)

        for t in targets:
            self.disconnect(t)
            self.startEvent.connect(
                t.onStart, error=self._log_event_error, keep_ref=True
            )
            self.dataEvent.connect(t.onData, error=self._log_event_error, keep_ref=True)
            t.feedbackEvent.connect(
                self.onFeedback, error=t._log_event_error, keep_ref=True
            )
            self._downstream_targets.append(t)

        return self

    def disconnect(self, *targets: Atom) -> Self:
        """Disconnect one or more directly connected downstream Atoms.

        ``-=`` is an alias. Connections and callbacks not created through
        :meth:`connect` are unaffected.

        Args:
            targets: Direct downstream Atoms to disconnect.

        Returns:
            This source Atom.
        """
        for t in targets:
            # the same target cannot be connected more than once
            self.startEvent.disconnect_obj(t)
            self.dataEvent.disconnect_obj(t)
            t.feedbackEvent.disconnect_obj(self)
            self._downstream_targets = [
                target for target in self._downstream_targets if target is not t
            ]
        return self

    def clear(self) -> None:
        """Remove all outgoing startup, data, and reverse-feedback connections.

        Reverse-feedback callbacks belonging to other upstream Atoms are
        preserved.
        """
        for target in tuple(self._downstream_targets):
            target.feedbackEvent.disconnect_obj(self)
        self._downstream_targets.clear()
        self.startEvent.clear()
        self.dataEvent.clear()

    def pipe(self, *targets: Atom) -> Pipe:
        """Create a linear Pipe beginning with this Atom.

        Unlike :meth:`connect`, which creates fan-out, this method connects
        each target to the preceding Atom.

        Args:
            targets: Atoms to append in data-flow order.

        Returns:
            Pipe whose first member is this Atom.

        Raises:
            TypeError: If a target is not an Atom or adjacent members are
                structurally incompatible.
        """
        return Pipe(self, *targets)

    __iadd__ = connect
    __isub__ = disconnect

    def __repr__(self) -> str:
        attrs = [
            f"{name}={value}"
            for name, value in self.__dict__.items()
            if not name.startswith("_")
            and name not in self.events
            and value
            and value != ActiveNext.ACTIVE
        ]
        contract = self.contract
        if contract is not None:
            attrs.append(f"contract={contract}")
        return f"{self.__class__.__name__}({', '.join(attrs)})"


class Pipe(Atom):
    """Connect several Atoms into one composable linear pipeline.

    Pass members in data-flow order. ``Pipe`` forwards input and startup to the
    first member, exposes the last member's output, and routes feedback from the
    last member to the first.

    Args:
        targets: One or more initialized Atoms in connection order.

    Note:
        Connection validation is performed between adjacent members while the
        pipe is assembled. When another Atom connects to the Pipe itself, the
        first member validates that prospective source. Fan-out semantics
        remain those of :class:`Atom`.
    """

    def __init__(self, *targets: Atom) -> None:
        if not targets:
            raise ValueError("Pipe requires at least one Atom")
        if not all(isinstance(target, Atom) for target in targets):
            raise TypeError("Pipe members must be Atom instances")
        self._members = targets
        self.first = self._members[0]
        self.last = self._members[-1]
        super().__init__()
        self._pipe()

    def _createEvents(self) -> None:
        # Pipe doesn't create its own events, but redirects events of member Atoms
        self.startEvent = self.last.startEvent
        self.dataEvent = self.last.dataEvent
        self.feedbackEvent = self.first.feedbackEvent

    def connect(self, *targets: Atom) -> Self:
        """Connect downstream targets to the Pipe's last member.

        Args:
            targets: Atoms connected as fan-out from the final member.

        Returns:
            This Pipe.
        """
        self.last.connect(*targets)
        return self

    def validate_source(self, source: Atom) -> None:
        """Ask the first member to validate a prospective upstream Atom.

        This preserves the first member's input constraint when callers compose
        an existing Pipe with ``source.connect(pipe)``.

        Args:
            source: Atom that would emit into the Pipe's first member.

        Raises:
            Exception: Propagates any incompatibility raised by the first
                member.
        """

        self.first.validate_source(source)

    def disconnect(self, *targets: Atom) -> Self:
        """Disconnect targets from the Pipe's last member.

        Args:
            targets: Direct downstream Atoms to disconnect.

        Returns:
            This Pipe.
        """
        self.last.disconnect(*targets)
        return self

    def clear(self) -> None:
        """Clear outgoing connections from the Pipe's last member."""

        self.last.clear()

    def onStart(self, data: Any, source: Atom | None = None) -> Awaitable[None] | None:
        """Forward workload startup to the Pipe's first member."""

        return self.first.onStart(data, source)

    def onData(self, data: Any, *args: Any) -> Awaitable[None] | None:
        """Forward one input to the Pipe's first member."""

        return self.first.onData(data, *args)

    def onFeedback(self, data: Any, *args: Any) -> Awaitable[None] | None:
        """Forward feedback from the Pipe's last member through the chain."""

        return self.last.onFeedback(data, *args)

    def _pipe(self) -> None:
        source = None
        for i, member in enumerate(self._members):
            if i > 0:
                assert source is not None
                source.connect(member)
            source = member

    def __getitem__(self, i: int) -> Atom:
        """Return the member at ``i`` in data-flow order."""

        return self._members[i]

    def __len__(self) -> int:
        """Return the number of members in the Pipe."""

        return len(self._members)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{tuple(i for i in self._members)}"
