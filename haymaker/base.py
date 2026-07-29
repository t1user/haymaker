from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    ClassVar,
    NamedTuple,
    Self,
    Sequence,
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
    pass


class ContractManagingDescriptor:
    # DON'T CHANGE THIS TO PROPERTY or it will screw up dataclasses
    # that inherit from Atom
    """
    Manage accessing `contract` property on :class:`Atom`.

    Contract needs to be qualified and their details obtaned before.
    This is being managed by :module:`Manager`, which puts correct
    contracts and details into :class:`ContractRegistry`.  The role of
    this descriptor is to pull the correct values from it.
    """

    def __set_name__(self, obj: type[Atom], name: str) -> None:
        self.name = f"_{name}_blueprint"

    def __set__(self, obj: Atom, value: ibi.Contract) -> None:
        if not isinstance(value, ibi.Contract):
            raise TypeError(f"attr contract must be ibi.Contract, not: {type(value)}")
        obj.__dict__[self.name] = value
        obj.contract_registry.register_blueprint(value)

    def __get__(self, obj: Atom, type=None) -> ibi.Contract | None:
        contract_blueprint = obj.__dict__.get(self.name)
        if contract_blueprint is None:
            return None
        try:
            return obj.contract_registry.get_contract(
                contract_blueprint, obj.which_contract
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
        startEvent (eventkit.Event): Downstream startup event. The base
            :meth:`onStart` emits the unchanged startup payload and this Atom as
            its source.
        dataEvent (eventkit.Event): Downstream data event. Subclasses emit it
            explicitly after producing output.
        feedbackEvent (eventkit.Event): Reverse-direction feedback event. The
            base :meth:`onFeedback` emits the supplied payload unchanged.
        contract (ib_insync.Contract | None): Optional contract associated with
            this component. Assignment registers the unqualified blueprint;
            access asks :attr:`contract_registry` for its current resolution.
            Before startup this may still be the blueprint; after qualification
            it is the selected concrete Contract. Components unrelated to a
            single instrument should leave it unset.
        which_contract (ActiveNext): Futures role returned by :attr:`contract`.
            The default is :attr:`~haymaker.enums.ActiveNext.ACTIVE`; use
            :attr:`~haymaker.enums.ActiveNext.NEXT` only when the component
            intentionally operates on the early-entry contract.
        ib (ib_insync.IB): Runtime broker client.
        book (Book): Runtime accounting and recovery service.
        contract_registry (ContractRegistry): Runtime contract qualification
            and selection registry.
        request_restart (Callable | None): Current supervisor restart callback,
            or ``None`` before one has been installed.
        contract_details (Details): Details for the resolved :attr:`contract`.
            These normally become available during startup. Missing details
            produce an empty ``Details`` value and an error log.
        contract_selector (AbstractBaseContractSelector | None): Selector
            registered for the contract blueprint. Access without a configured
            contract raises ``KeyError``.

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
        * ``validate_source(source)`` rejects a structurally incompatible
          prospective upstream Atom. The default accepts every Atom.
        * ``connect(*targets)`` connects this Atom directly to one or more
          downstream targets after all targets validate the source and returns
          this Atom.
        * ``disconnect(*targets)`` removes direct startup, data, and
          reverse-feedback connections and returns this Atom.
        * ``clear()`` removes this Atom's current downstream connections.
        * ``pipe(*targets)`` builds a linear :class:`Pipe` beginning with this
          Atom.
        * ``union(*targets)`` connects targets as fan-out and returns this Atom.
        * ``repr(atom)`` returns a concise representation of its non-default
          instance state and resolved contract.

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
        """
        Contract details received from the broker.

        if :attr:`contract` is not set empty :class:`Details` object
        will be returned.
        """
        details = self.contract_registry.get_details(self.contract)
        if details is None:
            log.error(f"Missing contract details for: {self.contract}")
            # empty details
            details = Details(ibi.ContractDetails())
        return details

    @property
    def contract_selector(self) -> AbstractBaseContractSelector | None:
        try:
            assert self._contract_blueprint
        except AssertionError:
            raise KeyError(
                f"contract_selector not available because contract not set on {self}"
            )
        return self.contract_registry.get_selector(self._contract_blueprint)

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
        """
        Will be called if contract object on `self.contract` changes.
        In particular, this happens when future contract is about to
        expire, and new on-the-run contract replaces old, expiring
        contract.  This method should be used to initialize any
        adjustment required on the object in relation to contract
        rolling.  Actual position rolling is taken care of by
        `Controller` object.
        """
        log.info(
            f"{self!s} {self.which_contract!s} contract changed: {old_contract.localSymbol} "
            f"--> {new_contract.localSymbol}"
        )
        self._roll_contract_data = ContractRollData(old_contract, new_contract)

    def validate_source(self, source: Atom) -> None:
        """Validate one prospective upstream connection.

        The default accepts every source. Built-in components override this
        only for structural incompatibilities that can be known before data
        arrives.

        Args:
            source: Atom that would emit into this Atom.
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
            TypeError: If any target is not an Atom or rejects the source.
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

        return self

    def disconnect(self, *targets: Atom) -> Self:
        """
        Disconnect passed :class:`Atom` objects, which are directly
        connected to this atom. Shorthand for this method is `-=`


        Args:
            targets (Atom): One or more :class:`Atom` objects to disconnect from.
        """
        for t in targets:
            # the same target cannot be connected more than once
            self.startEvent.disconnect_obj(t)
            self.dataEvent.disconnect_obj(t)
            t.feedbackEvent.disconnect_obj(self)
        return self

    def clear(self) -> None:
        connected_to = [i[0] for i in self.startEvent._slots]
        self.startEvent.clear()
        self.dataEvent.clear()
        for obj in connected_to:
            obj.feedbackEvent.clear()

    def pipe(self, *targets: Atom) -> Pipe:
        """
        Create a :class:`Pipe` or a chain of connected :class:`Atom` objects.
        Only first `target` will be directly connected to this object, second
        target will be connected to the first target and so on. It's different
        from :meth:`connect` method, which connects all targets directly to
        this object.

        Returns:
            Pipe: :class:`Pipe` object with all targets connected in a chain,
            where this object is the first and the last target is the last
            target in the list of passed targets.
        """
        return Pipe(self, *targets)

    def union(self, *targets: "Atom") -> Self:
        for t in targets:
            self.connect(t)
        return self

    __iadd__ = connect
    __isub__ = disconnect

    def __repr__(self) -> str:
        attrs = ", ".join(
            (
                f"{i}={j}"
                for i, j in self.__dict__.items()
                if ("Event" not in str(i))
                and (i != "_log")
                and (i != "_contract_blueprint")
                and j
                and j != ActiveNext.ACTIVE
            )
        )
        if self.contract is not None:
            attrs += f", contract={self.contract}"
        return f"{self.__class__.__name__}({attrs})"


class Pipe(Atom):
    """Connect several Atoms into one composable linear pipeline.

    Pass members in data-flow order. ``Pipe`` forwards input and startup to the
    first member, exposes the last member's output, and routes feedback from the
    last member to the first.

    Args:
        targets: One or more initialized Atoms in connection order.

    Note:
        Connection validation is performed between adjacent members while the
        pipe is assembled. Fan-out semantics remain those of :class:`Atom`.
    """

    def __init__(self, *targets: Atom):
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
        for target in targets:
            self.last.connect(target)
        return self

    def validate_source(self, source: Atom) -> None:
        """Validate an upstream connection against the first pipe member."""

        self.first.validate_source(source)

    def disconnect(self, *targets: Atom) -> Self:
        for target in targets:
            self.last.startEvent.disconnect_obj(target)
            self.last.dataEvent.disconnect_obj(target)
            target.feedbackEvent.disconnect_obj(self.last)
        return self

    def onStart(self, data: Any, *args: Any) -> None:
        self.first.onStart(data, *args)

    def onData(self, data: Any, *args: Any) -> None:
        self.first.onData(data, *args)

    def onFeedback(self, data: Any, *args: Any) -> None:
        self.last.onFeedback(data, *args)

    def _pipe(self) -> None:
        source = None
        for i, member in enumerate(self._members):
            if i > 0:
                assert source is not None
                source.connect(member)
            source = member

    def __getitem__(self, i: int) -> Atom:
        return self._members[i]

    def __len__(self) -> int:
        return len(self._members)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{tuple(i for i in self._members)}"
