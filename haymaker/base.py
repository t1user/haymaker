from __future__ import annotations

import logging
from datetime import datetime, timezone
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
    from .contract_selector import AbstractBaseContractSelector
    from .state_machine import StateMachine, Strategy

log = logging.getLogger(__name__)


class MissingContractError(Exception):
    pass


class ContractManagingDescriptor:
    # DON'T CHANGE THIS TO PROPERTY or it will screw up inherited dataclasses
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
                f"Unknown contract: {contract_blueprint} on {obj}"
            )


class ContractRollData(NamedTuple):
    old_contract: ibi.Contract
    new_contract: ibi.Contract


class Atom:
    """
    Abstract base object from which all other objects inherit. It's a basic building
    block for creating strategies in Haymaker. Every ``Atom`` represents a processing
    step typically for one traded instrument, thus allowing for separation of concerns.
    Connecting ``Atoms`` creates a processing pipeline.

    ``Atoms`` in a pipeline communicate with each other in an event-driven manner
    through three methods: :meth:`onStart`, :meth:`onData`, :meth:`onFeedback`.
    Those methods are called when appropriate :class:`eventkit.event.Event` objects are
    emitted (respectively :attr:`startEvent`, :attr:`dataEvent`, :attr:`feedbackEvent`),
    and emit their own events when they are done processing thus sending a signal to
    the next ``Atom`` in the pipeline that it can start processing.

    Users are free to put in those methods any processing logic they want, using any
    libraries and tools required. ``Atoms`` can be connected in any order; unions can be
    created by connecting more than one ``Atom``; pipelines can be created using
    auxiliary class :class:`Pipeline`.

    Attributes:
        contract (ib_insync.contract.Contract): The contract object associated with
            this Atom. This can be any :class:`ib_insync.contract.Contract`. On startup
            this contract will be qualified and available
            :class:`ib_insync.contract.ContractDetails` will be downloaded from broker
            and made available through :attr:`details` attribute. If contract
            is a :class:`ib_insync.contract.ContFuture` or
            :class:`ib_insync.contract.Future`, it will be replaced with on-the-run
            :class:`ib_insync.contract.Future`. `ContFuture` will pick contract that
            IB considers to be be current, `Future` allows for customization by tweaking
            :class:`FutureSelector`. Whichever method is chose, when contract
            to be rolled, :meth:`onContractChanged` method will be called.

            This attribute doesn't need to be set. If this Atom
            object is not related to any one particular contract, just don't assign any
            value to this attribute.

        which_contract (ActiveNext): default: ACTIVE; if NEXT chosen :attr:`contract`
            will return next contract in chain (relevant only for expiring contracts
            like futures or options) allowing for early usage of upcoming
            contracts for new positions a short period before they become active
            (number of days prior to expiry during which NEXT will be used can be
            configured in config.)

        ib (ClassVar[ibi.IB]): The instance of the :class:`ib_insync.ib.IB` client used
           for interacting with the broker. It can be used to communicate with
           the broker if neccessary.

        sm (ClassVar[StateMachine]): Access to :class:`StateMachine` which is
            Haymaker's central collection of information about current positions,
            orders and state of strategies.

        contracts (ClassVar[list[ibi.Contract]]): A collection of all contracts
            currently in use.

        events (ClassVar[Sequence[str]]): Collection of :class:`eventkit.Event` objects
            used by ``Haymaker``, i.e. :attr:`startEvent`, :attr:`dataEvent`,
            :attr:`feedbackEvent`, appropriate methods should use these events to
            communicate with other objects in the chain, e.g. :meth:`onStart` after
            processing incoming data should pass the result to the next object
            by calling `self.dataEvent.emit(data)`.
    """

    ib: ClassVar[ibi.IB]
    sm: ClassVar[StateMachine]
    contract_registry: ClassVar[ContractRegistry] = ContractRegistry()
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
    def set_init_data(cls, ib: ibi.IB, sm: StateMachine, cr: ContractRegistry) -> None:
        cls.ib = ib
        cls.sm = sm
        cls.contract_registry = cr

    def __init__(self) -> None:
        self._createEvents()
        self._log = logging.getLogger(f"strategy.{self.__class__.__name__}")
        if not getattr(self, "strategy", None):
            self.strategy = ""
        self.startup = False
        self._contract_memo: ibi.Contract | None = None
        self._roll_contract_data: ContractRollData | None = None

    @property
    def contract_details(self) -> Details:
        """
        Contract details received from the broker.

        If :attr:`contract` is set, :attr:`details` will be returned
        only for this contract, otherwise :attr:`details` will return
        a dictionary of all available details,
        ie. :class:`dict`[:class:`ibi.Contract`, :class:`Details`]

        :class:`Details` is a wrapper around
        :class:`ib_insync.contract.ContractDetails` with some
        additional methods and attributes.
        """
        try:
            return self.contract_registry.details[self.contract]
        except KeyError:
            log.error(f"Missing contract details for: {self.contract}")
            # empty details
            return Details(ibi.ContractDetails())

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

    def onStart(self, data: Any, *args: Any) -> Awaitable[None] | None:
        """
        Perform any initilization required on system (re)start.  It
        will be run automatically and it will be linked to
        :attr:`startEvent` of the preceding object in the chain.

        First `Atom` in a pipeline (typically a data streamer) will be
        called by system, which is an indication that (re)start is in
        progress and we have successfully connected to the broker.

        `data` by default is a dict.  Any information that needs to be
        passed to atoms down the chain, should be appended to `data`
        without removing any existing keys.

        If overriding the class, call superclass; call to
        :meth:`super().onStart(data)` should usually be the last line
        in overriden method as it will emit :attr:`startEvent` -
        basically do any processing required in sub-class and then
        call super-class, which will do standard processing and then
        emit the event to initialize processing in the next Atom down
        the chain; if you don't call superclass, make sure to emit
        :attr:`startEvent`, otherwise subsequent Atoms in the chain
        won't do startup initialization.

        This method can be synchronous as well as asynchronous (in the
        subclass it's ok to override it with `async def onData(self,
        data, *args)`).  If it's async, it will be put in the asyncio
        loop.
        """
        self._set_startup_attrs(data)
        self._process_contract_change()
        self.startEvent.emit(data, self)
        return None

    def _set_startup_attrs(self, data):
        # set strategy if not already set and present in `data` dict
        if (
            (self.strategy == "")
            and isinstance(data, dict)
            and (strategy := data.get("strategy"))
        ):
            self.strategy = strategy

        # set startup to whatever in dict
        if isinstance(data, dict) and (startup := data.get("startup")):
            self.startup = startup

    def _process_contract_change(self) -> None:
        if (self._contract_memo is not None) and (self._contract_memo != self.contract):
            # it will not fire if the system has been restarted after contract changed
            # cannot be relied on for rolls
            # MUST be used to ensure streamers and processors don't mix-up data
            # from old and new contracts
            log.debug(f"Future will reset: {self._contract_memo} --> {self.contract}")
            self._contractChangedEvent.emit(self._contract_memo, self.contract)
        self._contract_memo = self.contract

    def onData(self, data: Any, *args: Any) -> Awaitable[None] | None:
        """
        Connected to :attr:`dataEvent` of the preceding object in the chain.
        This is the entry point to any processing perfmormed by this
        object.  Result of this processing should be added to the
        `data` dict and passed to the subsequent object in the chain
        using :attr:`dataEvent` (by calling `self.dataEvent.emit(data)`).

        It's up to the user to emit `dataEvent` with appropriate data,
        this event will NOT be emitted by the system, so if it's not
        properly implemented, event chain will be broken.  This method
        must be obligatorily overriden in a subclass.

        Calling superclass on exit will add a timestamp with object's
        name to `data`, which may be useful for logging.

        This method can be synchronous as well as asynchronous (in the
        subclass it's ok to override it with `async def onData(self,
        data, *args)`).  If it's async, it will be put in the asyncio
        loop.
        """
        data[f"{self.__class__.__name__}_ts"] = datetime.now(tz=timezone.utc)
        return None

    def onFeedback(self, data: Any, *args: Any) -> Awaitable[None] | None:
        """
        Connected to :attr:`feedbackEvent` of the subsequent object in the
        chain.  Allows for passing of information about trading
        results.  It's optional to use it, if used, overriden method
        must emit `feedbackEvent` with appropriate data.  If not
        overriden, it will just pass received data to the previous
        object in the chain.

        This method can be synchronous as well as asynchronous (in the
        subclass it's ok to override it with `async def onData(self,
        data, *args)`).  If it's async, it will be put in the asyncio
        loop.
        """
        self.feedbackEvent.emit(data)
        return None

    def onContractChanged(
        self, old_contract: ibi.Future, new_contract: ibi.Future
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
        log.warning(
            f"Contract on {self} changed from {old_contract.localSymbol} "
            f"to new contract {new_contract.localSymbol}"
        )
        self._roll_contract_data = ContractRollData(old_contract, new_contract)
        return None

    @property
    def data(self) -> Strategy:
        strategy = getattr(self, "strategy", "")
        if not strategy:
            log.warning(f"{self} accessing data for empty strategy.")
        return self.sm.strategy[strategy]

    def connect(self, *targets: Atom) -> Self:
        """
        Connect appropriate events and methods to subsequent :class:`Atom`
        object(s) in the chain. Shorthand for this method is `+=`

        Args:
            targets (Atom): One or more :class:`Atom` objects to connect to.
                If more than one object passed, they will be connected directly
                in a `one-to-many` fashion. If the intention is to create
                a chain of objects, use :meth:`pipe` instead.

        Returns:
            Atom: The updated `Atom` object.
        """
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
                and j
                and j != ActiveNext.ACTIVE
            )
        )
        return f"{self.__class__.__name__}({attrs})"


class Pipe(Atom):
    """
    Auxiliary object for conneting several :class:`Atom` objects. Atoms to be connected
    need to passed in the right order at initialization.  :class:`Pipe` itself is
    a subclass of :class:`Atom`, so it can be connected
    to other :class:`Atom` (or :class:`Pipe`) and all :class:`Atom` attributes
    and methods are available.
    """

    def __init__(self, *targets: Atom):
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
