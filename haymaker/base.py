from __future__ import annotations

import collections.abc
import logging
from collections import UserDict
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    ClassVar,
    NamedTuple,
    Optional,
    Self,
    Sequence,
    cast,
)

import ib_insync as ibi

from .misc import hash_contract, is_active, next_active, process_trading_hours

if TYPE_CHECKING:
    from .state_machine import StateMachine, Strategy


log = logging.getLogger(__name__)


ContractOrSequence = Sequence[ibi.Contract] | ibi.Contract


class ContractManagingDescriptor:

    @staticmethod
    def _apply_to_contract_or_sequence(func):
        """
        Decorator that applies the `func` either to contract or to
        every contract in a sequence or if contract is neither
        :class:`ibi.Contract` nor a sequence return this contract.
        """

        @wraps(func)
        def wrapper(contract, *args, **kwargs):
            if getattr(contract, "__iter__", None):
                assert isinstance(contract, collections.abc.Sequence)
                return [func(c, *args, **kwargs) for c in contract]
            elif isinstance(contract, ibi.Contract):
                return func(contract, *args, **kwargs)
            else:
                return contract

        return wrapper

    def __set_name__(self, obj: Atom, name: str) -> None:
        self.name = name

    def __set__(
        self, obj: Atom, value: ibi.Contract | collections.abc.Sequence
    ) -> None:
        if not isinstance(value, (ibi.Contract, collections.abc.Sequence)):
            raise TypeError(
                f"attr contract must be ibi.Contract or sequence of ibi.Contract, "
                f"not: {type(value)}"
            )
        obj.__dict__[self.name] = value
        self._register_contract(obj, value)

    def __get__(self, obj: Atom, type=None) -> list[ibi.Contract] | ibi.Contract | None:
        # some methods rely on not raising an error here if attr self.name not set
        # if Atom has not set `contract` attribute, it shouldn't raise an error
        contract = obj.__dict__.get(self.name)
        return self._get_contract(contract, obj.contract_dict)

    def _register_contract(self, obj: Atom, value: ContractOrSequence) -> None:
        self._append_to_dict(value, obj.contract_dict, obj)

    @staticmethod
    @_apply_to_contract_or_sequence
    def _get_contract(
        contract: ibi.Contract | None, contract_dict: dict[int, ibi.Contract]
    ) -> list[ibi.Contract] | ibi.Contract | None:
        if contract is None:
            return None
        else:
            return contract_dict.get(hash_contract(contract))

    @staticmethod
    @_apply_to_contract_or_sequence
    def _append_to_dict(
        contract: ibi.Contract, contract_dict: dict[int, ibi.Contract], atom: Atom
    ) -> None:
        contract_dict[hash_contract(contract)] = contract
        log.debug(f"Contract registered: {contract}")


@dataclass
class Details:
    """
    Wrapper object for :class:`ib_insync.contract.ContractDetails` extracting
    and processing information that's most relevant for Haymaker.

    Attributes:
        details (ibi.ContractDetails): The original contract details object.
        trading_hours (list[tuple[datetime, datetime]]): List of tuples with
            start and end of trading hours for this contract.
        liquid_hours (list[tuple[datetime, datetime]]): List of tuples with
            start and end of liquid hours for this contract.
    """

    _fields: ClassVar[list[str]] = [f.name for f in fields(ibi.ContractDetails)]
    details: ibi.ContractDetails
    trading_hours: list[tuple[datetime, datetime]] = field(init=False, repr=False)
    liquid_hours: list[tuple[datetime, datetime]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.trading_hours = self._process_trading_hours(
            self.details.tradingHours, self.details.timeZoneId
        )
        self.liquid_hours = self._process_trading_hours(
            self.details.liquidHours, self.details.timeZoneId
        )

    def __getattr__(self, name: str) -> Any:
        if name in self._fields:
            return getattr(self.details, name)
        else:
            raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")

    def is_open(self, _now: Optional[datetime] = None) -> bool:
        """
        Given current time check if the market is open for underlying contract.

        Args:
            _now (Optional[datetime], optional): Defaults to None.
            If not provided current time will be used. Only situation when
            it's useful to provide `_now` is in testing.

        Returns:
            bool: True if market is open, False otherwise.
        """
        return self._is_active(self.trading_hours, _now)

    def is_liquid(self, _now: Optional[datetime] = None) -> bool:
        """
        Given current time check if the market is during liquid hours
        for underlying contract.

        Args:
            _now (Optional[datetime], optional): . Defaults to None.
            If not provided current time will be used. Only situation when
            it's useful to provide `_now` is in testing.

        Returns:
            bool: True if market is liquid, False otherwise.
        """
        return self._is_active(self.liquid_hours, _now)

    def next_open(self, _now: Optional[datetime] = None) -> datetime:
        """
        Return time of nearest market re-open (regardless if market is
        open now).  Should be used after it has been tested that
        :meth:`is_active` is False.

        Args:
            _now (Optional[datetime], optional): Defaults to None.
            If not provided current time will be used. Only situation when
            it's useful to provide `_now` is in testing.
        """
        return self._next_open(self.trading_hours, _now)

    _process_trading_hours = staticmethod(process_trading_hours)
    _is_active = staticmethod(is_active)
    _next_open = staticmethod(next_active)


class DetailsContainer(UserDict):
    def __setitem__(self, key: ibi.Contract, value: ibi.ContractDetails) -> None:
        super().__setitem__(key, Details(value))


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
            is a :class:`ib_insync.contract.ContFuture`,
            it will be replaced with on-the-run :class:`ib_insync.contract.Future`
            and whenever this contract needs to be rolled, :meth:`onContractChanged`
            method will be called. This attribute doesn't need to be set. If this Atom
            object is not related to any one particular contract, just don't assign any
            value to this attribute.

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
    contract_details: ClassVar[DetailsContainer] = DetailsContainer()
    contract_dict: dict[int, ibi.Contract] = {}
    events: ClassVar[Sequence[str]] = (
        "startEvent",
        "dataEvent",
        "feedbackEvent",
    )
    contract = cast(ibi.Contract, ContractManagingDescriptor())
    _contract_memo: Optional[ibi.Contract] = None
    _roll_contract_data: Optional[ContractRollData] = None

    @classmethod
    def set_init_data(cls, ib: ibi.IB, sm: StateMachine) -> None:
        cls.ib = ib
        cls.sm = sm

    def __init__(self) -> None:
        self._createEvents()
        self._log = logging.getLogger(f"strategy.{self.__class__.__name__}")

    @property
    def contracts(self):
        return self.contract_dict.values()

    @contracts.setter
    def contracts(self) -> None:
        raise ValueError("Forbidden to set values on Atom.contracts.")

    @property
    def details(self) -> Details | list[Details] | DetailsContainer:
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
        if self.contract:
            if getattr(self.contract, "__iter__", None):
                assert isinstance(self.contract, collections.abc.Sequence)
                return [self.contract_details[c] for c in self.contract]
            else:
                try:
                    return self.contract_details[self.contract]
                except KeyError:
                    log.error(f"Missing contract details for: {self.contract}")
                    return self.contract_details
        else:
            return self.contract_details

    def _createEvents(self) -> None:
        self.startEvent = ibi.Event("startEvent")
        self.dataEvent = ibi.Event("dataEvent")
        self.feedbackEvent = ibi.Event("feedbackEvent")
        # not chained, for internal use only
        self._contractChangedEvent = ibi.Event("contractChangedEvent")
        self._contractChangedEvent += self.onContractChanged

    def _log_event_error(self, event: ibi.Event, exception: Exception) -> None:
        self._log.error(f"Event error {event.name()}: {exception}", exc_info=True)

    def onStart(self, data: Any, *args: Any) -> None:
        """
        Perform any initilization required on system (re)start.  It
        will be run automatically and it will be linked to
        :attr:`startEvent` of the preceding object in the chain.

        First `Atom` in a pipeline (typically a data streamer) will
        be called by system, which is an indication that (re)start is
        in progress and we have successfully connected to the broker.

        `data` by default is a dict and all keys on this dict are
        being set as properties on the object.  Any information that
        needs to be passed to atoms down the chain, should be appended
        to `data` without removing any existing keys.

        If overriding the class, call superclass; call to
        :meth:`super().onStart(data)` should be the last line in overriden
        method; don't manually emit :attr:`startEvent` in subclass.

        This method can be synchronous as well as asynchronous (in the
        subclass it's ok to override it with `async def onData(self,
        data, *args)`).  If it's async, it will be put in the asyncio
        loop.
        """
        if isinstance(data, dict):
            self.__dict__.update(**data)
        if (self._contract_memo is not None) and (self._contract_memo != self.contract):
            # it will not fire if the system has been restarted after contract changed
            # cannot be relied on for rolls
            # MUST be used to ensure streamers and processors don't mix-up data
            # from old and new contracts
            self._contractChangedEvent.emit(self._contract_memo, self.contract)
        self._contract_memo = self.contract
        self.startEvent.emit(data, self)

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
                if "Event" not in str(i) and i != "_log" and j
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
