from __future__ import annotations

import collections.abc
import logging
from collections import UserDict
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Awaitable,
    ClassVar,
    NamedTuple,
    Optional,
    Sequence,
    Union,
    cast,
)

import ib_insync as ibi

from .misc import is_active, next_active, process_trading_hours

if TYPE_CHECKING:
    from .state_machine import StateMachine, Strategy


log = logging.getLogger(__name__)


ContractOrSequence = Union[Sequence[ibi.Contract], ibi.Contract]


class ContractManagingDescriptor:

    def __set_name__(self, obj: Atom, name: str) -> None:
        self.name = name

    def __set__(
        self, obj: Atom, value: Union[ibi.Contract, collections.abc.Sequence]
    ) -> None:
        if not isinstance(value, (ibi.Contract, collections.abc.Sequence)):
            raise TypeError(
                f"attr contract must be ibi.Contract or sequence of ibi.Contract, "
                f"not: {type(value)}"
            )
        obj.__dict__[self.name] = value
        self._register_contract(obj, value)

    def __get__(self, obj: Atom, type=None) -> Union[list[ibi.Contract], ibi.Contract]:
        # some methods rely on not raising an error here if attr self.name not set
        contract = obj.__dict__.get(self.name)
        return self._swap_contfuture(contract, obj.contracts)

    def _register_contract(self, obj: Atom, value: ContractOrSequence) -> None:
        log.debug(f"Registering contract: {value}")
        self._append(value, obj.contracts, obj)

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

    @staticmethod
    @_apply_to_contract_or_sequence
    def _swap_contfuture(
        contract: ibi.Contract, contract_list: list[ibi.Contract]
    ) -> ibi.Contract:
        if isinstance(contract, ibi.ContFuture) and contract.conId != 0:
            for i in contract_list:
                if isinstance(i, ibi.Future) and i.conId == contract.conId:
                    return i
            log.warning(f"Failed to replace ContFuture object: {contract}")
        return contract

    @staticmethod
    @_apply_to_contract_or_sequence
    def _append(contract: ibi.Contract, contract_list: list[ibi.Contract], atom: Atom):
        if contract not in contract_list:
            contract_list.append(contract)


@dataclass
class Details:
    _fields: ClassVar[list] = [f.name for f in fields(ibi.ContractDetails)]
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

    def __getattr__(self, name):
        if name in self._fields:
            return getattr(self.details, name)
        super().__getattr__(name)

    def is_open(self, _now: Optional[datetime] = None) -> bool:
        return self._is_active(self.trading_hours, _now)

    def is_liquid(self, _now: Optional[datetime] = None) -> bool:
        return self._is_active(self.liquid_hours, _now)

    def next_open(self, _now: Optional[datetime] = None) -> datetime:
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
    Abstract base object from which all other objects inherit.

    """

    ib: ClassVar[ibi.IB]
    sm: ClassVar[StateMachine]
    contract_details: ClassVar[DetailsContainer] = DetailsContainer()
    contracts: ClassVar[list[ibi.Contract]] = list()
    events: ClassVar[Sequence[str]] = (
        "startEvent",
        "dataEvent",
        "feedbackEvent",
        "contractChangedEvent",
    )
    contract = cast(ibi.Contract, ContractManagingDescriptor())
    _contract_memo: Optional[ibi.Contract] = None
    roll_contract_data: Optional[ContractRollData] = None

    @classmethod
    def set_init_data(cls, ib: ibi.IB, sm: StateMachine) -> None:
        cls.ib = ib
        cls.sm = sm

    def __init__(self) -> None:
        self._createEvents()
        self._log = logging.getLogger(f"strategy.{self.__class__.__name__}")

    @property
    def details(self) -> Union[Details, DetailsContainer]:
        """
        If :attr:`contract` is set :attr:`details` will be received
        only for this contract, otherwise :attr:`details` will return
        a dictionary of all available details, ie.  dict[ibi.Contract, Details]
        """
        if self.contract:
            try:
                return self.contract_details[self.contract]
            except KeyError:
                log.error(f"Missing contract details for: {self.contract}")
                return self.contract_details
        else:
            return self.contract_details

    def _createEvents(self):
        self.startEvent = ibi.Event("startEvent")
        self.dataEvent = ibi.Event("dataEvent")
        self.feedbackEvent = ibi.Event("feedbackEvent")
        self.contractChangedEvent = ibi.Event("contractChangedEvent")
        self.contractChangedEvent += self.onContractChanged

    def _log_event_error(self, event: ibi.Event, exception: Exception) -> None:
        self._log.error(f"Event error {event.name()}: {exception}", exc_info=True)

    def onStart(self, data, *args) -> None:
        if isinstance(data, dict):
            # for k, v in data.items():
            #     setattr(self, k, v)
            self.__dict__.update(**data)
        if (self._contract_memo is not None) and (self._contract_memo != self.contract):
            # it will not fire if the system has been restarted after contract changed
            # usless, TODO: consider removing
            # cannot be relied on for rolls
            self.contractChangedEvent.emit(self._contract_memo, self.contract)
        self._contract_memo = self.contract
        self.startEvent.emit(data, self)

    def onData(self, data: dict, *args) -> Union[Awaitable[None], None]:
        data[f"{self.__class__.__name__}_ts"] = datetime.now(tz=timezone.utc)
        return None

    def onFeedback(self, data, *args) -> Union[Awaitable[None], None]:
        return None

    def onContractChanged(
        self, old_contract: ibi.Future, new_contract: ibi.Future
    ) -> Union[Awaitable[None], None]:
        """This is not chained."""
        log.warning(
            f"Contract on {self} changed from {old_contract.symbol} "
            f"to new contract {new_contract.symbol}"
        )
        self.roll_contract_data = ContractRollData(old_contract, new_contract)
        return None

    @property
    def data(self) -> Strategy:
        strategy = getattr(self, "strategy", "")
        if not strategy:
            log.warning(f"{self} accessing data for empty strategy.")
        return self.sm.strategy[strategy]

    def connect(self, *targets: Atom) -> "Atom":
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

    def disconnect(self, *targets: Atom) -> "Atom":
        for t in targets:
            # the same target cannot be connected more than once
            self.startEvent.disconnect_obj(t)
            self.dataEvent.disconnect_obj(t)
            t.feedbackEvent.disconnect_obj(self)
        return self

    def clear(self):
        connected_to = [i[0] for i in self.startEvent._slots]
        self.startEvent.clear()
        self.dataEvent.clear()
        for obj in connected_to:
            obj.feedbackEvent.clear()

    def pipe(self, *targets: Atom) -> Pipe:
        return Pipe(self, *targets)

    def union(self, *targets: "Atom") -> "Atom":
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
    def __init__(self, *targets: Atom):
        self._members = targets
        self.first = self._members[0]
        self.last = self._members[-1]
        super().__init__()
        self.pipe()

    def _createEvents(self):
        self.startEvent = self.first.startEvent
        self.dataEvent = self.first.dataEvent
        self.feedbackEvent = self.first.feedbackEvent

    def connect(self, *targets: Atom) -> "Pipe":
        for target in targets:
            self.last.connect(target)
        return self

    def disconnect(self, *targets: Atom) -> "Pipe":
        for target in targets:
            self.last.startEvent.disconnect_obj(target)
            self.last.dataEvent.disconnect_obj(target)
            target.feedbackEvent.disconnect_obj(self.last)
        return self

    def onStart(self, data, *args) -> None:
        self.first.onStart(data, *args)

    def onData(self, data, *args) -> None:
        self.first.onData(data, *args)

    def onFeedback(self, data, *args) -> None:
        self.last.onFeedback(data, *args)

    def pipe(self):
        for i, member in enumerate(self._members):
            if i > 0:
                source.connect(member)  # noqa
            source = member  # noqa

    def __getitem__(self, i: int) -> Atom:
        return self._members[i]

    def __len__(self) -> int:
        return len(self._members)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{tuple(i for i in self._members)}"
