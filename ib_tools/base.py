from __future__ import annotations

import collections.abc
import dataclasses
import logging
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Optional,
    Protocol,
    Sequence,
    Type,
    Union,
    cast,
)

if TYPE_CHECKING:
    from ib_tools.manager import InitData

import ib_insync as ibi

log = logging.getLogger(__name__)


ContractOrSequence = Union[Sequence[ibi.Contract], ibi.Contract]


class ContractManagingDescriptor:
    def __set_name__(self, obj: Atom, name: str) -> None:
        self.name = name

    def __set__(self, obj: Atom, value: ibi.Contract) -> None:
        obj.__dict__[self.name] = value
        self._register_contract(obj, value)

    def __get__(self, obj: Atom, type=None) -> Optional[ibi.Contract]:
        contract = obj.__dict__.get(self.name)
        if isinstance(contract, ibi.ContFuture):
            for i in obj.contracts:
                if isinstance(i, ibi.Future) and i.conId == contract.conId:
                    obj.__dict__[f"_{self.name}"] = i
                    return i
        return contract

    def _register_contract(self, obj: Atom, value: ContractOrSequence) -> None:
        if getattr(value, "__iter__", None):
            assert isinstance(value, collections.abc.Sequence)
            for v in value:
                obj.contracts.append(v)
        else:
            assert isinstance(value, ibi.Contract)
            obj.contracts.append(value)


class Atom:
    """
    Abstract base object from which all other objects inherit.

    """

    ib: ClassVar[ibi.IB]
    _contract_details: ClassVar[dict[ibi.Contract, ibi.ContractDetails]]
    _trading_hours: ClassVar[dict[ibi.Contract, list[tuple[datetime, datetime]]]]
    events: ClassVar[Sequence[str]] = ("startEvent", "dataEvent")

    contracts: list[ibi.Contract] = list()
    contract = cast(ibi.Contract, ContractManagingDescriptor())

    @classmethod
    def set_init_data(cls, data: InitData) -> None:
        cls.ib = data.ib
        cls._trading_hours = data.trading_hours
        cls._contract_details = data.contract_details

    def __init__(self) -> None:
        self._createEvents()
        self._log = logging.getLogger(f"strategy.{self.__class__.__name__}")

    @property
    def trading_hours(self):
        """
        If :attr:`contract` is set  ``trading_hours`` will be
        received only for this contract, otherwise
        :attr:`trading_hours` will return all available trading hours
        """
        try:
            th = Atom._trading_hours
        except AttributeError:
            log.warning("Trading hours not set.")
            return {}
        try:
            th_for_contract_or_none = th.get(self.contract)
        except AttributeError:  # attribute contract is not set
            return th

        if th_for_contract_or_none:
            return th_for_contract_or_none
        else:
            log.warning(
                f"{self.__class__.__name__} no trading hours data for {self.contract}."
            )
            return th

    def _createEvents(self):
        self.startEvent = ibi.Event("startEven")
        self.dataEvent = ibi.Event("dataEvent")
        self.feedbackEvent = ibi.Event("feedbackEvent")

    def _log_event_error(self, event: ibi.Event, exception: Exception) -> None:
        self._log.error(f"Event error {event.name()}: {exception}", exc_info=True)

    def onStart(self, data, *args):
        if isinstance(data, dict):
            for k, v in data.items():
                setattr(self, k, v)
        self.startEvent.emit(data, self)

    def onData(self, data, *args):
        data[f"{self.__class__.__name__}_ts"] = datetime.now(tz=timezone.utc)

    def onFeedback(self, data, *args):
        pass

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


class IsDataclass(Protocol):
    __dataclass_fields__: dict[str, dataclasses.Field]


class AtomDataclass(Atom, IsDataclass):
    pass


class Strategy:
    # It's not finished
    # Should kwargs be given as a tree?
    # if not what if kwargs are the same for multiple classes

    _strings: dict[str, list[str]] = {}
    _objects: dict[str, Type[AtomDataclass]]
    _pipe: Pipe

    @classmethod
    def fromAtoms(cls, *targets: Type[AtomDataclass]):
        for obj in targets:
            cls._strings[obj.__name__] = list(obj.__dataclass_fields__.keys())
            cls._objects[obj.__name__] = obj

    def __init__(self, **kwargs):
        self._pipe = self.pipe(self.instantiate(**kwargs))

    def instantiate(self, **kwargs):
        for name, obj in self.objects.items():
            return obj(**self.strings.get[name])

    def to_yaml(self):
        pass
