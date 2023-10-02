from __future__ import annotations

import dataclasses
import logging
from typing import Any, ClassVar, Coroutine, Protocol, Sequence, Type, Union

import ib_insync as ibi

log = logging.getLogger(__name__)


ContractOrSequence = Union[Sequence[ibi.Contract], ibi.Contract]
CONTRACT_LIST: list[ibi.Contract] = list()


class Atom:
    """
    Abstract base object from which all other objects inherit.

    """

    ib: ClassVar[ibi.IB]
    _contract: ContractOrSequence
    events: ClassVar[Sequence[str]] = ("startEvent", "dataEvent")

    contracts = CONTRACT_LIST

    @classmethod
    def set_ib(cls, ib: ibi.IB) -> None:
        cls.ib = ib

    def __init__(self) -> None:
        self._createEvents()
        self._log = logging.getLogger(f"{self.__module__}.{self.__class__.__name__}")

    def __setattr__(self, prop, val):
        if prop == "contract":
            self._register_contract(val)
        super().__setattr__(prop, val)

    def _register_contract(self, value) -> None:
        if getattr(value, "__iter__", None):
            for v in value:
                self.contracts.append(v)
        else:
            self.contracts.append(value)

    def _createEvents(self):
        self.startEvent = ibi.Event("startEven")
        self.dataEvent = ibi.Event("dataEvent")

    def _log_event_error(self, event: ibi.Event, exception: Exception) -> None:
        self._log.error(f"Event error {event.name()}: {exception}", exc_info=True)

    def onStart(self, data, *args) -> None:
        if isinstance(data, dict):
            for k, v in data.items():
                setattr(self, k, v)
            self.startEvent.emit(data, self)

    def onData(self, data, *args) -> Union[Coroutine[Any, Any, None], None]:
        pass

    def connect(self, *targets: Atom) -> "Atom":
        for t in targets:
            self.startEvent.disconnect_obj(t)
            self.startEvent.connect(
                t.onStart, error=self._log_event_error, keep_ref=True
            )
            self.dataEvent.disconnect_obj(t)
            self.dataEvent.connect(t.onData, error=self._log_event_error, keep_ref=True)
        return self

    def disconnect(self, *targets) -> "Atom":
        for t in targets:
            # the same target cannot be connected more than once
            self.startEvent.disconnect_obj(t)
            self.dataEvent.disconnect_obj(t)
        return self

    def clear(self):
        self.startEvent.clear()
        self.dataEvent.clear()

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
                if "Event" not in str(i) and i != "_log"
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

    def connect(self, *targets: Atom) -> "Pipe":
        for target in targets:
            self.last.connect(target)
        return self

    def disconnect(self, *targets: Atom) -> "Pipe":
        for target in targets:
            self.last.startEvent.disconnect_obj(target)
            self.last.dataEvent.disconnect_obj(target)
        return self

    def onStart(self, data, *args) -> None:
        self.first.onStart(data, *args)

    def onData(self, data, *args) -> None:
        self.first.onData(data, *args)

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
