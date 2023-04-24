import dataclasses
from abc import ABC, abstractmethod
from typing import ClassVar, Dict, List, Optional, Protocol, Type

import ib_insync as ibi
from logbook import Logger  # type: ignore

log = Logger(__name__)


class NewEvent(ibi.Event):
    """
    Modified eventkit Event, that saves the object originating the Event and attaches it
    to every emission.
    """

    def __init__(
        self, name: str = "", _with_error_done_events: bool = True, *, source_object
    ):
        self.source_object = source_object
        super().__init__(name="", _with_error_done_events=True)

    def emit(self, *args):
        super().emit(*args, self.source_object)


class Atom:
    """
    Abstract base object from which all other objects inherit.

    """

    ib: ClassVar[ibi.IB]

    @classmethod
    def set_ib(cls, ib: ibi.IB) -> None:
        cls.ib = ib

    def __init__(self, wait_for_onStart: bool = False) -> None:
        self._createEvents()

    def _createEvents(self):
        self.startEvent = NewEvent("startEvent", source_object=self)
        self.dataEvent = NewEvent("dataEvent", source_object=self)

    def onStart(self, data, source: Optional["Atom"] = None) -> None:
        pass

    def onData(self, data, source: Optional["Atom"]) -> None:
        pass

    def connect(self, *targets) -> "Atom":
        for t in targets:
            self.startEvent.disconnect_obj(t)
            self.startEvent.connect(t.onStart, keep_ref=True)
            self.dataEvent.disconnect_obj(t)
            self.dataEvent.connect(t.onData, keep_ref=True)
        return self

    def disconnect(self, *targets) -> "Atom":
        for t in targets:
            self.startEvent.disconnect_obj(t)
            self.dataEvent.disconnect_obj(t)
        return self

    def clear(self):
        self.startEvent.clear()
        self.dataEvent.clear()

    def pipe(self, *targets: "Atom") -> "Pipe":
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
                if "Event" not in str(i)
                if str(i) != "_wait_for_onStart"
                if str(i) != "buffer"
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

    def connect(self, other) -> "Pipe":  # type: ignore
        self.last.startEvent.connect(other.onStart, keep_ref=True)
        self.last.dataEvent.connect(other.onData, keep_ref=True)
        return self

    def disconnect(self, other) -> "Pipe":  # type: ignore
        self.last.startEvent.disconnect_obj(other)
        self.last.dataEvent.disconnect_obj(other)
        return self

    def onStart(self, data, source: Optional[Atom] = None) -> None:
        self.first.onStart(data, source)

    def onData(self, data, source: Optional[Atom] = None) -> None:
        self.first.onData(data, source)

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
    __dataclass_fields__: Dict[str, dataclasses.Field]


class AtomDataclass(Atom, IsDataclass):
    pass


class Strategy:
    # It's not finished
    # Should kwargs be given as a tree?
    # if not what if kwargs are the same for multiple classes

    _strings: Dict[str, List[str]] = {}
    _objects: Dict[str, Type[AtomDataclass]]
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