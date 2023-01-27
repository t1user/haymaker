from abc import ABC, abstractmethod

from ib_insync import Event
from logbook import Logger  # type: ignore

log = Logger(__name__)


class Atom(ABC):

    """
    Abstract base object from which all other objects inherit.

    """

    def __init__(self):
        self._createEvents()

    def _createEvents(self):
        self.startEvent = Event("startEvent")
        self.dataEvent = Event("dataEvent")

    @abstractmethod
    def onStart(self, data, source: "Atom") -> None:
        pass

    @abstractmethod
    def onData(self, data, source: "Atom") -> None:
        pass

    def connect(self, *targets) -> "Atom":
        for t in targets:
            self.startEvent.connect(t.onStart, keep_ref=True)
            self.dataEvent.connect(t.onData, keep_ref=True)
        return self

    def disconnect(self, *targets) -> "Atom":
        for t in targets:
            self.startEvent.disconnect(t.onStart)
            self.dataEvent.connect(t.onData)
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
            (f"{i}={j}" for i, j in self.__dict__.items() if "Event" not in str(i))
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
        self.last.startEvent.disconnect(other.onStart)
        self.last.dataEvent.disconnect(other.onData)
        return self

    def onStart(self, data, source: Atom) -> None:
        self.first.onStart(data, source)

    def onData(self, data, source: Atom) -> None:
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
