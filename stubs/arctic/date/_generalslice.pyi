from _typeshed import Incomplete
from enum import Enum

class Intervals(Enum):
    OPEN_CLOSED: Incomplete
    CLOSED_OPEN: Incomplete
    OPEN_OPEN: Incomplete
    CLOSED_CLOSED: Incomplete

OPEN_CLOSED: Incomplete
CLOSED_OPEN: Incomplete
OPEN_OPEN: Incomplete
CLOSED_CLOSED: Incomplete

INTERVALS: Incomplete

class GeneralSlice:
    start: Incomplete
    end: Incomplete
    step: Incomplete
    interval: Incomplete
    def __init__(self, start, end, step: Incomplete | None = ..., interval=...) -> None: ...
    @property
    def startopen(self): ...
    @property
    def endopen(self): ...
