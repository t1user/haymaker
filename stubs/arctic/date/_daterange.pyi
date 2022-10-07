from ._generalslice import CLOSED_CLOSED as CLOSED_CLOSED, CLOSED_OPEN as CLOSED_OPEN, GeneralSlice as GeneralSlice, OPEN_CLOSED as OPEN_CLOSED, OPEN_OPEN as OPEN_OPEN
from ._parse import parse as parse
from _typeshed import Incomplete

INTERVAL_LOOKUP: Incomplete

class DateRange(GeneralSlice):
    def __init__(self, start: Incomplete | None = ..., end: Incomplete | None = ..., interval=...): ...
    @property
    def unbounded(self): ...
    def intersection(self, other): ...
    def as_dates(self): ...
    def mongo_query(self): ...
    def get_date_bounds(self): ...
    def __contains__(self, d): ...
    def __repr__(self): ...
    def __eq__(self, rhs): ...
    def __lt__(self, other): ...
    def __hash__(self): ...
    def __getitem__(self, key): ...
    def __str__(self): ...
    start: Incomplete
    end: Incomplete
    interval: Incomplete
    step: int
    def __setstate__(self, state) -> None: ...