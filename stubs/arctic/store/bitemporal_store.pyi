from _typeshed import Incomplete
from arctic.date._mktz import mktz as mktz
from arctic.multi_index import groupby_asof as groupby_asof
from typing import NamedTuple

class BitemporalItem(NamedTuple):
    symbol: Incomplete
    library: Incomplete
    data: Incomplete
    metadata: Incomplete
    last_updated: Incomplete

class BitemporalStore:
    _store: Incomplete
    observe_column: Incomplete
    def __init__(self, version_store, observe_column: str = ...) -> None: ...
    def read(self, symbol, as_of: Incomplete | None = ..., raw: bool = ..., **kwargs): ...
    def update(self, symbol, data, metadata: Incomplete | None = ..., upsert: bool = ..., as_of: Incomplete | None = ..., **kwargs) -> None: ...
    def write(self, *args, **kwargs) -> None: ...
    def _add_observe_dt_index(self, df, as_of): ...
