from .._util import are_equals as are_equals
from ..decorators import _get_host as _get_host
from ..exceptions import ConcurrentModificationException as ConcurrentModificationException, NoDataFoundException as NoDataFoundException
from .versioned_item import ChangedItem as ChangedItem, VersionedItem as VersionedItem
from _typeshed import Incomplete

logger: Incomplete

class DataChange:
    date_range: Incomplete
    new_data: Incomplete
    def __init__(self, date_range, new_data) -> None: ...

class ArcticTransaction:
    _version_store: Incomplete
    _symbol: Incomplete
    _user: Incomplete
    _log: Incomplete
    _audit: Incomplete
    base_ts: Incomplete
    _do_write: bool
    def __init__(self, version_store, symbol, user, log, modify_timeseries: Incomplete | None = ..., audit: bool = ..., *args, **kwargs) -> None: ...
    def change(self, symbol, data_changes, **kwargs) -> None: ...
    _write: Incomplete
    def write(self, symbol, data, prune_previous_version: bool = ..., metadata: Incomplete | None = ..., **kwargs) -> None: ...
    def __enter__(self): ...
    def __exit__(self, *args, **kwargs) -> None: ...
