from _typeshed import Incomplete
from arctic.exceptions import RequestDurationException as RequestDurationException
from enum import Enum

class AsyncRequestType(Enum):
    MODIFIER: str
    ACCESSOR: str

class AsyncRequest:
    id: Incomplete
    fun: Incomplete
    args: Incomplete
    kwargs: Incomplete
    kind: Incomplete
    library: Incomplete
    symbol: Incomplete
    future: Incomplete
    callback: Incomplete
    data: Incomplete
    exception: Incomplete
    is_running: bool
    is_completed: bool
    start_time: Incomplete
    end_time: Incomplete
    create_time: Incomplete
    mongo_retry: Incomplete
    def __init__(self, kind, library, fun, callback, *args, **kwargs) -> None: ...
    @property
    def execution_duration(self): ...
    @property
    def schedule_delay(self): ...
    @property
    def total_time(self): ...
    def __str__(self): ...
