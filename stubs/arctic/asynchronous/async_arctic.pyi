from ..decorators import mongo_retry as mongo_retry
from ..exceptions import AsyncArcticException as AsyncArcticException
from ._workers_pool import LazySingletonTasksCoordinator as LazySingletonTasksCoordinator
from .async_utils import AsyncRequest as AsyncRequest, AsyncRequestType as AsyncRequestType
from _typeshed import Incomplete

def _arctic_task_exec(request): ...

class AsyncArctic(LazySingletonTasksCoordinator):
    _instance: Incomplete
    _SINGLETON_LOCK: Incomplete
    _POOL_LOCK: Incomplete
    requests_per_library: Incomplete
    requests_by_id: Incomplete
    local_shutdown: bool
    deferred_requests: Incomplete
    def __init__(self, pool_size): ...
    def __reduce__(self): ...
    def _get_modifiers(self, library_name, symbol: Incomplete | None = ...): ...
    def _get_accessors(self, library_name, symbol: Incomplete | None = ...): ...
    @staticmethod
    def _verify_request(store, is_modifier, **kwargs): ...
    def _is_clashing(self, request): ...
    def _add_request(self, request) -> None: ...
    def _remove_request(self, request) -> None: ...
    def _schedule_request(self, request): ...
    def submit_arctic_request(self, store, fun, is_modifier, *args, **kwargs): ...
    def _reschedule_deferred(self) -> None: ...
    def _request_finished(self, request) -> None: ...
    def reset(self, pool_size: Incomplete | None = ..., timeout: Incomplete | None = ...): ...
    def shutdown(self, timeout: Incomplete | None = ...) -> None: ...
    def await_termination(self, timeout: Incomplete | None = ...) -> None: ...
    def total_pending_requests(self): ...
    @staticmethod
    def _wait_until_scheduled(requests, timeout: Incomplete | None = ..., check_interval: float = ...): ...
    @staticmethod
    def wait_request(request, do_raise: bool = ..., timeout: Incomplete | None = ...) -> None: ...
    @staticmethod
    def wait_requests(requests, do_raise: bool = ..., timeout: Incomplete | None = ...) -> None: ...
    @staticmethod
    def wait_any_request(requests, do_raise: bool = ..., timeout: Incomplete | None = ...) -> None: ...
    @staticmethod
    def filter_finished_requests(requests, do_raise: bool = ...): ...
    @staticmethod
    def raise_first_errored(requests) -> None: ...
    @staticmethod
    def filter_errored(requests): ...

ASYNC_ARCTIC: Incomplete
async_arctic_submit: Incomplete
async_wait_request: Incomplete
async_wait_requests: Incomplete
async_shutdown: Incomplete
async_await_termination: Incomplete
async_reset_pool: Incomplete
async_total_requests: Incomplete