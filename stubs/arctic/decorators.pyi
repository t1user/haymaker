from _typeshed import Incomplete

logger: Incomplete
_MAX_RETRIES: int

def _get_host(store): ...

_in_retry: bool
_retry_count: int

def mongo_retry(f): ...
def _handle_error(f, e, retry_count, **kwargs) -> None: ...
