import asyncio
import logging
from typing import Any, Callable, Coroutine, TypeVar

log = logging.getLogger(__name__)


R = TypeVar("R")

# Registry for active tasks
_tasks: set[asyncio.Task] = set()


async def _async_runner(func: Callable[..., R], *args: Any) -> R:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, func, *args)


def _create_task(coroutine: Coroutine[Any, Any, R], *, name: str | None = None) -> None:
    task: asyncio.Task[R] = asyncio.create_task(coroutine, name=name)

    # Below is preventing tasks from being prematurely garbage collected
    _tasks.add(task)

    def _on_done(t: asyncio.Task[R]) -> None:
        try:
            exc = t.exception()
            if exc:
                log.error(f"fire_and_forget error: {exc}")
        finally:
            _tasks.discard(t)

    task.add_done_callback(_on_done)


async def make_async(fn: Callable[..., R], *args) -> R:
    # can be used to make any callable async
    if not callable(fn):
        raise TypeError(f"{fn} is not callable")
    return await _async_runner(fn, *args)


def fire_and_forget(
    fn: Callable[..., Any], *args: Any, name: str | None = None
) -> None:
    # can be used on any callable that doesn't expect a return value
    if not callable(fn):
        raise TypeError(f"{fn} is not callable")
    _create_task(_async_runner(fn, *args), name=name)
