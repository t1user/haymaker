import asyncio
import itertools
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Coroutine, Generic, TypeVar

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
    """
    Schedule sync callable to run asynchronously in a different thread
    in executor.
    """
    # can be used on any callable that doesn't expect a return value
    if not callable(fn):
        raise TypeError(f"{fn} is not callable")
    _create_task(_async_runner(fn, *args), name=name)


T = TypeVar("T")


@dataclass
class QueueRunner(Generic[T]):
    """
    An asynchronous task queue with an automated worker lifecycle.

    This class manages an internal asyncio.Queue and a background worker task
    that processes items as they arrive. It provides backpressure support via
    maxsize and ensures that the worker remains active as long as there are
    tasks to process.

    Attrs:
    ------
        processing_func: An awaitable callable that accepts a single argument of
            type T. This is the logic executed for every item in the queue.
        owner: A unique identifier for the queue instance. If not provided,
            it is automatically generated using a global counter.
        maxsize: The maximum number of items allowed in the queue. If 0 (default),
            the queue size is infinite.
        max_failures: The number of consecutive failures after which the queue will
            crash
        _queue: The underlying asyncio.Queue instance.
        _worker_task: The asyncio.Task handle for the background consumer loop.
        _counter: A thread-safe class-level iterator for generating default owners.

    Methods:
    --------
        await put - add data to the queue and wait if queue full, can be used only
              in async functions
        push - just add data to the queue, make sure it never gets full, can be
              used in sync functions

    Example:
    -------
        async def my_processor(data: int):
            await asyncio.sleep(1)
            print(f"Processed {data}")

        q = Queue(processing_func=my_processor, maxsize=10)
        await q.put(42)

        or

        q.push(42)
    """

    processing_func: Callable[[T], Coroutine[Any, Any, None]]
    owner: str = ""
    maxsize: int = 0
    max_failures = 3

    _queue: asyncio.Queue[T] = field(repr=False, init=False)
    _worker_task: asyncio.Task | None = field(repr=False, init=False, default=None)
    _counter: ClassVar[itertools.count] = itertools.count()

    def __post_init__(self) -> None:
        self._queue = asyncio.Queue(maxsize=self.maxsize)
        if not self.owner:
            self.owner = str(next(self._counter))

    def _ensure_running_task(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(
                self._process_queue(), name=f"{self!s}"
            )

    async def put(self, data: T) -> None:
        """Adds data to the queue. Awaits if maxsize is reached."""
        await self._queue.put(data)
        self._ensure_running_task()

    def push(self, data: T) -> None:
        """Synchronous method to add data to the queue."""
        self._queue.put_nowait(data)
        self._ensure_running_task()

    async def _process_queue(self) -> None:
        consecutive_failures = 0

        try:
            while True:
                data = await self._queue.get()
                try:
                    await self.processing_func(data)
                    consecutive_failures = 0  # Reset on success
                except Exception as e:
                    consecutive_failures += 1
                    log.error(
                        f"{self!s} failed "
                        f"({consecutive_failures}/{self.max_failures}): {e}"
                    )

                    if consecutive_failures >= self.max_failures:
                        log.critical(f"{self!s} halting due to repeated failures.")
                        break  # Kill the worker task
                finally:
                    self._queue.task_done()
        except asyncio.CancelledError:
            pass

    async def close(self) -> None:
        """Wait for remaining tasks and cancel the worker."""
        await self._queue.join()
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    def qsize(self) -> int:
        return self._queue.qsize()

    def __str__(self) -> str:
        return f"<{self.__class__.__name__}-{self.owner}>"
