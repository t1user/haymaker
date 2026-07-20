from __future__ import annotations

import asyncio
import itertools
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Coroutine, Generic, TypeAlias, TypeVar

log = logging.getLogger(__name__)


T = TypeVar("T")
QUEUE_SHUTDOWN_TIMEOUT = 10.0


class QueueShutdownPolicy(Enum):
    """Determine queue failure and pending-item behavior at shutdown."""

    DRAIN = "drain"
    DISCARD = "discard"


class QueueProcessingError(RuntimeError):
    """Raised when a draining queue failed to process an item."""


class QueueDrainTimeoutError(TimeoutError):
    """Raised when a draining queue cannot finish within its shutdown timeout."""


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
        shutdown_policy: Whether final work is drained as critical or discarded
            as best effort. Draining queues propagate processing and timeout
            failures; discarding queues only log processing failures.
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
    max_failures: int = 3
    shutdown_policy: QueueShutdownPolicy = QueueShutdownPolicy.DRAIN

    _queue: asyncio.Queue[T] = field(repr=False, init=False)
    _worker_task: asyncio.Task | None = field(repr=False, init=False, default=None)
    _closed: bool = field(repr=False, init=False, default=False)
    _processing_error: Exception | None = field(repr=False, init=False, default=None)
    _fatal_error: Exception | None = field(repr=False, init=False, default=None)
    _counter: ClassVar[itertools.count] = itertools.count()
    _instances: ClassVar[list[QueueRunner]] = []

    def __post_init__(self) -> None:
        self._queue = asyncio.Queue(maxsize=self.maxsize)
        if not self.owner:
            self.owner = str(next(self._counter))
        self.__class__._instances.append(self)

    def _ensure_accepting_work(self) -> None:
        """Raise when this queue can no longer accept work."""

        if self._closed:
            raise RuntimeError(f"{self!s} is closed")
        if (
            self.shutdown_policy is QueueShutdownPolicy.DRAIN
            and self._fatal_error is not None
        ):
            raise QueueProcessingError(f"{self!s} processing has halted") from (
                self._fatal_error
            )

    def _start_worker(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(
                self._process_queue(), name=f"{self!s}"
            )

    async def put(self, data: T) -> None:
        """Adds data to the queue. Awaits if maxsize is reached."""
        self._ensure_accepting_work()
        await self._queue.put(data)
        self._start_worker()

    def push(self, data: T) -> None:
        """Synchronous method to add data to the queue."""
        self._ensure_accepting_work()
        self._queue.put_nowait(data)
        self._start_worker()

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
                    if self._processing_error is None:
                        self._processing_error = e
                    log.exception(
                        "%s failed (%d/%d).",
                        self,
                        consecutive_failures,
                        self.max_failures,
                    )

                    if consecutive_failures >= self.max_failures:
                        self._fatal_error = e
                        log.critical(f"{self!s} halting due to repeated failures.")
                        break  # Kill the worker task
                finally:
                    self._queue.task_done()
        except asyncio.CancelledError:
            log.debug(f"{self!s} cancelled.")

    async def close(self, timeout: float = QUEUE_SHUTDOWN_TIMEOUT) -> None:
        """Apply the queue's shutdown policy and stop its worker."""
        if self._closed:
            return
        self._closed = True

        drain_timeout: QueueDrainTimeoutError | None = None
        try:
            if self.shutdown_policy is QueueShutdownPolicy.DRAIN:
                if (
                    self._fatal_error is None
                    and self._queue.qsize()
                    and (self._worker_task is None or self._worker_task.done())
                ):
                    self._start_worker()
                if self._fatal_error is None:
                    try:
                        await asyncio.wait_for(self._queue.join(), timeout)
                    except TimeoutError:
                        drain_timeout = QueueDrainTimeoutError(
                            f"{self!s} did not drain within {timeout:.1f}s"
                        )
                        log.exception(
                            "%s; discarding %d pending items.",
                            drain_timeout,
                            self._queue.qsize(),
                        )

            self._discard_pending()
            if self._worker_task:
                self._worker_task.cancel()
                await asyncio.gather(self._worker_task, return_exceptions=True)
        finally:
            if self in self.__class__._instances:
                self.__class__._instances.remove(self)

        if self.shutdown_policy is QueueShutdownPolicy.DRAIN:
            if drain_timeout is not None:
                raise drain_timeout
            if self._processing_error is not None:
                raise QueueProcessingError(
                    f"{self!s} failed to process queued work"
                ) from self._processing_error

    def _discard_pending(self) -> None:
        """Remove queued items without processing them."""
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            else:
                self._queue.task_done()

    @classmethod
    async def close_all(cls) -> None:
        """
        Can be used system-level to ensure all queues have a chance to
        complete their tasks.
        """
        tasks = [qr.close() for qr in list(cls._instances)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        failures = [result for result in results if isinstance(result, Exception)]
        if failures:
            raise ExceptionGroup("Queue shutdown failed", failures)

    def qsize(self) -> int:
        return self._queue.qsize()

    def __str__(self) -> str:
        return f"<{self.__class__.__name__}-{self.owner}>"


_SyncTask: TypeAlias = tuple[Callable[..., None], tuple[Any, ...]]


@dataclass
class SyncQueueRunner:
    """
    Executes synchronous I/O-bound callables sequentially in a background thread,
    preserving enqueue order. Useful for offloading blocking I/O (e.g. database
    writes) from an async context without blocking the event loop.

    Internally wraps QueueRunner[_SyncTask], delegating execution to
    asyncio.to_thread for each enqueued callable.

    Attrs:
    ------
        owner: Identifier for this queue instance, used in logs and task names.
            Defaults to an auto-incremented integer string.
        maxsize: Maximum number of items allowed in the queue. 0 (default) means
            unlimited. Use to apply backpressure when the consumer is slower than
            the producer.
        max_failures: Number of consecutive failures after which the background
            worker halts. Defaults to 3.
        shutdown_policy: Whether final work is drained as critical or discarded
            as best effort.

    Methods:
    --------
        enqueue(fn, *args) - Schedule fn(*args) to run in a background thread.
            Safe to call from synchronous code. Preserves call order.
        close() - Apply the shutdown policy and stop the background worker. Must
            be awaited during application shutdown.

    Example:
    --------
        def save_to_db(record: dict) -> None:
            db.insert(record)

        queue = SyncQueueRunner(owner="db_writer", maxsize=100)
        queue.enqueue(save_to_db, {"id": 1, "value": "foo"})
        queue.enqueue(save_to_db, {"id": 2, "value": "bar"})
    """

    owner: str = ""
    maxsize: int = 0
    max_failures: int = 3
    shutdown_policy: QueueShutdownPolicy = QueueShutdownPolicy.DRAIN

    _queue_runner: QueueRunner[_SyncTask] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._queue_runner = QueueRunner(
            self._execute_store_task,
            self.owner,
            self.maxsize,
            self.max_failures,
            self.shutdown_policy,
        )

    def enqueue(self, fn: Callable[..., None], *args: Any) -> None:
        self._queue_runner.push((fn, args))

    async def _execute_store_task(self, data: _SyncTask) -> None:
        func, args = data
        await asyncio.to_thread(func, *args)

    async def close(self) -> None:
        await self._queue_runner.close()


R = TypeVar("R")

# Registry for active tasks
_tasks: set[asyncio.Task] = set()
_queue = SyncQueueRunner(
    "async_wrappers_queue", shutdown_policy=QueueShutdownPolicy.DISCARD
)


async def _async_runner(func: Callable[..., R], *args: Any) -> R:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, func, *args)


def create_background_task(
    coroutine: Coroutine[Any, Any, R], *, name: str | None = None
) -> asyncio.Task[R]:
    """Create and retain a detached task until it has completed."""
    task: asyncio.Task[R] = asyncio.create_task(coroutine, name=name)

    # Below is preventing tasks from being prematurely garbage collected
    _tasks.add(task)

    def _on_done(t: asyncio.Task[R]) -> None:
        try:
            if t.cancelled():
                return
            exc = t.exception()
            if exc:
                log.error("Background task failed: %s", exc)
        finally:
            _tasks.discard(t)

    task.add_done_callback(_on_done)
    return task


async def cancel_background_tasks() -> None:
    """Cancel and collect all detached framework background tasks."""
    tasks = [task for task in _tasks if not task.done()]
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def make_async(fn: Callable[..., R], *args) -> R:
    # can be used to make any callable async
    if not callable(fn):
        raise TypeError(f"{fn} is not callable")
    return await _async_runner(fn, *args)


async def finish_on_cancel(operation: Coroutine[Any, Any, R]) -> R:
    """Finish a started operation before propagating caller cancellation.

    Args:
        operation: Coroutine whose side effects must settle once scheduled.

    Returns:
        The operation result when the caller is not cancelled.

    Raises:
        asyncio.CancelledError: After a successful operation when the caller was
            cancelled while waiting.
        Exception: The operation failure, including when cancellation was
            already pending.
    """

    task = asyncio.create_task(operation)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


def fire_and_forget(fn: Callable[..., Any], *args: Any) -> None:
    """
    Schedule sync callable to run asynchronously in a different thread
    in executor.
    """
    # can be used on any callable that doesn't expect a return value
    if not callable(fn):
        raise TypeError(f"{fn} is not callable")
    _queue.enqueue(fn, *args)
