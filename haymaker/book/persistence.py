"""One ordered writer for all accounting state owners.

Return from an asynchronous save means queue acceptance, not database commit.
The first failed write halts dependent work. Startup reads and repairs happen
before enable_async(), and therefore finish before the runtime starts its loop.
"""

from datetime import datetime, timezone
from typing import Any

from ..async_wrappers import QueueProcessingError, QueueShutdownPolicy, SyncQueueRunner
from ..saver import AbstractBaseSaver

DEFAULT_ORDER_COLLECTION_NAME = "orders"
DEFAULT_STATE_COLLECTION_NAME = "state"


def utc_now() -> datetime:
    """Return an aware timestamp for an accounting state mutation."""
    return datetime.now(timezone.utc)


class PersistenceWriter:
    """Share fail-stop persistence mechanics without imposing collection semantics."""

    def __init__(self) -> None:
        """Begin in synchronous startup mode with no running worker."""
        self._queue: SyncQueueRunner | None = None
        self._failure: Exception | None = None

    def enable_async(self) -> None:
        """Use one critical queue after synchronous restoration is complete."""
        self.check_writable()
        if self._queue is not None:
            raise RuntimeError("Persistence writer is already asynchronous")
        self._queue = SyncQueueRunner(
            "Book", shutdown_policy=QueueShutdownPolicy.DRAIN, max_failures=1
        )

    def check_writable(self) -> None:
        """Reject work after an earlier persistence failure or queue shutdown."""
        if self._failure is not None:
            raise QueueProcessingError("Book persistence has halted") from self._failure
        if self._queue is not None:
            self._queue.check_accepting_work()

    def save(self, saver: AbstractBaseSaver, document: dict[str, Any]) -> None:
        """Write or enqueue an already serialized document in global call order."""
        self.check_writable()
        try:
            if self._queue is None:
                saver.save(document)
            else:
                self._queue.enqueue(saver.save, document)
        except Exception as exc:
            self._failure = exc
            raise

    async def close(self) -> None:
        """Drain accepted work and propagate any critical write failure."""
        if self._queue is not None:
            await self._queue.close()
        elif self._failure is not None:
            self.check_writable()
