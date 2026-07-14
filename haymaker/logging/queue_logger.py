"""
Source: https://www.zopatista.com/python/2019/05/11/asyncio-logging/
"""

import asyncio
import logging
import logging.handlers
from queue import SimpleQueue
from typing import List

_listener: logging.handlers.QueueListener | None = None


class LocalQueueHandler(logging.handlers.QueueHandler):
    def emit(self, record: logging.LogRecord) -> None:
        # Removed the call to self.prepare(), handle task cancellation
        try:
            self.enqueue(record)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.handleError(record)


def setup_logging_queue() -> None:
    """Move log handlers to a separate thread.

    Replace handlers on the root logger with a LocalQueueHandler,
    and start a logging.QueueListener holding the original
    handlers.

    """
    global _listener
    if _listener is not None:
        return

    queue: SimpleQueue[logging.LogRecord] = SimpleQueue()
    root = logging.getLogger()

    handlers: List[logging.Handler] = []

    handler = LocalQueueHandler(queue)
    root.addHandler(handler)
    for h in root.handlers[:]:
        if h is not handler:
            root.removeHandler(h)
            handlers.append(h)

    _listener = logging.handlers.QueueListener(
        queue, *handlers, respect_handler_level=True
    )
    _listener.start()


def shutdown_logging_queue() -> None:
    """Flush queued log records and stop the logging listener thread."""
    global _listener
    if _listener is None:
        return
    _listener.stop()
    _listener = None
