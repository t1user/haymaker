"""Threaded delivery for configured logging handlers."""

from __future__ import annotations

import atexit
import logging
import logging.handlers
from copy import copy
from dataclasses import dataclass, field
from queue import SimpleQueue
from typing import Iterable


class LocalQueueHandler(logging.handlers.QueueHandler):
    """Enqueue an unformatted record copy for a listener thread."""

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        """Copy a record without formatting it on the calling thread."""

        return copy(record)


class SafeQueueListener(logging.handlers.QueueListener):
    """Keep delivering records after one destination handler failure."""

    def handle(self, record: logging.LogRecord) -> None:
        """Dispatch a record and report handler failures without stopping."""

        record = self.prepare(record)
        for handler in self.handlers:
            if self.respect_handler_level and record.levelno < handler.level:
                continue
            try:
                handler.handle(record)
            except Exception:
                handler.handleError(record)


@dataclass(frozen=True)
class _HandlerBinding:
    """Connect one source logger to one threaded destination handler."""

    logger: logging.Logger
    destination: logging.Handler
    queue_handler: LocalQueueHandler
    listener: logging.handlers.QueueListener


@dataclass
class LoggingQueueRuntime:
    """Own queue/listener threads installed around logging handlers."""

    bindings: list[_HandlerBinding] = field(default_factory=list)
    _stopped: bool = field(default=False, init=False)

    def stop(self) -> None:
        """Restore configured handlers after flushing their queued records."""

        if self._stopped:
            return
        self._stopped = True

        for binding in self.bindings:
            handlers = binding.logger.handlers
            if binding.queue_handler in handlers:
                index = handlers.index(binding.queue_handler)
                handlers[index] = binding.destination

        for binding in self.bindings:
            binding.listener.stop()
            binding.destination.flush()


_logging_runtime: LoggingQueueRuntime | None = None


def _loggers_with_handlers() -> list[logging.Logger]:
    """Return root and named loggers that own configured handlers."""

    root = logging.getLogger()
    named_loggers = [
        logger
        for logger in root.manager.loggerDict.values()
        if isinstance(logger, logging.Logger) and logger.handlers
    ]
    loggers = [root] if root.handlers else []
    return [*loggers, *sorted(named_loggers, key=lambda logger: logger.name)]


def setup_logging_queue(
    loggers: Iterable[logging.Logger] | None = None,
) -> LoggingQueueRuntime:
    """Move every configured destination handler onto its own listener thread.

    Args:
        loggers: Optional explicit source loggers. Production setup discovers
            every logger with directly configured handlers; tests may provide a
            narrower collection.

    Returns:
        Runtime owning the installed queue handlers and listener threads.
    """

    global _logging_runtime
    if _logging_runtime is not None:
        return _logging_runtime

    runtime = LoggingQueueRuntime()
    source_loggers = list(loggers) if loggers is not None else _loggers_with_handlers()
    for logger in source_loggers:
        destinations = list(logger.handlers)
        queued_handlers: list[logging.Handler] = []
        for destination in destinations:
            queue: SimpleQueue[logging.LogRecord] = SimpleQueue()
            queue_handler = LocalQueueHandler(queue)
            queue_handler.setLevel(destination.level)
            listener = SafeQueueListener(queue, destination, respect_handler_level=True)
            listener.start()
            runtime.bindings.append(
                _HandlerBinding(logger, destination, queue_handler, listener)
            )
            queued_handlers.append(queue_handler)
        logger.handlers = queued_handlers

    _logging_runtime = runtime
    return runtime


def shutdown_logging_queue() -> None:
    """Flush queued records and stop every logging listener thread."""

    global _logging_runtime
    if _logging_runtime is None:
        return
    _logging_runtime.stop()
    _logging_runtime = None


atexit.register(shutdown_logging_queue)
