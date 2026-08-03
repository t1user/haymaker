"""Configure logging and own its queue-listener lifecycle."""

from __future__ import annotations

import atexit
import asyncio
import logging
import logging.config
import logging.handlers
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import SimpleQueue
from typing import Any, Iterable

import yaml

from .handlers import LocalQueueHandler, SafeQueueListener

__all__ = [
    "LoggingQueueRuntime",
    "UTCFormatter",
    "setup_asyncio_logging",
    "setup_logging",
    "setup_logging_queue",
    "shutdown_logging_queue",
]

log = logging.getLogger(__name__)

module_directory = Path(__file__).parent

level = 5
logging.addLevelName(5, "DATA")
logging.addLevelName(60, "NOTIFY")

logger = logging.getLogger("haymaker")
logger.setLevel(level)


def asyncio_exception_handler(
    loop: asyncio.AbstractEventLoop, context: dict[str, Any]
) -> None:
    """Route an unhandled event-loop failure through Haymaker logging.

    Args:
        loop: Event loop reporting the failure.
        context: Exception context supplied by asyncio.
    """

    del loop
    exception = context.get("exception")
    if isinstance(exception, ConnectionError) and str(exception) == "Not connected":
        return

    message = context.get("message", "Unhandled asyncio exception")
    task = context.get("task") or context.get("future")
    if exception is None:
        log.error("%s; task=%r; context=%r", message, task, context)
    else:
        log.error("%s; task=%r", message, task, exc_info=exception)


class UTCFormatter(logging.Formatter):
    """Format logging timestamps in UTC."""

    converter = time.gmtime  # type: ignore[assignment]


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


def setup_asyncio_logging(loop: asyncio.AbstractEventLoop) -> None:
    """Route unhandled failures from the active loop through Haymaker logging."""

    loop.set_exception_handler(asyncio_exception_handler)


def setup_logging(
    config_file: str | Path | None = None,
    directory: str = "logs",
    base_directory: str = "ib_data",
    level: str | int | None = None,
) -> None:
    """Configure logging and move output handlers onto listener threads.

    Args:
        config_file: Logging dictionary configuration file.
        directory: Logging directory below ``base_directory``.
        base_directory: Application data directory below the user's home.
        level: Optional package log level override.
    """

    shutdown_logging_queue()
    if not config_file:
        config_file = module_directory / "logging_config.yaml"
    elif Path(config_file).is_file():
        pass
    elif (module_directory / config_file).exists():
        config_file = module_directory / config_file
    else:
        raise ValueError(f"Cannot open logging config file: {config_file}")

    try:
        with Path(config_file).open() as config_stream:
            config = yaml.safe_load(config_stream.read())
    except Exception as exc:
        logging.basicConfig()
        log.exception(exc)
        raise

    for handler in config.get("handlers", {}).values():
        factory = handler.get("()")
        if factory in {
            "haymaker.logging.handlers.file_handler_setup",
            "haymaker.logging.handlers.timed_rotating_file_handler_setup",
        }:
            handler["_haymaker_base_directory"] = base_directory
            handler["_haymaker_logging_directory"] = directory

    logging.config.dictConfig(config)

    if isinstance(level, str):
        level = logging._nameToLevel.get(level.upper(), 0)
    if level:
        logging.getLogger("haymaker").setLevel(level)
    setup_logging_queue()


atexit.register(shutdown_logging_queue)
