import asyncio
import logging
import traceback

log = logging.getLogger(__name__)


def asyncio_exception_handler(
    loop: asyncio.AbstractEventLoop, context: dict, logger=None
) -> None:
    """
    Handle exceptions from asyncio tasks.

    Context dict may contain:
    - 'message': Error message
    - 'exception': The exception object
    - 'future' or 'task': The task/future that failed
    - 'source_traceback': Where the task was created
    - 'handle': The handle being executed
    """

    if logger is None:
        logger = log

    # Get the exception
    exception = context.get("exception")
    message = context.get("message", "Unhandled exception in async task")

    # Get the task
    task = context.get("task") or context.get("future")

    # Build detailed error message
    error_parts = [f"Asyncio exception: {message}"]

    if task:
        task_name = getattr(task, "get_name", lambda: "unnamed")()
        error_parts.append(f"Task: {task_name}")
        error_parts.append(f"Task repr: {task!r}")

    # Get source traceback (where task was created)
    source_traceback = context.get("source_traceback")
    if source_traceback:
        error_parts.append("\nTask was created at:")
        error_parts.append("".join(traceback.format_list(source_traceback)))

    # Log with full exception info
    if exception:
        error_parts.append(f"\nException: {exception!r}")
        logger.error("\n".join(error_parts), exc_info=exception)
    else:
        # No exception object, just dump the whole context
        error_parts.append(f"\nFull context: {context}")
        logger.error("\n".join(error_parts))


def setup_asyncio_logger() -> None:
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(asyncio_exception_handler)
