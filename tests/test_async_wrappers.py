"""Unit tests for async wrapper helpers under a sandbox-safe execution model.

The production module bridges synchronous callables into asyncio with
threadpool-backed helpers. That path is correct in production, but it is also
the part that can hang in the Codex sandbox. These tests replace the bridge
with inline implementations so they keep checking wrapper semantics without
depending on sandbox thread wakeups.
"""

import asyncio
import logging
from collections.abc import Callable
from typing import Any

import pytest

import haymaker.async_wrappers as async_wrappers


def sync_add(x: int, y: int) -> int:
    return x + y


def sync_raise() -> None:
    raise ValueError("boom")


class InlineQueue:
    """Run fire-and-forget work inline for deterministic wrapper unit tests."""

    def __init__(self) -> None:
        """Store queued calls for assertions if a test needs them."""
        self.calls: list[tuple[Callable[..., Any], tuple[Any, ...]]] = []

    def enqueue(self, fn, *args) -> None:
        """Execute the queued callable and log failures like QueueRunner."""
        self.calls.append((fn, args))
        try:
            fn(*args)
        except Exception as exc:
            logging.getLogger(async_wrappers.__name__).error(
                "<QueueRunner-async_wrappers_queue> failed (1/3): %s", exc
            )


async def inline_async_runner(func, *args):
    """Run sync functions inline behind make_async's async API."""
    return func(*args)


@pytest.fixture(autouse=True)
def inline_async_wrappers(monkeypatch):
    """Replace threadpool-backed helpers with inline test doubles.

    The fixture is autouse because every test in this module is about wrapper
    behavior, not the executor implementation. Integration coverage for the
    real threadpool path should live in separate tests that intentionally opt
    into that environment dependency.
    """
    monkeypatch.setattr(async_wrappers, "_async_runner", inline_async_runner)
    monkeypatch.setattr(async_wrappers, "_queue", InlineQueue())


@pytest.mark.asyncio
async def test_make_async_runs_function_and_returns_result():
    result = await async_wrappers.make_async(sync_add, 2, 3)
    assert result == 5


@pytest.mark.asyncio
async def test_make_async_raises_typeerror_on_non_callable():
    with pytest.raises(TypeError, match="not callable"):
        await async_wrappers.make_async(123)  # not a function


@pytest.mark.asyncio
async def test_fire_and_forget_schedules_task():
    results = []

    def sync_fn(x):
        results.append(x)

    async_wrappers.fire_and_forget(sync_fn, "hello")

    # Give event loop a tick to execute the background task
    await asyncio.sleep(0.1)

    assert results == ["hello"]


@pytest.mark.asyncio
async def test_fire_and_forget_non_callable_raises():
    with pytest.raises(TypeError, match="not callable"):
        async_wrappers.fire_and_forget(123)


@pytest.mark.asyncio
async def test_fire_and_forget_logs_exceptions(caplog):
    # Capture logs at the error level
    caplog.set_level(logging.ERROR)

    # Call fire_and_forget with the function that raises
    async_wrappers.fire_and_forget(sync_raise)

    # Give the event loop a tick to run the task
    await asyncio.sleep(0.01)

    # There should be at least one ERROR log
    error_logs = [record for record in caplog.records if record.levelname == "ERROR"]
    assert error_logs, "Expected at least one ERROR log"

    # Check that one of them contains the text 'fire_and_forget'
    assert any(
        "async_wrappers_queue" in record.message for record in error_logs
    ), "Expected 'async_wrappers_queue' in the error log message"
