import asyncio
import logging

import pytest
from haymaker.async_wrappers import fire_and_forget, make_async


def sync_add(x: int, y: int) -> int:
    return x + y


def sync_raise() -> None:
    raise ValueError("boom")


@pytest.mark.asyncio
async def test_make_async_runs_function_and_returns_result():
    result = await make_async(sync_add, 2, 3)
    assert result == 5


@pytest.mark.asyncio
async def test_make_async_raises_typeerror_on_non_callable():
    with pytest.raises(TypeError, match="not callable"):
        await make_async(123)  # not a function


@pytest.mark.asyncio
async def test_fire_and_forget_schedules_task():
    results = []

    def sync_fn(x):
        results.append(x)

    fire_and_forget(sync_fn, "hello")

    # Give event loop a tick to execute the background task
    await asyncio.sleep(0.1)

    assert results == ["hello"]


@pytest.mark.asyncio
async def test_fire_and_forget_non_callable_raises():
    with pytest.raises(TypeError, match="not callable"):
        fire_and_forget(123)


@pytest.mark.asyncio
async def test_fire_and_forget_logs_exceptions(caplog):
    # Capture logs at the error level
    caplog.set_level(logging.ERROR)

    # Call fire_and_forget with the function that raises
    fire_and_forget(sync_raise)

    # Give the event loop a tick to run the task
    await asyncio.sleep(0.01)

    # There should be at least one ERROR log
    error_logs = [record for record in caplog.records if record.levelname == "ERROR"]
    assert error_logs, "Expected at least one ERROR log"

    # Check that one of them contains the text 'fire_and_forget'
    assert any(
        "fire_and_forget" in record.message for record in error_logs
    ), "Expected 'fire_and_forget' in the error log message"
