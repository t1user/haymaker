import asyncio

import pytest

from haymaker.dataloader import connect


@pytest.mark.asyncio
async def test_runtime_runs_work():
    runs = []

    async def run_work():
        runs.append("run")

    runtime = connect.DataloaderRuntime(object(), run_work)
    await runtime.start()

    assert runs == ["run"]


@pytest.mark.asyncio
async def test_restart_resumes_by_starting_same_workload_again():
    release = asyncio.Event()
    runs = []

    async def run_work():
        runs.append("run")
        await release.wait()

    runtime = connect.DataloaderRuntime(object(), run_work)
    first_start = asyncio.create_task(runtime.start())
    while runs != ["run"]:
        await asyncio.sleep(0)
    await runtime.stop("socket disconnected")
    await first_start

    release = asyncio.Event()
    second_start = asyncio.create_task(runtime.start())
    while runs != ["run", "run"]:
        await asyncio.sleep(0)

    assert runs == ["run", "run"]
    release.set()
    await second_start


@pytest.mark.asyncio
async def test_runtime_runs_cleanup_before_restart():
    release = asyncio.Event()
    cleanups = []

    async def run_work():
        await release.wait()

    runtime = connect.DataloaderRuntime(
        object(), run_work, lambda: cleanups.append("cleanup")
    )
    run_task = asyncio.create_task(runtime.start())
    await asyncio.sleep(0)

    await runtime.stop("socket disconnected")

    assert cleanups == ["cleanup"]
    release.set()
    await run_task


@pytest.mark.asyncio
async def test_workload_failure_propagates():
    async def run_work():
        raise RuntimeError("broken workload")

    runtime = connect.DataloaderRuntime(object(), run_work)

    with pytest.raises(RuntimeError, match="broken workload"):
        await runtime.start()


def test_connection_module_exposes_runtime_only():
    """Dataloader connection handling should use the shared application."""

    assert not hasattr(connect, "DataloaderConnection")


async def test_runtime_close_stops_active_work():
    """Final runtime cleanup should collect any active dataloader work."""

    release = asyncio.Event()

    async def run_work():
        await release.wait()

    runtime = connect.DataloaderRuntime(object(), run_work)
    run_task = asyncio.create_task(runtime.start())
    await asyncio.sleep(0)

    await runtime.close()
    await run_task

    assert runtime._work_task is not None
    assert runtime._work_task.done()
