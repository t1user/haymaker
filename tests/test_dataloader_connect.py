import asyncio

import pytest

from haymaker.dataloader import connect


class FakeSupervisor:
    def __init__(self, *args):
        self.args = args
        self.stopped = False

    async def run(self):
        pass

    def stop(self):
        self.stopped = True


@pytest.fixture
def supervisor(monkeypatch):
    monkeypatch.setattr(connect, "ConnectionSupervisor", FakeSupervisor)


@pytest.mark.asyncio
async def test_reconnect_mode_runs_work_and_stops_supervisor(supervisor):
    runs = []

    async def run_work():
        runs.append("run")

    connection = connect.DataloaderConnection(object(), run_work)
    await connection.runtime.start()

    assert runs == ["run"]
    assert isinstance(connection.supervisor, FakeSupervisor)


@pytest.mark.asyncio
async def test_restart_resumes_by_starting_same_workload_again(supervisor):
    release = asyncio.Event()
    runs = []

    async def run_work():
        runs.append("run")
        await release.wait()

    connection = connect.DataloaderConnection(object(), run_work)
    first_start = asyncio.create_task(connection.runtime.start())
    while runs != ["run"]:
        await asyncio.sleep(0)
    await connection.runtime.stop("socket disconnected")
    await first_start

    release = asyncio.Event()
    second_start = asyncio.create_task(connection.runtime.start())
    while runs != ["run", "run"]:
        await asyncio.sleep(0)

    assert runs == ["run", "run"]
    release.set()
    await second_start


@pytest.mark.asyncio
async def test_reconnect_mode_runs_cleanup_before_restart(supervisor):
    release = asyncio.Event()
    cleanups = []

    async def run_work():
        await release.wait()

    connection = connect.DataloaderConnection(
        object(), run_work, lambda: cleanups.append("cleanup")
    )
    run_task = asyncio.create_task(connection.runtime.start())
    await asyncio.sleep(0)

    await connection.runtime.stop("socket disconnected")

    assert cleanups == ["cleanup"]
    release.set()
    await run_task


@pytest.mark.asyncio
async def test_workload_failure_propagates(supervisor):
    async def run_work():
        raise RuntimeError("broken workload")

    connection = connect.DataloaderConnection(object(), run_work)

    with pytest.raises(RuntimeError, match="broken workload"):
        await connection.runtime.start()


def test_watchdog_mode_is_rejected():
    with pytest.raises(ValueError, match="watchdog mode"):
        connect.connection(object(), lambda: None, run_mode="watchdog")
