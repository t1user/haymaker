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
async def test_wait_mode_leaves_existing_work_running_after_reconnect(supervisor):
    release = asyncio.Event()
    runs = []

    async def run_work():
        runs.append("run")
        await release.wait()

    connection = connect.DataloaderConnection(object(), run_work, run_mode="wait")
    first_start = asyncio.create_task(connection.runtime.start())
    await asyncio.sleep(0)
    second_start = asyncio.create_task(connection.runtime.start())
    await asyncio.sleep(0)

    assert runs == ["run"]
    release.set()
    await asyncio.gather(first_start, second_start)


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
async def test_wait_mode_does_not_run_cleanup_before_restart(supervisor):
    release = asyncio.Event()
    cleanups = []

    async def run_work():
        await release.wait()

    connection = connect.DataloaderConnection(
        object(), run_work, lambda: cleanups.append("cleanup"), run_mode="wait"
    )
    run_task = asyncio.create_task(connection.runtime.start())
    await asyncio.sleep(0)

    await connection.runtime.stop("socket disconnected")

    assert cleanups == []
    release.set()
    await run_task


def test_watchdog_mode_is_rejected():
    with pytest.raises(ValueError, match="watchdog mode"):
        connect.connection(object(), lambda: None, run_mode="watchdog")
