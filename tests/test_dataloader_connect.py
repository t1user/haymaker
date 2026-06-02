import asyncio

import pytest

from haymaker.dataloader import connect
from haymaker.supervisor import SupervisorState


class FakeSupervisor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.stopped = False
        self.state = SupervisorState.CONNECTED

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
    await connection.on_connected()
    await asyncio.sleep(0)

    assert runs == ["run"]
    assert connection.supervisor.stopped


@pytest.mark.asyncio
async def test_wait_mode_leaves_existing_work_running_after_reconnect(supervisor):
    release = asyncio.Event()
    runs = []

    async def run_work():
        runs.append("run")
        await release.wait()

    connection = connect.DataloaderConnection(object(), run_work, run_mode="wait")
    await connection.on_connected()
    await asyncio.sleep(0)
    await connection.on_connected()

    assert runs == ["run"]
    release.set()
    await asyncio.sleep(0)


def test_reconnect_mode_runs_cleanup_before_restart(supervisor):
    cleanups = []
    connection = connect.DataloaderConnection(
        object(), lambda: None, lambda: cleanups.append("cleanup")
    )

    connection.on_restarting("socket disconnected")

    assert cleanups == ["cleanup"]


def test_wait_mode_does_not_run_cleanup_before_restart(supervisor):
    cleanups = []
    connection = connect.DataloaderConnection(
        object(), lambda: None, lambda: cleanups.append("cleanup"), run_mode="wait"
    )

    connection.on_restarting("socket disconnected")

    assert cleanups == []


def test_watchdog_mode_is_rejected():
    with pytest.raises(ValueError, match="watchdog mode"):
        connect.connection(object(), lambda: None, run_mode="watchdog")
