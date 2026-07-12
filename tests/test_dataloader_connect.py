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
async def test_supervised_connection_runs_work_and_stops_supervisor(supervisor):
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
async def test_supervised_connection_runs_cleanup_before_restart(supervisor):
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


def test_supervised_connection_uses_default_client_id(supervisor, monkeypatch):
    monkeypatch.delitem(connect.CONFIG.maps[0], "clientId", raising=False)

    connection = connect.DataloaderConnection(object(), lambda: None)

    assert connection.supervisor.args[2].client_id == 1


def test_supervised_connection_uses_configured_client_id(supervisor, monkeypatch):
    monkeypatch.setitem(connect.CONFIG.maps[0], "clientId", 7)

    connection = connect.DataloaderConnection(object(), lambda: None)

    assert connection.supervisor.args[2].client_id == 7


def test_connection_module_exposes_app_like_connection_object_only():
    """The obsolete function wrapper should not duplicate the connection object."""

    assert not hasattr(connect, "connection")


def test_connection_run_suppresses_keyboard_interrupt_after_cleanup(
    supervisor, monkeypatch
):
    """A completed Ctrl-C cancellation should exit without a traceback."""

    def interrupt(coroutine):
        coroutine.close()
        raise KeyboardInterrupt

    monkeypatch.setattr(connect.asyncio, "run", interrupt)
    connection = connect.DataloaderConnection(object(), lambda: None)

    connection.run()
