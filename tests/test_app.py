import asyncio
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import ib_insync as ibi
import pytest

import haymaker.app as app_module
from haymaker.app import App, LiveRuntime
from haymaker.contract_registry import ContractRegistry
from haymaker.runtime import InitData, RuntimeContext, StartupJobs
from haymaker.streamers import Streamer
from haymaker.supervisor import ConnectionSettings


def make_runtime_context(atom_runtime) -> RuntimeContext:
    """Create an initialized runtime context for representation tests."""

    return RuntimeContext(
        config={
            "use_blotter": False,
            "controller": {},
            "app": {},
            "secret_value": "must-not-appear",
        },
        ib=atom_runtime.ib,
        contract_registry=atom_runtime.contract_registry,
        sm=atom_runtime.sm,
    )


def test_runtime_context_has_compact_repr_and_log_string(atom_runtime) -> None:
    """Runtime context representations must not dump its service graph."""

    context = make_runtime_context(atom_runtime)

    assert "must-not-appear" not in repr(context)
    assert "controller=" not in repr(context)
    assert "contract_registry=" not in repr(context)
    assert context.no_future_roll_strategies == []
    assert context.startup_jobs.streamers is Streamer.instances
    assert str(context) == (
        f"RuntimeContext<contracts=0, streamers={len(Streamer.instances)}>"
    )


def test_app_repr_avoids_duplicate_runtime_context(atom_runtime) -> None:
    """Application repr must not repeat its context through LiveRuntime."""

    context = make_runtime_context(atom_runtime)
    app = App(LiveRuntime(context), settings=ConnectionSettings(client_id=77))

    assert "RuntimeContext(" not in repr(app)
    assert "runtime=" not in repr(app)
    assert "supervisor=" not in repr(app)
    context_description = (
        f"RuntimeContext<contracts=0, streamers={len(Streamer.instances)}>"
    )
    assert str(app) == (
        f"App<client_id=77, runtime=LiveRuntime<{context_description}>>"
    )
    assert str(app.runtime) == f"LiveRuntime<{context_description}>"


def test_startup_jobs_separates_log_and_diagnostic_representations() -> None:
    """Startup jobs should use compact str and constructor-shaped repr."""

    ib = ibi.IB()
    jobs = StartupJobs(InitData(ib, ContractRegistry()), ib, [])

    assert str(jobs) == "StartupJobs()"
    assert repr(jobs).startswith("StartupJobs(init_data=InitData(")


def test_startup_jobs_observes_streamers_registered_after_construction() -> None:
    """Startup jobs created before strategy import must see later streamers."""

    ib = ibi.IB()
    streamers = []
    jobs = StartupJobs(InitData(ib, ContractRegistry()), ib, streamers)
    streamer = cast(Any, object())

    streamers.append(streamer)

    assert jobs.streamers == [streamer]


@pytest.mark.asyncio
async def test_app_closes_runtime_tasks_and_queues(monkeypatch) -> None:
    """Application shutdown should follow the shared process cleanup order."""

    events = []

    class FakeRuntime:
        ib = ibi.IB()
        request_restart: Callable[[str], bool | None] | None = None
        connection_unavailable: asyncio.Event | None = None

        def bind_supervisor(
            self,
            request_restart: Callable[[str], bool | None],
            connection_unavailable: asyncio.Event,
        ) -> None:
            self.request_restart = request_restart
            self.connection_unavailable = connection_unavailable

        async def start(self) -> None:
            pass

        async def stop(self, reason: str) -> None:
            pass

        async def close(self) -> None:
            events.append("runtime")

    class FakeSupervisor:
        def __init__(self, *args) -> None:
            self.connection_unavailable = asyncio.Event()

        def request_restart(self, reason: str) -> bool:
            return True

        async def run(self) -> None:
            events.append("supervisor")

    async def cancel_tasks() -> None:
        events.append("tasks")

    async def close_queues() -> None:
        events.append("queues")

    monkeypatch.setattr(app_module, "ConnectionSupervisor", FakeSupervisor)
    monkeypatch.setattr(app_module, "cancel_background_tasks", cancel_tasks)
    monkeypatch.setattr(app_module.QueueRunner, "close_all", close_queues)

    runtime = FakeRuntime()
    app = App(runtime, ConnectionSettings())
    await app._run()

    assert runtime.request_restart == app.supervisor.request_restart
    assert runtime.connection_unavailable is app.supervisor.connection_unavailable
    assert events == ["supervisor", "runtime", "tasks", "queues"]


@pytest.mark.asyncio
async def test_live_runtime_propagates_startup_failure() -> None:
    """Unexpected controller failures must reach the supervisor task."""

    class FailingController:
        async def run(self) -> bool:
            raise RuntimeError("controller failed")

    context = cast(RuntimeContext, SimpleNamespace(controller=FailingController()))
    runtime = LiveRuntime(context)

    with pytest.raises(RuntimeError, match="controller failed"):
        await runtime.start()


@pytest.mark.asyncio
async def test_live_runtime_runs_startup_jobs_after_controller() -> None:
    """Live startup should use the explicit StartupJobs.run entrypoint."""

    events = []

    class FakeController:
        async def run(self) -> None:
            events.append("controller")

    class FakeStartupJobs:
        async def run(self) -> None:
            events.append("startup-jobs")

    context = cast(
        RuntimeContext,
        SimpleNamespace(controller=FakeController(), startup_jobs=FakeStartupJobs()),
    )

    await LiveRuntime(context).start()

    assert events == ["controller", "startup-jobs"]


def test_app_run_propagates_unexpected_failure(monkeypatch) -> None:
    """Application failures must produce a failing process outcome."""

    failure = RuntimeError("application failed")

    def fail_run(coroutine) -> None:
        coroutine.close()
        raise failure

    monkeypatch.setattr(app_module.asyncio, "run", fail_run)

    app = object.__new__(App)
    with pytest.raises(RuntimeError, match="application failed"):
        app.run()


@pytest.mark.asyncio
async def test_sigterm_requests_supervisor_stop_and_restores_default(
    monkeypatch,
) -> None:
    """The first SIGTERM should request cleanup and expose default handling."""

    events: list[Any] = []

    class FakeLoop:
        signal_handler = None

        def set_exception_handler(self, handler) -> None:
            events.append("exception-handler")

        def add_signal_handler(self, signum, handler) -> None:
            events.append(("add-signal", signum))
            self.signal_handler = handler

        def remove_signal_handler(self, signum) -> bool:
            events.append(("remove-signal", signum))
            return True

    loop = FakeLoop()

    class FakeSupervisor:
        async def run(self) -> None:
            assert loop.signal_handler is not None
            loop.signal_handler()
            events.append("supervisor")

        def stop(self) -> None:
            events.append("stop")

    class FakeRuntime:
        async def close(self) -> None:
            events.append("runtime")

    async def cancel_tasks() -> None:
        events.append("tasks")

    async def close_queues() -> None:
        events.append("queues")

    monkeypatch.setattr(app_module.asyncio, "get_running_loop", lambda: loop)
    monkeypatch.setattr(app_module, "cancel_background_tasks", cancel_tasks)
    monkeypatch.setattr(app_module.QueueRunner, "close_all", close_queues)

    app = object.__new__(App)
    app.supervisor = cast(Any, FakeSupervisor())
    app.runtime = cast(Any, FakeRuntime())
    await app._run()

    assert events == [
        "exception-handler",
        ("add-signal", app_module.signal.SIGTERM),
        ("remove-signal", app_module.signal.SIGTERM),
        "stop",
        "supervisor",
        "runtime",
        "tasks",
        "queues",
        ("remove-signal", app_module.signal.SIGTERM),
    ]
