import ib_insync as ibi
import pytest

import haymaker.app as app_module
from haymaker.app import App, LiveRuntime
from haymaker.contract_registry import ContractRegistry
from haymaker.runtime import InitData, RuntimeContext, StartupJobs
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
    assert str(context) == (
        "RuntimeContext<contracts=0, streamers=0, startup_jobs_bound=False>"
    )


def test_app_repr_avoids_duplicate_runtime_context(atom_runtime) -> None:
    """Application repr must not repeat its context through LiveRuntime."""

    context = make_runtime_context(atom_runtime)
    app = App(LiveRuntime(context), settings=ConnectionSettings(client_id=77))

    assert "RuntimeContext(" not in repr(app)
    assert "runtime=" not in repr(app)
    assert "supervisor=" not in repr(app)
    assert str(app) == (
        "App<client_id=77, runtime=LiveRuntime<"
        "RuntimeContext<contracts=0, streamers=0, startup_jobs_bound=False>>>"
    )
    assert str(app.runtime) == (
        "LiveRuntime<"
        "RuntimeContext<contracts=0, streamers=0, startup_jobs_bound=False>>"
    )


def test_startup_jobs_separates_log_and_diagnostic_representations() -> None:
    """Startup jobs should use compact str and constructor-shaped repr."""

    ib = ibi.IB()
    jobs = StartupJobs(InitData(ib, ContractRegistry()), ib, [])

    assert str(jobs) == "StartupJobs()"
    assert repr(jobs).startswith("StartupJobs(init_data=InitData(")


@pytest.mark.asyncio
async def test_app_closes_runtime_tasks_and_queues(monkeypatch) -> None:
    """Application shutdown should follow the shared process cleanup order."""

    events = []

    class FakeRuntime:
        ib = ibi.IB()

        async def start(self) -> None:
            pass

        async def stop(self, reason: str) -> None:
            pass

        async def close(self) -> None:
            events.append("runtime")

    class FakeSupervisor:
        def __init__(self, *args) -> None:
            pass

        async def run(self) -> None:
            events.append("supervisor")

    async def cancel_tasks() -> None:
        events.append("tasks")

    async def close_queues() -> None:
        events.append("queues")

    monkeypatch.setattr(app_module, "ConnectionSupervisor", FakeSupervisor)
    monkeypatch.setattr(app_module, "cancel_background_tasks", cancel_tasks)
    monkeypatch.setattr(app_module.QueueRunner, "close_all", close_queues)

    app = App(FakeRuntime(), ConnectionSettings())
    await app._run()

    assert events == ["supervisor", "runtime", "tasks", "queues"]
