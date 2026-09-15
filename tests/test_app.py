import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import ib_insync as ibi
import pytest

import haymaker.app as app_module
from haymaker.app import App
from haymaker.base import Atom
from haymaker.config import LiveCommand, load_live_config
from haymaker.components import StandardOrderRole
from haymaker.controller.controller import SyncOutcome
from haymaker.contract_registry import ContractRegistry
from haymaker.databases import MongoService
from haymaker.datastore import MarketDataStoreFactory, SignalFramePersistenceFactory
from haymaker.runtime import InitData, LiveRuntime, RuntimeContext, StartupJobs
from haymaker.components.streamers import Streamer
from haymaker.supervisor import ConnectionSettings
from haymaker.trader import Trader


def live_config():
    """Return live config without persistence-backed blotter creation."""

    return load_live_config(
        LiveCommand(
            module_path=Path("strategy.py"),
            config_file=None,
            overrides=(("blotter.enabled", False),),
        ),
        environ={},
    )


def make_live_runtime(atom_runtime) -> LiveRuntime:
    """Create a live runtime from injected test services."""

    return LiveRuntime(
        live_config(),
        ib=atom_runtime.ib,
        contract_registry=atom_runtime.contract_registry,
        book=atom_runtime.book,
    )


def test_runtime_context_has_compact_repr_and_log_string(atom_runtime) -> None:
    """Runtime context representations must not dump its service graph."""

    context = RuntimeContext(
        ib=atom_runtime.ib,
        contract_registry=atom_runtime.contract_registry,
        book=atom_runtime.book,
        trader=Trader(atom_runtime.ib),
        frame_store_provider=atom_runtime.frame_store_provider,
        market_data_store_factory=atom_runtime.market_data_store_factory,
        signal_persistence_factory=atom_runtime.signal_persistence_factory,
        order_defaults=atom_runtime.order_defaults,
        timeout_policy=atom_runtime.timeout_policy,
    )

    assert "controller=" not in repr(context)
    assert "contract_registry=" not in repr(context)
    assert context.request_restart is None
    assert context.future_roll_policies == {}
    assert str(context) == "RuntimeContext<contracts=0>"


def test_live_runtime_builds_and_installs_ready_context(atom_runtime) -> None:
    """Live runtime should assemble services before user strategy import."""

    runtime = make_live_runtime(atom_runtime)

    assert Atom.runtime is runtime.context
    assert runtime.context.ib is atom_runtime.ib
    assert runtime.context.book is atom_runtime.book
    assert runtime.context.contract_registry is atom_runtime.contract_registry
    assert runtime.context.controller is not None
    assert runtime.context.frame_store_provider is runtime.frame_store_provider
    assert isinstance(
        runtime.context.market_data_store_factory,
        MarketDataStoreFactory,
    )
    assert runtime.context.market_data_store_factory.library == "market_data"
    assert isinstance(
        runtime.context.signal_persistence_factory,
        SignalFramePersistenceFactory,
    )
    assert runtime.context.signal_persistence_factory.library == "signal_data"
    assert runtime.startup_jobs.streamers is Streamer.instances
    assert not hasattr(runtime.context, "config")
    assert not hasattr(runtime.context, "startup_jobs")
    assert not hasattr(runtime.context, "store_factory")
    assert not hasattr(runtime.context, "dataframe_save_frequency")


def test_live_runtime_installs_injected_frame_store_provider(atom_runtime) -> None:
    """A supplied strategy provider should reach the ready runtime context."""

    runtime = LiveRuntime(
        live_config(),
        ib=atom_runtime.ib,
        contract_registry=atom_runtime.contract_registry,
        book=atom_runtime.book,
        frame_store_provider=atom_runtime.frame_store_provider,
    )

    assert runtime.context.frame_store_provider is atom_runtime.frame_store_provider


def test_live_runtime_keeps_mongo_service_out_of_context(atom_runtime) -> None:
    """Mongo lifecycle should remain private runtime infrastructure."""

    mongo_service = MongoService({"host": "mongo.example"})
    runtime = LiveRuntime(
        live_config(),
        ib=atom_runtime.ib,
        mongo_service=mongo_service,
        contract_registry=atom_runtime.contract_registry,
        book=atom_runtime.book,
    )

    assert runtime.mongo_service is mongo_service
    assert not hasattr(runtime.context, "mongo_service")


def test_app_repr_avoids_duplicate_runtime_context(atom_runtime) -> None:
    """Application repr must not repeat its context through LiveRuntime."""

    runtime = make_live_runtime(atom_runtime)
    app = App(runtime, settings=ConnectionSettings(client_id=77))

    assert "RuntimeContext(" not in repr(app)
    assert "runtime=" not in repr(app)
    assert "supervisor=" not in repr(app)
    assert str(app) == (
        f"App<client_id=77, runtime=LiveRuntime<contracts=0, "
        f"streamers={len(Streamer.instances)}>>"
    )
    assert str(app.runtime) == (
        f"LiveRuntime<contracts=0, streamers={len(Streamer.instances)}>"
    )


def test_startup_jobs_separates_log_and_diagnostic_representations() -> None:
    """Startup jobs should use compact str and constructor-shaped repr."""

    ib = ibi.IB()
    jobs = StartupJobs(InitData(ib, ContractRegistry()), ib, [])

    assert str(jobs) == "StartupJobs()"
    assert repr(jobs).startswith("StartupJobs(init_data=InitData(")


def test_startup_jobs_observes_streamers_registered_after_construction() -> None:
    """Startup jobs created before strategy import must see later streamers."""

    ib = ibi.IB()
    streamers: list[Any] = []
    jobs = StartupJobs(InitData(ib, ContractRegistry()), ib, streamers)
    streamer = cast(Any, object())

    streamers.append(streamer)

    assert jobs.streamers == [streamer]


@pytest.mark.asyncio
async def test_app_closes_runtime_tasks_and_queues(monkeypatch) -> None:
    """Application shutdown should follow the shared process cleanup order."""

    events: list[object] = []

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

    async def initialize() -> None:
        """Supply the contract-initialization phase without broker requests."""

    class FailingController:
        def set_future_roll_policies(self, policies: dict[str, bool]) -> None:
            pass

        async def run(self) -> bool:
            raise RuntimeError("controller failed")

    runtime = object.__new__(LiveRuntime)
    runtime.startup_jobs = cast(StartupJobs, SimpleNamespace(init_data=initialize))
    runtime.context = cast(
        RuntimeContext,
        SimpleNamespace(
            controller=FailingController(),
            future_roll_policies={},
            workload_generation=0,
        ),
    )

    with pytest.raises(RuntimeError, match="controller failed"):
        await runtime.start()


@pytest.mark.asyncio
async def test_live_runtime_runs_startup_jobs_after_controller(monkeypatch) -> None:
    """Live startup should apply policies and run monitoring after controller."""

    events: list[object] = []

    class FakeController:
        def set_future_roll_policies(self, policies: dict[str, bool]) -> None:
            events.append(("policies", dict(policies)))

        async def run(self) -> SyncOutcome:
            events.append("controller")
            return SyncOutcome.FAILED

    class FakeStartupJobs:
        async def init_data(self) -> None:
            """Make Contract details available before Controller recovery."""
            events.append("contract-details")

        async def run(self) -> None:
            events.append("startup-jobs")

    runtime = object.__new__(LiveRuntime)
    runtime.context = cast(
        RuntimeContext,
        SimpleNamespace(
            controller=FakeController(),
            future_roll_policies={"manual": False},
            workload_generation=0,
        ),
    )
    runtime.startup_jobs = cast(StartupJobs, FakeStartupJobs())
    monkeypatch.setattr(
        "haymaker.runtime.MarketDataTimeout._cancel_all",
        lambda: events.append("timeouts"),
    )

    await runtime.start()

    assert events == [
        ("policies", {"manual": False}),
        "contract-details",
        "controller",
        "startup-jobs",
        "timeouts",
    ]


@pytest.mark.asyncio
async def test_live_runtime_skips_startup_jobs_after_aborted_controller(
    monkeypatch,
) -> None:
    """A requested restart must end the workload before broker jobs start."""

    events: list[str] = []

    class FakeController:
        def set_future_roll_policies(self, policies: dict[str, bool]) -> None:
            pass

        async def run(self) -> SyncOutcome:
            events.append("controller")
            return SyncOutcome.ABORTED

    class FakeStartupJobs:
        async def init_data(self) -> None:
            """Prepare Contract details without starting streamers."""
            events.append("contract-details")

        async def run(self) -> None:
            events.append("startup-jobs")

    runtime = object.__new__(LiveRuntime)
    runtime.context = cast(
        RuntimeContext,
        SimpleNamespace(
            controller=FakeController(),
            future_roll_policies={},
            workload_generation=0,
        ),
    )
    runtime.startup_jobs = cast(StartupJobs, FakeStartupJobs())
    monkeypatch.setattr(
        "haymaker.runtime.MarketDataTimeout._cancel_all",
        lambda: events.append("timeouts"),
    )

    await runtime.start()

    assert events == ["contract-details", "controller", "timeouts"]


@pytest.mark.asyncio
async def test_live_runtime_stop_cancels_timeouts_before_controller_hold(
    monkeypatch,
) -> None:
    """Workload stop should disable stale-data callbacks before other cleanup."""

    events: list[str] = []

    class FakeController:
        def set_hold(self) -> None:
            events.append("hold")

    runtime = object.__new__(LiveRuntime)
    runtime.context = cast(
        RuntimeContext,
        SimpleNamespace(controller=FakeController()),
    )
    monkeypatch.setattr(
        "haymaker.runtime.MarketDataTimeout._cancel_all",
        lambda: events.append("timeouts"),
    )

    await runtime.stop("restart requested")

    assert events == ["timeouts", "hold"]


@pytest.mark.asyncio
async def test_live_runtime_close_warns_about_active_target_adjustments(
    caplog,
) -> None:
    """Final process close should expose unfinished direct execution work."""

    events: list[object] = []
    info = SimpleNamespace(
        orderId=17,
        permId=117,
        execution_model_name="serial",
        trade=SimpleNamespace(
            contract=ibi.Future(
                conId=123,
                symbol="ES",
                localSymbol="ESM6",
            )
        ),
        signed_working_quantity=-2,
    )

    class FakeBook:
        def __init__(self):
            """Expose the collection queries used by final shutdown checks."""
            self.orders = SimpleNamespace(active=self.active_orders)
            self.rolls = SimpleNamespace(all=self.roll_states)

        def active_orders(self, *, role=None):
            events.append(("query", role))
            return (info,) if role == StandardOrderRole.TARGET_ADJUSTMENT else ()

        def roll_states(self, *, active_only=False):
            events.append(("roll_states", active_only))
            return ()

        async def close(self) -> None:
            events.append("close")

    runtime = object.__new__(LiveRuntime)
    runtime.context = cast(RuntimeContext, SimpleNamespace(book=FakeBook()))

    with caplog.at_level(logging.WARNING, logger="haymaker.runtime"):
        await runtime.close()

    assert events == [
        ("query", StandardOrderRole.TARGET_ADJUSTMENT),
        ("roll_states", True),
        ("query", StandardOrderRole.ROLL),
        "close",
    ]
    assert "active TARGET_ADJUSTMENT" in caplog.text
    assert "serial" in caplog.text
    assert "orderId': 17" in caplog.text
    assert "conId': 123" in caplog.text
    assert "working_quantity': -2" in caplog.text


@pytest.mark.asyncio
async def test_live_runtime_close_is_quiet_without_target_adjustments(
    caplog,
) -> None:
    """Final process close should not warn when routed work is complete."""

    events: list[object] = []

    class FakeBook:
        def __init__(self):
            """Expose the collection queries used by final shutdown checks."""
            self.orders = SimpleNamespace(active=self.active_orders)
            self.rolls = SimpleNamespace(all=self.roll_states)

        def active_orders(self, *, role=None):
            events.append(("query", role))
            return ()

        def roll_states(self, *, active_only=False):
            events.append(("roll_states", active_only))
            return ()

        async def close(self) -> None:
            events.append("close")

    runtime = object.__new__(LiveRuntime)
    runtime.context = cast(RuntimeContext, SimpleNamespace(book=FakeBook()))

    with caplog.at_level(logging.WARNING, logger="haymaker.runtime"):
        await runtime.close()

    assert events == [
        ("query", StandardOrderRole.TARGET_ADJUSTMENT),
        ("roll_states", True),
        ("query", StandardOrderRole.ROLL),
        "close",
    ]
    assert "active TARGET_ADJUSTMENT" not in caplog.text


def test_live_runtime_binds_supervisor_controls(atom_runtime) -> None:
    """Supervisor controls should be installed on their owning services."""

    runtime = make_live_runtime(atom_runtime)
    unavailable = asyncio.Event()

    def request_restart(reason: str) -> bool:
        return True

    runtime.bind_supervisor(request_restart, unavailable)

    assert runtime.context.request_restart is request_restart
    assert runtime.context.controller._sync_abort_event is unavailable


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
