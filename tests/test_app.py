import ib_insync as ibi

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
    app = App(context, settings=ConnectionSettings(client_id=77))

    assert repr(app).count("RuntimeContext(") == 1
    assert "runtime=" not in repr(app)
    assert "supervisor=" not in repr(app)
    assert str(app) == (
        "App<client_id=77, contracts=0, streamers=0, future_roll_exclusions=0>"
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
