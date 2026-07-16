import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, cast

import ib_insync as ibi
import pytest

from haymaker.dataloader import runtime
from haymaker.config import DataloaderCommand, load_dataloader_config


class FakePacing:
    """Provide the broker error callback expected by the runtime."""

    def onErrEvent(self, *args: object) -> None:
        """Accept a broker error event without handling it."""


class FakeSession:
    """Provide controllable work and cleanup for runtime lifecycle tests."""

    def __init__(
        self,
        run_work: Callable[[], Awaitable[None]],
        cleanup: Callable[[], None] | None = None,
    ) -> None:
        self.run_work = run_work
        self.cleanup = cleanup
        self.pacing = FakePacing()

    async def run(self) -> None:
        """Run the injected async workload."""

        await self.run_work()

    def cancel_tasks(self) -> None:
        """Run the injected cleanup callback when configured."""

        if self.cleanup:
            self.cleanup()


def make_runtime(
    run_work: Callable[[], Awaitable[None]],
    cleanup: Callable[[], None] | None = None,
) -> runtime.DataloaderRuntime:
    """Return a runtime using an isolated fake dataloader session."""

    session = cast(Any, FakeSession(run_work, cleanup))
    config = load_dataloader_config(DataloaderCommand(None, ()), environ={})
    return runtime.DataloaderRuntime(config, ibi.IB(), session)


def test_runtime_decomposes_config_into_target_objects() -> None:
    """Runtime should route each dataloader option to its owning target."""

    config = load_dataloader_config(
        DataloaderCommand(
            None,
            (
                ("download.source", "custom.csv"),
                ("download.bar_size", "1 min"),
                ("download.what_to_show", "BID"),
                ("download.max_lookback_days", 30),
                ("download.gap_fill_mode", "heuristic"),
                ("download.use_rth", True),
                ("download.save_every_chunks", 3),
                ("download.number_of_workers", 4),
                ("pacing.no_restriction", True),
                ("pacing.allowance_fraction", 0.5),
                ("futures.selector", "current"),
                ("futures.current_index", -1),
            ),
        ),
        environ={},
    )

    dataloader_runtime = runtime.DataloaderRuntime(config, ibi.IB())
    assert dataloader_runtime.session is not None
    manager = dataloader_runtime.session.manager
    assert manager is not None
    assert dataloader_runtime.session.number_of_workers == 4
    assert manager.source == "custom.csv"
    assert manager.bar_size == "1 min"
    assert manager.wts == "BID"
    assert manager.max_lookback_days == 30
    assert manager.gap_fill_mode == "heuristic"
    assert manager.use_rth is True
    assert manager.save_every_chunks == 3
    assert manager.pacing is not None
    assert manager.pacing.no_restriction is True
    assert manager.pacing.allowance_fraction == 0.5
    assert manager.futures.selector == "current"
    assert manager.futures.current_index == -1


@pytest.mark.parametrize("section", ["download", "pacing"])
def test_runtime_rejects_unknown_target_option(section: str) -> None:
    config = load_dataloader_config(
        DataloaderCommand(None, ((f"{section}.unknown", True),)), environ={}
    )

    with pytest.raises(TypeError, match=f"Unknown {section} configuration"):
        runtime.DataloaderRuntime(config, ibi.IB())


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("download.number_of_workers", 0, "number_of_workers"),
        ("download.gap_fill_mode", "unknown", "gap-fill mode"),
        ("pacing.allowance_fraction", 0, "allowance_fraction"),
    ],
)
def test_runtime_preserves_target_policy_validation(
    path: str, value: object, message: str
) -> None:
    config = load_dataloader_config(
        DataloaderCommand(None, ((path, value),)), environ={}
    )

    with pytest.raises((TypeError, ValueError), match=message):
        runtime.DataloaderRuntime(config, ibi.IB())


@pytest.mark.asyncio
async def test_runtime_runs_work():
    runs = []

    async def run_work():
        runs.append("run")

    dataloader_runtime = make_runtime(run_work)
    await dataloader_runtime.start()

    assert runs == ["run"]


@pytest.mark.asyncio
async def test_restart_resumes_by_starting_same_workload_again():
    release = asyncio.Event()
    runs = []

    async def run_work():
        runs.append("run")
        await release.wait()

    dataloader_runtime = make_runtime(run_work)
    first_start = asyncio.create_task(dataloader_runtime.start())
    while runs != ["run"]:
        await asyncio.sleep(0)
    await dataloader_runtime.stop("socket disconnected")
    await first_start

    release = asyncio.Event()
    second_start = asyncio.create_task(dataloader_runtime.start())
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

    dataloader_runtime = make_runtime(run_work, lambda: cleanups.append("cleanup"))
    run_task = asyncio.create_task(dataloader_runtime.start())
    await asyncio.sleep(0)

    await dataloader_runtime.stop("socket disconnected")

    assert cleanups == ["cleanup"]
    release.set()
    await run_task


@pytest.mark.asyncio
async def test_workload_failure_propagates():
    async def run_work():
        raise RuntimeError("broken workload")

    dataloader_runtime = make_runtime(run_work)

    with pytest.raises(RuntimeError, match="broken workload"):
        await dataloader_runtime.start()


def test_runtime_module_exposes_runtime_only():
    """Dataloader runtime handling should use the shared application."""

    assert not hasattr(runtime, "DataloaderConnection")


async def test_runtime_close_stops_active_work():
    """Final runtime cleanup should collect any active dataloader work."""

    release = asyncio.Event()

    async def run_work():
        await release.wait()

    dataloader_runtime = make_runtime(run_work)
    run_task = asyncio.create_task(dataloader_runtime.start())
    await asyncio.sleep(0)

    await dataloader_runtime.close()
    await run_task

    assert dataloader_runtime._work_task is not None
    assert dataloader_runtime._work_task.done()
