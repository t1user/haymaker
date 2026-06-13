import asyncio
from typing import Any, cast

import ib_insync as ibi
import pytest
from helpers import wait_for_condition

from haymaker.supervisor.supervisor_one import ConnectionSettings, ConnectionSupervisor


class FakeIB:
    """Small IB stand-in exposing the event surface used by supervisor_one."""

    def __init__(self) -> None:
        self.errorEvent = ibi.Event()
        self.disconnectedEvent = ibi.Event()
        self.timeoutEvent = ibi.Event()
        self.connected = False
        self.connect_attempts = 0
        self.disconnect_count = 0
        self.fail_connect_attempts = 0
        self.cancel_connect_attempts = 0
        self.timeouts: list[float] = []
        self.probe_count = 0
        self.probe_results: list[list[object]] = [[object()]]

    async def connectAsync(self, *args: object, **kwargs: object) -> None:
        """Connect unless configured to fail this attempt."""

        self.connect_attempts += 1
        if self.cancel_connect_attempts:
            self.cancel_connect_attempts -= 1
            raise asyncio.CancelledError
        if self.fail_connect_attempts:
            self.fail_connect_attempts -= 1
            raise ConnectionError("connection failed")
        self.connected = True

    def disconnect(self) -> None:
        """Disconnect and emit the same event ib_insync emits."""

        if self.connected:
            self.connected = False
            self.disconnect_count += 1
            self.disconnectedEvent.emit()

    def isConnected(self) -> bool:
        """Return whether the fake socket is connected."""

        return self.connected

    def setTimeout(self, timeout: float) -> None:
        """Record the configured idle timeout."""

        self.timeouts.append(timeout)

    def reqHistoricalDataAsync(self, *args: object, **kwargs: object) -> object:
        """Return a probe coroutine."""

        async def request() -> list[object]:
            self.probe_count += 1
            if self.probe_results:
                return self.probe_results.pop(0)
            return [object()]

        return request()


class FakeWorkload:
    """Controllable workload used to observe supervisor lifecycle calls."""

    def __init__(self, complete_immediately: bool = False) -> None:
        self.complete_immediately = complete_immediately
        self.starts = 0
        self.stops: list[str] = []
        self._release: asyncio.Event | None = None

    async def start(self) -> None:
        """Run until stopped unless configured to complete immediately."""

        self.starts += 1
        if self.complete_immediately:
            return

        self._release = asyncio.Event()
        await self._release.wait()

    async def stop(self, reason: str) -> None:
        """Release the active workload and record the stop reason."""

        self.stops.append(reason)
        if self._release is not None:
            self._release.set()


def make_supervisor(
    fake_ib: FakeIB,
    workload: FakeWorkload | None = None,
    **settings: Any,
) -> ConnectionSupervisor:
    """Create a supervisor_one instance with fast deterministic timings."""

    default_settings: dict[str, Any] = {
        "retry_delay": 0,
        "app_timeout": 20,
        "probe_timeout": 0.01,
        "connection_lost_retry_delay": 60,
        "auto_recovery_grace_period": 0.02,
    }
    default_settings.update(settings)
    return ConnectionSupervisor(
        cast(ibi.IB, fake_ib),
        workload or FakeWorkload(),
        ConnectionSettings(**default_settings),
    )


async def stop_and_wait(
    supervisor: ConnectionSupervisor, task: asyncio.Task[None]
) -> None:
    """Request supervisor shutdown and wait for the run task to finish."""

    supervisor.stop()
    await asyncio.wait_for(task, timeout=1)


@pytest.mark.asyncio
async def test_completed_workload_stops_without_restart() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload(complete_immediately=True)
    supervisor = make_supervisor(fake_ib, workload)

    await asyncio.wait_for(supervisor.run(), timeout=1)

    assert workload.starts == 1
    assert workload.stops == []
    assert fake_ib.connect_attempts == 1
    assert fake_ib.disconnect_count == 1


@pytest.mark.asyncio
async def test_restart_disconnects_reconnects_and_reports_restart_reason() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: workload.starts == 1)

    supervisor.request_restart("manual restart")

    assert await wait_for_condition(lambda: workload.starts == 2)
    assert workload.stops == ["restart requested"]
    assert fake_ib.connect_attempts == 2
    assert fake_ib.disconnect_count == 1

    await stop_and_wait(supervisor, task)

    assert workload.stops == ["restart requested", "supervisor stopped"]
    assert fake_ib.disconnect_count == 2


@pytest.mark.asyncio
async def test_stop_interrupts_connection_lost_retry_sleep() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload, connection_lost_retry_delay=60)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: workload.starts == 1)

    fake_ib.disconnect()
    assert await wait_for_condition(lambda: fake_ib.disconnect_count == 1)
    assert await wait_for_condition(lambda: workload.stops == ["restart requested"])

    supervisor.stop()

    await asyncio.wait_for(task, timeout=1)

    assert workload.stops == ["restart requested"]
    assert fake_ib.connect_attempts == 1


@pytest.mark.asyncio
async def test_connection_cancelled_error_retries_without_stopping() -> None:
    fake_ib = FakeIB()
    fake_ib.cancel_connect_attempts = 1
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload, connection_lost_retry_delay=0)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: workload.starts == 1)

    assert fake_ib.connect_attempts == 2
    assert not task.done()

    await stop_and_wait(supervisor, task)


@pytest.mark.asyncio
async def test_external_run_cancellation_propagates_after_cleanup() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: workload.starts == 1)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert workload.stops == ["supervisor stopped"]
    assert fake_ib.connect_attempts == 1
    assert fake_ib.disconnect_count == 1


@pytest.mark.asyncio
async def test_watcher_handlers_are_detached_after_each_cycle() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: len(fake_ib.errorEvent) == 1)
    assert await wait_for_condition(lambda: len(fake_ib.timeoutEvent) == 1)

    supervisor.request_restart("manual restart")

    assert await wait_for_condition(lambda: workload.starts == 2)
    assert len(fake_ib.errorEvent) == 1
    assert len(fake_ib.timeoutEvent) == 1

    await stop_and_wait(supervisor, task)

    assert len(fake_ib.errorEvent) == 0
    assert len(fake_ib.timeoutEvent) == 0


@pytest.mark.asyncio
async def test_broker_data_maintained_continues_without_restart() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: workload.starts == 1)

    fake_ib.errorEvent.emit(-1, 1100, "Connectivity lost", ibi.Contract())
    assert await wait_for_condition(lambda: len(fake_ib.errorEvent) == 2)

    fake_ib.errorEvent.emit(-1, 1102, "Connectivity restored", ibi.Contract())

    assert await wait_for_condition(lambda: len(fake_ib.errorEvent) == 1)
    assert workload.starts == 1
    assert workload.stops == []
    assert fake_ib.connect_attempts == 1
    assert fake_ib.disconnect_count == 0

    await stop_and_wait(supervisor, task)


@pytest.mark.asyncio
async def test_restart_requests_are_blocked_during_broker_recovery_wait() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload, auto_recovery_grace_period=0.2)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: workload.starts == 1)

    fake_ib.errorEvent.emit(-1, 1100, "Connectivity lost", ibi.Contract())
    assert await wait_for_condition(lambda: len(fake_ib.errorEvent) == 2)

    assert not supervisor.request_restart("timeout while broker is recovering")
    assert not supervisor.restart_requested.is_set()
    assert workload.starts == 1
    assert workload.stops == []

    fake_ib.errorEvent.emit(-1, 1102, "Connectivity restored", ibi.Contract())
    assert await wait_for_condition(lambda: len(fake_ib.errorEvent) == 1)
    assert workload.starts == 1
    assert workload.stops == []

    assert supervisor.request_restart("manual restart after broker recovery")
    assert await wait_for_condition(lambda: workload.starts == 2)
    assert workload.stops == ["restart requested"]

    await stop_and_wait(supervisor, task)


@pytest.mark.asyncio
async def test_broker_wait_timeout_restarts_without_recovered_policy_fallthrough() -> (
    None
):
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload, auto_recovery_grace_period=0.01)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: workload.starts == 1)

    fake_ib.errorEvent.emit(-1, 1100, "Connectivity lost", ibi.Contract())

    assert await wait_for_condition(lambda: workload.starts == 2)
    assert workload.stops == ["restart requested"]
    assert fake_ib.connect_attempts == 2
    assert fake_ib.disconnect_count == 1

    await stop_and_wait(supervisor, task)
