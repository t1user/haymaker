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
        self.emit_disconnect_on_failed_connect = False
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
            if self.emit_disconnect_on_failed_connect:
                self.disconnectedEvent.emit()
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
        self.request_restart: Any = None
        self.connection_unavailable: asyncio.Event | None = None
        self._release: asyncio.Event | None = None

    def bind_supervisor(
        self, request_restart: Any, connection_unavailable: asyncio.Event
    ) -> None:
        """Record supervisor controls supplied to the workload."""

        self.request_restart = request_restart
        self.connection_unavailable = connection_unavailable

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

    def finish(self) -> None:
        """Let the active workload complete without a supervisor stop call."""

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


def test_workload_is_bound_to_supervisor_controls() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload)

    assert workload.request_restart("bound restart")
    assert supervisor.restart_requested.is_set()
    assert workload.connection_unavailable is supervisor.connection_unavailable
    assert supervisor.connection_unavailable.is_set()


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
async def test_watcher_probe_waits_for_idle_timeout() -> None:
    fake_ib = FakeIB()
    fake_ib.probe_results = [[]]
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: workload.starts == 1)
    assert await wait_for_condition(lambda: fake_ib.timeouts == [20])
    assert fake_ib.probe_count == 0

    fake_ib.timeoutEvent.emit(20)

    assert await wait_for_condition(lambda: fake_ib.probe_count == 1)
    assert await wait_for_condition(lambda: workload.starts == 2)
    assert workload.stops == ["restart requested"]
    assert fake_ib.connect_attempts == 2

    await stop_and_wait(supervisor, task)


@pytest.mark.asyncio
async def test_restart_disconnects_reconnects_and_reports_restart_reason() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: workload.starts == 1)
    assert not supervisor.connection_unavailable.is_set()

    supervisor.request_restart("manual restart")
    assert supervisor.connection_unavailable.is_set()

    assert await wait_for_condition(lambda: workload.starts == 2)
    assert not supervisor.connection_unavailable.is_set()
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
async def test_disconnect_restart_wins_when_workload_finishes_during_cleanup() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload, connection_lost_retry_delay=0)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: workload.starts == 1)

    fake_ib.disconnect()
    workload.finish()

    assert await wait_for_condition(lambda: workload.starts == 2)
    assert workload.stops == ["restart requested"]
    assert fake_ib.connect_attempts == 2
    assert not supervisor._workload_completed

    await stop_and_wait(supervisor, task)


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
async def test_failed_connect_disconnect_event_is_cleared_by_watcher_cleanup() -> None:
    fake_ib = FakeIB()
    fake_ib.fail_connect_attempts = 1
    fake_ib.emit_disconnect_on_failed_connect = True
    workload = FakeWorkload()
    supervisor = make_supervisor(
        fake_ib, workload, retry_delay=0, connection_lost_retry_delay=0
    )
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: workload.starts == 2)
    await asyncio.sleep(0.01)

    assert fake_ib.connect_attempts == 3
    assert workload.starts == 2
    assert workload.stops == ["restart requested"]
    assert not supervisor.restart_requested.is_set()
    assert not supervisor._delay_next_restart

    await stop_and_wait(supervisor, task)


@pytest.mark.asyncio
async def test_run_cancellation_stops_after_cleanup() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload, connection_lost_retry_delay=0)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: workload.starts == 1)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert workload.starts == 1
    assert workload.stops == ["supervisor stopped"]
    assert fake_ib.connect_attempts == 1
    assert fake_ib.disconnect_count == 1
    assert supervisor.stop_requested.is_set()


@pytest.mark.asyncio
async def test_unexpected_recovery_cap_stops_supervisor() -> None:
    fake_ib = FakeIB()
    fake_ib.cancel_connect_attempts = 2
    workload = FakeWorkload()
    supervisor = make_supervisor(
        fake_ib,
        workload,
        connection_lost_retry_delay=0,
        max_recoveries=1,
    )

    await asyncio.wait_for(supervisor.run(), timeout=1)

    assert fake_ib.connect_attempts == 2
    assert workload.starts == 0
    assert supervisor.stop_requested.is_set()


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
    assert not supervisor.connection_unavailable.is_set()

    fake_ib.errorEvent.emit(-1, 1100, "Connectivity lost", ibi.Contract())
    assert supervisor.connection_unavailable.is_set()

    fake_ib.errorEvent.emit(-1, 1102, "Connectivity restored", ibi.Contract())

    assert await wait_for_condition(
        lambda: not supervisor.connection_unavailable.is_set()
    )
    assert workload.starts == 1
    assert workload.stops == []
    assert fake_ib.connect_attempts == 1
    assert fake_ib.disconnect_count == 0

    await stop_and_wait(supervisor, task)


@pytest.mark.parametrize("code", [2103, 2105, 2157, 10182, 2104, 2106, 2158])
@pytest.mark.asyncio
async def test_weak_data_farm_messages_do_not_change_lifecycle(code: int) -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: workload.starts == 1)
    assert not supervisor.connection_unavailable.is_set()

    fake_ib.errorEvent.emit(-1, code, "Weak farm message", ibi.Contract())
    await asyncio.sleep(0)

    assert await wait_for_condition(lambda: len(fake_ib.errorEvent) == 1)
    assert not supervisor.connection_unavailable.is_set()
    assert not supervisor.restart_requested.is_set()
    assert fake_ib.probe_count == 0
    assert workload.starts == 1
    assert workload.stops == []
    assert fake_ib.connect_attempts == 1
    assert fake_ib.disconnect_count == 0

    await stop_and_wait(supervisor, task)


@pytest.mark.asyncio
async def test_live_update_failure_and_farm_ok_messages_do_not_restart() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: workload.starts == 1)
    assert not supervisor.connection_unavailable.is_set()

    fake_ib.errorEvent.emit(
        -1,
        2105,
        "HMDS data farm connection is broken:euhmds",
        ibi.Contract(),
    )

    fake_ib.errorEvent.emit(
        -1,
        10182,
        "Failed to request live updates (disconnected).",
        ibi.Contract(),
    )
    fake_ib.errorEvent.emit(
        -1,
        2106,
        "HMDS data farm connection is OK:euhmds",
        ibi.Contract(),
    )
    await asyncio.sleep(0)

    assert not supervisor.connection_unavailable.is_set()
    assert not supervisor.restart_requested.is_set()
    assert workload.starts == 1
    assert workload.stops == []
    assert fake_ib.probe_count == 0
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
    assert supervisor.connection_unavailable.is_set()

    assert not supervisor.request_restart("timeout while broker is recovering")
    assert not supervisor.restart_requested.is_set()
    assert workload.starts == 1
    assert workload.stops == []

    fake_ib.errorEvent.emit(-1, 1102, "Connectivity restored", ibi.Contract())
    assert await wait_for_condition(
        lambda: not supervisor.connection_unavailable.is_set()
    )
    assert workload.starts == 1
    assert workload.stops == []

    assert supervisor.request_restart("manual restart after broker recovery")
    assert await wait_for_condition(lambda: workload.starts == 2)
    assert workload.stops == ["restart requested"]

    await stop_and_wait(supervisor, task)


@pytest.mark.asyncio
async def test_broker_data_maintained_is_not_missed_before_waiter_starts() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload, auto_recovery_grace_period=0.01)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: workload.starts == 1)

    fake_ib.errorEvent.emit(-1, 1100, "Connectivity lost", ibi.Contract())
    fake_ib.errorEvent.emit(-1, 1102, "Connectivity restored", ibi.Contract())

    assert await wait_for_condition(
        lambda: not supervisor.connection_unavailable.is_set()
    )
    await asyncio.sleep(0.03)

    assert workload.starts == 1
    assert workload.stops == []
    assert fake_ib.connect_attempts == 1
    assert fake_ib.disconnect_count == 0

    await stop_and_wait(supervisor, task)


@pytest.mark.asyncio
async def test_socket_disconnect_bypasses_broker_recovery_restart_block() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(
        fake_ib,
        workload,
        auto_recovery_grace_period=60,
        connection_lost_retry_delay=0,
    )
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: workload.starts == 1)

    fake_ib.errorEvent.emit(-1, 1100, "Connectivity lost", ibi.Contract())
    assert supervisor.connection_unavailable.is_set()

    fake_ib.disconnect()

    assert await wait_for_condition(lambda: workload.starts == 2)
    assert workload.stops == ["restart requested"]
    assert fake_ib.connect_attempts == 2
    assert fake_ib.disconnect_count == 1

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


@pytest.mark.asyncio
async def test_data_lost_message_restarts_immediately() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: workload.starts == 1)

    fake_ib.errorEvent.emit(-1, 1101, "Data lost", ibi.Contract())

    assert await wait_for_condition(lambda: workload.starts == 2)
    assert workload.stops == ["restart requested"]
    assert fake_ib.connect_attempts == 2
    assert fake_ib.disconnect_count == 1

    await stop_and_wait(supervisor, task)


@pytest.mark.asyncio
async def test_socket_reset_message_restarts_immediately() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: workload.starts == 1)

    fake_ib.errorEvent.emit(-1, 1300, "Socket reset", ibi.Contract())

    assert await wait_for_condition(lambda: workload.starts == 2)
    assert workload.stops == ["restart requested"]
    assert fake_ib.connect_attempts == 2
    assert fake_ib.disconnect_count == 1

    await stop_and_wait(supervisor, task)


@pytest.mark.asyncio
async def test_data_maintained_message_restarts_when_configured() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(
        fake_ib, workload, restart_on_recovered_connection=True
    )
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: workload.starts == 1)

    fake_ib.errorEvent.emit(-1, 1102, "Connectivity restored", ibi.Contract())

    assert await wait_for_condition(lambda: workload.starts == 2)
    assert workload.stops == ["restart requested"]
    assert fake_ib.connect_attempts == 2
    assert fake_ib.disconnect_count == 1

    await stop_and_wait(supervisor, task)
