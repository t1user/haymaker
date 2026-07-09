import asyncio
from typing import Any, cast

import ib_insync as ibi
import pytest
from helpers import wait_for_condition

from haymaker.supervisor import ConnectionSettings, ConnectionSupervisor
from haymaker.supervisor.supervisor import SupervisorRace
from haymaker.supervisor.states import (
    AbstractState,
    BackoffRestartingState,
    BrokerConnectivityLostState,
    ConnectingState,
    ConnectedState,
    ProbingState,
    RestartingState,
    StoppedState,
    StoppingState,
)


class FakeIB:
    """Small IB stand-in exposing the event surface used by ConnectionSupervisor."""

    def __init__(self, emit_global_disconnect_error: bool = False) -> None:
        self.errorEvent = ibi.Event()
        self.disconnectedEvent = ibi.Event()
        self.timeoutEvent = ibi.Event()
        self.updateEvent = ibi.Event()
        self.emit_global_disconnect_error = emit_global_disconnect_error
        self.connected = False
        self.connect_attempts = 0
        self.disconnect_count = 0
        self.fail_connect_attempts = 0
        self.timeouts: list[float] = []
        self.probe_results: list[list[object]] = [[object()]]
        self.probe_delays: list[float] = []
        self.probe_count = 0

    async def connectAsync(self, *args: object, **kwargs: object) -> None:
        """Connect unless configured to fail this attempt."""

        self.connect_attempts += 1
        if self.fail_connect_attempts:
            self.fail_connect_attempts -= 1
            raise ConnectionError("connection failed")
        self.connected = True

    def disconnect(self) -> None:
        """Disconnect and emit the same event ib_insync emits."""

        if self.connected:
            self.connected = False
            self.disconnect_count += 1
            if self.emit_global_disconnect_error:
                ibi.util.globalErrorEvent.emit(ConnectionError("Socket disconnect"))
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
            if self.probe_delays:
                await asyncio.sleep(self.probe_delays.pop(0))
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
        self.stop_started: asyncio.Event | None = None
        self.release_stop: asyncio.Event | None = None
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
        if self.stop_started is not None:
            self.stop_started.set()
        if self.release_stop is not None:
            await self.release_stop.wait()
        if self._release is not None:
            self._release.set()


class RestartAfterHandleState(AbstractState):
    """State that requests restart before returning a normal transition."""

    async def handle(self) -> type[AbstractState]:
        """Request restart during state handling."""

        self.context.request_restart("late restart")
        return ConnectedState


class StopAfterHandleState(AbstractState):
    """State that requests stop before returning a normal transition."""

    async def handle(self) -> type[AbstractState]:
        """Request stop during state handling."""

        self.context.stop()
        return ConnectedState


def make_supervisor(
    fake_ib: FakeIB,
    workload: FakeWorkload | None = None,
    **settings: Any,
) -> ConnectionSupervisor:
    """Create a supervisor with fast deterministic test timings."""

    default_settings: dict[str, Any] = {
        "retry_delay": 0,
        "app_timeout": 20,
        "probe_timeout": 0.01,
        "connection_lost_retry_delay": 0.02,
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


def current_state(supervisor: ConnectionSupervisor) -> type[AbstractState]:
    """Return the current state type for state-machine assertions."""

    return type(supervisor._state)


async def run_supervisor_race(
    supervisor: ConnectionSupervisor,
) -> type[AbstractState]:
    """Run one supervisor race for isolated transition assertions."""

    async with SupervisorRace(
        supervisor._state,
        supervisor._stop_requested,
        supervisor._restart_requested,
        supervisor._workload_task,
    ) as race:
        return await race.wait()


def test_workload_is_bound_to_supervisor_controls() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload)
    supervisor._state = ConnectedState(supervisor)

    assert workload.request_restart("bound restart")
    assert supervisor._restart_requested.is_set()
    assert workload.connection_unavailable is supervisor.connection_unavailable
    assert supervisor.connection_unavailable.is_set()


@pytest.mark.asyncio
async def test_run_connects_probes_and_starts_workload() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload)

    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: current_state(supervisor) is ConnectedState)
    assert await wait_for_condition(lambda: fake_ib.timeouts == [20])
    assert not supervisor.connection_unavailable.is_set()
    assert workload.starts == 1
    assert fake_ib.connect_attempts == 1
    assert fake_ib.probe_count == 1

    await stop_and_wait(supervisor, task)

    assert current_state(supervisor) is StoppedState
    assert workload.stops == ["supervisor stopped"]
    assert fake_ib.disconnect_count == 1


@pytest.mark.asyncio
async def test_completed_workload_stops_supervisor() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload(complete_immediately=True)
    supervisor = make_supervisor(fake_ib, workload)

    await supervisor.run()

    assert current_state(supervisor) is StoppedState
    assert workload.starts == 1
    assert workload.stops == []
    assert fake_ib.disconnect_count == 1


@pytest.mark.asyncio
async def test_restart_stops_workload_disconnects_and_reconnects() -> None:
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


@pytest.mark.asyncio
async def test_duplicate_restart_requests_coalesce() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: workload.starts == 1)

    supervisor.request_restart("first restart")
    supervisor.request_restart("second restart")

    assert await wait_for_condition(lambda: workload.starts == 2)
    assert workload.stops == ["restart requested"]

    await stop_and_wait(supervisor, task)


@pytest.mark.asyncio
async def test_pending_restart_overrides_state_transition() -> None:
    fake_ib = FakeIB()
    supervisor = make_supervisor(fake_ib)
    supervisor._state = RestartAfterHandleState(supervisor)

    transition = await run_supervisor_race(supervisor)

    assert transition is RestartingState
    assert not supervisor._restart_requested.is_set()


@pytest.mark.asyncio
async def test_pending_stop_overrides_state_transition() -> None:
    fake_ib = FakeIB()
    supervisor = make_supervisor(fake_ib)
    supervisor._state = StopAfterHandleState(supervisor)

    transition = await run_supervisor_race(supervisor)

    assert transition is StoppingState


@pytest.mark.asyncio
async def test_stop_request_overrides_pending_restart() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: workload.starts == 1)

    supervisor.request_restart("manual restart")
    supervisor.stop()

    await asyncio.wait_for(task, timeout=1)

    assert current_state(supervisor) is StoppedState
    assert workload.stops == ["supervisor stopped"]
    assert fake_ib.connect_attempts == 1
    assert fake_ib.disconnect_count == 1


@pytest.mark.asyncio
async def test_pending_stop_overrides_later_restart_request() -> None:
    fake_ib = FakeIB()
    supervisor = make_supervisor(fake_ib)
    supervisor._state = ConnectedState(supervisor)

    supervisor.stop()
    supervisor.request_restart("restart after stop")
    assert supervisor._restart_requested.is_set()

    transition = await run_supervisor_race(supervisor)

    assert transition is StoppingState
    assert not supervisor._restart_requested.is_set()


@pytest.mark.asyncio
async def test_stop_during_restart_cleanup_prevents_reconnect() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    workload.stop_started = asyncio.Event()
    workload.release_stop = asyncio.Event()
    supervisor = make_supervisor(fake_ib, workload)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: workload.starts == 1)

    supervisor.request_restart("manual restart")
    await asyncio.wait_for(workload.stop_started.wait(), timeout=1)
    assert not supervisor.request_restart("ignored during restart cleanup")
    assert not supervisor._restart_requested.is_set()
    supervisor.stop()

    await asyncio.sleep(0.05)
    assert workload.stops == ["restart requested"]
    assert fake_ib.disconnect_count == 0
    assert not task.done()

    workload.release_stop.set()

    await asyncio.wait_for(task, timeout=1)

    assert current_state(supervisor) is StoppedState
    assert workload.stops == ["restart requested"]
    assert fake_ib.connect_attempts == 1
    assert fake_ib.disconnect_count == 1
    assert not supervisor._restart_requested.is_set()


@pytest.mark.asyncio
async def test_unexpected_disconnect_restarts_workload() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: workload.starts == 1)

    fake_ib.disconnect()

    assert await wait_for_condition(lambda: workload.starts == 2)
    assert workload.stops == ["restart requested"]
    assert fake_ib.connect_attempts == 2
    assert fake_ib.disconnect_count == 1

    await stop_and_wait(supervisor, task)


@pytest.mark.asyncio
async def test_global_socket_disconnect_error_does_not_stop_supervisor() -> None:
    fake_ib = FakeIB(emit_global_disconnect_error=True)
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload)
    task = asyncio.create_task(supervisor.run())
    previous_global_error = ibi.util.globalErrorEvent._value

    try:
        assert await wait_for_condition(lambda: workload.starts == 1)

        fake_ib.disconnect()

        assert await wait_for_condition(lambda: workload.starts == 2)
        assert not task.done()
        assert workload.stops == ["restart requested"]
        assert fake_ib.connect_attempts == 2
        assert fake_ib.disconnect_count == 1
    finally:
        if not task.done():
            await stop_and_wait(supervisor, task)
        else:
            task.exception()
        ibi.util.globalErrorEvent._value = previous_global_error


@pytest.mark.asyncio
async def test_connect_retries_until_success() -> None:
    fake_ib = FakeIB()
    fake_ib.fail_connect_attempts = 2
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: workload.starts == 1)
    assert fake_ib.connect_attempts == 3

    await stop_and_wait(supervisor, task)


@pytest.mark.asyncio
async def test_stop_interrupts_connect_retry_wait() -> None:
    fake_ib = FakeIB()
    fake_ib.fail_connect_attempts = 100
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload, retry_delay=60)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: fake_ib.connect_attempts == 1)

    supervisor.stop()

    await asyncio.wait_for(task, timeout=1)

    assert current_state(supervisor) is StoppedState
    assert fake_ib.connect_attempts == 1
    assert workload.starts == 0


@pytest.mark.asyncio
async def test_restart_interrupts_in_flight_probe() -> None:
    fake_ib = FakeIB()
    fake_ib.probe_delays = [60]
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: fake_ib.probe_count == 1)

    supervisor.request_restart("probe interrupted")

    assert await wait_for_condition(lambda: workload.starts == 1)
    assert fake_ib.connect_attempts == 2
    assert fake_ib.disconnect_count == 1

    await stop_and_wait(supervisor, task)


@pytest.mark.parametrize("code", [1100, 2110])
@pytest.mark.asyncio
async def test_broker_connectivity_lost_message_enters_lost_state(code: int) -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload, auto_recovery_grace_period=60)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: current_state(supervisor) is ConnectedState)
    assert not supervisor.connection_unavailable.is_set()

    fake_ib.errorEvent.emit(-1, code, "Connectivity lost", ibi.Contract())
    assert supervisor.connection_unavailable.is_set()

    assert await wait_for_condition(
        lambda: current_state(supervisor) is BrokerConnectivityLostState
    )
    assert fake_ib.disconnect_count == 0
    assert workload.starts == 1

    await stop_and_wait(supervisor, task)


@pytest.mark.asyncio
async def test_broker_connectivity_lost_1102_moves_to_probing() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload, auto_recovery_grace_period=60)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: current_state(supervisor) is ConnectedState)

    fake_ib.probe_delays = [60]
    fake_ib.errorEvent.emit(-1, 1100, "Connectivity lost", ibi.Contract())
    assert await wait_for_condition(
        lambda: current_state(supervisor) is BrokerConnectivityLostState
    )

    fake_ib.errorEvent.emit(-1, 1102, "Connectivity restored", ibi.Contract())

    assert await wait_for_condition(lambda: fake_ib.probe_count == 2)
    assert current_state(supervisor) is ProbingState
    assert fake_ib.disconnect_count == 0
    assert workload.starts == 1

    await stop_and_wait(supervisor, task)


@pytest.mark.asyncio
async def test_broker_connectivity_lost_grace_expiry_moves_to_probing() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: current_state(supervisor) is ConnectedState)

    fake_ib.probe_delays = [60]
    fake_ib.errorEvent.emit(-1, 1100, "Connectivity lost", ibi.Contract())
    assert await wait_for_condition(
        lambda: current_state(supervisor) is BrokerConnectivityLostState
    )

    assert await wait_for_condition(lambda: fake_ib.probe_count == 2)
    assert current_state(supervisor) is ProbingState
    assert fake_ib.disconnect_count == 0
    assert workload.starts == 1

    await stop_and_wait(supervisor, task)


@pytest.mark.asyncio
async def test_update_event_does_not_wake_broker_connectivity_lost_state() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload, auto_recovery_grace_period=60)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: current_state(supervisor) is ConnectedState)

    fake_ib.errorEvent.emit(-1, 1100, "Connectivity lost", ibi.Contract())

    assert await wait_for_condition(
        lambda: current_state(supervisor) is BrokerConnectivityLostState
    )

    fake_ib.updateEvent.emit()
    await asyncio.sleep(0)

    assert current_state(supervisor) is BrokerConnectivityLostState
    assert fake_ib.probe_count == 1
    assert fake_ib.disconnect_count == 0
    assert workload.starts == 1

    await stop_and_wait(supervisor, task)


@pytest.mark.parametrize("code", [2103, 2105, 2157, 10182, 2104, 2106, 2158])
@pytest.mark.asyncio
async def test_weak_data_farm_messages_do_not_change_state(
    code: int, caplog: pytest.LogCaptureFixture
) -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: current_state(supervisor) is ConnectedState)
    assert not supervisor.connection_unavailable.is_set()

    caplog.set_level("DEBUG", logger="haymaker.supervisor.supervisor")
    caplog.clear()
    fake_ib.errorEvent.emit(-1, code, "Weak farm message", ibi.Contract())
    await asyncio.sleep(0)

    assert current_state(supervisor) is ConnectedState
    assert not supervisor.connection_unavailable.is_set()
    assert not supervisor._restart_requested.is_set()
    assert fake_ib.probe_count == 1
    assert fake_ib.disconnect_count == 0
    assert workload.starts == 1
    assert f"broker message {code}: Weak farm message" in caplog.text

    await stop_and_wait(supervisor, task)


@pytest.mark.asyncio
async def test_weak_data_farm_logging_can_be_disabled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload, log_datafarm_status=False)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: current_state(supervisor) is ConnectedState)
    assert not supervisor.connection_unavailable.is_set()

    caplog.set_level("DEBUG", logger="haymaker.supervisor.supervisor")
    caplog.clear()
    fake_ib.errorEvent.emit(-1, 2105, "Weak farm message", ibi.Contract())
    await asyncio.sleep(0)

    assert current_state(supervisor) is ConnectedState
    assert not supervisor.connection_unavailable.is_set()
    assert not supervisor._restart_requested.is_set()
    assert "Weak farm message" not in caplog.text

    await stop_and_wait(supervisor, task)


@pytest.mark.asyncio
async def test_live_update_failure_restart_is_disabled_by_zero_delay() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload, live_update_failure_restart_delay=0)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: current_state(supervisor) is ConnectedState)

    fake_ib.errorEvent.emit(-1, 10182, "Failed to request live updates", ibi.Contract())
    await asyncio.sleep(0.05)

    assert current_state(supervisor) is ConnectedState
    assert fake_ib.disconnect_count == 0
    assert workload.starts == 1

    await stop_and_wait(supervisor, task)


@pytest.mark.asyncio
async def test_live_update_failure_requests_restart_after_quiet_period(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(
        fake_ib, workload, live_update_failure_restart_delay=0.01
    )
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: current_state(supervisor) is ConnectedState)

    caplog.set_level("DEBUG", logger="haymaker.supervisor.states")
    fake_ib.errorEvent.emit(-1, 10182, "Failed to request live updates", ibi.Contract())

    assert current_state(supervisor) is ConnectedState
    assert fake_ib.disconnect_count == 0
    assert await wait_for_condition(lambda: fake_ib.disconnect_count == 1)
    assert (
        "Live-update failure quiet period elapsed; requesting workload restart."
        in caplog.text
    )

    await stop_and_wait(supervisor, task)


@pytest.mark.asyncio
async def test_live_update_failure_quiet_period_resets_on_repeated_message() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(
        fake_ib, workload, live_update_failure_restart_delay=0.05
    )
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: current_state(supervisor) is ConnectedState)

    fake_ib.errorEvent.emit(-1, 10182, "Failed to request live updates", ibi.Contract())
    await asyncio.sleep(0.03)
    fake_ib.errorEvent.emit(-1, 10182, "Failed to request live updates", ibi.Contract())
    await asyncio.sleep(0.03)

    assert fake_ib.disconnect_count == 0
    assert current_state(supervisor) is ConnectedState
    assert await wait_for_condition(lambda: fake_ib.disconnect_count == 1)

    await stop_and_wait(supervisor, task)


@pytest.mark.asyncio
async def test_live_update_failure_timer_stops_when_connected_state_exits() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(
        fake_ib,
        workload,
        live_update_failure_restart_delay=0.05,
        probe_timeout=1,
    )
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: current_state(supervisor) is ConnectedState)

    fake_ib.errorEvent.emit(-1, 10182, "Failed to request live updates", ibi.Contract())
    await asyncio.sleep(0.01)
    fake_ib.probe_delays = [60]
    fake_ib.timeoutEvent.emit(20)

    assert await wait_for_condition(lambda: current_state(supervisor) is ProbingState)
    await asyncio.sleep(0.06)

    assert current_state(supervisor) is ProbingState
    assert fake_ib.disconnect_count == 0

    await stop_and_wait(supervisor, task)


@pytest.mark.asyncio
async def test_broker_connectivity_loss_wins_over_pending_live_update_restart() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(
        fake_ib,
        workload,
        live_update_failure_restart_delay=0.05,
        auto_recovery_grace_period=1,
    )
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: current_state(supervisor) is ConnectedState)

    fake_ib.errorEvent.emit(-1, 10182, "Failed to request live updates", ibi.Contract())
    await asyncio.sleep(0.01)
    fake_ib.errorEvent.emit(-1, 1100, "Connectivity lost", ibi.Contract())

    assert await wait_for_condition(
        lambda: current_state(supervisor) is BrokerConnectivityLostState
    )
    await asyncio.sleep(0.06)

    assert current_state(supervisor) is BrokerConnectivityLostState
    assert fake_ib.disconnect_count == 0
    assert workload.starts == 1

    await stop_and_wait(supervisor, task)


@pytest.mark.asyncio
async def test_timeout_event_in_connected_state_moves_to_probing() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: current_state(supervisor) is ConnectedState)

    fake_ib.probe_delays = [60]
    fake_ib.timeoutEvent.emit(20)

    assert await wait_for_condition(lambda: fake_ib.probe_count == 2)
    assert current_state(supervisor) is ProbingState
    assert fake_ib.disconnect_count == 0
    assert workload.starts == 1

    await stop_and_wait(supervisor, task)


@pytest.mark.asyncio
async def test_failed_probe_transitions_to_backoff_restart() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload, connection_lost_retry_delay=60)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: current_state(supervisor) is ConnectedState)

    fake_ib.probe_results = [[]]
    fake_ib.timeoutEvent.emit(20)

    assert await wait_for_condition(
        lambda: current_state(supervisor) is BackoffRestartingState
    )
    assert await wait_for_condition(lambda: workload.stops == ["restart requested"])
    assert await wait_for_condition(lambda: fake_ib.disconnect_count == 1)
    assert supervisor.connection_unavailable.is_set()

    await stop_and_wait(supervisor, task)


@pytest.mark.asyncio
async def test_backoff_restart_ignores_restart_but_honors_stop() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload, connection_lost_retry_delay=60)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: current_state(supervisor) is ConnectedState)

    fake_ib.probe_results = [[]]
    fake_ib.timeoutEvent.emit(20)

    assert await wait_for_condition(
        lambda: current_state(supervisor) is BackoffRestartingState
    )
    assert not supervisor.request_restart("ignored during backoff")
    assert not supervisor._restart_requested.is_set()

    supervisor.stop()
    await asyncio.wait_for(task, timeout=1)

    assert current_state(supervisor) is StoppedState
    assert fake_ib.connect_attempts == 1


@pytest.mark.asyncio
async def test_connecting_ignores_redundant_restart_but_honors_stop() -> None:
    fake_ib = FakeIB()
    fake_ib.fail_connect_attempts = 100
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload, retry_delay=60)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(
        lambda: current_state(supervisor) is ConnectingState
    )
    assert await wait_for_condition(lambda: fake_ib.connect_attempts == 1)

    assert not supervisor.request_restart("ignored while connecting")
    assert not supervisor._restart_requested.is_set()

    supervisor.stop()
    await asyncio.wait_for(task, timeout=1)

    assert current_state(supervisor) is StoppedState
    assert workload.starts == 0


@pytest.mark.asyncio
async def test_restart_request_overrides_pending_connectivity_loss() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: current_state(supervisor) is ConnectedState)

    fake_ib.errorEvent.emit(-1, 1100, "Connectivity lost", ibi.Contract())
    supervisor.request_restart("manual restart")

    assert await wait_for_condition(lambda: workload.starts == 2)
    assert workload.stops == ["restart requested"]
    assert fake_ib.connect_attempts == 2
    assert fake_ib.disconnect_count == 1

    await stop_and_wait(supervisor, task)


@pytest.mark.asyncio
async def test_stop_during_backoff_restart_cleanup_prevents_reconnect() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    workload.stop_started = asyncio.Event()
    workload.release_stop = asyncio.Event()
    supervisor = make_supervisor(fake_ib, workload)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: current_state(supervisor) is ConnectedState)

    fake_ib.probe_results = [[]]
    fake_ib.timeoutEvent.emit(20)
    await asyncio.wait_for(workload.stop_started.wait(), timeout=1)

    supervisor.stop()

    await asyncio.sleep(0.05)
    assert workload.stops == ["restart requested"]
    assert fake_ib.disconnect_count == 0
    assert not task.done()

    workload.release_stop.set()

    await asyncio.wait_for(task, timeout=1)

    assert current_state(supervisor) is StoppedState
    assert workload.stops == ["restart requested"]
    assert fake_ib.connect_attempts == 1
    assert fake_ib.disconnect_count == 1


@pytest.mark.asyncio
async def test_update_event_is_ignored_while_connected() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: current_state(supervisor) is ConnectedState)

    fake_ib.updateEvent.emit()
    await asyncio.sleep(0)

    assert current_state(supervisor) is ConnectedState
    assert fake_ib.probe_count == 1
    assert workload.starts == 1

    await stop_and_wait(supervisor, task)


@pytest.mark.asyncio
async def test_data_maintained_message_resumes_without_restart_by_default() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload, auto_recovery_grace_period=60)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: current_state(supervisor) is ConnectedState)

    fake_ib.errorEvent.emit(-1, 1100, "Connectivity lost", ibi.Contract())
    assert await wait_for_condition(
        lambda: current_state(supervisor) is BrokerConnectivityLostState
    )

    fake_ib.errorEvent.emit(-1, 1102, "Connectivity restored", ibi.Contract())

    assert await wait_for_condition(lambda: current_state(supervisor) is ConnectedState)
    assert fake_ib.connect_attempts == 1
    assert fake_ib.disconnect_count == 0
    assert workload.starts == 1

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


def test_connection_settings_from_config_uses_flat_mapping_and_client_id() -> None:
    settings = ConnectionSettings.from_config(
        {
            "host": "gateway",
            "port": 4001,
            "clientId": 999,
            "connectTimeout": 3,
            "restart_time": 5,
            "retryDelay": 7,
            "appTimeout": 11,
            "probeTimeout": 13,
            "connection_lost_retry": 15,
            "auto_recovery_grace_period": 17,
            "max_recoveries": 21,
            "recovery_warning_after": 19,
            "recovery_warning_interval": 23,
            "restart_on_recovered_connection": True,
            "live_update_failure_restart_delay": 29,
            "log_datafarm_status": False,
        },
        client_id=42,
    )

    assert settings.host == "gateway"
    assert settings.port == 4001
    assert settings.client_id == 42
    assert settings.connect_timeout == 3
    assert settings.retry_delay == 7
    assert settings.app_timeout == 11
    assert settings.probe_timeout == 13
    assert settings.connection_lost_retry_delay == 15
    assert settings.auto_recovery_grace_period == 17
    assert settings.max_recoveries == 21
    assert settings.restart_on_recovered_connection is True
    assert settings.live_update_failure_restart_delay == 29
    assert settings.log_datafarm_status is False
    assert not hasattr(settings, "restart_delay")
    assert not hasattr(settings, "recovery_warning_after")
    assert not hasattr(settings, "recovery_warning_interval")


def test_connection_settings_from_config_uses_defaults() -> None:
    settings = ConnectionSettings.from_config({}, client_id=51)

    assert settings.host == "127.0.0.1"
    assert settings.port == 4002
    assert settings.client_id == 51
    assert settings.connect_timeout == 15
    assert settings.retry_delay == 30
    assert settings.app_timeout == 90
    assert settings.probe_timeout == 15
    assert settings.connection_lost_retry_delay == 90
    assert settings.auto_recovery_grace_period == 120
    assert settings.max_recoveries == 10
    assert settings.restart_on_recovered_connection is False
    assert settings.live_update_failure_restart_delay == 0
    assert settings.log_datafarm_status is True


def test_request_restart_returns_true_when_state_supervisor_accepts_request() -> None:
    fake_ib = FakeIB()
    supervisor = make_supervisor(fake_ib)
    supervisor._state = ConnectedState(supervisor)

    assert supervisor.request_restart("manual restart") is True
    assert supervisor._restart_requested.is_set()
