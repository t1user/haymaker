import asyncio
from typing import Any, cast

import ib_insync as ibi
import pytest
from helpers import wait_for_condition

from haymaker.supervisor import ConnectionSettings, ConnectionSupervisor
from haymaker.supervisor.states import (
    ConnectedState,
    StoppedState,
    WaitingForBrokerState,
)


class FakeIB:
    """Small IB stand-in exposing the event surface used by ConnectionSupervisor."""

    def __init__(self) -> None:
        self.errorEvent = ibi.Event()
        self.disconnectedEvent = ibi.Event()
        self.timeoutEvent = ibi.Event()
        self.updateEvent = ibi.Event()
        self.connected = False
        self.connect_attempts = 0
        self.disconnect_count = 0
        self.fail_connect_attempts = 0
        self.timeouts: list[float] = []
        self.probe_results: list[list[object]] = [[object()]]
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
    """Create a supervisor with fast deterministic test timings."""

    return ConnectionSupervisor(
        cast(ibi.IB, fake_ib),
        workload or FakeWorkload(),
        ConnectionSettings(
            retry_delay=0,
            app_timeout=20,
            probe_timeout=0.01,
            auto_recovery_grace_period=0.02,
            **settings,
        ),
    )


async def stop_and_wait(
    supervisor: ConnectionSupervisor, task: asyncio.Task[None]
) -> None:
    """Request supervisor shutdown and wait for the run task to finish."""

    supervisor.stop()
    await asyncio.wait_for(task, timeout=1)


@pytest.mark.asyncio
async def test_run_connects_probes_and_starts_workload() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload)

    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: supervisor.state is ConnectedState)
    assert workload.starts == 1
    assert fake_ib.connect_attempts == 1
    assert fake_ib.probe_count == 1
    assert fake_ib.timeouts == [20]

    await stop_and_wait(supervisor, task)

    assert supervisor.state is StoppedState
    assert workload.stops == ["supervisor stopped"]
    assert fake_ib.disconnect_count == 1


@pytest.mark.asyncio
async def test_completed_workload_stops_supervisor() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload(complete_immediately=True)
    supervisor = make_supervisor(fake_ib, workload)

    await supervisor.run()

    assert supervisor.state is StoppedState
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

    supervisor.request_restart("manual restart")

    assert await wait_for_condition(lambda: workload.starts == 2)
    assert workload.stops == ["manual restart"]
    assert fake_ib.connect_attempts == 2
    assert fake_ib.disconnect_count == 1

    await stop_and_wait(supervisor, task)


@pytest.mark.asyncio
async def test_duplicate_restart_requests_keep_first_reason() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: workload.starts == 1)

    supervisor.request_restart("first restart")
    supervisor.request_restart("second restart")

    assert await wait_for_condition(lambda: workload.starts == 2)
    assert workload.stops == ["first restart"]

    await stop_and_wait(supervisor, task)


@pytest.mark.asyncio
async def test_unexpected_disconnect_restarts_workload() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: workload.starts == 1)

    fake_ib.disconnect()

    assert await wait_for_condition(lambda: workload.starts == 2)
    assert workload.stops == ["IB socket disconnected"]
    assert fake_ib.connect_attempts == 2
    assert fake_ib.disconnect_count == 1

    await stop_and_wait(supervisor, task)


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
async def test_timeout_with_recent_broker_message_waits_for_recovery() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: supervisor.state is ConnectedState)

    fake_ib.errorEvent.emit(-1, 1100, "Connectivity lost", ibi.Contract())
    fake_ib.timeoutEvent.emit(20)

    assert await wait_for_condition(lambda: supervisor.state is WaitingForBrokerState)
    assert fake_ib.disconnect_count == 0
    assert workload.starts == 1

    fake_ib.updateEvent.emit()

    assert await wait_for_condition(lambda: supervisor.state is ConnectedState)
    assert fake_ib.disconnect_count == 0
    assert workload.starts == 1

    await stop_and_wait(supervisor, task)


@pytest.mark.asyncio
async def test_data_maintained_message_resumes_without_restart_by_default() -> None:
    fake_ib = FakeIB()
    workload = FakeWorkload()
    supervisor = make_supervisor(fake_ib, workload)
    task = asyncio.create_task(supervisor.run())

    assert await wait_for_condition(lambda: supervisor.state is ConnectedState)

    fake_ib.errorEvent.emit(-1, 1100, "Connectivity lost", ibi.Contract())
    fake_ib.timeoutEvent.emit(20)
    assert await wait_for_condition(lambda: supervisor.state is WaitingForBrokerState)

    fake_ib.errorEvent.emit(-1, 1102, "Connectivity restored", ibi.Contract())

    assert await wait_for_condition(lambda: supervisor.state is ConnectedState)
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
    assert workload.stops == [
        "broker connectivity restored with data maintained (1102)"
    ]
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
    assert workload.stops == ["broker connectivity restored with data lost (1101)"]
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
    assert workload.stops == ["broker reset API socket port (1300)"]
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
            "auto_recovery_grace_period": 17,
            "recovery_warning_after": 19,
            "recovery_warning_interval": 23,
            "restart_on_recovered_connection": True,
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
    assert settings.auto_recovery_grace_period == 17
    assert settings.restart_on_recovered_connection is True
    assert not hasattr(settings, "restart_delay")
    assert not hasattr(settings, "recovery_warning_after")
    assert not hasattr(settings, "recovery_warning_interval")


def test_connection_settings_from_config_uses_defaults() -> None:
    settings = ConnectionSettings.from_config({}, client_id=51)

    assert settings.host == "127.0.0.1"
    assert settings.port == 4002
    assert settings.client_id == 51
    assert settings.connect_timeout == 2
    assert settings.retry_delay == 2
    assert settings.app_timeout == 20
    assert settings.probe_timeout == 4
    assert settings.auto_recovery_grace_period == 120
    assert settings.restart_on_recovered_connection is False
