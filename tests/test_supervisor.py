import asyncio
import logging
from dataclasses import replace

import ib_insync as ibi
import pytest
from helpers import wait_for_condition

import haymaker.supervisor as supervisor_module
from haymaker.supervisor import (
    ConnectionSettings,
    ConnectionSupervisor,
    SupervisorState,
)


class FakeIB:
    def __init__(self):
        self.errorEvent = ibi.Event()
        self.disconnectedEvent = ibi.Event()
        self.timeoutEvent = ibi.Event()
        self.connected = False
        self.connect_count = 0
        self.disconnect_count = 0
        self.timeouts = []
        self.probe_result = [object()]
        self.probe_count = 0

    async def connectAsync(self, *args, **kwargs):
        self.connected = True
        self.connect_count += 1

    def disconnect(self):
        if self.connected:
            self.connected = False
            self.disconnect_count += 1
            self.disconnectedEvent.emit()

    def isConnected(self):
        return self.connected

    def setTimeout(self, timeout):
        self.timeouts.append(timeout)

    def reqHistoricalDataAsync(self, *args, **kwargs):
        async def request():
            self.probe_count += 1
            return self.probe_result

        return request()


@pytest.fixture
def fake_ib():
    return FakeIB()


class FakeWorkload:
    stop_supervisor_on_completion = False

    def __init__(self):
        self.starts = []
        self.stops = []

    async def start(self):
        self.starts.append("started")

    async def stop(self, reason):
        self.stops.append(reason)


@pytest.fixture
def supervisor(fake_ib):
    return ConnectionSupervisor(
        fake_ib,
        FakeWorkload(),
        ConnectionSettings(
            restart_delay=0,
            retry_delay=0,
            auto_recovery_grace_period=0.01,
        ),
    )


@pytest.mark.asyncio
async def test_waits_for_broker_auto_recovery_without_restarting(supervisor):
    supervisor._running = True

    supervisor.onErrEvent(-1, 1100, "Connectivity lost", ibi.Contract())

    assert supervisor.state == SupervisorState.WAITING_FOR_BROKER
    assert not supervisor._restart_requested.is_set()

    supervisor.onErrEvent(-1, 1102, "Connectivity restored", ibi.Contract())
    await asyncio.sleep(0.02)

    assert supervisor.state == SupervisorState.CONNECTED
    assert not supervisor._restart_requested.is_set()
    supervisor.stop()


@pytest.mark.asyncio
async def test_requests_restart_after_auto_recovery_grace_period(supervisor):
    supervisor._running = True

    supervisor.onErrEvent(-1, 2110, "Connectivity broken", ibi.Contract())
    assert await wait_for_condition(lambda: supervisor._restart_requested.is_set())

    assert supervisor.state == SupervisorState.WAITING_FOR_BROKER
    supervisor.stop()


def test_data_lost_message_requests_restart(supervisor):
    supervisor._running = True

    supervisor.onErrEvent(-1, 1101, "Data lost", ibi.Contract())

    assert supervisor._restart_requested.is_set()
    supervisor.stop()


def test_duplicate_restart_requests_are_coalesced(fake_ib):
    supervisor = ConnectionSupervisor(fake_ib, FakeWorkload())
    supervisor._running = True

    supervisor.request_restart("first")
    supervisor.request_restart("second")

    assert supervisor._pending_restart_reason == "first"
    supervisor.stop()


def test_connection_settings_from_config_uses_flat_mapping_and_explicit_client_id():
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
        },
        client_id=42,
    )

    assert settings.host == "gateway"
    assert settings.port == 4001
    assert settings.client_id == 42
    assert settings.connect_timeout == 3
    assert settings.restart_delay == 5
    assert settings.retry_delay == 7
    assert settings.app_timeout == 11
    assert settings.probe_timeout == 13
    assert settings.auto_recovery_grace_period == 17
    assert settings.recovery_warning_after == 19
    assert settings.recovery_warning_interval == 23


def test_connection_settings_from_config_uses_live_defaults():
    settings = ConnectionSettings.from_config({}, client_id=51)

    assert settings.host == "127.0.0.1"
    assert settings.port == 4002
    assert settings.client_id == 51
    assert settings.connect_timeout == 2
    assert settings.restart_delay == 30
    assert settings.retry_delay == 2
    assert settings.app_timeout == 20
    assert settings.probe_timeout == 4


@pytest.mark.asyncio
async def test_restart_cycle_reconnects_and_runs_workload_again(fake_ib):
    workload = FakeWorkload()

    supervisor = ConnectionSupervisor(
        fake_ib,
        workload,
        ConnectionSettings(restart_delay=0, retry_delay=0),
    )
    runner = asyncio.create_task(supervisor.run())
    assert await wait_for_condition(lambda: len(workload.starts) == 1)

    supervisor.request_restart("test restart")
    assert await wait_for_condition(lambda: len(workload.starts) == 2)

    supervisor.stop()
    await runner
    assert fake_ib.connect_count == 2
    assert fake_ib.probe_count == 2
    assert fake_ib.disconnect_count == 2
    assert "test restart" in workload.stops


@pytest.mark.asyncio
async def test_dataloader_settings_probe_before_workload(fake_ib):
    workload = FakeWorkload()
    supervisor = ConnectionSupervisor(
        fake_ib,
        workload,
        ConnectionSettings.from_config({"restart_time": 0, "retryDelay": 0}, 77),
    )

    runner = asyncio.create_task(supervisor.run())
    assert await wait_for_condition(lambda: workload.starts == ["started"])

    supervisor.stop()
    await runner
    assert fake_ib.probe_count == 1


@pytest.mark.asyncio
async def test_timeout_does_not_probe_while_waiting_for_broker(supervisor, fake_ib):
    supervisor._running = True
    supervisor.state = SupervisorState.WAITING_FOR_BROKER

    supervisor.onTimeoutEvent(20)
    await asyncio.sleep(0)

    assert fake_ib.probe_count == 0
    supervisor.stop()


@pytest.mark.asyncio
async def test_failed_timeout_probe_requests_restart(supervisor, fake_ib):
    supervisor._running = True
    fake_ib.connected = True
    fake_ib.probe_result = []
    supervisor.state = SupervisorState.CONNECTED

    supervisor.onTimeoutEvent(20)
    assert await wait_for_condition(lambda: supervisor._restart_requested.is_set())

    assert supervisor.state == SupervisorState.CONNECTED
    supervisor.stop()


def test_delayed_recovery_warning_is_throttled(
    supervisor, caplog, monkeypatch: pytest.MonkeyPatch
):
    caplog.set_level(logging.WARNING)
    supervisor.settings = replace(
        supervisor.settings,
        recovery_warning_after=10,
        recovery_warning_interval=20,
    )
    supervisor._recovery_started_at = 0

    monkeypatch.setattr(supervisor_module, "monotonic", lambda: 11)
    supervisor._warn_if_recovery_delayed()
    supervisor._warn_if_recovery_delayed()

    assert caplog.text.count("IB connection recovery is still pending.") == 1
