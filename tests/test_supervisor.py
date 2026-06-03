import asyncio
import logging
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import ib_insync as ibi
import pytest
from helpers import wait_for_condition

import haymaker.supervisor as supervisor_module
from haymaker.supervisor import (
    ConnectionSettings,
    ConnectionSupervisor,
    SupervisorState,
    contract_refresh_is_overdue,
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
    stop_on_completion = False

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
    assert fake_ib.disconnect_count == 2
    assert "test restart" in workload.stops


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


def test_contract_refresh_age_is_checked_without_broker_request():
    now = datetime(2026, 6, 1, tzinfo=timezone.utc)

    assert contract_refresh_is_overdue(now - timedelta(hours=25), 24 * 60 * 60, now)
    assert not contract_refresh_is_overdue(now - timedelta(hours=23), 24 * 60 * 60, now)
    assert not contract_refresh_is_overdue(None, 24 * 60 * 60, now)
