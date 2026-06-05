"""Connection lifecycle management for Interactive Brokers clients."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from time import monotonic
from typing import Any, Protocol

import ib_insync as ibi

log = logging.getLogger(__name__)


class SupervisorState(str, Enum):
    """Current connection supervisor lifecycle state."""

    STOPPED = "stopped"
    CONNECTING = "connecting"
    PROBING = "probing"
    CONNECTED = "connected"
    WAITING_FOR_BROKER = "waiting_for_broker"
    RESTARTING = "restarting"
    STOPPING = "stopping"


ALLOWED_SUPERVISOR_TRANSITIONS: Mapping[SupervisorState, frozenset[SupervisorState]] = {
    SupervisorState.STOPPED: frozenset({SupervisorState.CONNECTING}),
    SupervisorState.CONNECTING: frozenset(
        {SupervisorState.PROBING, SupervisorState.STOPPING}
    ),
    SupervisorState.PROBING: frozenset(
        {
            SupervisorState.CONNECTED,
            SupervisorState.RESTARTING,
            SupervisorState.STOPPING,
        }
    ),
    SupervisorState.CONNECTED: frozenset(
        {
            SupervisorState.WAITING_FOR_BROKER,
            SupervisorState.RESTARTING,
            SupervisorState.STOPPING,
        }
    ),
    SupervisorState.WAITING_FOR_BROKER: frozenset(
        {
            SupervisorState.CONNECTED,
            SupervisorState.RESTARTING,
            SupervisorState.STOPPING,
        }
    ),
    SupervisorState.RESTARTING: frozenset(
        {SupervisorState.CONNECTING, SupervisorState.STOPPING}
    ),
    SupervisorState.STOPPING: frozenset({SupervisorState.STOPPED}),
}


class SupervisorWorkload(Protocol):
    """Workload run by :class:`ConnectionSupervisor` after IB is usable."""

    stop_supervisor_on_completion: bool

    async def start(self) -> None:
        """Start or resume work after a usable IB connection is available."""

    async def stop(self, reason: str) -> None:
        """Release active work before the supervisor reconnects or exits."""


@dataclass(frozen=True)
class BrokerMessage:
    """Recent IB broker status message used for recovery decisions."""

    code: int
    message: str
    received_at: float


@dataclass(frozen=True)
class ConnectionSettings:
    """Connection and recovery settings for :class:`ConnectionSupervisor`."""

    host: str = "127.0.0.1"
    port: int = 4002
    client_id: int = 0
    connect_timeout: float = 2
    restart_delay: float = 30
    retry_delay: float = 2
    app_timeout: float = 20
    probe_contract: ibi.Contract = field(default_factory=lambda: ibi.Forex("EURUSD"))
    probe_timeout: float = 4
    auto_recovery_grace_period: float = 120
    recovery_warning_after: float = 300
    recovery_warning_interval: float = 900
    restart_on_recovered_connection: bool = False

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], client_id: int
    ) -> ConnectionSettings:
        """Create connection settings from a flat config mapping.

        Args:
            config: Mapping with connection keys directly available.
            client_id: IB API client ID chosen by the caller.
        """

        return cls(
            host=config.get("host", "127.0.0.1"),
            port=config.get("port", 4002),
            client_id=client_id,
            connect_timeout=config.get("connectTimeout", 2),
            restart_delay=config.get("restart_time", 30),
            retry_delay=config.get("retryDelay", 2),
            app_timeout=config.get("appTimeout", 20),
            probe_contract=config.get("probeContract") or ibi.Forex("EURUSD"),
            probe_timeout=config.get("probeTimeout", 4),
            auto_recovery_grace_period=config.get("auto_recovery_grace_period", 120),
            recovery_warning_after=config.get("recovery_warning_after", 300),
            recovery_warning_interval=config.get("recovery_warning_interval", 900),
            restart_on_recovered_connection=config.get(
                "restart_on_recovered_connection", False
            ),
        )


@dataclass
class ConnectionSupervisor:
    """Maintain an owned IB socket connection around one workload.

    ``run()`` is the only execution entry point. It connects the configured
    ``ib_insync.IB`` client, verifies broker usability with a probe, starts the
    workload, performs requested reconnect cycles, and owns final cleanup.
    ``stop()`` is a synchronous request for the active ``run()`` loop to exit; it
    does not disconnect the socket or await workload cleanup itself.
    ``request_restart()`` coalesces reconnect requests for the active ``run()``
    loop.

    The supervisor deliberately does not manage the TWS or IB Gateway process.
    It is only for sockets owned by Haymaker, such as live trading and managed
    dataloader runs. Attached dataloader work should not use this class because
    it must not connect, disconnect, or request restarts on a borrowed socket.
    A workload is stopped only when an active workload task is interrupted for a
    restart or shutdown; a workload that completes normally owns its own cleanup.

    Args:
        ib: Shared Interactive Brokers client.
        workload: Workload started after the connection has been verified.
        settings: Connection, probe, and recovery settings.
    """

    ib: ibi.IB
    workload: SupervisorWorkload
    settings: ConnectionSettings = field(default_factory=ConnectionSettings)
    state: SupervisorState = field(default=SupervisorState.STOPPED, init=False)
    _running: bool = field(default=False, init=False, repr=False)
    _restart_requested: asyncio.Event = field(
        default_factory=asyncio.Event, init=False, repr=False
    )
    _pending_restart_reason: str | None = field(default=None, init=False, repr=False)
    _intentional_disconnect: bool = field(default=False, init=False, repr=False)
    _auto_recovery_handle: asyncio.TimerHandle | None = field(
        default=None, init=False, repr=False
    )
    _probe_task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _workload_task: asyncio.Task[None] | None = field(
        default=None, init=False, repr=False
    )
    _recovery_started_at: float | None = field(default=None, init=False, repr=False)
    _last_warning_at: float | None = field(default=None, init=False, repr=False)
    _recent_broker_messages: deque[BrokerMessage] = field(
        default_factory=deque, init=False, repr=False
    )

    RECOVERABLE_BROKER_CODES = frozenset({1100, 2110, 2103, 2105, 2157, 10182})
    SOCKET_RESET_CODE = 1300
    DATA_LOST_CODE = 1101
    DATA_MAINTAINED_CODE = 1102
    RECENT_BROKER_MESSAGE_TTL = 300
    RECENT_BROKER_MESSAGE_LIMIT = 20

    def __post_init__(self) -> None:
        # IB calls this errorEvent, but most payloads are broker messages, not
        # actionable application errors. Keep "error" out of callback logs.
        self.ib.errorEvent += self.onErrEvent
        self.ib.disconnectedEvent += self.onDisconnectedEvent
        self.ib.timeoutEvent += self.onTimeoutEvent
        self.ib.updateEvent += self.onUpdateEvent

    async def run(self) -> None:
        """Run connection and restart cycles until explicitly stopped."""

        if self._running:
            raise RuntimeError("Connection supervisor is already running.")

        self._running = True
        self._restart_requested.clear()
        self._pending_restart_reason = None
        try:
            while self._running:
                if not await self._connect_and_verify():
                    break

                if self._restart_requested.is_set():
                    reason = self._consume_restart_reason()
                    if self._running:
                        await self._prepare_restart(reason)
                    continue

                self._start_workload()
                reason = await self._wait_for_restart_or_completion()
                if self._workload_task is not None:
                    await self._stop_workload(reason)
                if self._running:
                    await self._prepare_restart(reason)
        finally:
            self._running = False
            if self.state not in {
                SupervisorState.STOPPED,
                SupervisorState.STOPPING,
            }:
                self._transition_to(SupervisorState.STOPPING)
            self._cancel_connection_recovery()
            if self._workload_task is not None:
                await self._stop_workload("supervisor stopped")
            self._disconnect()
            if self.state != SupervisorState.STOPPED:
                self._transition_to(SupervisorState.STOPPED)

    def stop(self) -> None:
        """Request supervisor shutdown.

        The run loop performs awaited workload cleanup and marks STOPPED.
        """

        if not self._running and self.state == SupervisorState.STOPPED:
            return
        self._running = False
        if self.state not in {
            SupervisorState.STOPPED,
            SupervisorState.STOPPING,
        }:
            self._transition_to(SupervisorState.STOPPING)
        self._restart_requested.set()

    def _cancel_connection_recovery(self) -> None:
        """Cancel pending recovery/probe work for a connection cycle."""

        self._cancel_auto_recovery_wait()
        self._cancel_probe()

    def request_restart(self, reason: str) -> None:
        """Request one restart cycle, coalescing concurrent requests."""

        if not self._running:
            log.debug(f"Ignoring restart request while stopped: {reason}")
            return

        if self.state in {
            SupervisorState.RESTARTING,
            SupervisorState.STOPPING,
        }:
            log.debug(f"Restart already in progress; additional reason: {reason}")
            return

        if not self._restart_requested.is_set():
            self._pending_restart_reason = reason
            self._start_recovery_clock()
            log.debug(f"Restart requested: {reason}")
            self._restart_requested.set()
        else:
            log.debug(f"Restart already requested; additional reason: {reason}")

    def onErrEvent(
        self,
        req_id: int,
        code: int,
        message: str,
        contract: ibi.Contract,
    ) -> None:
        """Translate selected broker messages into lifecycle actions."""

        self._remember_broker_message(code, message)
        if code == self.DATA_LOST_CODE:
            self.request_restart(
                f"broker connectivity restored with data lost ({code})"
            )
        elif code == self.SOCKET_RESET_CODE:
            self.request_restart(f"broker reset API socket port ({code})")
        elif code == self.DATA_MAINTAINED_CODE:
            self._handle_data_maintained_message(code)

    def onDisconnectedEvent(self) -> None:
        """Restart after an unexpected API socket disconnection."""

        if self._running and not self._intentional_disconnect:
            self.request_restart("IB socket disconnected")

    def onTimeoutEvent(self, idle_period: float) -> None:
        """Probe an otherwise idle connection before requesting a restart."""

        if not self._running:
            return

        if self.state == SupervisorState.WAITING_FOR_BROKER:
            self._warn_if_recovery_delayed()
            return

        if self.state != SupervisorState.CONNECTED:
            return

        broker_message = self._latest_recoverable_broker_message()
        if broker_message:
            self._wait_for_broker_auto_recovery(broker_message)
            return

        self._request_probe(f"No IB traffic for {idle_period}s")

    def onUpdateEvent(self) -> None:
        """Probe broker recovery when IB traffic resumes during degraded state."""

        if self._running and self.state == SupervisorState.WAITING_FOR_BROKER:
            self._request_probe("IB traffic resumed while waiting for broker recovery")

    async def _connect_and_verify(self) -> bool:
        self._transition_to(SupervisorState.CONNECTING)
        if not await self._connect():
            return False

        self._transition_to(SupervisorState.PROBING)
        await self._wait_until_probe_succeeds()

        if not self._running:
            return False

        if self._restart_requested.is_set():
            return True

        if not self.ib.isConnected():
            return False

        self._mark_connected()
        if self.settings.app_timeout:
            self.ib.setTimeout(self.settings.app_timeout)
        return True

    def _start_workload(self) -> None:
        if self._workload_task is None or self._workload_task.done():
            self._workload_task = asyncio.create_task(
                self.workload.start(), name="connection-supervisor-workload"
            )

    async def _wait_for_restart_or_completion(self) -> str:
        while self._running:
            restart_wait = asyncio.create_task(
                self._restart_requested.wait(),
                name="connection-supervisor-restart-wait",
            )
            tasks: set[asyncio.Task[Any]] = {restart_wait}
            if self._workload_task and not self._workload_task.done():
                tasks.add(self._workload_task)

            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            if restart_wait in pending:
                restart_wait.cancel()
                await asyncio.gather(restart_wait, return_exceptions=True)

            if restart_wait in done and self._restart_requested.is_set():
                return self._consume_restart_reason()

            if self._workload_task in done:
                self._consume_workload_result()
                self._workload_task = None
                if self.workload.stop_supervisor_on_completion:
                    self._running = False
                    return "workload completed"
                log.debug("Connection workload completed; waiting for restart.")

        return "supervisor stopping"

    def _consume_restart_reason(self) -> str:
        if not self._running:
            self._pending_restart_reason = None
            self._restart_requested.clear()
            return "supervisor stopping"

        reason = self._pending_restart_reason or "restart requested"
        self._pending_restart_reason = None
        self._restart_requested.clear()
        return reason

    def _consume_workload_result(self) -> None:
        if self._workload_task is None:
            return

        try:
            self._workload_task.result()
        except asyncio.CancelledError:
            log.debug("Connection workload task cancelled.")
        except Exception:
            log.exception("Connection workload task failed.")

    async def _stop_workload(self, reason: str) -> None:
        task = self._workload_task
        if task is None:
            return

        await self.workload.stop(reason)
        if task.done():
            self._consume_workload_result()
        else:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._workload_task = None

    async def _prepare_restart(self, reason: str) -> None:
        self._transition_to(SupervisorState.RESTARTING)
        log.debug(f"Restarting connection: {reason}")
        self._cancel_connection_recovery()
        self._disconnect()
        await asyncio.sleep(self.settings.restart_delay)

    async def _connect(self) -> bool:
        while self._running and not self.ib.isConnected():
            try:
                log.debug(
                    f"Connecting to IB at {self.settings.host}:{self.settings.port}."
                )
                await self.ib.connectAsync(
                    self.settings.host,
                    self.settings.port,
                    clientId=self.settings.client_id,
                    timeout=self.settings.connect_timeout,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.debug(f"IB connection attempt failed: {exc!r}")
                self._start_recovery_clock()
                self._warn_if_recovery_delayed()
                await asyncio.sleep(self.settings.retry_delay)
        return self._running

    async def _wait_until_probe_succeeds(self) -> None:
        while self._running and self.ib.isConnected():
            if await self._probe():
                return
            if self._restart_requested.is_set() or not self.ib.isConnected():
                return
            log.debug("Connection probe failed. Waiting before retry.")
            self._start_recovery_clock()
            self._warn_if_recovery_delayed()
            await asyncio.sleep(5)

    def _request_probe(self, reason: str) -> None:
        if self._probe_task is not None and not self._probe_task.done():
            return

        self._probe_task = asyncio.create_task(
            self._run_probe(reason),
            name="connection-supervisor-probe",
        )

    async def _run_probe(self, reason: str) -> None:
        current_task = asyncio.current_task()
        log.debug(f"{reason}. Probing connection.")
        try:
            if await self._probe():
                self._handle_successful_probe()
            else:
                self._handle_failed_probe()
        finally:
            if self._probe_task is current_task:
                self._probe_task = None

    def _handle_successful_probe(self) -> None:
        log.debug("Connection probe succeeded.")
        self._mark_connected()
        if self.settings.app_timeout:
            self.ib.setTimeout(self.settings.app_timeout)

    def _handle_failed_probe(self) -> None:
        if self.state == SupervisorState.WAITING_FOR_BROKER:
            log.debug("Connection probe failed; continuing to wait for broker.")
            self._warn_if_recovery_delayed()
            return

        broker_message = self._latest_recoverable_broker_message()
        if broker_message:
            self._wait_for_broker_auto_recovery(broker_message)
            return

        self.request_restart("connection probe failed")

    def _handle_data_maintained_message(self, code: int) -> None:
        if (
            self.settings.restart_on_recovered_connection
            and self._running
            and self.state
            in {
                SupervisorState.CONNECTED,
                SupervisorState.WAITING_FOR_BROKER,
            }
        ):
            self._cancel_auto_recovery_wait()
            self.request_restart(
                f"broker connectivity restored with data maintained ({code})"
            )
            return

        self._mark_connected()

    async def _probe(self) -> bool:
        try:
            probe = self.ib.reqHistoricalDataAsync(
                self.settings.probe_contract, "", "30 S", "5 secs", "MIDPOINT", False
            )
            bars = await asyncio.wait_for(probe, self.settings.probe_timeout)
        except (asyncio.TimeoutError, ConnectionError) as exc:
            log.debug(f"Connection probe did not complete: {exc!r}")
            return False
        return bool(bars)

    def _wait_for_broker_auto_recovery(self, broker_message: BrokerMessage) -> None:
        if not self._running or self._restart_requested.is_set():
            return

        self._transition_to(SupervisorState.WAITING_FOR_BROKER)
        self._start_recovery_clock()
        self._cancel_auto_recovery_wait()
        loop = asyncio.get_running_loop()
        self._auto_recovery_handle = loop.call_later(
            self.settings.auto_recovery_grace_period,
            self.request_restart,
            f"broker auto-recovery timed out after message {broker_message.code}",
        )
        log.debug(
            f"Recent broker message {broker_message.code}; waiting "
            f"{self.settings.auto_recovery_grace_period}s "
            "for automatic recovery."
        )

    def _mark_connected(self) -> None:
        self._cancel_auto_recovery_wait()
        self._recent_broker_messages.clear()
        self._recovery_started_at = None
        self._last_warning_at = None
        if (
            self._running
            and not self._restart_requested.is_set()
            and self.state
            in {
                SupervisorState.PROBING,
                SupervisorState.CONNECTED,
                SupervisorState.WAITING_FOR_BROKER,
            }
        ):
            self._transition_to(SupervisorState.CONNECTED)

    def _remember_broker_message(self, code: int, message: str) -> None:
        self._recent_broker_messages.append(BrokerMessage(code, message, monotonic()))
        self._discard_stale_broker_messages()
        while len(self._recent_broker_messages) > self.RECENT_BROKER_MESSAGE_LIMIT:
            self._recent_broker_messages.popleft()

    def _latest_recoverable_broker_message(self) -> BrokerMessage | None:
        self._discard_stale_broker_messages()
        for broker_message in reversed(self._recent_broker_messages):
            if broker_message.code in self.RECOVERABLE_BROKER_CODES:
                return broker_message
        return None

    def _discard_stale_broker_messages(self) -> None:
        cutoff = monotonic() - self.RECENT_BROKER_MESSAGE_TTL
        while (
            self._recent_broker_messages
            and self._recent_broker_messages[0].received_at < cutoff
        ):
            self._recent_broker_messages.popleft()

    def _disconnect(self) -> None:
        if self.ib.isConnected():
            self._intentional_disconnect = True
            try:
                self.ib.disconnect()
            finally:
                self._intentional_disconnect = False

    def _cancel_auto_recovery_wait(self) -> None:
        if self._auto_recovery_handle:
            self._auto_recovery_handle.cancel()
            self._auto_recovery_handle = None

    def _cancel_probe(self) -> None:
        if self._probe_task and not self._probe_task.done():
            self._probe_task.cancel()
        self._probe_task = None

    def _start_recovery_clock(self) -> None:
        if self._recovery_started_at is None:
            self._recovery_started_at = monotonic()

    def _warn_if_recovery_delayed(self) -> None:
        if self._recovery_started_at is None:
            return

        now = monotonic()
        if now - self._recovery_started_at < self.settings.recovery_warning_after:
            return

        if (
            self._last_warning_at is None
            or now - self._last_warning_at >= self.settings.recovery_warning_interval
        ):
            log.warning("IB connection recovery is still pending.")
            self._last_warning_at = now

    def _transition_to(self, state: SupervisorState) -> None:
        if state == self.state:
            return

        allowed_states = ALLOWED_SUPERVISOR_TRANSITIONS[self.state]
        if state not in allowed_states:
            raise RuntimeError(
                "Invalid supervisor state transition: "
                f"{self.state.value} -> {state.value}"
            )
        self.state = state
