"""Connection lifecycle management for Interactive Brokers clients."""

from __future__ import annotations

import asyncio
import logging
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


class SupervisorWorkload(Protocol):
    """Workload run by :class:`ConnectionSupervisor` after IB is usable."""

    stop_supervisor_on_completion: bool

    async def start(self) -> None:
        """Start or resume work after a usable IB connection is available."""

    async def stop(self, reason: str) -> None:
        """Release work before the supervisor disconnects or exits."""


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

    @classmethod
    def from_live_config(cls, config: Mapping[str, Any]) -> ConnectionSettings:
        """Create live-trading connection settings from Haymaker config."""

        app_config = config.get("app") or {}
        return cls(
            host=app_config.get("host", "127.0.0.1"),
            port=app_config.get("port", 4002),
            client_id=app_config.get("clientId", 0),
            connect_timeout=app_config.get("connectTimeout", 2),
            restart_delay=app_config.get("restart_time", 30),
            retry_delay=app_config.get("retryDelay", 2),
            app_timeout=app_config.get("appTimeout", 20),
            probe_contract=app_config.get("probeContract") or ibi.Forex("EURUSD"),
            probe_timeout=app_config.get("probeTimeout", 4),
            auto_recovery_grace_period=app_config.get(
                "auto_recovery_grace_period", 120
            ),
            recovery_warning_after=app_config.get("recovery_warning_after", 300),
            recovery_warning_interval=app_config.get("recovery_warning_interval", 900),
        )

    @classmethod
    def from_dataloader_config(
        cls, config: Mapping[str, Any], client_id: int
    ) -> ConnectionSettings:
        """Create dataloader connection settings from Haymaker config."""

        return cls(
            host=config.get("host", "localhost"),
            port=config.get("port", 4002),
            client_id=client_id,
            connect_timeout=config.get("connectTimeout", 2),
            restart_delay=config.get("restart_time", 60),
            retry_delay=config.get("retryDelay", 60),
            app_timeout=0,
            auto_recovery_grace_period=config.get("auto_recovery_grace_period", 120),
            recovery_warning_after=config.get("recovery_warning_after", 300),
            recovery_warning_interval=config.get("recovery_warning_interval", 900),
        )


@dataclass
class ConnectionSupervisor:
    """Maintain an IB socket connection and restart workloads when needed.

    The supervisor deliberately does not manage the TWS or IB Gateway process.
    It reconnects the API client and runs one explicit workload task for each
    usable connection cycle.

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
    _runner: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
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

    AUTO_RECOVERY_CODES = frozenset({1100, 2110})
    DATA_LOST_CODE = 1101
    DATA_MAINTAINED_CODE = 1102

    def __post_init__(self) -> None:
        # IB calls this errorEvent, but most payloads are broker messages, not
        # actionable application errors. Keep "error" out of callback logs.
        self.ib.errorEvent += self.onErrEvent
        self.ib.disconnectedEvent += self.onDisconnectedEvent
        self.ib.timeoutEvent += self.onTimeoutEvent

    def start(self) -> asyncio.Task[None]:
        """Start the supervisor in the current event loop."""

        if self._runner is None or self._runner.done():
            self._runner = asyncio.ensure_future(self.run())
            self._runner.set_name("connection-supervisor")
        return self._runner

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

                self._start_workload()
                reason = await self._wait_for_restart_or_completion()
                await self._stop_workload(reason)
                if self._running:
                    await self._prepare_restart(reason)
        finally:
            self._running = False
            self._transition_to(SupervisorState.STOPPING)
            self._cancel_auto_recovery_wait()
            self._cancel_probe()
            await self._stop_workload("supervisor stopped")
            self._disconnect()
            self._transition_to(SupervisorState.STOPPED)

    def stop(self) -> None:
        """Stop reconnecting and disconnect the IB socket."""

        self._running = False
        self._transition_to(SupervisorState.STOPPING)
        self._restart_requested.set()
        self._cancel_auto_recovery_wait()
        self._cancel_probe()
        self._disconnect()

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

        if code in self.AUTO_RECOVERY_CODES:
            self._wait_for_broker_auto_recovery(code)
        elif code == self.DATA_LOST_CODE:
            self.request_restart(
                f"broker connectivity restored with data lost ({code})"
            )
        elif code == self.DATA_MAINTAINED_CODE:
            self._mark_connected()

    def onDisconnectedEvent(self) -> None:
        """Restart after an unexpected API socket disconnection."""

        if self._running and not self._intentional_disconnect:
            self.request_restart("IB socket disconnected")

    def onTimeoutEvent(self, idle_period: float) -> None:
        """Probe an otherwise idle connection before requesting a restart."""

        if (
            self._running
            and self.state == SupervisorState.CONNECTED
            and (self._probe_task is None or self._probe_task.done())
        ):
            self._probe_task = asyncio.create_task(
                self._probe_after_timeout(idle_period),
                name="connection-supervisor-probe",
            )

    async def _connect_and_verify(self) -> bool:
        self._transition_to(SupervisorState.CONNECTING)
        if not await self._connect():
            return False

        self._transition_to(SupervisorState.PROBING)
        await self._wait_until_probe_succeeds()

        if not self._running or not self.ib.isConnected():
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
        await self.workload.stop(reason)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._workload_task = None

    async def _prepare_restart(self, reason: str) -> None:
        self._transition_to(SupervisorState.RESTARTING)
        log.debug(f"Restarting connection: {reason}")
        self._cancel_auto_recovery_wait()
        self._cancel_probe()
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
            log.debug("Connection probe failed. Waiting before retry.")
            self._start_recovery_clock()
            self._warn_if_recovery_delayed()
            await asyncio.sleep(5)

    async def _probe_after_timeout(self, idle_period: float) -> None:
        log.debug(f"No IB traffic for {idle_period}s. Probing connection.")
        if await self._probe():
            log.debug("Connection probe succeeded.")
            if self.settings.app_timeout:
                self.ib.setTimeout(self.settings.app_timeout)
        else:
            self.request_restart("connection probe failed")

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

    def _wait_for_broker_auto_recovery(self, code: int) -> None:
        if not self._running or self._restart_requested.is_set():
            return

        self._transition_to(SupervisorState.WAITING_FOR_BROKER)
        self._start_recovery_clock()
        self._cancel_auto_recovery_wait()
        loop = asyncio.get_running_loop()
        self._auto_recovery_handle = loop.call_later(
            self.settings.auto_recovery_grace_period,
            self.request_restart,
            f"broker auto-recovery timed out after message {code}",
        )
        log.debug(
            f"Broker message {code}; waiting "
            f"{self.settings.auto_recovery_grace_period}s "
            "for automatic recovery."
        )

    def _mark_connected(self) -> None:
        self._cancel_auto_recovery_wait()
        self._recovery_started_at = None
        self._last_warning_at = None
        if not self._restart_requested.is_set():
            self._transition_to(SupervisorState.CONNECTED)

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
        self.state = state
