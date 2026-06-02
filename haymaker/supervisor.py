"""Connection lifecycle management for Interactive Brokers clients."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from time import monotonic

import ib_insync as ibi

log = logging.getLogger(__name__)

AsyncCallback = Callable[[], Awaitable[None]]
RestartCallback = Callable[[str], None]


def contract_refresh_is_overdue(
    last_refresh: datetime | None,
    max_age: float,
    now: datetime | None = None,
) -> bool:
    """Return whether successful contract initialization is too old."""

    if last_refresh is None:
        return False

    now = now or datetime.now(tz=timezone.utc)
    return now - last_refresh > timedelta(seconds=max_age)


class SupervisorState(str, Enum):
    """Current connection supervisor state."""

    STOPPED = "stopped"
    CONNECTED = "connected"
    WAITING_FOR_BROKER = "waiting_for_broker"
    RESTARTING = "restarting"


@dataclass
class ConnectionSupervisor:
    """Maintain an IB socket connection and restart workloads when needed.

    The supervisor deliberately does not manage the TWS or IB Gateway process.
    It reconnects the API client and invokes application-specific callbacks
    around the established Haymaker restart cycle.

    Args:
        ib: Shared Interactive Brokers client.
        on_connected: Async callback run after a usable connection is available.
        on_restarting: Optional callback run once when a restart is requested.
        host: Hostname for the TWS or IB Gateway API socket.
        port: Port for the TWS or IB Gateway API socket.
        client_id: Interactive Brokers API client identifier.
        connect_timeout: Seconds allowed for each socket connection attempt.
        restart_delay: Seconds to wait before reconnecting after a restart.
        retry_delay: Seconds to wait after a failed connection attempt.
        app_timeout: Idle-traffic interval after which the connection is probed.
        probe_contract: Contract used for historical-data probes.
        probe_timeout: Seconds allowed for each probe request.
        probe_on_connect: Whether to verify upstream connectivity after connecting.
        auto_recovery_grace_period: Seconds to wait for broker auto-recovery.
        recovery_warning_after: Seconds before delayed recovery is warned about.
        recovery_warning_interval: Minimum seconds between repeated warnings.
    """

    ib: ibi.IB
    on_connected: AsyncCallback
    on_restarting: RestartCallback | None = None
    host: str = "127.0.0.1"
    port: int = 4002
    client_id: int = 0
    connect_timeout: float = 2
    restart_delay: float = 30
    retry_delay: float = 2
    app_timeout: float = 20
    probe_contract: ibi.Contract = field(default_factory=lambda: ibi.Forex("EURUSD"))
    probe_timeout: float = 4
    probe_on_connect: bool = True
    auto_recovery_grace_period: float = 120
    recovery_warning_after: float = 300
    recovery_warning_interval: float = 900
    state: SupervisorState = field(default=SupervisorState.STOPPED, init=False)
    _running: bool = field(default=False, init=False, repr=False)
    _runner: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _restart_requested: asyncio.Event = field(
        default_factory=asyncio.Event, init=False, repr=False
    )
    _restart_reason: str = field(default="", init=False, repr=False)
    _intentional_disconnect: bool = field(default=False, init=False, repr=False)
    _auto_recovery_handle: asyncio.TimerHandle | None = field(
        default=None, init=False, repr=False
    )
    _probe_task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _recovery_started_at: float | None = field(default=None, init=False, repr=False)
    _last_warning_at: float | None = field(default=None, init=False, repr=False)

    AUTO_RECOVERY_CODES = frozenset({1100, 2110})
    DATA_LOST_CODE = 1101
    DATA_MAINTAINED_CODE = 1102

    def __post_init__(self) -> None:
        self.ib.errorEvent += self.on_broker_message
        self.ib.disconnectedEvent += self.on_disconnected
        self.ib.timeoutEvent += self.on_timeout

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
        first_connection = True
        try:
            while self._running:
                if not first_connection:
                    await self._prepare_restart()
                first_connection = False

                if not await self._connect():
                    break

                if self.probe_on_connect:
                    await self._wait_until_probe_succeeds()
                if not self._running or not self.ib.isConnected():
                    continue

                self._mark_connected()
                if self.app_timeout:
                    self.ib.setTimeout(self.app_timeout)
                await self.on_connected()

                if self._running:
                    await self._restart_requested.wait()
        finally:
            self._running = False
            self._cancel_auto_recovery_wait()
            self._cancel_probe()
            self._disconnect()
            self.state = SupervisorState.STOPPED

    def stop(self) -> None:
        """Stop reconnecting and disconnect the IB socket."""

        self._running = False
        self._restart_requested.set()
        self._cancel_auto_recovery_wait()
        self._cancel_probe()
        self._disconnect()
        self.state = SupervisorState.STOPPED

    def request_restart(self, reason: str) -> None:
        """Request one restart cycle, coalescing concurrent requests."""

        if not self._running:
            log.debug(f"Ignoring restart request while stopped: {reason}")
            return

        if not self._restart_requested.is_set():
            self._restart_reason = reason
            self._start_recovery_clock()
            self.state = SupervisorState.RESTARTING
            log.debug(f"Restart requested: {reason}")
            if self.on_restarting:
                self.on_restarting(reason)
            self._restart_requested.set()
        else:
            log.debug(f"Restart already requested; additional reason: {reason}")

    def on_broker_message(
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

    def on_disconnected(self) -> None:
        """Restart after an unexpected API socket disconnection."""

        if self._running and not self._intentional_disconnect:
            self.request_restart("IB socket disconnected")

    def on_timeout(self, idle_period: float) -> None:
        """Probe an otherwise idle connection before requesting a restart."""

        if (
            self._running
            and self.state != SupervisorState.WAITING_FOR_BROKER
            and (self._probe_task is None or self._probe_task.done())
        ):
            self._probe_task = asyncio.create_task(
                self._probe_after_timeout(idle_period),
                name="connection-supervisor-probe",
            )

    async def _prepare_restart(self) -> None:
        self.state = SupervisorState.RESTARTING
        reason = self._restart_reason or "restart requested"
        log.debug(f"Restarting connection: {reason}")
        self._restart_requested.clear()
        self._cancel_auto_recovery_wait()
        self._cancel_probe()
        self._disconnect()
        await asyncio.sleep(self.restart_delay)

    async def _connect(self) -> bool:
        while self._running and not self.ib.isConnected():
            try:
                log.debug(f"Connecting to IB at {self.host}:{self.port}.")
                await self.ib.connectAsync(
                    self.host,
                    self.port,
                    clientId=self.client_id,
                    timeout=self.connect_timeout,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.debug(f"IB connection attempt failed: {exc!r}")
                self._start_recovery_clock()
                self._warn_if_recovery_delayed()
                await asyncio.sleep(self.retry_delay)
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
            if self.app_timeout:
                self.ib.setTimeout(self.app_timeout)
        else:
            self.request_restart("connection probe failed")

    async def _probe(self) -> bool:
        try:
            probe = self.ib.reqHistoricalDataAsync(
                self.probe_contract, "", "30 S", "5 secs", "MIDPOINT", False
            )
            bars = await asyncio.wait_for(probe, self.probe_timeout)
        except (asyncio.TimeoutError, ConnectionError) as exc:
            log.debug(f"Connection probe did not complete: {exc!r}")
            return False
        return bool(bars)

    def _wait_for_broker_auto_recovery(self, code: int) -> None:
        if not self._running or self._restart_requested.is_set():
            return

        self.state = SupervisorState.WAITING_FOR_BROKER
        self._start_recovery_clock()
        self._cancel_auto_recovery_wait()
        loop = asyncio.get_running_loop()
        self._auto_recovery_handle = loop.call_later(
            self.auto_recovery_grace_period,
            self.request_restart,
            f"broker auto-recovery timed out after message {code}",
        )
        log.debug(
            f"Broker message {code}; waiting {self.auto_recovery_grace_period}s "
            "for automatic recovery."
        )

    def _mark_connected(self) -> None:
        self._cancel_auto_recovery_wait()
        self._recovery_started_at = None
        self._last_warning_at = None
        if not self._restart_requested.is_set():
            self.state = SupervisorState.CONNECTED

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
        if now - self._recovery_started_at < self.recovery_warning_after:
            return

        if (
            self._last_warning_at is None
            or now - self._last_warning_at >= self.recovery_warning_interval
        ):
            log.warning("IB connection recovery is still pending.")
            self._last_warning_at = now
