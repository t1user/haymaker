"""State-machine supervisor for owned IB socket recovery."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, Protocol

import ib_insync as ibi

from .states import (
    AbstractState,
    ConnectingState,
    StoppedError,
    StoppingState,
)

log = getLogger(__name__)


INITIAL_STATE = ConnectingState


class SupervisorWorkload(Protocol):
    """Workload run by :class:`ConnectionSupervisor` after IB is usable."""

    async def start(self) -> None:
        """Start or resume work after a usable IB connection is available."""

    async def stop(self, reason: str) -> None:
        """Release active work before the supervisor reconnects or exits."""


@dataclass(frozen=True)
class ConnectionSettings:
    """Connection and recovery settings for :class:`ConnectionSupervisor`.

    Attributes:
        host: Hostname or IP address of the TWS/IB Gateway API endpoint.
        port: Port number of the TWS/IB Gateway API endpoint.
        client_id: IB API client ID used for this supervised socket.
        connect_timeout: Seconds to wait for one socket connection attempt.
        retry_delay: Seconds to wait between failed connection attempts.
        app_timeout: Seconds of no IB traffic before running connection health checks.
        probe_contract: Contract used for the small historical-data readiness probe.
        probe_timeout: Seconds to wait for the readiness probe to complete.
        auto_recovery_grace_period: Seconds to wait for broker-side recovery before reconnecting.
        restart_on_recovered_connection: Whether to restart even after IB reports data was maintained.
    """

    host: str = "127.0.0.1"
    port: int = 4002
    client_id: int = 0
    connect_timeout: float = 2
    retry_delay: float = 2
    app_timeout: float = 20
    probe_contract: ibi.Contract = field(default_factory=lambda: ibi.Forex("EURUSD"))
    probe_timeout: float = 4
    auto_recovery_grace_period: float = 120
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
            retry_delay=config.get("retryDelay", 2),
            app_timeout=config.get("appTimeout", 20),
            probe_contract=config.get("probeContract") or ibi.Forex("EURUSD"),
            probe_timeout=config.get("probeTimeout", 4),
            auto_recovery_grace_period=config.get("auto_recovery_grace_period", 120),
            restart_on_recovered_connection=config.get(
                "restart_on_recovered_connection", False
            ),
        )


@dataclass
class ConnectionSupervisor:
    """Run one workload under an owned IB socket connection.

    Public api:

        - ``run()`` start connection and run managed workload; owns
          connection setup, workload execution, restart cycles, and
          final cleanup. It returns only after the supervisor reaches
          the stopped state, and it re-raises unexpected supervisor
          failures after cleanup.

        - ``stop()`` sent a shutdown request.

        - ``request_restart()`` asks the active state to return a
          reconnect/rebuild transition that stops active work,
          disconnects the owned socket, and reconnects immediately.

    The supervisor does not start, stop, or restart TWS/IB Gateway
    itself, and it does not know whether the workload is live trading
    or dataloader work. Broker messages are categorized into restart
    requests, broker-recovery wait signals, or recovery hints;
    non-restart messages are interpreted immediately by the active
    state.
    """

    ib: ibi.IB
    workload: SupervisorWorkload
    settings: ConnectionSettings = field(default_factory=ConnectionSettings)
    _state: AbstractState = field(init=False)
    _workload_task: asyncio.Task[None] | None = field(
        init=False, default=None, repr=False
    )
    _stop_requested: asyncio.Event = field(
        default_factory=asyncio.Event, init=False, repr=False
    )
    _intentional_disconnect: bool = field(default=False, init=False, repr=False)

    BROKER_WAIT_CODES = frozenset({1100, 2110, 2103, 2105, 2157, 10182})
    DATA_LOST_CODE = 1101
    DATA_MAINTAINED_CODE = 1102
    SOCKET_RESET_CODE = 1300

    def __post_init__(self) -> None:
        self._state = INITIAL_STATE(self)
        self.ib.errorEvent += self.onErrEvent
        self.ib.disconnectedEvent += self.onDisconnectedEvent
        self.ib.timeoutEvent += self.onTimeoutEvent
        self.ib.updateEvent += self.onUpdateEvent

    @property
    def stop_requested(self) -> bool:
        """Return whether supervisor shutdown has been requested."""

        return self._stop_requested.is_set()

    async def run(self) -> None:
        """Run connection, workload, restart, and shutdown states."""

        try:
            while True:
                transition = await self._state.handle()
                self._transition_to(transition)
        except StoppedError:
            return
        except asyncio.CancelledError:
            self.stop()
            await self._cleanup()
            raise
        except Exception:
            log.exception("Connection supervisor failed; stopping.")
            await self._cleanup()
            raise

    def _transition_to(self, next_state: type[AbstractState]) -> AbstractState:
        """Transition to a new state class and return the state instance."""

        if next_state != type(self._state):
            log.debug(
                f"Supervisor transition: {type(self._state).__name__} -> "
                f"{next_state.__name__}"
            )
        self._state = next_state(self)
        return self._state

    def stop(self) -> None:
        """Request supervisor shutdown."""

        self.mark_stop_requested()
        self._state.request_stop()

    def mark_stop_requested(self) -> None:
        """Record supervisor shutdown intent."""

        self._stop_requested.set()

    def request_restart(self, reason: str = "") -> None:
        """Request a reconnect/rebuild cycle."""

        if self.stop_requested:
            log.debug("Restart ignored because stop is already pending.")
            return
        restart_reason = reason or "restart requested"
        log.debug(f"Restart requested: {restart_reason}")
        self._state.request_restart()

    def onDisconnectedEvent(self) -> None:
        """Request restart after unexpected API socket disconnection."""

        if not self._intentional_disconnect and not self.stop_requested:
            self.request_restart("IB socket disconnected")

    def onTimeoutEvent(self, idle_period: float) -> None:
        """Dispatch an idle timeout to the active supervisor state."""

        log.debug(f"No IB traffic for {idle_period}s.")
        self._state.on_timeout(idle_period)

    def onUpdateEvent(self) -> None:
        """Dispatch resumed IB traffic to the active supervisor state."""

        self._state.on_update()

    def onErrEvent(
        self,
        req_id: int,
        code: int,
        message: str,
        contract: ibi.Contract,
    ) -> None:
        """Translate selected broker messages into supervisor signals."""

        if code == self.DATA_LOST_CODE:
            self.request_restart(
                f"broker connectivity restored with data lost ({code})"
            )
        elif code == self.SOCKET_RESET_CODE:
            self.request_restart(f"broker reset API socket port ({code})")
        elif code == self.DATA_MAINTAINED_CODE:
            if self.settings.restart_on_recovered_connection:
                self.request_restart(
                    f"broker connectivity restored with data maintained ({code})"
                )
            else:
                self._state.on_broker_message(code, message)
        else:
            self._state.on_broker_message(code, message)

    def start_workload(self) -> None:
        """Start the supervised workload as a tracked task."""

        if self._workload_task is None or self._workload_task.done():
            self._workload_task = asyncio.create_task(
                self.workload.start(), name="connection-supervisor-workload"
            )

    async def stop_workload(self, reason: str = "") -> None:
        """Stop and cancel the active workload task when one exists."""

        task = self._workload_task
        if task is None:
            return

        if task.done():
            self.consume_workload_result()
            return

        await self.workload.stop(reason)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        self._workload_task = None

    def consume_workload_result(self) -> None:
        """Log workload failure and clear a completed workload task."""

        if self._workload_task is None:
            return

        try:
            self._workload_task.result()
        except asyncio.CancelledError:
            log.debug("Connection workload task cancelled.")
        except Exception:
            log.exception("Connection workload task failed.")
        self._workload_task = None

    def disconnect(self) -> None:
        """Disconnect the owned IB client without treating it as unexpected."""

        if self.ib.isConnected():
            self._intentional_disconnect = True
            try:
                self.ib.disconnect()
            finally:
                self._intentional_disconnect = False

    async def wait_or_stop(self, delay: float) -> bool:
        """Wait for ``delay`` seconds or return early when shutdown is requested."""

        try:
            await asyncio.wait_for(self._stop_requested.wait(), delay)
        except asyncio.TimeoutError:
            return False
        return True

    async def _cleanup(self) -> None:
        """Run terminal cleanup and leave the supervisor in the stopped state."""

        next_state = await self._transition_to(StoppingState).handle()
        self._transition_to(next_state)
