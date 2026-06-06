"""State-machine supervisor for owned IB socket recovery."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from logging import getLogger
from time import monotonic
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
class BrokerMessage:
    """Recent IB broker status message used for recovery decisions."""

    code: int
    message: str
    received_at: float


@dataclass(frozen=True)
class ConnectionSettings:
    """Connection and recovery settings for :class:`ConnectionSupervisor`.

    ``retry_delay`` is the only reconnect pacing knob. Restarting after a
    timeout, socket reset, or data-loss message immediately re-enters the
    connecting state; failed connection attempts wait for ``retry_delay`` before
    trying again.
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

    Public callers use a deliberately small lifecycle interface:

    - ``run()`` owns connection setup, workload execution, restart cycles, and
      final cleanup. It returns only after the supervisor reaches the stopped
      state, and it re-raises unexpected supervisor failures after cleanup.
    - ``stop()`` is a request only. It wakes the run loop by setting the stop
      signal, but shutdown work remains owned by ``run()``.
    - ``request_restart()`` asks the run loop to stop active work, disconnect the
      owned socket, and reconnect immediately. Failed connection attempts wait
      for ``settings.retry_delay`` before retrying.

    The supervisor does not start, stop, or restart TWS/IB Gateway itself, and
    it does not know whether the workload is live trading or dataloader work.
    Broker messages provide recovery context; timeout and probe results decide
    whether to wait for broker auto-recovery or rebuild the socket/workload.
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
    _restart_requested: asyncio.Event = field(
        default_factory=asyncio.Event, init=False, repr=False
    )
    _timeout_received: asyncio.Event = field(
        default_factory=asyncio.Event, init=False, repr=False
    )
    _traffic_resumed: asyncio.Event = field(
        default_factory=asyncio.Event, init=False, repr=False
    )
    _broker_recovered: asyncio.Event = field(
        default_factory=asyncio.Event, init=False, repr=False
    )
    _pending_restart_reason: str | None = field(default=None, init=False, repr=False)
    _intentional_disconnect: bool = field(default=False, init=False, repr=False)
    _recent_broker_messages: deque[BrokerMessage] = field(
        default_factory=deque, init=False, repr=False
    )

    RECOVERABLE_BROKER_CODES = frozenset({1100, 2110, 2103, 2105, 2157, 10182})
    DATA_LOST_CODE = 1101
    DATA_MAINTAINED_CODE = 1102
    SOCKET_RESET_CODE = 1300
    RECENT_BROKER_MESSAGE_TTL = 300
    RECENT_BROKER_MESSAGE_LIMIT = 20

    def __post_init__(self) -> None:
        self._state = INITIAL_STATE(self)
        self.ib.errorEvent += self.onErrEvent
        self.ib.disconnectedEvent += self.onDisconnectedEvent
        self.ib.timeoutEvent += self.onTimeoutEvent
        self.ib.updateEvent += self.onUpdateEvent

    @property
    def state(self) -> type[AbstractState]:
        """Return the current supervisor state class."""

        return type(self._state)

    async def run(self) -> None:
        """Run connection, workload, restart, and shutdown states."""

        try:
            while True:
                next_state = await self._state.handle()
                self.transition_to(next_state)
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

    def transition_to(self, state: type[AbstractState]) -> AbstractState:
        """Transition to a new state class and return the state instance."""

        if state != type(self._state):
            log.debug(
                f"Supervisor transition: {type(self._state).__name__} -> "
                f"{state.__name__}"
            )
        self._state = state(self)
        return self._state

    def stop(self) -> None:
        """Request supervisor shutdown."""

        self._stop_requested.set()

    def request_restart(self, reason: str = "") -> None:
        """Request a reconnect/rebuild cycle."""

        if self._restart_requested.is_set():
            log.debug(f"Restart already pending: {self._pending_restart_reason}")
            return
        self._pending_restart_reason = reason or "restart requested"
        log.debug(f"Restart requested: {self._pending_restart_reason}")
        self._restart_requested.set()

    def consume_restart_reason(self) -> str:
        """Return and clear the pending restart reason."""

        reason = self._pending_restart_reason or "restart requested"
        self._pending_restart_reason = None
        self._restart_requested.clear()
        return reason

    def onDisconnectedEvent(self) -> None:
        """Request restart after unexpected API socket disconnection."""

        if not self._intentional_disconnect and not self._stop_requested.is_set():
            self.request_restart("IB socket disconnected")

    def onTimeoutEvent(self, idle_period: float) -> None:
        """Record an idle timeout for the connected state to evaluate."""

        log.debug(f"No IB traffic for {idle_period}s.")
        self._timeout_received.set()

    def onUpdateEvent(self) -> None:
        """Record resumed IB traffic during broker-degraded recovery."""

        self._traffic_resumed.set()

    def onErrEvent(
        self,
        req_id: int,
        code: int,
        message: str,
        contract: ibi.Contract,
    ) -> None:
        """Translate selected broker messages into supervisor signals."""

        self._remember_broker_message(code, message)
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
                self.clear_broker_degraded_context()
                self._broker_recovered.set()

    async def connect(self) -> bool:
        """Connect the owned IB client, retrying until stopped."""

        while not self._stop_requested.is_set() and not self.ib.isConnected():
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
                await self.wait_or_stop(self.settings.retry_delay)
        return self.ib.isConnected()

    async def probe(self) -> bool:
        """Return whether the broker accepted a small historical-data request."""

        try:
            probe = self.ib.reqHistoricalDataAsync(
                self.settings.probe_contract, "", "30 S", "5 secs", "MIDPOINT", False
            )
            bars = await asyncio.wait_for(probe, self.settings.probe_timeout)
        except (asyncio.TimeoutError, ConnectionError) as exc:
            log.debug(f"Connection probe did not complete: {exc!r}")
            return False
        return bool(bars)

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

    def recent_recoverable_broker_message(self) -> BrokerMessage | None:
        """Return the latest recent broker-degraded message, if any."""

        self._discard_stale_broker_messages()
        for broker_message in reversed(self._recent_broker_messages):
            if broker_message.code in self.RECOVERABLE_BROKER_CODES:
                return broker_message
        return None

    def clear_broker_degraded_context(self) -> None:
        """Clear broker-degraded recovery context and related wake-up signals."""

        self._recent_broker_messages.clear()
        self.clear_recovery_signals()

    def clear_recovery_signals(self) -> None:
        """Clear broker-recovery wake-up signals without dropping message context."""

        self._broker_recovered.clear()
        self._traffic_resumed.clear()

    def _remember_broker_message(self, code: int, message: str) -> None:
        """Remember recent broker messages used to choose recovery behavior."""

        self._recent_broker_messages.append(BrokerMessage(code, message, monotonic()))
        self._discard_stale_broker_messages()
        while len(self._recent_broker_messages) > self.RECENT_BROKER_MESSAGE_LIMIT:
            self._recent_broker_messages.popleft()

    def _discard_stale_broker_messages(self) -> None:
        """Drop broker messages that are too old to guide recovery."""

        cutoff = monotonic() - self.RECENT_BROKER_MESSAGE_TTL
        while (
            self._recent_broker_messages
            and self._recent_broker_messages[0].received_at < cutoff
        ):
            self._recent_broker_messages.popleft()

    async def _cleanup(self) -> None:
        """Run terminal cleanup and leave the supervisor in the stopped state."""

        next_state = await self.transition_to(StoppingState).handle()
        self.transition_to(next_state)


Supervisor = ConnectionSupervisor
