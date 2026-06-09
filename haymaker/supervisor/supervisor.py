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
    RestartingState,
    StoppedError,
    StoppedState,
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


class SupervisorRace:
    """Own per-state race tasks and classify the winning interrupt source."""

    def __init__(
        self,
        state: AbstractState,
        requests: Mapping[str, asyncio.Event],
        workload_task: asyncio.Task[None] | None,
    ) -> None:
        """Configure the active supervisor race.

        Args:
            state: State whose handler is being run.
            requests: Lifecycle request events enabled for this state.
            workload_task: Workload task when completion can stop supervision.
        """

        self._state = state
        self._requests = requests
        self._workload_task = workload_task
        self._state_task: asyncio.Task[type[AbstractState]] | None = None
        self._request_tasks: dict[str, asyncio.Task[bool]] = {}

    async def __aenter__(self) -> SupervisorRace:
        """Create tasks participating in the current supervisor race."""

        self._state_task = asyncio.create_task(
            self._state.handle(),
            name="connection-supervisor-state",
        )
        self._request_tasks = {
            name: asyncio.create_task(
                event.wait(),
                name=f"connection-supervisor-{name}-request",
            )
            for name, event in self._requests.items()
        }
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        """Cancel tasks that did not win the race."""

        tasks: list[asyncio.Task[Any]] = [*self._request_tasks.values()]
        if self._state_task is not None:
            tasks.append(self._state_task)

        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def wait(self) -> type[AbstractState]:
        """Wait for the first active task and return the proposed next state."""

        if self._state_task is None:
            raise RuntimeError("SupervisorRace must be entered before waiting.")

        tasks: set[asyncio.Future[Any]] = {
            self._state_task,
            *self._request_tasks.values(),
        }
        if self._workload_task is not None:
            tasks.add(self._workload_task)

        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        if self._request_tasks.get("stop") in done:
            return StoppingState

        if self._request_tasks.get("restart") in done:
            return RestartingState

        if self._state_task in done and (
            self._state_task.cancelled()
            or self._state_task.exception() is not None
        ):
            return self._state_task.result()

        if self._workload_task is not None and self._workload_task in done:
            return StoppingState

        return self._state_task.result()


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

    Public API:

        - ``run()`` starts connection and runs managed workload; it owns
          connection setup, workload execution, restart cycles, and
          final cleanup. It returns only after the supervisor reaches
          the stopped state, and it re-raises unexpected supervisor
          failures after cleanup.

        - ``stop()`` sends a shutdown request; ``run()`` performs cleanup.

        - ``request_restart()`` asks the supervisor to interrupt the
          active state and run a reconnect/rebuild transition that
          stops active work, disconnects the owned socket, and
          reconnects immediately.

    State-facing helpers such as ``start_workload()``,
    ``cleanup_workload()``, ``has_workload``, and ``disconnect()`` are
    part of the supervisor/state contract, not the external lifecycle
    API. Use one supervisor for one owned IB socket. Attached
    dataloader work should run without a supervisor because it must not
    connect, disconnect, or restart an externally owned socket.

    The supervisor does not start, stop, or restart TWS/IB Gateway
    itself, and it does not know whether the workload is live trading
    or dataloader work. Broker messages are categorized into restart
    requests, broker-recovery wait signals, or recovery hints;
    non-restart messages are interpreted immediately by the active
    state.

    States return their proposed next state, but supervisor-owned
    lifecycle requests keep final priority: stop overrides restart, and
    restart is ignored once restart or shutdown cleanup is active.
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
    # True while the supervisor closes its own socket, so disconnectedEvent is
    # not classified as an unexpected outage.
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

    def _state_race(self) -> SupervisorRace:
        """Return the race configuration enabled for the active state."""

        if isinstance(self._state, RestartingState):
            return SupervisorRace(self._state, {"stop": self._stop_requested}, None)

        if isinstance(self._state, (StoppingState, StoppedState)):
            return SupervisorRace(self._state, {}, None)

        return SupervisorRace(
            self._state,
            {
                "stop": self._stop_requested,
                "restart": self._restart_requested,
            },
            self._workload_task,
        )

    async def run(self) -> None:
        """Run connection, workload, restart, and shutdown states."""

        try:
            while True:
                transition = await self._run_state()
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

    async def _run_state(self) -> type[AbstractState]:
        """Race state work against supervisor-owned lifecycle events."""

        async with self._state_race() as race:
            transition = await race.wait()

        if self._restart_requested.is_set() and not issubclass(
            transition, (StoppingState, StoppedState)
        ):
            transition = RestartingState
        self._restart_requested.clear()
        return transition

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
        """Request supervisor shutdown; the run loop performs cleanup."""

        self._stop_requested.set()

    def request_restart(self, reason: str = "") -> None:
        """Request a reconnect/rebuild cycle."""

        if self._stop_requested.is_set():
            log.debug("Restart ignored because stop is already pending.")
            return
        if isinstance(self._state, (RestartingState, StoppingState, StoppedState)):
            log.debug("Restart ignored because lifecycle cleanup is already active.")
            return
        restart_reason = reason or "restart requested"
        log.debug(f"Restart requested: {restart_reason}")
        self._restart_requested.set()

    def onDisconnectedEvent(self) -> None:
        """Request restart after unexpected API socket disconnection."""

        if not self._intentional_disconnect and not self._stop_requested.is_set():
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

    @property
    def has_workload(self) -> bool:
        """Return whether the supervisor currently tracks a workload task."""

        return self._workload_task is not None

    async def cleanup_workload(self, reason: str = "") -> None:
        """Stop active workload or collect its completed result."""

        task = self._workload_task
        if task is None:
            return

        if task.done():
            self._log_workload_result(task)
            self._workload_task = None
            return

        await self.workload.stop(reason)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        self._workload_task = None

    def _log_workload_result(self, task: asyncio.Task[None]) -> None:
        """Log failure from a completed workload task."""

        try:
            task.result()
        except asyncio.CancelledError:
            log.debug("Connection workload task cancelled.")
        except Exception:
            log.exception("Connection workload task failed.")

    def disconnect(self) -> None:
        """Disconnect the owned IB socket without triggering restart handling."""

        if self.ib.isConnected():
            self._intentional_disconnect = True
            try:
                self.ib.disconnect()
            finally:
                self._intentional_disconnect = False

    async def _cleanup(self) -> None:
        """Run terminal cleanup and leave the supervisor in the stopped state."""

        next_state = await self._transition_to(StoppingState).handle()
        self._transition_to(next_state)
