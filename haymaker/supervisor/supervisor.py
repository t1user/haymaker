"""State-machine supervisor for owned IB socket recovery."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, Protocol

import ib_insync as ibi

from .codes import (
    DATA_LOST_CODE,
    DATA_MAINTAINED_CODE,
    LIVE_UPDATE_FAILURE_CODE,
    SOCKET_RESET_CODE,
    WEAK_DATA_FARM_CODES,
)
from .settings import ConnectionSettings
from .states import (
    AbstractState,
    ConnectingState,
    RestartingState,
    StoppedError,
    StoppingState,
)

log = getLogger(__name__)


INITIAL_STATE = ConnectingState


class SupervisorWorkload(Protocol):
    """Workload managed by a connection supervisor."""

    async def start(self) -> None:
        """Start or resume work after a usable IB connection is available."""

    async def stop(self, reason: str) -> None:
        """Release active work before the supervisor reconnects or exits."""


class SupervisorRace:
    """Race state work against service-interruption signals.

    State handlers often wait on broker I/O, sleeps, or state-local wakeups, but
    service interruptions arrive through independent signals: explicit stop
    requests, restart requests from broker events, and workload completion. The
    race gives those signals one centralized policy point instead of spreading
    interruption checks across states and event handlers.

    Each run creates request waiters first, checks already-pending lifecycle
    requests, then creates a task for ``state.handle()`` if state work should
    actually begin. The first completed task wins; request events are then
    prioritized over normal state completion, with stop taking precedence over
    restart.
    """

    def __init__(
        self,
        state: AbstractState,
        stop_requested: asyncio.Event,
        restart_requested: asyncio.Event,
        workload_task: asyncio.Task[None] | None,
    ) -> None:
        """Configure the active supervisor race.

        Args:
            state: State whose handler is being run.
            stop_requested: Supervisor shutdown signal.
            restart_requested: Supervisor restart signal.
            workload_task: Current workload task. The race decides whether its
                completion is an active signal for the current state.
        """

        self._state = state
        self._stop_requested = stop_requested
        self._restart_requested = restart_requested
        self._workload_task = workload_task
        self._state_task: asyncio.Task[type[AbstractState]] | None = None
        self._request_tasks: dict[str, asyncio.Task[bool]] = {}

    async def __aenter__(self) -> SupervisorRace:
        """Create tasks participating in the current supervisor race."""

        self._request_tasks = {
            name: asyncio.create_task(event.wait(), name=f"{name}-request")
            for name, event in self._active_requests().items()
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

        request_transition = self._request_transition()
        if request_transition is not None:
            return request_transition

        self._state_task = asyncio.create_task(
            self._state.handle(),
            name="connection-supervisor-state",
        )
        tasks: set[asyncio.Future[Any]] = {
            self._state_task,
            *self._request_tasks.values(),
        }
        workload_task = self._active_workload_task()
        if workload_task is not None:
            tasks.add(workload_task)

        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        request_transition = self._request_transition()
        if request_transition is not None:
            return request_transition

        if self._state_task in done:
            return self._state_task.result()

        if workload_task is not None and workload_task in done:
            return StoppingState

        return self._state_task.result()

    def _active_requests(self) -> dict[str, asyncio.Event]:
        """Return lifecycle request events that may interrupt this race."""

        requests: dict[str, asyncio.Event] = {}
        if self._state.accepts_stop:
            requests["connection-supervisor-stop"] = self._stop_requested
        if self._state.accepts_restart:
            requests["connection-supervisor-restart"] = self._restart_requested
        return requests

    def _active_workload_task(self) -> asyncio.Task[None] | None:
        """Return workload completion when it is an external race signal."""

        if not self._state.observes_workload:
            return None

        return self._workload_task

    def _request_transition(self) -> type[AbstractState] | None:
        """Return the highest-priority lifecycle request for this
        race.

        The purpose of this method is to ensure that:

            - pending stop/restart has priority over normal state
              completion;

            - state work does not begin if a lifecycle request is
              already pending;

            - restart requests are not accidentally carried into later
              states that should not inherit them.
        """

        if self._stop_requested.is_set() and self._state.accepts_stop:
            self._restart_requested.clear()
            return StoppingState

        if self._restart_requested.is_set() and self._state.accepts_restart:
            self._restart_requested.clear()
            return RestartingState

        if self._restart_requested.is_set() and not self._state.accepts_restart:
            self._restart_requested.clear()

        return None


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

        - ``request_restart()`` records a reconnect/rebuild request.
          The active supervisor race decides whether and when to
          consume that request.

    State-facing helpers such as ``start_workload()``,
    ``cleanup_workload()``, ``has_workload``, and ``disconnect()`` are
    part of the supervisor/state contract, not the external lifecycle
    API. Use one supervisor for one owned IB socket. Attached
    dataloader work should run without a supervisor because it must not
    connect, disconnect, or restart an externally owned socket.

    The supervisor does not start, stop, or restart TWS/IB Gateway
    itself, and it does not know whether the workload is live trading
    or dataloader work. Broker messages are categorized into restart
    requests, broker-connectivity-lost signals, or informational
    data-farm messages; non-restart messages are interpreted immediately
    by the active state or logged as context.

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
    connection_unavailable: asyncio.Event = field(
        default_factory=asyncio.Event, init=False, repr=False
    )
    # True while the supervisor closes its own socket, so disconnectedEvent is
    # not classified as an unexpected outage.
    _intentional_disconnect: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self._state = INITIAL_STATE(self)
        self.connection_unavailable.set()
        self.ib.errorEvent += self.onErrEvent
        self.ib.disconnectedEvent += self.onDisconnectedEvent
        self.ib.timeoutEvent += self.onTimeoutEvent
        self.ib.updateEvent += self.onUpdateEvent

    def __str__(self) -> str:
        """Return a compact supervisor status suitable for logs."""

        return (
            f"ConnectionSupervisor<state={type(self._state).__name__}, "
            f"connected={self.ib.isConnected()}, "
            f"workload={type(self.workload).__name__}>"
        )

    def __repr__(self) -> str:
        """Return a diagnostic representation of supervisor dependencies."""

        return (
            f"ConnectionSupervisor(ib={self.ib!r}, workload={self.workload!r}, "
            f"settings={self.settings!r}, state={type(self._state).__name__})"
        )

    async def run(self) -> None:
        """Run connection, workload, restart, and shutdown states."""

        try:
            while True:
                async with SupervisorRace(
                    self._state,
                    self._stop_requested,
                    self._restart_requested,
                    self._workload_task,
                ) as race:
                    transition = await race.wait()
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
            log.debug(f"{type(self._state).__name__} -> {next_state.__name__}")
        self._state = next_state(self)
        return self._state

    def stop(self) -> None:
        """Request supervisor shutdown; the run loop performs cleanup."""

        self.mark_connection_unavailable("supervisor stop requested")
        self._stop_requested.set()

    def request_restart(self, reason: str = "") -> bool:
        """Record a reconnect/rebuild request."""

        restart_reason = reason or "restart requested"
        if not self._state.accepts_restart:
            log.debug(
                f"Restart request ignored by {type(self._state).__name__}: "
                f"{restart_reason}"
            )
            return False

        log.debug(f"Restart requested: {restart_reason}")
        self.mark_connection_unavailable(restart_reason)
        self._restart_requested.set()
        return True

    def mark_connection_available(self, reason: str) -> None:
        """Allow workload sync after the supervisor has a usable connection."""

        self.connection_unavailable.clear()

    def mark_connection_unavailable(self, reason: str) -> None:
        """Abort workload sync while the supervisor cannot trust the connection."""

        self.connection_unavailable.set()

    def onDisconnectedEvent(self) -> None:
        """Request restart after unexpected API socket disconnection."""

        if not self._intentional_disconnect:
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

        if code == DATA_LOST_CODE:
            self.request_restart(
                f"broker connectivity restored with data lost ({code})"
            )
        elif code == SOCKET_RESET_CODE:
            self.request_restart(f"broker reset API socket port ({code})")
        elif code == DATA_MAINTAINED_CODE:
            if self.settings.restart_on_recovered_connection:
                self.request_restart(
                    f"broker connectivity restored with data maintained ({code})"
                )
            else:
                self._state.on_broker_message(code, message)
        elif code in WEAK_DATA_FARM_CODES:
            if code == LIVE_UPDATE_FAILURE_CODE:
                self._state.on_broker_message(code, message)
            if self.settings.log_datafarm_status:
                log.debug(f"broker message {code}: {message}")
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

        await self.workload.stop(reason)

        if task.done():
            self._log_workload_result(task)
            self._workload_task = None
            return

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
            self.mark_connection_unavailable("disconnecting IB socket")
            self._intentional_disconnect = True
            try:
                self.ib.disconnect()
            finally:
                self._intentional_disconnect = False

    async def _cleanup(self) -> None:
        """Run terminal cleanup and leave the supervisor in the stopped state."""

        next_state = await self._transition_to(StoppingState).handle()
        self._transition_to(next_state)
