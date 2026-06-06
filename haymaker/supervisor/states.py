"""Connection supervisor states.

The supervisor owns events, workload tasks, and socket cleanup. States only
wait for relevant signals and return the next state class.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .supervisor import ConnectionSupervisor


class StoppedError(Exception):
    """Raised by the terminal stopped state."""


class AbstractState(ABC):
    """Base class for connection supervisor states."""

    transition_priority: int = 0

    def __init__(self, context: ConnectionSupervisor) -> None:
        self.context = context
        self.settings = context.settings
        self.ib = context.ib

    @abstractmethod
    async def handle(self) -> type[AbstractState]:
        """Run state work and return the next state class."""

    def on_timeout(self, idle_period: float) -> None:
        """Handle an IB idle timeout while this state is active."""

    def on_update(self) -> None:
        """Handle resumed IB traffic while this state is active."""

    def on_broker_message(self, code: int, message: str) -> None:
        """Handle a broker message while this state is active."""

    def _priority_transition(self) -> type[AbstractState] | None:
        """Return the highest-priority pending transition, if any."""

        if self.context._stop_requested.is_set():
            return StoppingState

        return self.context.consume_state_transition()

    def __str__(self) -> str:
        return self.__class__.__name__.upper()


class ConnectingState(AbstractState):
    """Connect the owned IB client."""

    async def handle(self) -> type[AbstractState]:
        if self.ib.isConnected():
            return ProbingState

        if await self.context.connect():
            return ProbingState
        return StoppingState


class ProbingState(AbstractState):
    """Verify that the broker connection is usable."""

    transition_priority = 10

    async def handle(self) -> type[AbstractState]:
        if next_state := self._priority_transition():
            return next_state

        if not self.ib.isConnected():
            return ConnectingState

        probe_succeeded = await self.context.probe()

        if next_state := self._priority_transition():
            return next_state

        if probe_succeeded:
            if self.context._workload_task is None:
                return StartingWorkloadState
            return ConnectedState

        if not self.ib.isConnected():
            return ConnectingState

        return RestartingState

    def on_broker_message(self, code: int, message: str) -> None:
        """Wait for broker recovery if degradation is reported while probing."""

        if code in self.context.BROKER_WAIT_CODES:
            self.context.request_state_transition(WaitingForBrokerState)


class StartingWorkloadState(AbstractState):
    """Start the supervised workload after a successful probe."""

    async def handle(self) -> type[AbstractState]:
        self.context.start_workload()
        return ConnectedState


class ConnectedState(AbstractState):
    """Wait for restart, timeout, stop, or workload completion."""

    async def handle(self) -> type[AbstractState]:
        if self.settings.app_timeout:
            self.ib.setTimeout(self.settings.app_timeout)

        done = await self._wait_for_activity()

        if next_state := self._priority_transition():
            return next_state

        if self._workload_completed(done):
            self.context.consume_workload_result()
            return StoppingState

        return ConnectedState

    def on_timeout(self, idle_period: float) -> None:
        """Request a probe after an IB idle timeout."""

        self.context.request_state_transition(ProbingState)

    def on_broker_message(self, code: int, message: str) -> None:
        """Wait for broker recovery when degradation is reported."""

        if code in self.context.BROKER_WAIT_CODES:
            self.context.request_state_transition(WaitingForBrokerState)

    async def _wait_for_activity(self) -> set[asyncio.Future[Any]]:
        """Wait until connected-state work needs supervisor attention."""

        tasks: set[asyncio.Future[Any]] = {
            asyncio.create_task(
                self.context._stop_requested.wait(),
                name="connection-supervisor-stop-wait",
            ),
            asyncio.create_task(
                self.context._state_transition_requested.wait(),
                name="connection-supervisor-state-transition-wait",
            ),
        }
        if self.context._workload_task is not None:
            tasks.add(self.context._workload_task)

        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        waiters = [task for task in pending if task is not self.context._workload_task]
        for task in waiters:
            task.cancel()
        await asyncio.gather(*waiters, return_exceptions=True)
        return done

    def _workload_completed(self, done: set[asyncio.Future[Any]]) -> bool:
        """Return whether the supervised workload task finished."""

        return (
            self.context._workload_task is not None
            and self.context._workload_task in done
        )


class WaitingForBrokerState(AbstractState):
    """Wait briefly for broker-side auto-recovery before rebuilding."""

    transition_priority = 20

    async def handle(self) -> type[AbstractState]:
        done = await self._wait_for_recovery_signal()

        if next_state := self._priority_transition():
            return next_state

        if self._workload_completed(done):
            self.context.consume_workload_result()
            return StoppingState

        return RestartingState

    def on_update(self) -> None:
        """Request a probe when IB traffic resumes during broker recovery."""

        self.context.request_state_transition(ProbingState)

    def on_broker_message(self, code: int, message: str) -> None:
        """Request a probe when IB reports data-maintained recovery."""

        if code == self.context.DATA_MAINTAINED_CODE:
            self.context.request_state_transition(ProbingState)

    async def _wait_for_recovery_signal(self) -> set[asyncio.Future[Any]]:
        """Wait for broker recovery, timeout, restart, stop, or workload completion."""

        tasks: set[asyncio.Future[Any]] = {
            asyncio.create_task(
                self.context._stop_requested.wait(),
                name="connection-supervisor-stop-wait",
            ),
            asyncio.create_task(
                self.context._state_transition_requested.wait(),
                name="connection-supervisor-state-transition-wait",
            ),
            asyncio.create_task(
                asyncio.sleep(self.settings.auto_recovery_grace_period),
                name="connection-supervisor-auto-recovery-wait",
            ),
        }
        if self.context._workload_task is not None:
            tasks.add(self.context._workload_task)

        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        waiters = [task for task in pending if task is not self.context._workload_task]
        for task in waiters:
            task.cancel()
        await asyncio.gather(*waiters, return_exceptions=True)
        return done

    def _workload_completed(self, done: set[asyncio.Future[Any]]) -> bool:
        """Return whether the supervised workload task finished while waiting."""

        return (
            self.context._workload_task is not None
            and self.context._workload_task in done
        )


class RestartingState(AbstractState):
    """Stop active work and disconnect before reconnecting immediately."""

    transition_priority = 30

    async def handle(self) -> type[AbstractState]:
        reason = self.context._take_restart_reason()
        await self.context.stop_workload(reason)
        self.context.disconnect()
        return ConnectingState


class StoppingState(AbstractState):
    """Final cleanup for supervisor shutdown."""

    async def handle(self) -> type[AbstractState]:
        self.context._stop_requested.set()
        await self.context.stop_workload("supervisor stopped")
        self.context.disconnect()
        return StoppedState


class StoppedState(AbstractState):
    """Terminal stopped state."""

    async def handle(self) -> type[AbstractState]:
        raise StoppedError
