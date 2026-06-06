"""Connection supervisor states.

The supervisor owns events, workload tasks, and socket cleanup. States only
wait for relevant signals and return the next state class.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .supervisor import Supervisor


class StoppedError(Exception):
    """Raised by the terminal stopped state."""


class AbstractState(ABC):
    """Base class for connection supervisor states."""

    def __init__(self, context: Supervisor) -> None:
        self.context = context
        self.settings = context.settings
        self.ib = context.ib

    @abstractmethod
    async def handle(self) -> type[AbstractState]:
        """Run state work and return the next state class."""

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

    async def handle(self) -> type[AbstractState]:
        if self.context._stop_requested.is_set():
            return StoppingState

        if self.context._restart_requested.is_set():
            return RestartingState

        if not self.ib.isConnected():
            return ConnectingState

        if await self.context.probe():
            self.context.clear_broker_degraded_context()
            if self.context._workload_task is None:
                return StartingWorkloadState
            return ConnectedState

        if self.context._restart_requested.is_set():
            return RestartingState

        if not self.ib.isConnected():
            return ConnectingState

        if self.context.recent_recoverable_broker_message():
            return WaitingForBrokerState

        return RestartingState


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

        self.context._timeout_received.clear()
        done = await self._wait_for_activity()

        if self.context._stop_requested.is_set():
            return StoppingState

        if self.context._restart_requested.is_set():
            return RestartingState

        if self._workload_completed(done):
            self.context.consume_workload_result()
            return StoppingState

        if self.context._timeout_received.is_set():
            if self.context.recent_recoverable_broker_message():
                return WaitingForBrokerState
            return ProbingState

        return ConnectedState

    async def _wait_for_activity(self) -> set[asyncio.Future[Any]]:
        """Wait until connected-state work needs supervisor attention."""

        tasks: set[asyncio.Future[Any]] = {
            asyncio.create_task(
                self.context._stop_requested.wait(),
                name="connection-supervisor-stop-wait",
            ),
            asyncio.create_task(
                self.context._restart_requested.wait(),
                name="connection-supervisor-restart-wait",
            ),
            asyncio.create_task(
                self.context._timeout_received.wait(),
                name="connection-supervisor-timeout-wait",
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

    async def handle(self) -> type[AbstractState]:
        self.context.clear_recovery_signals()
        done = await self._wait_for_recovery_signal()

        if self.context._stop_requested.is_set():
            return StoppingState

        if self.context._restart_requested.is_set():
            return RestartingState

        if self._workload_completed(done):
            self.context.consume_workload_result()
            return StoppingState

        if (
            self.context._broker_recovered.is_set()
            or self.context._traffic_resumed.is_set()
        ):
            return ProbingState

        return RestartingState

    async def _wait_for_recovery_signal(self) -> set[asyncio.Future[Any]]:
        """Wait for broker recovery, timeout, restart, stop, or workload completion."""

        tasks: set[asyncio.Future[Any]] = {
            asyncio.create_task(
                self.context._stop_requested.wait(),
                name="connection-supervisor-stop-wait",
            ),
            asyncio.create_task(
                self.context._restart_requested.wait(),
                name="connection-supervisor-restart-wait",
            ),
            asyncio.create_task(
                self.context._broker_recovered.wait(),
                name="connection-supervisor-broker-recovered-wait",
            ),
            asyncio.create_task(
                self.context._traffic_resumed.wait(),
                name="connection-supervisor-traffic-resumed-wait",
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

    async def handle(self) -> type[AbstractState]:
        reason = self.context.consume_restart_reason()
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
