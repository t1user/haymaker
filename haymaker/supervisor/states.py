"""Connection supervisor states.

The supervisor owns events, workload tasks, and socket cleanup. States only
wait for relevant signals and return the next state.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from logging import getLogger
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from .supervisor import ConnectionSupervisor

StateResult: TypeAlias = "type[AbstractState]"

log = getLogger(__name__)


class StoppedError(Exception):
    """Raised by the terminal stopped state."""


class AbstractState(ABC):
    """Base class for connection supervisor states."""

    def __init__(self, context: ConnectionSupervisor) -> None:
        self.context = context
        self.settings = context.settings
        self.ib = context.ib
        self.wakeup = asyncio.Event()
        self._restart_requested = False

    @abstractmethod
    async def handle(self) -> StateResult:
        """Run state work and return the next state."""

    def on_timeout(self, idle_period: float) -> None:
        """Handle an IB idle timeout while this state is active."""

    def on_update(self) -> None:
        """Handle resumed IB traffic while this state is active."""

    def on_broker_message(self, code: int, message: str) -> None:
        """Handle a broker message while this state is active."""

    def request_stop(self) -> None:
        """Wake this state after supervisor shutdown is requested."""

        self.wakeup.set()

    def request_restart(self) -> None:
        """Request a restart transition from this state."""

        self._restart_requested = True
        self.wakeup.set()

    def requested_lifecycle_state(self) -> StateResult | None:
        """Return a requested stop or restart transition, if one is pending."""

        if self.context.stop_requested:
            return StoppingState

        if self._restart_requested:
            return RestartingState

        return None

    def __str__(self) -> str:
        return self.__class__.__name__.upper()


class ConnectingState(AbstractState):
    """Connect the owned IB client."""

    async def handle(self) -> StateResult:
        if next_state := self.requested_lifecycle_state():
            return next_state

        if self.ib.isConnected():
            return ProbingState

        if await self._connect():
            if next_state := self.requested_lifecycle_state():
                return next_state
            return ProbingState
        return StoppingState

    async def _connect(self) -> bool:
        """Connect the owned IB client, retrying until stopped."""

        while not self.context.stop_requested and not self.ib.isConnected():
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
                await self.context.wait_or_stop(self.settings.retry_delay)
        return self.ib.isConnected()


class ProbingState(AbstractState):
    """Verify that the broker connection is usable."""

    def __init__(self, context: ConnectionSupervisor) -> None:
        super().__init__(context)
        self._broker_wait_requested = False

    async def handle(self) -> StateResult:
        if next_state := self.requested_lifecycle_state():
            return next_state

        if self._broker_wait_requested:
            return WaitingForBrokerState

        if not self.ib.isConnected():
            return ConnectingState

        probe_succeeded = await self._probe()

        if next_state := self.requested_lifecycle_state():
            return next_state

        if self._broker_wait_requested:
            return WaitingForBrokerState

        if probe_succeeded:
            if self.context._workload_task is None:
                return StartingWorkloadState
            return ConnectedState

        if not self.ib.isConnected():
            return ConnectingState

        return RestartingState

    async def _probe(self) -> bool:
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

    def on_broker_message(self, code: int, message: str) -> None:
        """Wait for broker recovery if degradation is reported while probing."""

        if code in self.context.BROKER_WAIT_CODES:
            self._broker_wait_requested = True
            self.wakeup.set()


class StartingWorkloadState(AbstractState):
    """Start the supervised workload after a successful probe."""

    async def handle(self) -> StateResult:
        if next_state := self.requested_lifecycle_state():
            return next_state

        self.context.start_workload()
        return ConnectedState


class ConnectedState(AbstractState):
    """Wait for restart, timeout, stop, or workload completion."""

    def __init__(self, context: ConnectionSupervisor) -> None:
        super().__init__(context)
        self._timeout_requested = False
        self._broker_wait_requested = False

    async def handle(self) -> StateResult:
        if self.settings.app_timeout:
            self.ib.setTimeout(self.settings.app_timeout)

        done = await self._wait_for_activity()

        if next_state := self.requested_lifecycle_state():
            return next_state

        if self._broker_wait_requested:
            return WaitingForBrokerState

        if self._timeout_requested:
            return ProbingState

        if self._workload_completed(done):
            self.context.consume_workload_result()
            return StoppingState

        return ConnectedState

    def on_timeout(self, idle_period: float) -> None:
        """Request a probe after an IB idle timeout."""

        self._timeout_requested = True
        self.wakeup.set()

    def on_broker_message(self, code: int, message: str) -> None:
        """Wait for broker recovery when degradation is reported."""

        if code in self.context.BROKER_WAIT_CODES:
            self._broker_wait_requested = True
            self.wakeup.set()

    async def _wait_for_activity(self) -> set[asyncio.Future[Any]]:
        """Wait until connected-state work needs supervisor attention."""

        tasks: set[asyncio.Future[Any]] = {
            asyncio.create_task(
                self.wakeup.wait(),
                name="connection-supervisor-state-wakeup",
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

    def __init__(self, context: ConnectionSupervisor) -> None:
        super().__init__(context)
        self._probe_requested = False

    async def handle(self) -> StateResult:
        done = await self._wait_for_recovery_signal()

        if next_state := self.requested_lifecycle_state():
            return next_state

        if self._probe_requested:
            return ProbingState

        if self._workload_completed(done):
            self.context.consume_workload_result()
            return StoppingState

        return RestartingState

    def on_update(self) -> None:
        """Request a probe when IB traffic resumes during broker recovery."""

        self._probe_requested = True
        self.wakeup.set()

    def on_broker_message(self, code: int, message: str) -> None:
        """Request a probe when IB reports data-maintained recovery."""

        if code == self.context.DATA_MAINTAINED_CODE:
            self._probe_requested = True
            self.wakeup.set()

    async def _wait_for_recovery_signal(self) -> set[asyncio.Future[Any]]:
        """Wait for broker recovery, timeout, restart, stop, or workload completion."""

        tasks: set[asyncio.Future[Any]] = {
            asyncio.create_task(
                self.wakeup.wait(),
                name="connection-supervisor-state-wakeup",
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

    async def handle(self) -> StateResult:
        await self.context.stop_workload("restart requested")
        self.context.disconnect()
        return ConnectingState


class StoppingState(AbstractState):
    """Final cleanup for supervisor shutdown."""

    async def handle(self) -> StateResult:
        self.context.mark_stop_requested()
        await self.context.stop_workload("supervisor stopped")
        self.context.disconnect()
        return StoppedState


class StoppedState(AbstractState):
    """Terminal stopped state."""

    async def handle(self) -> StateResult:
        raise StoppedError
