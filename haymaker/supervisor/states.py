"""Connection supervisor states.

The supervisor owns lifecycle events, workload tasks, and socket cleanup. States
wait for state-local signals and return the next state.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable
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

    @abstractmethod
    async def handle(self) -> StateResult:
        """Run state work and return the next state."""

    def on_timeout(self, idle_period: float) -> None:
        """Handle an IB idle timeout while this state is active."""

    def on_update(self) -> None:
        """Handle resumed IB traffic while this state is active."""

    def on_broker_message(self, code: int, message: str) -> None:
        """Handle a broker message while this state is active."""

    async def wait_for_wakeup_or(
        self,
        *awaitables: Awaitable[Any] | asyncio.Future[Any],
    ) -> set[asyncio.Future[Any]]:
        """Wait until this state is woken or supplied awaitables complete."""

        wakeup_task = asyncio.create_task(
            self.wakeup.wait(),
            name="connection-supervisor-state-wakeup",
        )
        supplied_tasks: set[asyncio.Future[Any]] = {
            asyncio.ensure_future(awaitable) for awaitable in awaitables
        }
        tasks: set[asyncio.Future[Any]] = {wakeup_task, *supplied_tasks}

        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if wakeup_task in done:
            wakeup_task.result()
        return done & supplied_tasks

    def __str__(self) -> str:
        return self.__class__.__name__.upper()


class ConnectingState(AbstractState):
    """Connect the owned IB client."""

    async def handle(self) -> StateResult:
        if self.ib.isConnected():
            return ProbingState

        if await self._connect():
            return ProbingState
        return StoppingState

    async def _connect(self) -> bool:
        """Connect the owned IB client, retrying until stopped."""

        while not self.ib.isConnected():
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
                await asyncio.sleep(self.settings.retry_delay)
        return self.ib.isConnected()


class ProbingState(AbstractState):
    """Verify that the broker connection is usable."""

    def __init__(self, context: ConnectionSupervisor) -> None:
        super().__init__(context)
        self._broker_wait_requested = False

    async def handle(self) -> StateResult:
        if not self.ib.isConnected():
            return ConnectingState

        probe_task = asyncio.create_task(
            self._probe(), name="connection-supervisor-probe"
        )
        done = await self.wait_for_wakeup_or(probe_task)

        if self._broker_wait_requested:
            return WaitingForBrokerState

        if probe_task not in done:
            return ProbingState

        probe_succeeded = probe_task.result()

        if probe_succeeded:
            self.context.mark_connection_available("connection probe succeeded")
            if not self.context.has_workload:
                return StartingWorkloadState
            return ConnectedState

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
            self.context.mark_connection_unavailable(
                f"broker recovery wait requested by code {code}"
            )
            self.wakeup.set()


class StartingWorkloadState(AbstractState):
    """Start the supervised workload after a successful probe."""

    async def handle(self) -> StateResult:
        self.context.start_workload()
        return ConnectedState


class ConnectedState(AbstractState):
    """Wait for timeout or broker degradation."""

    def __init__(self, context: ConnectionSupervisor) -> None:
        super().__init__(context)
        self._timeout_registered = False
        self._broker_wait_requested = False

    async def handle(self) -> StateResult:
        if self.settings.app_timeout:
            self.ib.setTimeout(self.settings.app_timeout)

        await self.wait_for_wakeup_or()

        if self._broker_wait_requested:
            self.context.mark_connection_unavailable("broker recovery wait")
            return WaitingForBrokerState

        if self._timeout_registered:
            return ProbingState

        return ConnectedState

    def on_timeout(self, idle_period: float) -> None:
        """Request a probe after an IB idle timeout."""

        self._timeout_registered = True
        self.wakeup.set()

    def on_broker_message(self, code: int, message: str) -> None:
        """Wait for broker recovery when degradation is reported."""

        if code in self.context.BROKER_WAIT_CODES:
            self._broker_wait_requested = True
            self.context.mark_connection_unavailable(
                f"broker recovery wait requested by code {code}"
            )
            self.wakeup.set()


class WaitingForBrokerState(AbstractState):
    """Wait briefly for broker-side auto-recovery before rebuilding."""

    def __init__(self, context: ConnectionSupervisor) -> None:
        super().__init__(context)
        self._probe_requested = False

    async def handle(self) -> StateResult:
        self.context.mark_connection_unavailable("broker recovery wait")
        await self.wait_for_wakeup_or(
            asyncio.sleep(self.settings.auto_recovery_grace_period)
        )

        if self._probe_requested:
            return ProbingState

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


class RestartingState(AbstractState):
    """Stop active work and disconnect before reconnecting immediately."""

    async def handle(self) -> StateResult:
        self.context.mark_connection_unavailable("restart requested")
        await self.context.cleanup_workload("restart requested")
        self.context.disconnect()
        return ConnectingState


class StoppingState(AbstractState):
    """Final cleanup for supervisor shutdown."""

    async def handle(self) -> StateResult:
        self.context.mark_connection_unavailable("supervisor stopped")
        await self.context.cleanup_workload("supervisor stopped")
        self.context.disconnect()
        return StoppedState


class StoppedState(AbstractState):
    """Terminal stopped state."""

    async def handle(self) -> StateResult:
        raise StoppedError
