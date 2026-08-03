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

import eventkit as ev  # type: ignore

from .codes import (
    BROKER_CONNECTIVITY_LOST_CODES,
    DATA_MAINTAINED_CODE,
    LIVE_UPDATE_FAILURE_CODE,
)

if TYPE_CHECKING:
    from .supervisor import ConnectionSupervisor

StateResult: TypeAlias = "type[AbstractState]"

log = getLogger(__name__)

STALE_SUBSCRIPTION_RESTART_DELAY = 180


class StoppedError(Exception):
    """Raised by the terminal stopped state."""


class AbstractState(ABC):
    """Base class for connection supervisor states."""

    accepts_stop = True
    accepts_restart = True
    observes_workload = True

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

    def handle_broker_connectivity_lost_code(self, code: int) -> bool:
        """Handle a broker message code indicating broker connectivity loss.

        Args:
            code: IB broker message code.

        Returns:
            True if the code indicated broker connectivity loss, otherwise False.
        """

        if code not in BROKER_CONNECTIVITY_LOST_CODES:
            return False

        log.debug(f"Broker connectivity lost by code {code}; waiting for recovery.")
        self.context.mark_connection_unavailable()
        self.wakeup.set()
        return True

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

    accepts_restart = False

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
            except TimeoutError:
                log.debug("IB connection attempt failed: Timeout")
                await asyncio.sleep(self.settings.retry_delay)
            except ConnectionError:
                log.debug("IB connection attempt failed: Connection unavailable")
                await asyncio.sleep(self.settings.retry_delay)
            except Exception:
                log.exception("Unexpected IB connection attempt failure.")
                await asyncio.sleep(self.settings.retry_delay)
        return self.ib.isConnected()


class ProbingState(AbstractState):
    """Verify that the broker connection is usable."""

    def __init__(self, context: ConnectionSupervisor) -> None:
        super().__init__(context)
        self._connectivity_lost_requested = False

    async def handle(self) -> StateResult:
        if not self.ib.isConnected():
            return ConnectingState

        probe_task = asyncio.create_task(
            self._probe(), name="connection-supervisor-probe"
        )
        done = await self.wait_for_wakeup_or(probe_task)

        if self._connectivity_lost_requested:
            return ConnectionLostState

        if probe_task not in done:
            return ProbingState

        probe_succeeded = probe_task.result()

        if probe_succeeded:
            self.context.mark_connection_available()
            if not self.context.has_workload:
                return StartingWorkloadState
            return ConnectedState

        return BackoffRestartCleanupState

    async def _probe(self) -> bool:
        """Return whether the broker accepted a small historical-data request."""

        try:
            probe_contract = await self.context.probe_contract()
            probe = self.ib.reqHistoricalDataAsync(
                probe_contract, "", "30 S", "5 secs", "MIDPOINT", False
            )
            bars = await asyncio.wait_for(probe, self.settings.probe_timeout)
        except asyncio.TimeoutError:
            log.debug("Connection probe did not complete: Timeout")
            return False
        except ConnectionError:
            log.debug("Connection probe did not complete: Connection unavailable")
            return False
        return bool(bars)

    def on_broker_message(self, code: int, message: str) -> None:
        """Wait for broker connectivity recovery if loss is reported."""

        if self.handle_broker_connectivity_lost_code(code):
            self._connectivity_lost_requested = True


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
        self._connectivity_lost_requested = False
        self._stale_subscription_signal: ev.Event | None = None
        self._stale_subscription_timeout: ev.Timeout | None = None
        self._stale_subscription_restart_due = False

    async def handle(self) -> StateResult:
        if self.settings.app_timeout:
            self.ib.setTimeout(self.settings.app_timeout)

        try:
            await self.wait_for_wakeup_or()
        finally:
            self._cancel_stale_subscription_timer()

        if self._connectivity_lost_requested:
            return ConnectionLostState

        if self._timeout_registered:
            return ProbingState

        if self._stale_subscription_restart_due:
            log.debug(
                "Stale subscription quiet period elapsed; "
                "requesting workload restart."
            )
            return RestartingState

        return ConnectedState

    def on_timeout(self, idle_period: float) -> None:
        """Request a probe after an IB idle timeout."""

        self._timeout_registered = True
        self.wakeup.set()

    def on_broker_message(self, code: int, message: str) -> None:
        """React to broker messages that matter while connected."""

        if code == LIVE_UPDATE_FAILURE_CODE:
            self._register_stale_subscription_signal(message)
            return

        if self.handle_broker_connectivity_lost_code(code):
            self._connectivity_lost_requested = True

    def _register_stale_subscription_signal(self, message: str) -> None:
        """Start or reset the quiet-period timer after IB ``10182``."""

        if not self.context.has_workload:
            return

        if self._stale_subscription_signal is None:
            self._stale_subscription_signal = ev.Event(
                "connection-supervisor-stale-subscription"
            )
            self._stale_subscription_timeout = self._stale_subscription_signal.timeout(
                STALE_SUBSCRIPTION_RESTART_DELAY
            )
            self._stale_subscription_timeout.connect(
                self._on_stale_subscription_quiet_period_elapsed
            )
            if self.settings.log_datafarm_status:
                log.debug(
                    "Stale subscription restart scheduled in "
                    f"{STALE_SUBSCRIPTION_RESTART_DELAY}s after quiet period: "
                    f"{message}"
                )

        self._stale_subscription_signal.emit()

    def _on_stale_subscription_quiet_period_elapsed(self) -> None:
        """Wake the state after the configured quiet period expires."""

        self._stale_subscription_restart_due = True
        self.wakeup.set()

    def _cancel_stale_subscription_timer(self) -> None:
        """Cancel pending stale-subscription timeout owned by this state."""

        if self._stale_subscription_signal is not None:
            self._stale_subscription_signal.set_done()
        elif (
            self._stale_subscription_timeout is not None
            and not self._stale_subscription_timeout.done()
        ):
            self._stale_subscription_timeout.set_done()

        self._stale_subscription_signal = None
        self._stale_subscription_timeout = None


class ConnectionLostState(AbstractState):
    """Wait briefly for broker-side connectivity to recover before probing."""

    async def handle(self) -> StateResult:
        self.context.mark_connection_unavailable()
        await self.wait_for_wakeup_or(
            asyncio.sleep(self.settings.auto_recovery_grace_period)
        )

        return ProbingState

    def on_update(self) -> None:
        """Ignore generic IB traffic while broker connectivity is lost."""

    def on_broker_message(self, code: int, message: str) -> None:
        """Request a probe when IB reports data-maintained recovery."""

        if code == DATA_MAINTAINED_CODE:
            log.debug(
                f"Broker connectivity restored with data maintained ({code}); probing."
            )
            self.wakeup.set()


class RestartingState(AbstractState):
    """Stop active work and disconnect before reconnecting immediately."""

    accepts_stop = False
    accepts_restart = False
    observes_workload = False

    async def handle(self) -> StateResult:
        await self._cleanup_for_restart()
        return self.next_state()

    def next_state(self) -> StateResult:
        """Return the state to enter after restart cleanup completes."""

        return ConnectingState

    async def _cleanup_for_restart(self) -> None:
        """Stop active work and disconnect the owned socket before reconnecting."""

        self.context.mark_connection_unavailable()
        await self.context.cleanup_workload("restart requested")
        self.context.disconnect()


class BackoffRestartingState(AbstractState):
    """Pause before reconnecting after failed-probe cleanup has finished."""

    accepts_stop = True
    accepts_restart = False
    observes_workload = False

    async def handle(self) -> StateResult:
        delay = self.settings.connection_lost_retry_delay
        log.debug(f"Waiting {delay}s before reconnecting after failed probe.")
        await asyncio.sleep(delay)
        return ConnectingState


class BackoffRestartCleanupState(RestartingState):
    """Stop active work and disconnect before a delayed reconnect."""

    def next_state(self) -> StateResult:
        """Return the backoff-wait state after cleanup completes."""

        return BackoffRestartingState


class StoppingState(AbstractState):
    """Final cleanup for supervisor shutdown."""

    accepts_stop = False
    accepts_restart = False
    observes_workload = False

    async def handle(self) -> StateResult:
        self.context.mark_connection_unavailable()
        await self.context.cleanup_workload("supervisor stopped")
        self.context.disconnect()
        return StoppedState


class StoppedState(AbstractState):
    """Terminal stopped state."""

    accepts_stop = False
    accepts_restart = False
    observes_workload = False

    async def handle(self) -> StateResult:
        raise StoppedError
