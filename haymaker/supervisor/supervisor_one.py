"""Onion-layer supervisor for owned IB socket recovery."""

from __future__ import annotations

import asyncio
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum, auto
from logging import getLogger
from typing import Self

import ib_insync as ibi

from .codes import (
    BROKER_CONNECTIVITY_LOST_CODES,
    DATA_LOST_CODE,
    DATA_MAINTAINED_CODE,
    SOCKET_RESET_CODE,
    WEAK_DATA_FARM_CODES,
)
from .settings import (
    ConnectionSettings,
    SupervisorWorkload,
    bind_supervisor_controls,
)

log = getLogger(__name__)


class CycleOutcome(Enum):
    """Result from one supervisor onion cycle."""

    RESTART = auto()
    STOP = auto()
    WORKLOAD_DONE = auto()


class BrokerRecoveryOutcome(Enum):
    """Result from waiting for broker-side connectivity recovery."""

    TIMEOUT = auto()
    DATA_MAINTAINED = auto()
    DATA_LOST = auto()


@dataclass
class ConnectionSupervisor:
    ib: ibi.IB
    workload: SupervisorWorkload
    settings: ConnectionSettings = field(default_factory=ConnectionSettings)
    workload_task: asyncio.Task[None] | None = field(default=None, repr=False)
    onion_task: asyncio.Task[CycleOutcome] | None = field(default=None, repr=False)
    stop_requested: asyncio.Event = field(
        default_factory=asyncio.Event, init=False, repr=False
    )
    restart_requested: asyncio.Event = field(
        default_factory=asyncio.Event, init=False, repr=False
    )
    connection_unavailable: asyncio.Event = field(
        default_factory=asyncio.Event, init=False, repr=False
    )
    _workload_completed: bool = False
    # True while the supervisor closes its own socket, so disconnectedEvent is
    # not classified as an unexpected outage.
    _intentional_disconnect: bool = field(default=False, init=False, repr=False)
    _resets_blocked: bool = field(default=False, init=False, repr=False)
    _delay_next_restart: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        OnionLayer.set_context(self)
        self.connection_unavailable.set()
        bind_supervisor_controls(
            self.workload,
            self.request_restart,
            self.connection_unavailable,
        )
        self.ib.disconnectedEvent += self.onDisconnectedEvent

    async def run(self):
        recoveries = 0
        while not self._workload_completed:
            try:
                self.restart_requested.clear()

                log.debug("Supervisor cycle starting.")
                self.onion_task = asyncio.create_task(
                    self._run_onion(), name="ConnectionSupervisor-onion-task"
                )
                # Do not let external cancellation decide onion cleanup; run()
                # classifies the interruption and owns the recovery policy.
                outcome = await asyncio.shield(self.onion_task)
                log.debug(f"Supervisor cycle outcome: {outcome.name}.")
                if outcome in (CycleOutcome.STOP, CycleOutcome.WORKLOAD_DONE):
                    log.debug(f"Supervisor run finished with {outcome.name}.")
                    return
                recoveries = 0
                if outcome is CycleOutcome.RESTART and self._delay_next_restart:
                    self._delay_next_restart = False
                    if await self.wait_before_reconnect(
                        self.settings.connection_lost_retry_delay
                    ):
                        break
                    log.debug(
                        "Reconnect delay finished; starting new supervisor cycle."
                    )

            except asyncio.CancelledError as exc:
                if self.stop_requested.is_set():
                    break
                task = asyncio.current_task()
                if task is not None and task.cancelling():
                    log.debug("Supervisor run cancelled; stopping.")
                    self.stop()
                    await self.cleanup_onion()
                    raise
                recoveries += 1
                if await self.recover_from_unexpected_error(exc, recoveries):
                    break
            except Exception as exc:
                if self.stop_requested.is_set():
                    break
                recoveries += 1
                if await self.recover_from_unexpected_error(exc, recoveries):
                    break

    async def _run_onion(self) -> CycleOutcome:
        async with Connection():
            async with Watcher():
                async with Workload():
                    async with Waiter() as waiter:
                        return await waiter.wait()

    async def recover_from_unexpected_error(
        self, exc: BaseException, recovery_count: int
    ) -> bool:
        """
        Clean up after an unexpected cycle failure.

        Return True when the supervisor should stop instead of attempting
        another recovery cycle.
        """

        if recovery_count > self.settings.max_recoveries:
            log.exception(
                "Maximum supervisor recoveries exceeded; stopping.",
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            self.stop()
            await self.cleanup_onion()
            return True

        log.exception(
            "Unexpected supervisor cycle failure; restarting.",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        self.restart_requested.set()
        await self.cleanup_onion()
        self.restart_requested.clear()
        return await self.wait_before_reconnect(
            self.settings.connection_lost_retry_delay
        )

    async def cleanup_onion(self) -> None:
        """Cancel and collect the current onion task when it is still running."""

        if self.onion_task is not None and not self.onion_task.done():
            self.onion_task.cancel()
            try:
                await self.onion_task
            except asyncio.CancelledError:
                pass

    def stop(self) -> None:
        """Request supervisor shutdown; the run loop performs cleanup."""
        log.debug("Stop requested")
        self.mark_connection_unavailable("supervisor stop requested")
        self.stop_requested.set()
        if self.onion_task is not None:
            self.onion_task.cancel()

    def request_restart(self, reason: str = "") -> bool:
        """Record a reconnect/rebuild request if this phase accepts one."""

        restart_reason = reason or "restart requested"
        if self._resets_blocked:
            log.debug(
                f"Restart request blocked during broker recovery: {restart_reason}"
            )
            return False
        log.debug(f"Restart requested: {restart_reason}")
        self.mark_connection_unavailable(restart_reason)
        self.restart_requested.set()
        return True

    def _request_restart_now(self, reason: str = "") -> bool:
        """Record a supervisor-owned immediate reconnect/rebuild request."""

        self.block_resets_clear()
        return self.request_restart(reason)

    def mark_connection_available(self, reason: str) -> None:
        """Allow workload sync after the supervisor has a usable connection."""

        if self.connection_unavailable.is_set():
            log.debug(f"Connection marked available: {reason}")
        self.connection_unavailable.clear()

    def mark_connection_unavailable(self, reason: str) -> None:
        """Abort workload sync while the supervisor cannot trust the connection."""

        if not self.connection_unavailable.is_set():
            log.debug(f"Connection marked unavailable: {reason}")
        self.connection_unavailable.set()

    def clear_cycle_restart_request(self) -> None:
        """Discard restart requests after the current watcher cycle has ended."""

        if self.restart_requested.is_set():
            log.debug("Clearing restart request after watcher cleanup.")
        self.restart_requested.clear()

    def block_resets_set(self) -> None:
        """Block restart requests while broker recovery owns the lifecycle decision."""

        self._resets_blocked = True

    def block_resets_clear(self) -> None:
        """Allow restart requests after broker recovery wait has finished."""

        self._resets_blocked = False

    def set_workload_completed(self) -> None:
        log.debug("Supervised workload completed.")
        self._workload_completed = True

    def onDisconnectedEvent(self, *args) -> None:
        if not self._intentional_disconnect:
            if self._request_restart_now("IB socket disconnected"):
                self._delay_next_restart = True

    async def wait_before_reconnect(self, delay: float) -> bool:
        """
        Wait before reconnecting for ``delay`` seconds, while watching
        for stop request. Return True immediately if stop request
        registered, False otherwise.
        """

        log.debug(f"Will try reconnection in {delay}s")
        try:
            await asyncio.wait_for(self.stop_requested.wait(), timeout=delay)
        except asyncio.TimeoutError:
            pass

        if self.stop_requested.is_set():
            return True
        else:
            return False


class OnionLayer(ABC):
    # supervisor_one assumes one active supervisor instance per process; these
    # module-local layers share that active context instead of carrying it through
    # every context-manager constructor.
    ct: ConnectionSupervisor

    @classmethod
    def set_context(cls, context: ConnectionSupervisor) -> None:
        cls.ct = context

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        pass


class Connection(OnionLayer):

    async def __aenter__(self) -> Self:
        while not self.ct.ib.isConnected():
            try:
                log.debug(
                    f"Connecting to IB at {self.ct.settings.host}:{self.ct.settings.port}."
                )
                await self.ct.ib.connectAsync(
                    self.ct.settings.host,
                    self.ct.settings.port,
                    clientId=self.ct.settings.client_id,
                    timeout=self.ct.settings.connect_timeout,
                )
                log.debug("IB socket connected.")
                self.ct.mark_connection_available("IB socket connected")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.debug(f"IB connection attempt failed: {exc!r}")
                if await self.ct.wait_before_reconnect(self.ct.settings.retry_delay):
                    raise asyncio.CancelledError()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        if self.ct.ib.isConnected():
            self.ct.mark_connection_unavailable("disconnecting IB socket")
            self.ct._intentional_disconnect = True
            try:
                self.ct.ib.disconnect()
            finally:
                self.ct._intentional_disconnect = False


class Watcher(OnionLayer):

    _connection_issue_manager: ConnectionIssueManager | None
    _connection_issue_manager_task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> Self:
        self._connection_issue_manager = ConnectionIssueManager(self.ct)
        self._connection_issue_manager_task = asyncio.create_task(
            self._connection_issue_manager.run()
        )
        return self

    async def __aexit__(self, *args):
        if (
            self._connection_issue_manager_task is not None
            and not self._connection_issue_manager_task.done()
        ):
            self._connection_issue_manager_task.cancel()
            try:
                await self._connection_issue_manager_task
            except asyncio.CancelledError:
                pass
        self._connection_issue_manager_task = None
        if self._connection_issue_manager is not None:
            self._connection_issue_manager.close()
            self._connection_issue_manager = None
        self.ct.clear_cycle_restart_request()


class Workload(OnionLayer):

    async def __aenter__(self) -> Self:
        """Start the supervised workload as a tracked task."""

        if self.ct.workload_task is None or self.ct.workload_task.done():
            log.debug("Starting supervised workload.")
            self.ct.workload_task = asyncio.create_task(
                self.ct.workload.start(), name="connection-supervisor-workload"
            )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        """Stop active workload or collect its completed result."""

        task = self.ct.workload_task
        if task is None:
            return

        reason = self._stop_reason(exc_type, exc)
        should_stop = (
            self.ct.stop_requested.is_set()
            or self.ct.restart_requested.is_set()
            or exc_type is not None
        )
        if should_stop:
            log.debug(f"Stopping supervised workload: {reason}")
            await self.ct.workload.stop(reason)

        if task.done():
            log.debug("Supervised workload task already finished during cleanup.")
            self._log_workload_result(task)
            self.ct.workload_task = None
            return

        if not should_stop:
            log.debug(f"Stopping supervised workload: {reason}")
            await self.ct.workload.stop(reason)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        self.ct.workload_task = None

    def _stop_reason(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
    ) -> str:
        """Return the lifecycle reason reported to workload cleanup."""

        if self.ct.stop_requested.is_set():
            return "supervisor stopped"
        if self.ct.restart_requested.is_set():
            return "restart requested"
        if exc_type is not None:
            return f"{exc_type.__name__}: {exc}"
        return "supervisor stopped"

    @staticmethod
    def _log_workload_result(task: asyncio.Task[None]) -> None:
        """Log failure from a completed workload task."""

        try:
            task.result()
        except asyncio.CancelledError:
            log.debug("Connection workload task cancelled.")
        except Exception:
            log.exception("Connection workload task failed.")


class Waiter(OnionLayer):
    _event_waiters: tuple[asyncio.Task[bool], asyncio.Task[bool]] | None = None

    async def __aenter__(self) -> Self:
        self._event_waiters = (
            asyncio.create_task(
                self.ct.restart_requested.wait(),
                name="supervisor-restart-requested-waiter",
            ),
            asyncio.create_task(
                self.ct.stop_requested.wait(), name="supervisor-stop-requested-waiter"
            ),
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        if self._event_waiters is not None:
            for task in self._event_waiters:
                task.cancel()
            await asyncio.gather(*self._event_waiters, return_exceptions=True)
        self._event_waiters = None

    async def wait(self) -> CycleOutcome:
        assert self.ct.workload_task is not None
        assert self._event_waiters is not None
        waiting_tasks = (
            *self._event_waiters,
            self.ct.workload_task,
        )
        done, _ = await asyncio.wait(
            waiting_tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if self.ct.stop_requested.is_set():
            return CycleOutcome.STOP
        if self.ct.restart_requested.is_set():
            return CycleOutcome.RESTART
        if self.ct.workload_task in done:
            self.ct.set_workload_completed()
            return CycleOutcome.WORKLOAD_DONE
        return CycleOutcome.RESTART


class ConnectionIssueManager:

    def __init__(self, ct: ConnectionSupervisor) -> None:
        self.ct = ct
        ct.ib.errorEvent += self.onErrEvent
        ct.ib.timeoutEvent += self.onTimeoutEvent
        self.wakeup = asyncio.Event()
        self._broker_recovery_event = asyncio.Event()
        self._broker_wait_requested = False
        self._broker_outcome: BrokerRecoveryOutcome | None = None

    def set_timeout(self):
        if self.ct.settings.app_timeout:
            self.ct.ib.setTimeout(self.ct.settings.app_timeout)

    async def run(self):
        while True:
            self.set_timeout()
            await self.wakeup.wait()
            self.wakeup.clear()

            if self._broker_wait_requested:
                self.ct.mark_connection_unavailable("broker recovery wait")
                log.debug(
                    "Entering broker recovery wait for "
                    f"{self.ct.settings.auto_recovery_grace_period}s."
                )
                broker_outcome = await self.wait_for_broker()
                self.clear_flags()
                if broker_outcome is BrokerRecoveryOutcome.TIMEOUT:
                    self.restart(
                        f"Broker grace period of "
                        f"{self.ct.settings.auto_recovery_grace_period}s "
                        f"expired without re-establishing connection."
                    )
                    break
                if broker_outcome is BrokerRecoveryOutcome.DATA_MAINTAINED:
                    if self.ct.settings.restart_on_recovered_connection:
                        self.restart(
                            "Broker re-connected with data maintained; "
                            "restarting due to policy."
                        )
                        break
                    log.debug("Broker re-connected with data maintained, no restart.")
                    self.ct.mark_connection_available(
                        "broker re-connected with data maintained"
                    )
                    continue
                if broker_outcome is BrokerRecoveryOutcome.DATA_LOST:
                    self.restart("Broker re-connected with data lost; restarting.")
                    break
            else:
                await self.probe()

    def clear_flags(self) -> None:
        """Clear broker recovery flags after one broker-wait episode."""

        self._broker_wait_requested = False
        self._broker_outcome = None
        self._broker_recovery_event.clear()
        self.ct.block_resets_clear()

    def onErrEvent(
        self, reqId: int, code: int, message: str, contract: ibi.Contract
    ) -> None:
        if code in BROKER_CONNECTIVITY_LOST_CODES:
            log.debug(f"Broker recovery wait requested by code {code}: {message}")
            self._broker_wait_requested = True
            self.ct.mark_connection_unavailable(
                f"broker recovery wait requested by code {code}"
            )
            self.ct.block_resets_set()
            self.wakeup.set()
        if code == SOCKET_RESET_CODE:
            log.debug(f"Broker reports API socket reset: {message}")
            self.restart(f"broker reset API socket port ({code})")
        elif code == DATA_MAINTAINED_CODE:
            log.debug(f"Broker reports data maintained after recovery: {message}")
            if self._broker_wait_requested:
                self._broker_outcome = BrokerRecoveryOutcome.DATA_MAINTAINED
                self._broker_recovery_event.set()
            elif self.ct.settings.restart_on_recovered_connection:
                self.restart(
                    f"broker connectivity restored with data maintained ({code})"
                )
        elif code == DATA_LOST_CODE:
            log.debug(f"Broker reports data lost after recovery: {message}")
            if self._broker_wait_requested:
                self._broker_outcome = BrokerRecoveryOutcome.DATA_LOST
                self._broker_recovery_event.set()
            else:
                self.restart(f"broker connectivity restored with data lost ({code})")
        elif code in WEAK_DATA_FARM_CODES:
            if self.ct.settings.log_datafarm_status:
                log.debug(f"broker message {code}: {message}")

    def onTimeoutEvent(self, idle_period: float) -> None:
        self.wakeup.set()

    async def wait_for_broker(self) -> BrokerRecoveryOutcome:
        if self._broker_outcome is not None:
            return self._broker_outcome

        try:
            await asyncio.wait_for(
                self._broker_recovery_event.wait(),
                self.ct.settings.auto_recovery_grace_period,
            )
        except asyncio.TimeoutError:
            return BrokerRecoveryOutcome.TIMEOUT
        return self._broker_outcome or BrokerRecoveryOutcome.TIMEOUT

    async def probe(self) -> None:
        if not await self._probe():
            self.restart("Connection probe failed.")

    async def _probe(self) -> bool:
        """Return whether the broker accepted a small historical-data request."""

        try:
            probe = self.ct.ib.reqHistoricalDataAsync(
                self.ct.settings.probe_contract, "", "30 S", "5 secs", "MIDPOINT", False
            )
            bars = await asyncio.wait_for(probe, self.ct.settings.probe_timeout)
        except (asyncio.TimeoutError, ConnectionError) as exc:
            log.debug(f"Connection probe did not complete: {exc!r}")
            return False
        return bool(bars)

    def restart(self, reason: str = "") -> None:
        self.ct._request_restart_now(reason)

    def close(self) -> None:
        self.ct.block_resets_clear()
        self.ct.ib.errorEvent -= self.onErrEvent
        self.ct.ib.timeoutEvent -= self.onTimeoutEvent
