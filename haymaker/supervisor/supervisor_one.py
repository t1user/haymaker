"""State-machine supervisor for owned IB socket recovery."""

from __future__ import annotations

import asyncio
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum, auto
from logging import getLogger
from typing import Self

import ib_insync as ibi

from .settings import ConnectionSettings, SupervisorWorkload

log = getLogger(__name__)


class CycleOutcome(Enum):
    """Result from one supervisor onion cycle."""

    RESTART = auto()
    STOP = auto()
    WORKLOAD_DONE = auto()


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
    _workload_completed: bool = False
    # True while the supervisor closes its own socket, so disconnectedEvent is
    # not classified as an unexpected outage.
    _intentional_disconnect: bool = field(default=False, init=False, repr=False)
    _resets_blocked: bool = field(default=False, init=False, repr=False)
    _delay_next_restart: bool = field(default=False, init=False, repr=False)

    BROKER_WAIT_CODES = frozenset({1100, 2110, 2103, 2105, 2157, 10182})
    DATA_LOST_CODE = 1101
    DATA_MAINTAINED_CODE = 1102
    SOCKET_RESET_CODE = 1300
    CONNECTION_RESTORED_CODES = frozenset({DATA_MAINTAINED_CODE, DATA_LOST_CODE})

    def __post_init__(self) -> None:
        OnionLayer.set_context(self)
        self.ib.disconnectedEvent += self.onDisconnectedEvent

    async def run(self):
        while not self._workload_completed:
            try:
                self.restart_requested.clear()

                onion_task = asyncio.create_task(
                    self._run_onion(), name="ConnectionSupervisor-onion-task"
                )
                self.onion_task = onion_task
                # Do not let external cancellation cancel the onion before stop()
                # marks shutdown intent; run() handles that cancellation path below.
                outcome = await asyncio.shield(onion_task)
                if outcome in (CycleOutcome.STOP, CycleOutcome.WORKLOAD_DONE):
                    log.debug(f"Supervisor cycle finished with {outcome.name}.")
                    return
                if outcome is CycleOutcome.RESTART and self._delay_next_restart:
                    self._delay_next_restart = False
                    await self._wait_before_reconnect()
                    if self.stop_requested.is_set():
                        break

            except (ConnectionError, asyncio.CancelledError) as exc:
                if self.stop_requested.is_set():
                    break
                if isinstance(exc, asyncio.CancelledError) and self._onion_running():
                    self.stop()
                    if self.onion_task is not None:
                        try:
                            await self.onion_task
                        except asyncio.CancelledError:
                            pass
                    raise
                self.restart_requested.clear()
                await self._wait_before_reconnect()
                if self.stop_requested.is_set():
                    break
            except Exception as e:
                log.exception(e)
                break

    async def _run_onion(self) -> CycleOutcome:
        async with Connection():
            async with Watcher():
                async with Workload():
                    async with Waiter() as waiter:
                        return await waiter.wait()

    def stop(self) -> None:
        """Request supervisor shutdown; the run loop performs cleanup."""
        log.debug("Stop requested")
        self.stop_requested.set()
        if self.onion_task is not None:
            self.onion_task.cancel()

    def request_restart(self, reason: str = "") -> bool:
        """Record a reconnect/rebuild request."""

        restart_reason = reason or "restart requested"
        if self._resets_blocked:
            log.debug(
                f"Restart request blocked during broker recovery: {restart_reason}"
            )
            return False
        log.debug(f"Restart requested: {restart_reason}")
        self.restart_requested.set()
        return True

    def block_resets_set(self) -> None:
        """Block restart requests while broker recovery owns the lifecycle decision."""

        self._resets_blocked = True

    def block_resets_clear(self) -> None:
        """Allow restart requests after broker recovery wait has finished."""

        self._resets_blocked = False

    def set_workload_completed(self) -> None:
        self._workload_completed = True

    def onDisconnectedEvent(self, *args) -> None:
        if not self._intentional_disconnect:
            if self.request_restart("IB socket disconnected"):
                self._delay_next_restart = True

    async def _wait_before_reconnect(self) -> None:
        """Wait before reconnecting after connection loss or restart cancellation."""

        delay = self.settings.connection_lost_retry_delay
        log.debug(f"Connection lost... will retry in {delay}")
        await self._sleep_until_stop(delay)

    def _onion_running(self) -> bool:
        """Return whether the active onion task is still running."""

        return self.onion_task is not None and not self.onion_task.done()

    async def _sleep_until_stop(self, delay: float) -> None:
        """Sleep for retry delay, returning early when shutdown is requested."""

        try:
            await asyncio.wait_for(self.stop_requested.wait(), timeout=delay)
        except asyncio.TimeoutError:
            pass


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
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.debug(f"IB connection attempt failed: {exc!r}")
                await asyncio.sleep(self.ct.settings.retry_delay)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        if self.ct.ib.isConnected():
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


class Workload(OnionLayer):

    async def __aenter__(self) -> Self:
        """Start the supervised workload as a tracked task."""

        if self.ct.workload_task is None or self.ct.workload_task.done():
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

        if task.done():
            self._log_workload_result(task)
            self.ct.workload_task = None
            return

        await self.ct.workload.stop(self._stop_reason(exc_type, exc))
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
        assert self.ct.workload_task
        if self.ct.workload_task.done():
            self.ct.set_workload_completed()

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
            return CycleOutcome.WORKLOAD_DONE
        return CycleOutcome.RESTART


class ConnectionIssueManager:

    def __init__(self, ct: ConnectionSupervisor) -> None:
        self.ct = ct
        ct.ib.errorEvent += self.onErrEvent
        ct.ib.timeoutEvent += self.onTimeoutEvent
        self.wakeup = asyncio.Event()
        self._broker_wait_requested = False
        self._data_maintained = False

    def set_timeout(self):
        if self.ct.settings.app_timeout:
            self.ct.ib.setTimeout(self.ct.settings.app_timeout)

    async def run(self):
        while True:
            await self.probe()
            self.set_timeout()
            await self.wakeup.wait()
            self.wakeup.clear()

            if self._broker_wait_requested:
                log.debug(
                    "Entering broker recovery wait for "
                    f"{self.ct.settings.auto_recovery_grace_period}s."
                )
                broker_recovered = await self.wait_for_broker()
                data_maintained = self._data_maintained
                self.clear_flags()
                if not broker_recovered:
                    self.restart(
                        f"Broker grace period of "
                        f"{self.ct.settings.auto_recovery_grace_period}s "
                        f"expired without re-establishing connection."
                    )
                    break
                if (
                    data_maintained
                    and not self.ct.settings.restart_on_recovered_connection
                ):
                    log.debug("Broker re-connected with data maintained, no restart.")
                    continue
                else:
                    self.restart(
                        f"Broker re-connected, restarting due to policy {data_maintained=}"
                    )
                    break

    def clear_flags(self) -> None:
        """Clear broker recovery flags after one broker-wait episode."""

        self._broker_wait_requested = False
        self._data_maintained = False
        self.ct.block_resets_clear()

    def onErrEvent(
        self, reqId: int, code: int, message: str, contract: ibi.Contract
    ) -> None:
        if code in self.ct.BROKER_WAIT_CODES:
            log.debug(f"Broker recovery wait requested by code {code}: {message}")
            self._broker_wait_requested = True
            self.ct.block_resets_set()
            self.wakeup.set()
        if code == self.ct.SOCKET_RESET_CODE:
            # this is meant to test live behaviour and change if necessary
            log.debug("Socket reset code received.")
        if code == self.ct.DATA_MAINTAINED_CODE:
            log.debug(f"Broker reports data maintained after recovery: {message}")
            self._data_maintained = True
        if code == self.ct.DATA_LOST_CODE:
            log.debug(f"Broker reports data lost after recovery: {message}")
            self._data_maintained = False

    def onTimeoutEvent(self, idle_period: float) -> None:
        self.wakeup.set()

    async def wait_for_broker(self) -> bool:
        waiting_object = BrokerWaiter(self.ct)
        broker_recovered = False
        try:
            await asyncio.wait_for(
                waiting_object.wait(),
                self.ct.settings.auto_recovery_grace_period,
            )
            broker_recovered = True
        except asyncio.TimeoutError:
            pass
        finally:
            waiting_object.close()
        return broker_recovered

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
        self.ct.block_resets_clear()
        self.ct.request_restart(reason)

    def close(self) -> None:
        self.ct.block_resets_clear()
        self.ct.ib.errorEvent -= self.onErrEvent
        self.ct.ib.timeoutEvent -= self.onTimeoutEvent


class BrokerWaiter:

    def __init__(self, ct: ConnectionSupervisor) -> None:
        self.ct = ct
        self.connection_restored_event = asyncio.Event()
        ct.ib.errorEvent += self.onErrEvent

    def onErrEvent(
        self, reqId: int, code: int, message: str, contract: ibi.Contract
    ) -> None:
        if code in self.ct.CONNECTION_RESTORED_CODES:
            self.connection_restored_event.set()

    async def wait(self) -> None:
        await self.connection_restored_event.wait()

    def close(self):
        self.ct.ib.errorEvent -= self.onErrEvent
