"""State-machine supervisor for owned IB socket recovery."""

from __future__ import annotations

import asyncio
from abc import ABC
from collections.abc import Mapping
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, Protocol, Self

import ib_insync as ibi

log = getLogger(__name__)


class SupervisorWorkload(Protocol):
    """Workload run by :class:`ConnectionSupervisor` after IB is usable."""

    async def start(self) -> None:
        """Start or resume work after a usable IB connection is available."""

    async def stop(self, reason: str) -> None:
        """Release active work before the supervisor reconnects or exits."""


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
        connection_lost_retry_delay: Seconds to wait after lost connection before reconnecting
        auto_recovery_grace_period: Seconds to wait for broker-side recovery before reconnecting.
        restart_on_recovered_connection: Whether to restart even after IB reports data was maintained.
    """

    host: str = "127.0.0.1"
    port: int = 4002
    client_id: int = 0
    connect_timeout: float = 15
    retry_delay: float = 30
    app_timeout: float = 90
    probe_contract: ibi.Contract = field(default_factory=lambda: ibi.Forex("EURUSD"))
    probe_timeout: float = 15
    connection_lost_retry_delay: float = 90
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
            connect_timeout=config.get("connectTimeout", 15),
            retry_delay=config.get("retryDelay", 30),
            app_timeout=config.get("appTimeout", 90),
            probe_contract=config.get("probeContract") or ibi.Forex("EURUSD"),
            probe_timeout=config.get("probeTimeout", 15),
            connection_lost_retry_delay=config.get("connection_lost_retry", 90),
            auto_recovery_grace_period=config.get("auto_recovery_grace_period", 120),
            restart_on_recovered_connection=config.get(
                "restart_on_recovered_connection", False
            ),
        )


@dataclass
class ConnectionSupervisor:
    ib: ibi.IB
    workload: SupervisorWorkload
    settings: ConnectionSettings = field(default_factory=ConnectionSettings)
    workload_task: asyncio.Task[None] | None = field(default=None, repr=False)
    run_task: asyncio.Task[None] | None = field(default=None, repr=False)
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

                self.run_task = asyncio.create_task(
                    self._run_onion(), name="ConnectionSupervisor-run-task"
                )
                if self.run_task is not None:
                    await self.run_task

                if self.stop_requested.is_set():
                    log.debug("Workload completed. Exiting.")
                    return

            except (ConnectionError, asyncio.CancelledError):
                if self.stop_requested.is_set():
                    break
                delay = self.settings.connection_lost_retry_delay
                log.debug(f"Connection lost... will retry in {delay}")
                await self._sleep_until_stop(delay)
                if self.stop_requested.is_set():
                    break
            except Exception as e:
                log.exception(e)
                break

    async def _run_onion(self):
        async with Connection():
            async with Watcher():
                async with Workload():
                    async with Waiter() as waiter:
                        await waiter.wait()

    def stop(self) -> None:
        """Request supervisor shutdown; the run loop performs cleanup."""
        log.debug("Stop requested")
        self.stop_requested.set()
        if self.run_task is not None:
            self.run_task.cancel()

    def request_restart(self, reason: str = "") -> None:
        """Record a reconnect/rebuild request."""

        restart_reason = reason or "restart requested"
        log.debug(f"Restart requested: {restart_reason}")
        self.restart_requested.set()

    def set_workload_completed(self) -> None:
        self._workload_completed = True

    def onDisconnectedEvent(self, *args) -> None:
        if not self._intentional_disconnect and self.run_task is not None:
            self.request_restart("IB socket disconnected")
            self.run_task.cancel()

    async def _sleep_until_stop(self, delay: float) -> None:
        """Sleep for retry delay, returning early when shutdown is requested."""

        try:
            await asyncio.wait_for(self.stop_requested.wait(), timeout=delay)
        except asyncio.TimeoutError:
            pass


class OnionLayer(ABC):
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

    async def wait(self) -> None:
        assert self.ct.workload_task is not None
        assert self._event_waiters is not None
        waiting_tasks = (
            *self._event_waiters,
            self.ct.workload_task,
        )
        await asyncio.wait(
            waiting_tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )


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

    def clear_flags(self) -> None:
        """Clear broker recovery flags after one broker-wait episode."""

        self._broker_wait_requested = False
        self._data_maintained = False

    def onErrEvent(
        self, reqId: int, code: int, message: str, contract: ibi.Contract
    ) -> None:
        if code in self.ct.BROKER_WAIT_CODES:
            log.debug(f"Broker recovery wait requested by code {code}: {message}")
            self._broker_wait_requested = True
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
        self.ct.request_restart(reason)

    def close(self) -> None:
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
