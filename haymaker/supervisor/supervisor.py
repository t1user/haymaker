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
class Supervisor:
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
        while True:
            try:
                self.restart_requested.clear()

                self.run_task = asyncio.create_task(
                    self._run_onion(), name="Supervisor-run-task"
                )
                if self.run_task is not None:
                    await self.run_task

                if self.stop_requested.is_set():
                    log.debug("Workload completed. Exiting.")
                    return

            except (ConnectionError, asyncio.CancelledError):
                delay = self.settings.connection_lost_retry_delay
                log.debug(f"Connection lost... will retry in {delay}")
                await asyncio.sleep(delay)
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

    def request_restart(self, reason: str = "") -> None:
        """Record a reconnect/rebuild request."""

        restart_reason = reason or "restart requested"
        log.debug(f"Restart requested: {restart_reason}")
        self.restart_requested.set()

    def onDisconnectedEvent(self, *args) -> None:
        if self.run_task is not None:
            self.run_task.cancel()


class OnionLayer(ABC):
    ct: Supervisor

    @classmethod
    def set_context(cls, context: Supervisor) -> None:
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


class Watcher(OnionLayer):

    _connection_issue_manager: ConnectionIssueManager | None
    _connection_issue_manager_task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> Self:
        self._connection_issue_manager = ConnectionIssueManager(self.ct)
        self._connecton_issue_manager_task = asyncio.create_task(
            self._connection_issue_manager.run()
        )
        return self

    async def __aexit__(self, *args):
        if (
            self._connection_issue_manager_task is not None
            and not self._connection_issue_manager_task.done()
        ):
            self._connection_issue_manager_task.cancel()
        self._connection_issue_manager_task = None
        if self._connection_issue_manager is not None:
            del self._connection_issue_manager


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
            self._workload_task = None
            return

        await self.ct.workload.stop(f"{exc_type} - {exc}")
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        self.ct.workload_task = None

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

    async def __aenter__(self) -> Self:
        return self

    async def wait(self) -> None:
        while self._should_wait():
            await asyncio.sleep(0)

    def _should_wait(self) -> bool:
        assert self.ct.workload_task is not None
        return not any(
            (
                self.ct.restart_requested.is_set(),
                self.ct.stop_requested.is_set(),
                self.ct.workload_task.done(),
            )
        )


class ConnectionIssueManager:

    def __init__(self, ct: Supervisor) -> None:
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

            if self._broker_wait_requested:
                await self.wait_for_broker()
                self._broker_wait_requested = False
                if (
                    self._data_maintained
                    and not self.ct.settings.restart_on_recovered_connection
                ):
                    continue
                else:
                    self.restart(f"Broker re-connected, restarting due to policy")

    def onErrEvent(
        self, reqId: int, code: int, message: str, contract: ibi.Contract
    ) -> None:
        if code in self.ct.BROKER_WAIT_CODES:
            self._broker_wait_requested = True
            self.wakeup.set()
        if code == self.ct.DATA_MAINTAINED_CODE:
            self._data_maintained = True
        if code == self.ct.DATA_LOST_CODE:
            self._data_maintained = False

    def onTimeoutEvent(self, idle_period: float) -> None:
        self.wakeup.set()

    async def wait_for_broker(self):
        waiting_object = BrokerWaiter(self.ct)
        try:
            await asyncio.wait_for(
                waiting_object.wait(),
                self.ct.settings.auto_recovery_grace_period,
            )
        except asyncio.TimeoutError:
            self.restart(
                f"Broker grace period of "
                f"{self.ct.settings.auto_recovery_grace_period}s "
                f"expired without re-establishing connection."
            )
        finally:
            del waiting_object

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

    def __del__(self) -> None:
        self.ct.ib.errorEvent += self.onErrEvent
        self.ct.ib.timeoutEvent += self.onTimeoutEvent


class BrokerWaiter:

    def __init__(self, ct: Supervisor) -> None:
        self.ct = ct
        self.connection_restored_event = asyncio.Event()
        ct.ib.errorEvent += self.onErrEvent

    def onErrEvent(self, code: int, message: str) -> None:
        if code in self.ct.CONNECTION_RESTORED_CODES:
            self.connection_restored_event.set()

    async def wait(self) -> None:
        await self.connection_restored_event.wait()

    def __del__(self):
        self.ct.ib.errorEvent -= self.onErrEvent
