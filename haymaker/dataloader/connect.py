"""Dataloader connection lifecycle management."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from logging import getLogger

from ib_insync import IB

from haymaker.config import CONFIG
from haymaker.supervisor import ConnectionSettings, ConnectionSupervisor

log = getLogger(__name__)

DEFAULT_CLIENT_ID = 1


@dataclass
class DataloaderRuntime:
    """Run dataloader work for one or more supervisor connection cycles.

    Args:
        func: Async dataloader workload to run after connection.
        cleanup: Optional callback used to release active work before restart.
    """

    func: Callable[[], Awaitable[None]]
    cleanup: Callable | None = None
    _work_task: asyncio.Task | None = field(default=None, init=False)

    async def start(self) -> None:
        """Start or resume dataloader work after connection."""

        if self._work_task and not self._work_task.done():
            log.debug("Dataloader work is still active after reconnection.")
        else:
            log.debug(f"Running {self.func}.")
            self._work_task = asyncio.create_task(
                self._run_work(), name="dataloader-work"
            )

        if self._work_task:
            try:
                await self._work_task
            except asyncio.CancelledError:
                log.debug("Dataloader work interrupted during connection recovery.")

    async def _run_work(self) -> None:
        """Run one dataloader workload and stop after normal completion."""

        try:
            await self.func()
        except asyncio.CancelledError:
            log.debug("Dataloader work interrupted during connection recovery.")
            raise

    async def stop(self, reason: str) -> None:
        """Release current work before reconnecting when configured to rerun."""

        log.debug(f"Restarting dataloader connection: {reason}")
        has_active_work = self._work_task is not None and not self._work_task.done()
        if self.cleanup and has_active_work:
            self.cleanup()

        if self._work_task and has_active_work:
            self._work_task.cancel()
            try:
                await self._work_task
            except asyncio.CancelledError:
                pass


@dataclass
class DataloaderConnection:
    """Run dataloader work under an owned supervised IB connection.

    Args:
        ib: Interactive Brokers client used by the dataloader.
        func: Async dataloader workload to run after connection.
        cleanup: Optional callback used to release work after disconnection.
    """

    ib: IB
    func: Callable
    cleanup: Callable | None = None
    runtime: DataloaderRuntime = field(init=False)
    supervisor: ConnectionSupervisor = field(init=False)

    def __post_init__(self) -> None:
        client_id = CONFIG.get("clientId", DEFAULT_CLIENT_ID)
        self.runtime = DataloaderRuntime(self.func, self.cleanup)
        self.supervisor = ConnectionSupervisor(
            self.ib,
            self.runtime,
            ConnectionSettings.from_config(CONFIG, client_id),
        )

    def run(self) -> None:
        """Run until the dataloader workload finishes."""

        asyncio.run(self.supervisor.run())


def connection(
    ib: IB,
    func: Callable,
    cleanup: Callable | None = None,
) -> DataloaderConnection:
    """Run dataloader work under the connection supervisor."""

    log.debug("Running dataloader under supervised connection.")
    dataloader_connection = DataloaderConnection(ib, func, cleanup)
    dataloader_connection.run()
    return dataloader_connection
