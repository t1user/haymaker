"""Dataloader connection lifecycle management."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from logging import getLogger
from typing import Literal, TypeAlias

from ib_insync import IB

from haymaker.config import CONFIG
from haymaker.supervisor import ConnectionSettings, ConnectionSupervisor

log = getLogger(__name__)

Mode: TypeAlias = Literal["reconnect", "wait"]


@dataclass
class DataloaderRuntime:
    """Run dataloader work for one or more supervisor connection cycles.

    Args:
        func: Async dataloader workload to run after connection.
        cleanup: Optional callback used to release work after disconnection.
        run_mode: Re-run work after reconnect, or wait for in-place recovery.
    """

    func: Callable
    cleanup: Callable | None = None
    run_mode: Mode = "reconnect"
    _started: bool = field(default=False, init=False)
    _work_task: asyncio.Task | None = field(default=None, init=False)

    async def start(self) -> None:
        """Start or resume dataloader work after connection."""

        if self.run_mode == "wait" and self._started:
            log.debug("Reconnected. Waiting for IB to resume sending data.")
        elif self._work_task and not self._work_task.done():
            log.debug("Dataloader work is still active after reconnection.")
        else:
            self._started = True
            log.debug(f"Running {self.func}.")
            self._work_task = asyncio.create_task(
                self._run_work(), name="dataloader-work"
            )

        if self._work_task:
            try:
                if self.run_mode == "wait":
                    await asyncio.shield(self._work_task)
                else:
                    await self._work_task
            except asyncio.CancelledError:
                log.debug("Dataloader work interrupted during connection recovery.")

    async def _run_work(self) -> None:
        """Run one dataloader workload and stop after normal completion."""

        try:
            await self.func()
        except (asyncio.CancelledError, ConnectionError):
            log.debug("Dataloader work interrupted during connection recovery.")
        except Exception as exc:
            log.exception(f"Dataloader work failed: {exc}")

    async def stop(self, reason: str) -> None:
        """Release current work before reconnecting when configured to rerun."""

        log.debug(f"Restarting dataloader connection: {reason}")
        has_active_work = self._work_task is not None and not self._work_task.done()
        if self.run_mode == "reconnect" and self.cleanup and has_active_work:
            self.cleanup()

        if self.run_mode == "reconnect" and self._work_task and has_active_work:
            self._work_task.cancel()
            try:
                await self._work_task
            except asyncio.CancelledError:
                pass


@dataclass
class DataloaderConnection:
    """Run dataloader work under an owned managed IB connection supervisor.

    Args:
        ib: Interactive Brokers client used by the dataloader.
        func: Async dataloader workload to run after connection.
        cleanup: Optional callback used to release work after disconnection.
        run_mode: Re-run work after reconnect, or wait for in-place recovery.
    """

    ib: IB
    func: Callable
    cleanup: Callable | None = None
    run_mode: Mode = "reconnect"
    runtime: DataloaderRuntime = field(init=False)
    supervisor: ConnectionSupervisor = field(init=False)

    def __post_init__(self) -> None:
        client_id = 51
        self.runtime = DataloaderRuntime(self.func, self.cleanup, self.run_mode)
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
    run_mode: Mode = "reconnect",
) -> DataloaderConnection:
    """Run dataloader work with shared connection supervision."""

    if run_mode not in ("reconnect", "wait"):
        raise ValueError(
            f"Unknown mode: {run_mode}. The gateway-managing watchdog mode "
            "is no longer supported."
        )

    log.debug(f"Running in {run_mode} mode.")
    dataloader_connection = DataloaderConnection(ib, func, cleanup, run_mode)
    dataloader_connection.run()
    return dataloader_connection
