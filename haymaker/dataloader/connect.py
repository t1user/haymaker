"""Dataloader connection lifecycle management."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from logging import getLogger

from ib_insync import IB

log = getLogger(__name__)


@dataclass
class DataloaderRuntime:
    """Run dataloader work for one or more supervisor connection cycles.

    Args:
        ib: Broker connection owned by this runtime.
        func: Async dataloader workload to run after connection.
        cleanup: Optional synchronous callback used to request cancellation
            before the supervisor stops the active workload.
    """

    ib: IB
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
                pass

    async def _run_work(self) -> None:
        """Run one dataloader workload and stop after normal completion."""

        try:
            await self.func()
        except asyncio.CancelledError:
            log.debug("Dataloader work interrupted.")
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

    async def close(self) -> None:
        """Ensure dataloader work has stopped before process shutdown."""
        await self.stop("dataloader runtime closing")
