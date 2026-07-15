"""Dataloader runtime construction and lifecycle management."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING

from ib_insync import IB

if TYPE_CHECKING:
    from .dataloader import DataloaderSession

log = getLogger(__name__)

DEFAULT_CLIENT_ID = 1


def _create_ib() -> IB:
    """Return the broker client owned by a standalone dataloader runtime."""

    return IB()


@dataclass
class DataloaderRuntime:
    """Construct and run dataloader work under connection supervision.

    Args:
        ib: Optional broker connection owned by this runtime.
        session: Optional preconfigured dataloader session. When omitted, the
            runtime constructs its own session for ``ib``.
    """

    ib: IB = field(default_factory=_create_ib)
    session: DataloaderSession | None = field(default=None, repr=False)
    _work_task: asyncio.Task | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Create and wire the configured dataloader session when needed."""

        if self.session is None:
            from .dataloader import DataloaderSession

            self.session = DataloaderSession(self.ib)
        self.ib.errorEvent += self.session.pacing.onErrEvent
        log.debug("Dataloader runtime initialized.")

    def bind_supervisor(
        self,
        request_restart: Callable[[str], bool | None],
        connection_unavailable: asyncio.Event,
    ) -> None:
        """Accept lifecycle controls unused by the dataloader runtime."""

    async def start(self) -> None:
        """Start or resume dataloader work after connection."""

        assert self.session is not None
        if self._work_task and not self._work_task.done():
            log.debug("Dataloader work is still active after reconnection.")
        else:
            log.debug("Running %s.", self.session.run)
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

        assert self.session is not None
        try:
            await self.session.run()
        except asyncio.CancelledError:
            log.debug("Dataloader work interrupted.")
            raise

    async def stop(self, reason: str) -> None:
        """Release current work before reconnecting when configured to rerun."""

        log.debug(f"Restarting dataloader connection: {reason}")
        has_active_work = self._work_task is not None and not self._work_task.done()
        if has_active_work:
            assert self.session is not None
            self.session.cancel_tasks()

        if self._work_task and has_active_work:
            self._work_task.cancel()
            try:
                await self._work_task
            except asyncio.CancelledError:
                pass

    async def close(self) -> None:
        """Ensure dataloader work has stopped before process shutdown."""
        await self.stop("dataloader runtime closing")
