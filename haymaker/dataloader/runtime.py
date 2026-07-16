"""Dataloader runtime construction and lifecycle management."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING

from ib_insync import IB

from haymaker.config.settings import DataloaderSettings
from haymaker.databases import StoreFactory

if TYPE_CHECKING:
    from .dataloader import DataloaderSession

log = getLogger(__name__)


def _create_ib() -> IB:
    """Return the broker client owned by a standalone dataloader runtime."""

    return IB()


@dataclass
class DataloaderRuntime:
    """Construct and run dataloader work under connection supervision.

    Args:
        settings: Validated framework settings for this dataloader process.
        ib: Optional broker connection owned by this runtime.
        session: Optional preconfigured dataloader session. When omitted, the
            runtime constructs its own session for ``ib``.
    """

    settings: DataloaderSettings = field(repr=False)
    ib: IB = field(default_factory=_create_ib)
    session: DataloaderSession | None = field(default=None, repr=False)
    store_factory: StoreFactory = field(init=False, repr=False)
    _work_task: asyncio.Task | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Create and wire the configured dataloader session when needed."""

        self.store_factory = StoreFactory(self.settings.storage)
        if self.session is None:
            from .dataloader import DataloaderSession, Manager
            from .pacer import RequestPacing

            download = self.settings.download
            pacing = RequestPacing(
                self.ib,
                no_restriction=self.settings.pacing.no_restriction,
                allowance_fraction=self.settings.pacing.allowance_fraction,
            )
            manager = Manager(
                self.ib,
                pacing=pacing,
                store_factory=self.store_factory,
                futures=self.settings.futures,
                source=download.source,
                gap_fill_mode=download.gap_fill_mode,
                use_rth=download.use_rth,
                max_lookback_days=download.max_lookback_days,
                save_every_chunks=download.save_every_chunks,
                wts=download.what_to_show,
                bar_size=download.bar_size,
                pacer_no_restriction=self.settings.pacing.no_restriction,
                pacer_allowance_fraction=self.settings.pacing.allowance_fraction,
            )
            self.session = DataloaderSession(
                self.ib,
                manager=manager,
                number_of_workers=download.number_of_workers,
            )
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
