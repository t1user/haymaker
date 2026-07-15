import asyncio
import logging
import signal
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import ib_insync as ibi

from .async_wrappers import (
    QueueRunner,
    cancel_background_tasks,
)
from .logging import setup_asyncio_logger
from .runtime import RuntimeContext
from .supervisor import ConnectionSettings, ConnectionSupervisor

log = logging.getLogger(__name__)


class Runtime(Protocol):
    """Application runtime managed by the shared process runner."""

    @property
    def ib(self) -> ibi.IB:
        """Return the broker connection owned by this runtime."""

    def bind_supervisor(
        self,
        request_restart: Callable[[str], bool | None],
        connection_unavailable: asyncio.Event,
    ) -> None:
        """Receive supervisor restart and connection lifecycle controls."""

    async def start(self) -> None:
        """Start or resume work after a usable IB connection is available."""

    async def stop(self, reason: str) -> None:
        """Release active work before the supervisor reconnects or exits."""

    async def close(self) -> None:
        """Release runtime-owned resources before application shutdown."""


@dataclass
class LiveRuntime:
    """Run live controller and streamer work for one supervisor cycle."""

    context: RuntimeContext

    @property
    def ib(self) -> ibi.IB:
        """Return the broker connection owned by this runtime."""
        return self.context.ib

    def bind_supervisor(
        self,
        request_restart: Callable[[str], bool | None],
        connection_unavailable: asyncio.Event,
    ) -> None:
        """Bind supervisor controls used by runtime components."""

        self.context.bind_restart_handler(request_restart)
        self.context.controller.set_sync_abort_event(connection_unavailable)

    async def start(self) -> None:
        """Start controller and strategy jobs after connectivity is verified."""

        log.debug("Will run controller...")
        await self.context.controller.run()
        await self.context.require_startup_jobs()()

    async def stop(self, reason: str) -> None:
        """Put controller on hold while supervised runtime work stops."""

        self.context.controller.set_hold()
        log.debug(f"Stopping live runtime: {reason}")

    async def close(self) -> None:
        """Flush final live-runtime state before process shutdown."""
        self.context.close()

    def __str__(self) -> str:
        """Return a compact live-runtime description suitable for logs."""

        return f"LiveRuntime<{self.context!s}>"


@dataclass
class App:
    runtime: Runtime = field(repr=False)
    settings: ConnectionSettings
    supervisor: ConnectionSupervisor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.supervisor = ConnectionSupervisor(
            self.runtime.ib, self.runtime, self.settings
        )
        self.runtime.bind_supervisor(
            self.supervisor.request_restart,
            self.supervisor.connection_unavailable,
        )
        log.debug("App initialized: %s", self)

    async def _run(self) -> None:
        """Run the supervisor and complete process-level async cleanup."""

        loop = asyncio.get_running_loop()
        setup_asyncio_logger(loop)

        def request_sigterm_stop() -> None:
            """Request graceful cleanup; a second SIGTERM uses Linux defaults."""

            loop.remove_signal_handler(signal.SIGTERM)
            self.supervisor.stop()

        loop.add_signal_handler(signal.SIGTERM, request_sigterm_stop)
        try:
            await self.supervisor.run()
        finally:
            try:
                await self.runtime.close()
            finally:
                try:
                    await cancel_background_tasks()
                finally:
                    try:
                        await QueueRunner.close_all()
                    finally:
                        loop.remove_signal_handler(signal.SIGTERM)

    def run(self) -> None:
        # this is the main entry point into strategy
        log.debug("Initializing connection supervisor.")
        try:
            asyncio.run(self._run())
        except KeyboardInterrupt:
            log.info("Keyboard interrupt received; stopping application.")

    def __str__(self) -> str:
        """Return a compact application description suitable for logs."""

        return f"App<client_id={self.settings.client_id}, runtime={self.runtime!s}>"
