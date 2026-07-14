import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import ib_insync as ibi

from .async_wrappers import (
    QueueRunner,
    cancel_background_tasks,
)
from .logging import shutdown_logging_queue
from .runtime import RuntimeContext
from .supervisor import ConnectionSettings, ConnectionSupervisor, Runtime

log = logging.getLogger(__name__)


@dataclass
class LiveRuntime:
    """Run live controller and streamer work for one supervisor cycle."""

    context: RuntimeContext

    @property
    def ib(self) -> ibi.IB:
        """Return the broker connection owned by this runtime."""
        return self.context.ib

    @classmethod
    def from_context(cls, context: RuntimeContext) -> "LiveRuntime":
        """Create live runtime settings from the process context."""

        return cls(context=context)

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
        try:
            await self.context.controller.run()
            await self.context.require_startup_jobs()()
        except asyncio.CancelledError:
            log.debug("Live runtime task cancelled.")
            raise
        except ConnectionError as ce:
            log.info(f"Connection fault: {ce}")
        except Exception as e:
            log.exception(e)

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
        log.debug("App initialized: %s", self)

    async def _run(self) -> None:
        """Run the supervisor and complete process-level async cleanup."""
        try:
            await self.supervisor.run()
        finally:
            try:
                await self.runtime.close()
            finally:
                await cancel_background_tasks()
                await QueueRunner.close_all()

    def run(self) -> None:
        # this is the main entry point into strategy
        log.debug("Initializing connection supervisor.")
        try:
            asyncio.run(self._run())
        except KeyboardInterrupt:
            log.info("Keyboard interrupt received; stopping application.")
        finally:
            shutdown_logging_queue()

    def __str__(self) -> str:
        """Return a compact application description suitable for logs."""

        return f"App<client_id={self.settings.client_id}, runtime={self.runtime!s}>"
