import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from .runtime import RuntimeContext
from .supervisor import ConnectionSettings, ConnectionSupervisor

log = logging.getLogger(__name__)


@dataclass
class LiveRuntime:
    """Run live controller and streamer work for one supervisor cycle."""

    context: RuntimeContext
    exit_on_failed_sync: bool = False

    @classmethod
    def from_context(cls, context: RuntimeContext) -> "LiveRuntime":
        """Create live runtime settings from the process context."""

        app_config = context.config.get("app") or {}
        return cls(
            context=context,
            exit_on_failed_sync=app_config.get("exit_on_failed_sync", False),
        )

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
            controller_started = await self.context.controller.run()
            if not controller_started and self.exit_on_failed_sync:
                log.debug("Controller did not start; jobs will not be started.")
                return
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


@dataclass
class App:
    context: RuntimeContext
    runtime: LiveRuntime | None = None
    settings: ConnectionSettings | None = None
    supervisor: ConnectionSupervisor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.runtime is None:
            self.runtime = LiveRuntime.from_context(self.context)
        if self.settings is None:
            self.settings = ConnectionSettings.from_config(
                self.context.config.get("app") or {}, 0
            )
        self.supervisor = ConnectionSupervisor(
            self.context.ib, self.runtime, self.settings
        )
        log.debug(f"App initiated: {self}")

    def run(self) -> None:
        # this is the main entry point into strategy
        log.debug("Initializing connection supervisor.")
        try:
            asyncio.run(self.supervisor.run())
        except KeyboardInterrupt:
            log.info("Keyboard interrupt received; stopping application.")
            self.supervisor.stop()
