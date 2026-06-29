import asyncio
import datetime
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import eventkit as ev  # type: ignore
import ib_insync as ibi

from .config import CONFIG as config
from .handlers import IBHandlers
from .logging import setup_logging
from .supervisor import ConnectionSettings, ConnectionSupervisor

if TYPE_CHECKING:
    from .controller import Controller

# Don't change the order here!
# You want manager namespace to be logged,
# but dont' want to setup logging inside manager
# because tests import manager (and then all tests will get logged);
# MODULE app MUSTN'T BE IMPORTED BY ANY TESTS
setup_logging(config.get("logging_config"))

from .manager import CONTROLLER, IB, JOBS, Jobs  # noqa: E402
from .timeout import Timeout  # noqa: E402

ibi.util.patchAsyncio()

log = logging.getLogger(__name__)


if config.get("log_broker"):
    broker_logger = IBHandlers(IB)


@dataclass
class LiveRuntime:
    """Run live controller and streamer work for one supervisor cycle."""

    jobs: Jobs = field(default_factory=lambda: JOBS)
    controller: "Controller" = field(default_factory=lambda: CONTROLLER)
    future_roll_time: tuple[int, int] = (10, 0)
    future_roll_timezone: str = "America/New_York"
    no_future_roll_strategies: list[str] = field(default_factory=list)
    request_restart: Callable[[str], bool | None] | None = field(
        default=None, init=False, repr=False
    )
    exit_on_failed_sync: bool = False
    _future_roll_timer: ev.Event | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_config(cls, top_config: Mapping[str, Any]) -> "LiveRuntime":
        """Create live runtime settings from Haymaker config."""

        app_config = top_config.get("app") or {}
        roll_hour, roll_minute = app_config.get("future_roll_time", [10, 0])
        return cls(
            future_roll_time=(roll_hour, roll_minute),
            future_roll_timezone=app_config.get(
                "future_roll_timezone", "America/New_York"
            ),
        )

    def set_restart_handler(
        self, request_restart: Callable[[str], bool | None]
    ) -> None:
        """Set the restart callback used by stale-streamer timeouts."""

        self.request_restart = request_restart
        Timeout.set_restart_handler(request_restart)

    def set_no_future_roll_strategies(self, strategies: list[str]) -> None:
        """Set strategy names that should be excluded from futures rolls."""

        self.no_future_roll_strategies = strategies

    async def start(self) -> None:
        """Start controller and strategy jobs after connectivity is verified."""

        log.debug("Will run controller...")
        try:
            controller_started = await self.controller.run()
            if not controller_started and self.exit_on_failed_sync:
                log.debug("Controller did not start; jobs will not be started.")
                return
            await self.jobs()
        except asyncio.CancelledError:
            log.debug("Live runtime task cancelled.")
            raise
        except ConnectionError as ce:
            log.info(f"Connection fault: {ce}")
        except Exception as e:
            log.exception(e)

    async def stop(self, reason: str) -> None:
        """Put controller on hold while supervised runtime work stops."""

        self.controller.set_hold()
        log.debug(f"Stopping live runtime: {reason}")

    def schedule_future_roll(self) -> None:
        """Schedule daily futures rolls for the app lifetime."""

        roll_hour, roll_minute = self.future_roll_time
        roll_timezone = ZoneInfo(self.future_roll_timezone)
        rt = datetime.time(hour=roll_hour, minute=roll_minute, tzinfo=roll_timezone)
        self.controller.set_no_future_roll_strategies(self.no_future_roll_strategies)
        self._future_roll_timer = ev.Event.timerange(
            start=rt, step=datetime.timedelta(days=1)  # type: ignore
        )
        self._future_roll_timer += self.controller.roll_futures
        log.debug(f"Future roll scheduled for {rt} {rt.tzinfo.key}")  # type: ignore


@dataclass
class App:
    no_future_roll_strategies: list[str] = field(default_factory=list)
    ib: ibi.IB = IB
    runtime: LiveRuntime = field(
        default_factory=lambda: LiveRuntime.from_config(config)
    )
    settings: ConnectionSettings = field(
        default_factory=lambda: ConnectionSettings.from_config(
            config.get("app") or {}, 0
        )
    )
    supervisor: ConnectionSupervisor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.runtime.set_no_future_roll_strategies(self.no_future_roll_strategies)
        self.runtime.schedule_future_roll()
        self.supervisor = ConnectionSupervisor(self.ib, self.runtime, self.settings)
        self.runtime.set_restart_handler(self.supervisor.request_restart)
        log.debug(f"App initiated: {self}")

    def run(self) -> None:
        # this is the main entry point into strategy
        log.debug("Initializing connection supervisor.")
        asyncio.run(self.supervisor.run())
