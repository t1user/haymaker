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

CONFIG = config.get("app") or {}


if config.get("log_broker"):
    broker_logger = IBHandlers(IB)


@dataclass
class LiveRuntime:
    """Run live controller and streamer work for one supervisor cycle."""

    ib: ibi.IB = IB
    jobs: Jobs = JOBS
    controller: "Controller" = CONTROLLER
    app_config: Mapping[str, Any] = field(default_factory=lambda: CONFIG)
    contract_refresh_max_age: float = CONFIG.get("contract_refresh_max_age", 86400)
    contract_refresh_check_interval: float = CONFIG.get(
        "contract_refresh_check_interval", 900
    )
    no_future_roll_strategies: list[str] = field(default_factory=list)
    request_restart: Callable[[str], None] | None = field(
        default=None, init=False, repr=False
    )
    stop_on_completion: bool = False
    _future_roll_timer: ev.Event | None = field(default=None, init=False, repr=False)
    _contract_refresh_timer: ev.Timer | None = field(
        default=None, init=False, repr=False
    )
    _scheduled: bool = field(default=False, init=False, repr=False)

    def set_restart_handler(self, request_restart: Callable[[str], None]) -> None:
        """Set the restart callback used by timeout and contract freshness checks."""

        self.request_restart = request_restart
        Timeout.set_restart_handler(request_restart)

    async def start(self) -> None:
        """Start controller and strategy jobs after connectivity is verified."""

        self.start_scheduled_tasks()
        log.debug("Probe successful. Will run controller...")
        try:
            controller_started = await self.controller.run()
            if not controller_started:
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
        """Put controller on hold and stop scheduled runtime checks."""

        self.controller.set_hold()
        self.stop_scheduled_tasks()
        log.debug(f"Stopping live runtime: {reason}")

    def start_scheduled_tasks(self) -> None:
        """Start scheduled live-runtime checks for the current cycle."""

        if not self._scheduled:
            self.schedule_future_roll()
            self.schedule_contract_refresh_check()
            self._scheduled = True

    def stop_scheduled_tasks(self) -> None:
        """Stop scheduled live-runtime checks for the current cycle."""

        if self._future_roll_timer:
            self._future_roll_timer.set_done()
            self._future_roll_timer = None
        if self._contract_refresh_timer:
            self._contract_refresh_timer.set_done()
            self._contract_refresh_timer = None
        self._scheduled = False

    def schedule_future_roll(self) -> None:
        """Schedule daily futures rolls while live runtime is active."""

        roll_hour, roll_minute = self.app_config.get("future_roll_time", [10, 0])
        roll_timezone = ZoneInfo(
            self.app_config.get("future_roll_timezone", "America/New_York")
        )
        rt = datetime.time(hour=roll_hour, minute=roll_minute, tzinfo=roll_timezone)
        self.controller.set_no_future_roll_strategies(self.no_future_roll_strategies)
        self._future_roll_timer = ev.Event.timerange(
            start=rt, step=datetime.timedelta(days=1)  # type: ignore
        )
        self._future_roll_timer += self.controller.roll_futures
        log.debug(f"Future roll scheduled for {rt} {rt.tzinfo.key}")  # type: ignore

    def schedule_contract_refresh_check(self) -> None:
        """Schedule checks that ensure contract metadata is refreshed daily."""

        if self.contract_refresh_check_interval:
            self._contract_refresh_timer = ev.Timer(
                self.contract_refresh_check_interval
            )
            self._contract_refresh_timer += self.ensure_contract_refresh  # type: ignore

    def ensure_contract_refresh(self, *args) -> None:
        """Request a restart when the last contract refresh is too old."""

        if self.request_restart and self.jobs.contract_refresh_is_overdue(
            self.contract_refresh_max_age
        ):
            self.request_restart("contract metadata refresh overdue")


@dataclass
class App:
    ib: ibi.IB = IB
    runtime: LiveRuntime = field(default_factory=LiveRuntime)
    settings: ConnectionSettings = field(
        default_factory=lambda: ConnectionSettings.from_live_config(config)
    )
    no_future_roll_strategies: list[str] = field(default_factory=list)
    supervisor: ConnectionSupervisor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        ibi.util.globalErrorEvent += self.onGlobalErrEvent
        if self.no_future_roll_strategies:
            self.runtime.no_future_roll_strategies = self.no_future_roll_strategies
        self.supervisor = ConnectionSupervisor(self.ib, self.runtime, self.settings)
        self.runtime.set_restart_handler(self.supervisor.request_restart)
        log.debug(f"App initiated: {self}")

    def onGlobalErrEvent(self, *args, **kwargs):
        """Log ib_insync global errors without making them operational errors."""

        log.debug(f"Global err: {args} {kwargs}")

    def run(self) -> None:
        # this is the main entry point into strategy
        self.ib.run(self.main())

    async def main(self) -> None:
        """Run the live application under the connection supervisor."""

        log.debug("Initializing connection supervisor.")
        await self.supervisor.run()
