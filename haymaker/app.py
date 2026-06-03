import datetime
import logging
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

import eventkit as ev  # type: ignore
import ib_insync as ibi

from .config import CONFIG as config
from .handlers import IBHandlers
from .logging import setup_logging
from .supervisor import ConnectionSupervisor

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
class App:
    ib: ibi.IB = IB
    jobs: Jobs = JOBS
    host: str = CONFIG.get("host", "127.0.0.1")
    port: int = CONFIG.get("port", 4002)
    clientId: int = CONFIG.get("clientId", 0)
    connectTimeout: float = CONFIG.get("connectTimeout", 2)
    restart_time: float = CONFIG.get("restart_time", 30)
    appTimeout: float = CONFIG.get("appTimeout", 20)
    retryDelay: float = CONFIG.get("retryDelay", 2)
    probeContract: ibi.Contract = CONFIG.get("probeContract") or ibi.Forex("EURUSD")
    probeTimeout: float = CONFIG.get("probeTimeout", 4)
    auto_recovery_grace_period: float = CONFIG.get("auto_recovery_grace_period", 120)
    recovery_warning_after: float = CONFIG.get("recovery_warning_after", 300)
    recovery_warning_interval: float = CONFIG.get("recovery_warning_interval", 900)
    contract_refresh_max_age: float = CONFIG.get("contract_refresh_max_age", 86400)
    contract_refresh_check_interval: float = CONFIG.get(
        "contract_refresh_check_interval", 900
    )
    no_future_roll_strategies: list[str] = field(default_factory=list)
    supervisor: ConnectionSupervisor = field(init=False, repr=False)
    _contract_refresh_timer: ev.Timer | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        ibi.util.globalErrorEvent += self.onGlobalErr

        self.supervisor = ConnectionSupervisor(
            ib=self.ib,
            on_connected=self._run,
            on_restarting=self.onRestarting,
            host=self.host,
            port=self.port,
            client_id=self.clientId,
            connect_timeout=self.connectTimeout,
            restart_delay=self.restart_time,
            retry_delay=self.retryDelay,
            app_timeout=self.appTimeout,
            probe_contract=self.probeContract,
            probe_timeout=self.probeTimeout,
            auto_recovery_grace_period=self.auto_recovery_grace_period,
            recovery_warning_after=self.recovery_warning_after,
            recovery_warning_interval=self.recovery_warning_interval,
        )
        Timeout.set_restart_handler(self.supervisor.request_restart)

        self.schedule_future_roll()
        self.schedule_contract_refresh_check()

        log.debug(f"App initiated: {self}")

    def schedule_future_roll(self) -> None:
        roll_hour, roll_minute = CONFIG.get("future_roll_time", [10, 0])
        roll_timezone = ZoneInfo(CONFIG.get("future_roll_timezone", "America/New_York"))
        rt = datetime.time(hour=roll_hour, minute=roll_minute, tzinfo=roll_timezone)
        CONTROLLER.set_no_future_roll_strategies(self.no_future_roll_strategies)
        scheduler = ev.Event.timerange(
            start=rt, step=datetime.timedelta(days=1)  # type: ignore
        )
        scheduler += CONTROLLER.roll_futures
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

        if self.jobs.contract_refresh_is_overdue(self.contract_refresh_max_age):
            self.supervisor.request_restart("contract metadata refresh overdue")

    def onGlobalErr(self, *args, **kwargs):
        log.debug(f"Global err: {args} {kwargs}")

    def onRestarting(self, reason: str) -> None:
        """Put the controller on hold when an actual restart begins."""

        CONTROLLER.set_hold()
        log.debug(f"Restarting application: {reason}")

    def run(self) -> None:
        # this is the main entry point into strategy
        log.debug("Initializing connection supervisor.")
        self.supervisor.start()
        self.ib.run()

    async def _run(self) -> None:
        """Run application startup after the supervisor verifies connectivity."""

        log.debug("Probe successful. Will run controller...")
        try:
            controller_started = await CONTROLLER.run()
            if not controller_started:
                log.debug("Controller did not start; jobs will not be started.")
                return
            await self.jobs()
        except ConnectionError as ce:
            log.info(f"Connection fault: {ce}")
        except Exception as e:
            log.exception(e)
            # raise
