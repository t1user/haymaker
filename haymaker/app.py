import asyncio
import datetime
import logging
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Protocol, cast
from zoneinfo import ZoneInfo

import eventkit as ev  # type: ignore
import ib_insync as ibi

from .config import CONFIG as config
from .handlers import IBHandlers
from .logging import setup_logging

# Don't change the order here!
# You want manager namespace to be logged,
# but dont' want to setup logging inside manager
# because tests import manager (and then all tests will get logged);
# MODULE app MUSTN'T BE IMPORTED BY ANY TESTS
setup_logging(config.get("logging_config"))

from .manager import CONTROLLER, IB, JOBS, Jobs  # noqa: E402

ibi.util.patchAsyncio()

log = logging.getLogger(__name__)

CONFIG = config.get("app") or {}


if config.get("log_broker"):
    broker_logger = IBHandlers(IB)


class IBC(Protocol):
    async def startAsync(self) -> None: ...

    async def terminateAsync(self) -> None: ...


@dataclass
class FakeIBC(IBC):
    restart_time: int = cast(int, CONFIG.get("restart_time"))

    async def startAsync(self) -> None:
        pass

    async def terminateAsync(self) -> None:
        CONTROLLER.set_hold()
        log.debug(f"Pausing {self.restart_time} secs before restart...")
        await asyncio.sleep(self.restart_time)


@dataclass
class App:
    ib: ibi.IB = IB
    jobs: Jobs = JOBS
    ibc: IBC = field(default_factory=FakeIBC)
    host: str = CONFIG.get("host") or "127.0.0.1"
    port: int = CONFIG.get("port") or 4002
    clientId: int = CONFIG.get("cliendId") or 0
    appStartupTime: float = CONFIG.get("appStartupTime") or 0
    appTimeout: float = CONFIG.get("appTimeout") or 20
    retryDelay: float = CONFIG.get("retryDelay") or 2
    probeContract: ibi.Contract = CONFIG.get("probeContract") or ibi.Forex("EURUSD")
    probeTimeout: float = CONFIG.get("probeTimeout") or 4
    no_future_roll_strategies: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # IB events
        self.ib.errorEvent += self.onErr
        self.ib.connectedEvent += self.onConnected
        self.ib.disconnectedEvent += self.onDisconnected

        self.watchdog = ibi.Watchdog(
            self.ibc,  # type: ignore
            self.ib,
            port=self.port,
            clientId=self.clientId,
            appStartupTime=self.appStartupTime,
            appTimeout=self.appTimeout,
            retryDelay=self.retryDelay,
            probeContract=self.probeContract,
            probeTimeout=self.probeTimeout,
        )
        # Watchdog events
        self.watchdog.startingEvent += self.onStarting
        self.watchdog.startedEvent += self.onStarted
        self.watchdog.stoppingEvent += self.onStopping
        self.watchdog.stoppedEvent += self.onStopped
        self.watchdog.softTimeoutEvent += self.onSoftTimeout
        self.watchdog.hardTimeoutEvent += self.onHardTimeout

        self.schedule_future_roll()

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

    def onErr(  # don't want word 'error' in logs, unless it's a real error
        self, reqId: int, errorCode: int, errorString: str, contract: ibi.Contract
    ) -> None:
        if errorCode in config.get("ignore_errors", []):
            return
        elif "URGENT" in errorString:
            log.error(f"{errorString} {reqId=} code={errorCode} {contract=}")
        else:
            log.debug(f"{errorString} {reqId=} code={errorCode} {contract=}")

    def onStarting(self, watchdog: ibi.Watchdog) -> None:
        log.debug("# # # # # # # # # ( R E ) S T A R T... # # # # # # # # # ")

    def onStarted(self, *args) -> None:
        log.debug("Watchdog started...")

    def onStopping(self, *args) -> None:
        log.debug("Watchdog stopping")

    def onStopped(self, *args) -> None:
        log.debug("Watchdog stopped...")
        debug_string = " | ".join(
            [
                (
                    task.get_name()
                    if not task.get_name().startswith("Task-")
                    else str(task.get_coro())
                )
                for task in asyncio.all_tasks()
            ]
        )
        log.debug(f"tasks: {debug_string}")

    def onSoftTimeout(self, watchdog: ibi.Watchdog) -> None:
        log.debug("Soft timeout event.")

    def onHardTimeout(self, watchdog: ibi.Watchdog) -> None:
        log.debug("Hard timeout event.")

    def onConnected(self, *args) -> None:
        log.debug("IB Connected")

    def onDisconnected(self, *args) -> None:
        log.debug(f"IB Disconnected {args}")

    def _log_event_error(self, event: ibi.Event, exception: Exception) -> None:
        log.error(f"Event error {event.name()}: {exception}", exc_info=True)

    async def connection_probe(self) -> bool:
        probe = self.ib.reqHistoricalDataAsync(
            self.probeContract, "", "30 S", "5 secs", "MIDPOINT", False
        )
        bars = None
        with suppress(asyncio.TimeoutError):
            bars = await asyncio.wait_for(probe, self.probeTimeout)
        if bars:
            return True
        else:
            return False

    def run(self) -> None:
        # this is the main entry point into strategy
        self.watchdog.startedEvent.connect(self._run, error=self._log_event_error)
        log.debug("initializing watchdog...")
        self.watchdog.start()
        log.debug("watchdog initialized")
        self.ib.run()

    async def _run(self, *args) -> None:
        # watchdog connects when api has connection with IB gateway
        # but does IB gateway have connection to IB?
        # whatever documentation tells us,
        # no way to know other than actually probe it
        while not await self.connection_probe():
            log.debug("Connection probe failed. Holding...")
            if not self.ib.isConnected():
                log.debug("IB not connected, breaking out of probe loop.")
                return
            await asyncio.sleep(5)
        log.debug("Probe successful. Will run controller...")
        try:
            await CONTROLLER.run()
            await self.jobs()
        except Exception as e:
            log.exception(e)
            # raise
