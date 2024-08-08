import asyncio
import logging
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Protocol, cast

import ib_insync as ibi

from .config import CONFIG as config
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


class IBC(Protocol):
    async def startAsync(self): ...

    async def terminateAsync(self): ...


@dataclass
class FakeIBC(IBC):
    restart_time: int = cast(int, CONFIG.get("restart_time"))

    async def startAsync(self):
        pass

    async def terminateAsync(self):
        CONTROLLER.set_hold()
        log.debug(f"Pausing {self.restart_time} secs before restart...")
        await asyncio.sleep(self.restart_time)


@dataclass
class App:
    ib: ibi.IB = IB
    jobs: Jobs = JOBS
    ibc: IBC = field(default_factory=FakeIBC)
    host: str = CONFIG.get("host") or "127.0.0.1"
    port: float = CONFIG.get("port") or 4002
    clientId: float = CONFIG.get("cliendId") or 0
    appStartupTime: float = CONFIG.get("appStartupTime") or 0
    appTimeout: float = CONFIG.get("appTimeout") or 20
    retryDelay: float = CONFIG.get("retryDelay") or 2
    probeContract: ibi.Contract = CONFIG.get("probeContract") or ibi.Forex("EURUSD")
    probeTimeout: float = CONFIG.get("probeTimeout") or 4
    _connections: int = 0

    def __post_init__(self):
        self.ib.errorEvent += self.onError
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

        self.watchdog.startingEvent += self.onStarting
        self.watchdog.startedEvent += self.onStarted
        self.watchdog.stoppingEvent += self.onStopping
        self.watchdog.stoppedEvent += self.onStopped
        self.watchdog.softTimeoutEvent += self.onSoftTimeout
        self.watchdog.hardTimeoutEvent += self.onHardTimeout

        log.debug(f"App initiated: {self}")

    def onError(
        self, reqId: int, errorCode: int, errorString: str, contract: ibi.Contract
    ) -> None:
        if errorCode in config.get("ignore_errors", []):
            return
        elif "URGENT" in errorString:
            log.error(f"IB error: {reqId=} {errorCode} {errorString} {contract=}")
        else:
            log.debug(f"IB warning: {reqId=} {errorCode} {errorString} {contract=}")

    def onStarting(self, watchdog: ibi.Watchdog) -> None:
        log.debug("# # # # # # # # # ( R E ) S T A R T... # # # # # # # # # ")

    def onStarted(self, *args) -> None:
        self._connections += 1
        log.debug(f"Watchdog started... connections: {self._connections}")

    def onStopping(self, *args) -> None:
        log.debug("Watchdog stopping")

    def onStopped(self, *args) -> None:
        self._connections -= 1
        log.debug(f"Watchdog stopped... connections: {self._connections}")
        log.debug(f"tasks: {asyncio.all_tasks()}")
        # for task in self.jobs._tasks:
        #     task.cancel("Watchdog stopped cancellation.")
        # # for task in asyncio.all_tasks():
        #     task.cancel("Watchdog stopped cancellation.")

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

    def run(self):
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
            raise
