import asyncio
import logging
from dataclasses import dataclass, field
from typing import Protocol, cast

import ib_insync as ibi

from ib_tools.config import CONFIG

# from ib_tools.blotter import Blotter
from ib_tools.controller import CONTROLLER
from ib_tools.manager import IB, JOBS, Jobs

log = logging.getLogger(__name__)

ibi.util.patchAsyncio()


class IBC(Protocol):
    async def startAsync(self):
        ...

    async def terminateAsync(self):
        ...


@dataclass
class FakeIBC(IBC):
    restart_time: int = cast(int, CONFIG.get("restart_time"))

    async def startAsync(self):
        pass

    async def terminateAsync(self):
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
    # blotter: bool = True

    def __post_init__(self):
        self.ib.errorEvent += self.onError
        self.ib.connectedEvent += self.onConnected

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
        self.watchdog.softTimeoutEvent += self.onSoftTimeout
        self.watchdog.hardTimeoutEvent += self.onHardTimeout

    def onError(
        self, reqId: int, errorCode: int, errorString: str, contract: ibi.Contract
    ) -> None:
        if errorCode not in (
            2104,  # Market data farm connection is ok
            2106,  # data farm connection is OK
            2108,  # Market data farm [...] is inactive but [...~ok]
            2158,  # Sec-def data farm connection is OK
        ):
            log.debug(f"Error event: {reqId} {errorCode} {errorString} {contract}")
            # Consider handling 2103 - Market data connection is broken

    def onSoftTimeout(self, watchdog: ibi.Watchdog) -> None:
        log.debug("Soft timeout event.")

    def onHardTimeout(self, watchdog: ibi.Watchdog) -> None:
        log.debug("Hard timeout event.")

    def onStarting(self, watchdog: ibi.Watchdog) -> None:
        log.debug("Starting...")

    def onStarted(self, *args) -> None:
        log.debug("Watchdog started")

    def onConnected(self, *args) -> None:
        log.debug("IB Connected")

    def run(self):
        # this is the main entry point into strategy
        self.watchdog.startedEvent += self._run
        log.debug("initializing watchdog...")
        self.watchdog.start()
        log.debug("watchdog initialized")
        self.ib.run()

    async def _run(self, *args) -> None:
        await CONTROLLER.sync()
        await self.jobs()
