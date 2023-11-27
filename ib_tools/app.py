import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, Protocol

import ib_insync as ibi

from ib_tools.blotter import AbstractBaseBlotter
from ib_tools.logging import setup_logging
from ib_tools.manager import CONTROLLER, IB, JOBS, Jobs

log = logging.getLogger(__name__)

ibi.util.patchAsyncio()

setup_logging()


class IBC(Protocol):
    async def startAsync(self):
        ...

    async def terminateAsync(self):
        ...


@dataclass
class FakeIBC:
    restart_time: int = 60

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
    host: str = "127.0.0.1"
    port: float = 4002
    clientId: float = 0
    appStartupTime: float = 0
    appTimeout: float = 20
    retryDelay: float = 2
    probeContract: ibi.Contract = ibi.Forex("EURUSD")
    probeTimeout: float = 4
    blotter: Optional[AbstractBaseBlotter] = None

    def __post_init__(self):
        self.ib.errorEvent += self.onError
        self.ib.connectedEvent += self.onConnected

        if self.blotter:
            CONTROLLER.config(blotter=self.blotter)

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
        await CONTROLLER.init()
        await self.jobs()
