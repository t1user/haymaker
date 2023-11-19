import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, Protocol

import ib_insync as ibi

from ib_tools.blotter import AbstractBaseBlotter
from ib_tools.logging import setup_logging_queue
from ib_tools.manager import CONTROLLER, IB, JOBS, Jobs

ibi.util.patchAsyncio()

log = logging.getLogger(__name__)

# ------------ logging setup ---------------
level = 5
logging.addLevelName(5, "DATA")
logging.addLevelName(60, "NOTIFY")

logger = logging.getLogger("ib_tools")
logger.setLevel(level)


formatter = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(name)-23s | %(message)s"
    " | %(module)s %(funcName)s %(lineno)d"
)

rfh = logging.handlers.TimedRotatingFileHandler(
    "/home/tomek/ib_data/test_logs/dummy.log", when="D"
)
rfh.setLevel(level)
rfh.setFormatter(formatter)

sh = logging.StreamHandler()
sh.setLevel(level)
sh.setFormatter(formatter)

logger.addHandler(rfh)
logger.addHandler(sh)

# logging.basicConfig(
#     format="%(asctime)s | %(levelname)-8s | %(name)-23s | %(message)s"
#     " | %(module)s %(funcName)s %(lineno)d",
#     level=5,
# )

# shut up foreign loggers
# logging.getLogger("ib_insync").setLevel(logging.ERROR)

# stream_handler = logging.StreamHandler()
# stream_handler.setLevel(logging.INFO)
# watchdog_logger = logging.getLogger("ib_insync.Watchdog")
# watchdog_logger.setLevel(logging.INFO)
# watchdog_logger.addHandler(stream_handler)

# logging.getLogger("asyncio").setLevel(logging.INFO)
# logging.getLogger("numba").setLevel(logging.CRITICAL)

setup_logging_queue()
# ------------ end logging setup ---------------


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

    def onError(
        self, reqId: int, errorCode: int, errorString: str, contract: ibi.Contract
    ) -> None:
        log.debug(f"Error event: {reqId} {errorCode} {errorString} {contract}")

    def handle_soft_timeout(self, watchdog: ibi.Watchdog):
        log.debug("Soft timeout event.")

    def handle_hard_timeout(self, watchdog: ibi.Watchdog):
        log.debug("Hard timeout event.")

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
