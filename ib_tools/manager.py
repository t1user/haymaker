from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final, Optional, Protocol, cast

import ib_insync as ibi

from ib_tools import misc
from ib_tools.base import Atom
from ib_tools.blotter import AbstractBaseBlotter
from ib_tools.controller import Controller
from ib_tools.logging import setup_logging_queue

# from ib_tools.runner import App
from ib_tools.state_machine import StateMachine
from ib_tools.streamers import Streamer

log = logging.getLogger(__name__)

ibi.util.patchAsyncio()

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


@dataclass
class InitData:
    ib: ibi.IB
    contract_list: list[ibi.Contract]
    contract_details: dict[ibi.Contract, ibi.ContractDetails] = field(
        default_factory=dict
    )
    trading_hours: dict[ibi.Contract, list[tuple[datetime, datetime]]] = field(
        default_factory=dict
    )

    async def __call__(self) -> "InitData":
        await self.qualify_contracts()
        await self.acquire_contract_details()
        self.process_trading_hours()
        return self

    async def qualify_contracts(self) -> "InitData":
        await self.ib.qualifyContractsAsync(*self.contract_list)
        log.debug(f"contracts qualified {set([c.symbol for c in self.contract_list])}")
        return self

    async def acquire_contract_details(self) -> "InitData":
        for contract in set(self.contract_list):
            log.debug(f"Acquiring details for: {contract.symbol}")
            details_ = await IB.reqContractDetailsAsync(contract)
            try:
                assert len(details_) == 1
            except AssertionError:
                log.exception(f"Ambiguous contract: {contract}. Critical error.")

            details = details_[0]
            self.contract_details[cast(ibi.Contract, details.contract)] = details
        log.debug(
            f"Details acquired: {set([k.symbol for k in self.contract_details.keys()])}"
        )
        return self

    _process_trading_hours = staticmethod(misc.process_trading_hours)

    def process_trading_hours(self) -> "InitData":
        for contract, details in self.contract_details.items():
            self.trading_hours[contract] = self._process_trading_hours(
                details.tradingHours, details.timeZoneId
            )
        log.debug(
            f"Trading hours processed for: "
            f"{[c.symbol for c in self.trading_hours.keys()]}"
        )
        return self


class Jobs:
    _tasks: set = set()

    def __init__(self, init_data: InitData):
        self.init_data = init_data
        self.streamers = Streamer.instances

    def _handle_error(self, task: asyncio.Task):
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.exception(e)

    async def __call__(self):
        await self.init_data()

        log.info(
            f"Open positions on restart: "
            f"{ {p.contract.symbol: p.position for p in IB.positions()} }"
        )
        order_dict = {
            t.contract.symbol: (
                t.order.orderId,
                t.order.orderType,
                t.order.action,
                t.order.totalQuantity,
            )
            for t in IB.openTrades()
        }
        log.info(f"Orders on restart: {order_dict}")

        for streamer in self.streamers:
            task = asyncio.create_task(streamer.run(), name=f"{streamer!s}, ")
            log.debug(f"Task created: {task}")

            # Add task to the set. This creates a strong reference.
            self._tasks.add(task)

            # To prevent keeping references to finished tasks forever,
            # make each task remove its own reference from the set after
            # completion:
            task.add_done_callback(self._tasks.discard)
            # ensure errors are logged for debugging
            task.add_done_callback(self._handle_error)
        await asyncio.gather(*self._tasks, return_exceptions=False)


IB: Final[ibi.IB] = ibi.IB()
STATE_MACHINE: Final[StateMachine] = StateMachine()
CONTROLLER: Final[Controller] = Controller(STATE_MACHINE, IB)
INIT_DATA = InitData(IB, Atom.contracts)
JOBS = Jobs(INIT_DATA)
Atom.set_init_data(INIT_DATA)


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
