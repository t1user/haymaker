from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Final

import ib_insync as ibi

from ib_tools import misc
from ib_tools.base import CONTRACT_LIST, Atom, Pipe
from ib_tools.controller import Controller
from ib_tools.logging import create_task, setup_logging_queue
from ib_tools.runner import App
from ib_tools.state_machine import StateMachine
from ib_tools.streamers import Streamer

log = logging.getLogger(__name__)

print("About to setup logger...")
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s: | %(name)20s | %(message)s"
    " | %(module)s %(funcName)s %(lineno)d",
    level=5,
)
print("...logger set up")

logging.addLevelName(5, "DATA")
logging.addLevelName(60, "NOTIFY")

# shut up foreign loggers
logging.getLogger("ib_insync").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.INFO)
logging.getLogger("numba").setLevel(logging.CRITICAL)

setup_logging_queue()

IB: Final[ibi.IB] = ibi.IB()
Atom.set_ib(IB)
STATE_MACHINE: Final[StateMachine] = StateMachine()
CONTROLLER: Final[Controller] = Controller(STATE_MACHINE, IB)
CONTRACT_DETAILS: dict[ibi.Contract, ibi.ContractDetails] = {}
TRADING_HOURS: dict[ibi.Contract, list[tuple[datetime, datetime]]]


@dataclass
class Manager:
    pipe: Pipe

    def __post_init__(self):
        pass


def acquire_contract_details():
    global CONTRACT_DETAILS
    for contract in set(CONTRACT_LIST):
        details = IB.reqContractDetails(contract)
        assert len(details) == 1, f"Ambiguous contract: {contract}. Critical error."
        details = details[0]
        try:
            CONTRACT_DETAILS[details.contract] = details
        except Exception:
            log.exception()
            raise


def set_trading_hours() -> None:
    global TRADING_HOURS
    TRADING_HOURS = dict()
    for contract, details in CONTRACT_DETAILS.items():
        TRADING_HOURS[contract] = misc.process_trading_hours(details.tradingHours)


async def prepare():
    await IB.qualifyContractsAsync(*CONTRACT_LIST)
    log.debug(f"contracts qualified {set([c.symbol for c in CONTRACT_LIST])}")
    acquire_contract_details()
    # log.debug(f"Details acquired: {list(CONTRACT_DETAILS.keys())}")
    log.debug(f"Details aquired: {set([k.symbol for k in CONTRACT_DETAILS.keys()])}")
    set_trading_hours()
    Atom.trading_hours = TRADING_HOURS

    await asyncio.gather(
        *[
            create_task(s, logger=log, message="asyncio error")
            for s in Streamer.awaitables()
        ],
        return_exceptions=True,
    )


log.debug("Will instantiate the App...")
app = App(IB, prepare)
