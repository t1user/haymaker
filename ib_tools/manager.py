from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Final

import ib_insync as ibi

from ib_tools.base import CONTRACT_LIST, Atom, Pipe
from ib_tools.controller import Controller
from ib_tools.runner import App
from ib_tools.state_machine import StateMachine
from ib_tools.streamers import Streamer

log = logging.getLogger(__name__)

print("About to setup logger...")
logging.basicConfig(
    format="[%(asctime)s] %(levelname)8s: %(name)20s | %(message)s"
    " | %(module)s %(funcName)s %(lineno)d",
    level=logging.DEBUG,
)
print("...logger set up")

ib_log = logging.getLogger("ib_insync")
ib_log.setLevel(logging.CRITICAL)
asyncio_log = logging.getLogger("asyncio")
asyncio_log.setLevel(logging.DEBUG)


IB: Final[ibi.IB] = ibi.IB()
Atom.set_ib(IB)
STATE_MACHINE: Final[StateMachine] = StateMachine()
CONTROLLER: Final[Controller] = Controller(STATE_MACHINE, IB)
CONTRACT_DETAILS: dict[ibi.Contract, ibi.ContractDetails] = {}


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


async def prepare():
    await IB.qualifyContractsAsync(*CONTRACT_LIST)
    log.debug(f"contracts qualified {CONTRACT_LIST}")
    acquire_contract_details()
    log.debug(f"Details aquired: {CONTRACT_DETAILS}")
    await asyncio.gather(*Streamer.awaitables(), return_exceptions=True)


log.debug("Will instantiate the App...")
app = App(IB, prepare)
