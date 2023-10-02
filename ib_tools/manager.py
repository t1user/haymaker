from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import ib_insync as ibi

from ib_tools.base import CONTRACT_LIST, Atom, Pipe
from ib_tools.controller import Controller
from ib_tools.state_machine import StateMachine

log = logging.getLogger(__name__)

# ### Singletons ####
IB: Final[ibi.IB] = ibi.IB()
STATE_MACHINE: Final[StateMachine] = StateMachine()
CONTROLLER: Final[Controller] = Controller(STATE_MACHINE, IB)
CONTRACT_DETAILS: dict[ibi.Contract, ibi.ContractDetails] = {}
# ###################

Atom.set_ib(IB)

logging.basicConfig(level=logging.DEBUG)


@dataclass
class Manager:
    pipe: Pipe

    def __post_init__(self):
        pass
