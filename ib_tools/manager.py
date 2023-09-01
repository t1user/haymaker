import logging
from typing import Final

import ib_insync as ibi

from ib_tools.base import Atom
from ib_tools.controller import Controller
from ib_tools.state_machine import StateMachine

log = logging.getLogger(__name__)


IB: Final[ibi.IB] = ibi.IB()
Atom.set_ib(IB)
STATE_MACHINE: Final[StateMachine] = StateMachine()
CONTROLLER: Final[Controller] = Controller(STATE_MACHINE)
