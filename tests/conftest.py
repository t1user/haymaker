from types import SimpleNamespace

import ib_insync as ibi
import pytest

from ib_tools.base import Atom
from ib_tools.state_machine import StateMachine


@pytest.fixture
def state_machine():
    # ensure any existing singleton is destroyed
    # mere module imports will create an instance
    # so using yield and subsequent tear-down
    # will not work
    if StateMachine._instance:
        StateMachine._instance = None
    return StateMachine()


@pytest.fixture()
def atom(state_machine):
    data = SimpleNamespace()
    data.ib = ibi.IB()

    data.trading_hours = {}
    data.contract_details = {}
    sm = state_machine
    Atom.set_init_data(data, sm)
    return Atom
