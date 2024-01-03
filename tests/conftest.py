import pytest

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
