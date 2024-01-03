import pytest

from ib_tools.state_machine import StateMachine


@pytest.fixture
def state_machine():
    print("Getting new state_machine")
    # ensure any existing singleton is destroyed
    if StateMachine._instance:
        StateMachine._instance = None
    return StateMachine()
