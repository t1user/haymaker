import asyncio

import ib_insync as ibi
import pytest

from ib_tools.manager import IB, STATE_MACHINE
from ib_tools.state_machine import StateMachine


def test_StateMachine_is_singleton():
    """
    Atempt to instantiate :class:`StateMachine` more than once
    should raise `TypeError`
    """
    with pytest.raises(TypeError):
        StateMachine()


@pytest.mark.asyncio
async def test_StateMachine_linked_to_ib_newOrderEvent(caplog):
    IB.newOrderEvent.emit(ibi.Trade(order=ibi.Order(orderId=123)))
    await asyncio.sleep(0.2)
    assert "123" in caplog.text


def test_StateMachine_lined_to_ib_orderStatusEvent(caplog):
    """TODO: This doesnt test anythin yet."""
    IB.orderStatusEvent.emit(456)
