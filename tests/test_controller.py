import asyncio
import random
from itertools import count

import ib_insync as ibi
import pytest

from ib_tools.controller import Controller
from ib_tools.manager import IB


@pytest.fixture
def trades_and_positions():
    conId = count(1).__next__
    contract_str = ["NQ", "ES", "YM", "RTY"]
    contracts = [ibi.Contract(c, conId=conId()) for c in contract_str]
    positions = [
        ibi.Position(
            account="123",
            contract=c,
            position=1,
            avgCost=random.randrange(10000, 11000) / 10,
        )
        for c in contracts
    ]
    trades = [ibi.Trade(c, ibi.StopOrder("SELL", 1, stopPrice=5)) for c in contracts]
    return trades, positions


@pytest.fixture()
def controller():
    return Controller()


# def test_positions_and_stop_losses_no_diff(trades_and_positions, controller):
#     trades, positions = trades_and_positions
#     diff = controller.positions_and_stop_losses(trades, positions)
#     assert diff == {}


# def test_positions_and_stop_losses_missing_stop(trades_and_positions, controller):
#     trades, positions = trades_and_positions
#     missing_trade = trades.pop()
#     diff = controller.positions_and_stop_losses(trades, positions)
#     assert diff == {missing_trade.contract: (1, 0)}


# def test_positions_and_stop_losses_redundant_stop(trades_and_positions, controller):
#     trades, positions = trades_and_positions
#     missing_position = positions.pop()
#     diff = controller.positions_and_stop_losses(trades, positions)
#     assert diff == {missing_position.contract: (0, 1)}


# def test_positions_and_stop_losses_stop_wrong_side(trades_and_positions, controller):
#     trades, positions = trades_and_positions
#     wrong_trade = trades[-1]
#     wrong_trade.order.action = "BUY"
#     diff = controller.positions_and_stop_losses(trades, positions)
#     assert diff == {wrong_trade.contract: (1, -1)}


# def test_positions_and_stop_losses_stop_wrong_amount(trades_and_positions, controller):
#     trades, positions = trades_and_positions
#     wrong_trade = trades[-1]
#     wrong_trade.order.totalQuantity = 2
#     diff = controller.positions_and_stop_losses(trades, positions)
#     assert diff == {wrong_trade.contract: (1, 2)}


# def test_positions_and_stop_losses_stop_wrong_side_and_amount(
#     trades_and_positions, controller
# ):
#     trades, positions = trades_and_positions
#     wrong_trade = trades[-1]
#     wrong_trade.order.action = "BUY"
#     wrong_trade.order.totalQuantity = 2
#     diff = controller.positions_and_stop_losses(trades, positions)
#     assert diff == {wrong_trade.contract: (1, -2)}


# def test_check_for_orphan_trades_no_orphan(trades_and_positions, controller):
#     trades, positions = trades_and_positions
#     orphan_trades = controller.check_for_orphan_trades(trades, positions)
#     assert orphan_trades == []


# def test_check_for_orphan_trades(trades_and_positions, controller):
#     trades, positions = trades_and_positions
#     positions.pop()
#     orphan_trades = controller.check_for_orphan_trades(trades, positions)
#     assert orphan_trades == [trades[-1]]


# def test_check_for_orphan_positions_no_orphan(trades_and_positions, controller):
#     trades, positions = trades_and_positions
#     orphan_positions = controller.check_for_orphan_positions(trades, positions)
#     assert orphan_positions == []


# def test_check_for_orphan_positions(trades_and_positions, controller):
#     trades, positions = trades_and_positions
#     trades.pop()
#     orphan_positions = controller.check_for_orphan_positions(trades, positions)
#     assert orphan_positions == [positions[-1]]


@pytest.mark.asyncio
async def test_StateMachine_linked_to_ib_newOrderEvent(caplog):
    IB.newOrderEvent.emit(ibi.Trade(order=ibi.Order(orderId=123)))
    await asyncio.sleep(0.2)
    assert "123" in caplog.text


def test_StateMachine_lined_to_ib_orderStatusEvent(caplog):
    """TODO: This doesnt test anythin yet."""
    IB.orderStatusEvent.emit(456)
