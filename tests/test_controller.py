import random
from itertools import count

import ib_insync as ibi
import pytest

from ib_tools.controller import Controller


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


def test_positions_and_stop_losses_no_diff(trades_and_positions):
    trades, positions = trades_and_positions
    controller = Controller("sm", "ib")
    diff = controller.positions_and_stop_losses(trades, positions)
    assert diff == {}


def test_positions_and_stop_losses_missing_stop(trades_and_positions):
    trades, positions = trades_and_positions
    missing_trade = trades.pop()
    controller = Controller("sm", "ib")
    diff = controller.positions_and_stop_losses(trades, positions)
    assert diff == {missing_trade.contract: (1, 0)}


def test_positions_and_stop_losses_redundant_stop(trades_and_positions):
    trades, positions = trades_and_positions
    missing_position = positions.pop()
    controller = Controller("sm", "ib")
    diff = controller.positions_and_stop_losses(trades, positions)
    assert diff == {missing_position.contract: (0, 1)}


def test_positions_and_stop_losses_stop_wrong_side(trades_and_positions):
    trades, positions = trades_and_positions
    wrong_trade = trades[-1]
    wrong_trade.order.action = "BUY"
    controller = Controller("sm", "ib")
    diff = controller.positions_and_stop_losses(trades, positions)
    assert diff == {wrong_trade.contract: (1, -1)}


def test_positions_and_stop_losses_stop_wrong_amount(trades_and_positions):
    trades, positions = trades_and_positions
    wrong_trade = trades[-1]
    wrong_trade.order.totalQuantity = 2
    controller = Controller("sm", "ib")
    diff = controller.positions_and_stop_losses(trades, positions)
    assert diff == {wrong_trade.contract: (1, 2)}


def test_positions_and_stop_losses_stop_wrong_side_and_amount(trades_and_positions):
    trades, positions = trades_and_positions
    wrong_trade = trades[-1]
    wrong_trade.order.action = "BUY"
    wrong_trade.order.totalQuantity = 2
    controller = Controller("sm", "ib")
    diff = controller.positions_and_stop_losses(trades, positions)
    assert diff == {wrong_trade.contract: (1, -2)}


def test_check_for_orphan_trades_no_orphan(trades_and_positions):
    trades, positions = trades_and_positions
    controller = Controller("sm", "ib")
    orphan_trades = controller.check_for_orphan_trades(trades, positions)
    assert orphan_trades == []


def test_check_for_orphan_trades(trades_and_positions):
    trades, positions = trades_and_positions
    positions.pop()
    controller = Controller("sm", "ib")
    orphan_trades = controller.check_for_orphan_trades(trades, positions)
    assert orphan_trades == [trades[-1]]


def test_check_for_orphan_positions_no_orphan(trades_and_positions):
    trades, positions = trades_and_positions
    controller = Controller("sm", "ib")
    orphan_positions = controller.check_for_orphan_positions(trades, positions)
    assert orphan_positions == []


def test_check_for_orphan_positions(trades_and_positions):
    trades, positions = trades_and_positions
    trades.pop()
    controller = Controller("sm", "ib")
    orphan_positions = controller.check_for_orphan_positions(trades, positions)
    assert orphan_positions == [positions[-1]]
