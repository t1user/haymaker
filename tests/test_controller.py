import asyncio
import random
from copy import deepcopy
from itertools import count
from types import SimpleNamespace

import ib_insync as ibi
import pytest
from helpers import wait_for_condition

from haymaker.controller import sync_routines
from haymaker.controller.controller import Controller
from haymaker.controller.objects import SyncResult
from haymaker.controller.sync_routines import (
    OrderReconciliationSync,
    OrderSyncStrategy,
    PositionSyncStrategy,
)
from haymaker.state_machine import OrderInfo
from haymaker.trader import Trader

# from haymaker.manager import IB


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


# @pytest.fixture()
# def controller(Controller):
#     return Controller()


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
async def test_StateMachine_linked_to_ib_newOrderEvent(caplog, Atom):
    controller = Controller(Trader(Atom.ib))  # noqa
    Atom.ib.newOrderEvent.emit(ibi.Trade(order=ibi.Order(orderId=123, permId=45678)))
    assert await wait_for_condition(lambda: "123" in caplog.text)


def test_register_position_skips_already_accounted_execution(controller, trade):
    strategy = controller.sm.strategy["coolstrategy"]
    controller.sm.order[trade.order.orderId] = OrderInfo(
        "coolstrategy", "OPEN", trade, {}
    )

    controller.register_position(strategy, trade, trade.fills[0])
    controller.register_position(strategy, trade, trade.fills[0])

    assert strategy.position == 1
    assert controller.sm.order[trade.order.orderId].accounted_exec_ids == [
        trade.fills[0].execution.execId
    ]


@pytest.mark.asyncio
async def test_sync_timeout_disables_trading(controller, monkeypatch):
    async def pending_positions():
        await asyncio.sleep(1)
        return []

    controller.broker_request_timeout = 0.01
    monkeypatch.setattr(controller.ib, "isConnected", lambda: True)
    monkeypatch.setattr(controller.ib, "positions", lambda: [])
    monkeypatch.setattr(controller.ib, "reqPositionsAsync", pending_positions)

    result = await controller.sync()

    assert not result.ok
    assert controller._trading_disabled


@pytest.mark.asyncio
async def test_broker_position_source_disagreement_disables_trading(
    controller, trade, monkeypatch
):
    position = ibi.Position(
        account="DU123",
        contract=trade.contract,
        position=1,
        avgCost=1,
    )

    async def requested_positions():
        return []

    monkeypatch.setattr(controller.ib, "positions", lambda: [position])
    monkeypatch.setattr(controller.ib, "reqPositionsAsync", requested_positions)

    result = await controller.sync()

    assert not result.ok
    assert controller._trading_disabled


def test_disabled_trading_does_not_register_order(controller, trade, monkeypatch):
    called = False

    def fake_trade(contract, order):
        nonlocal called
        called = True
        return trade

    controller.disable_trading("test")
    monkeypatch.setattr(controller.trader, "trade", fake_trade)

    result = controller.trade(
        "coolstrategy",
        trade.contract,
        ibi.MarketOrder("BUY", 1),
        "OPEN",
        {},
    )

    assert result is None
    assert not called
    assert list(controller.sm.order.values()) == []


def test_unknown_broker_orders_are_cancelled_without_disabling_trading(
    controller, trade, monkeypatch
):
    cancelled = []

    def fake_cancel(received_trade):
        cancelled.append(received_trade)

    monkeypatch.setattr(sync_routines, "CANCEL_UNKNOWN_TRADES", True)
    monkeypatch.setattr(controller, "cancel", fake_cancel)

    controller.order_sync_handlers.handle_unknown_broker_orders([trade])

    assert cancelled == [trade]
    assert not controller._trading_disabled


def test_unknown_broker_orders_can_be_left_active_by_config(
    controller, trade, monkeypatch
):
    cancelled = []

    def fake_cancel(received_trade):
        cancelled.append(received_trade)

    monkeypatch.setattr(sync_routines, "CANCEL_UNKNOWN_TRADES", False)
    monkeypatch.setattr(controller, "cancel", fake_cancel)

    controller.order_sync_handlers.handle_unknown_broker_orders([trade])

    assert cancelled == []
    assert not controller._trading_disabled


@pytest.mark.asyncio
async def test_unknown_broker_orders_skip_correction_trades(
    controller, trade, monkeypatch
):
    report = OrderSyncStrategy(controller.ib, controller.sm)
    report.unknown.append(trade)
    reconciled = []

    async def requested_positions():
        return []

    def fake_order_sync_run(cls, ib, sm):
        return report

    def fake_reconciliation_run(cls, ct):
        reconciled.append(ct)

    monkeypatch.setattr(controller.ib, "isConnected", lambda: True)
    monkeypatch.setattr(controller.ib, "positions", lambda: [])
    monkeypatch.setattr(controller.ib, "reqPositionsAsync", requested_positions)
    monkeypatch.setattr(OrderSyncStrategy, "run", classmethod(fake_order_sync_run))
    monkeypatch.setattr(
        PositionSyncStrategy,
        "run",
        classmethod(lambda cls, ib, sm: SimpleNamespace(errors={})),
    )
    monkeypatch.setattr(
        OrderReconciliationSync, "run", classmethod(fake_reconciliation_run)
    )
    monkeypatch.setattr(sync_routines, "CANCEL_UNKNOWN_TRADES", False)

    result = await controller.sync()

    assert result == SyncResult(True, "recovery completed")
    assert reconciled == []


def test_emergency_close_blocked_when_broker_position_protected(
    controller, trade, monkeypatch
):
    strategy = controller.sm.strategy["coolstrategy"]
    strategy.position = 1
    strategy.active_contract = trade.contract

    protective_trade = deepcopy(trade)
    protective_trade.order = ibi.Order(
        action="SELL", totalQuantity=1, orderType="TRAIL"
    )
    protective_trade.orderStatus = ibi.OrderStatus(status="Submitted")

    monkeypatch.setattr(controller.trader, "position_for_contract", lambda contract: 1)
    monkeypatch.setattr(
        controller.trader, "trades_for_contract", lambda contract: [protective_trade]
    )

    assert not controller.can_emergency_close_strategy(strategy)


# def test_StateMachine_lined_to_ib_orderStatusEvent(caplog):
#     """TODO: This doesnt test anythin yet."""
#     IB.orderStatusEvent.emit(456)
