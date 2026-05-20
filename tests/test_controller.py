import asyncio
import datetime as dt
import random
from copy import deepcopy
from itertools import count

import ib_insync as ibi
import pytest
from helpers import wait_for_condition

from haymaker.controller.controller import Controller, ControllerError
from haymaker.controller.sync_actions import (
    BracketActionExecutor,
    EmergencyCloseGuard,
    OrderSyncApplier,
)
from haymaker.controller.sync_types import (
    BracketFindings,
    BracketIssue,
    BrokerSnapshot,
    OrderFindings,
    SyncResult,
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


def make_broker_snapshot(
    positions: tuple[ibi.Position, ...] = (),
    open_trades: tuple[ibi.Trade, ...] = (),
    trades: tuple[ibi.Trade, ...] = (),
    fills: tuple[ibi.Fill, ...] = (),
) -> BrokerSnapshot:
    """Create a broker snapshot for focused controller tests."""
    return BrokerSnapshot(
        positions=positions,
        open_trades=open_trades,
        trades=trades,
        fills=fills,
        captured_at=dt.datetime.now(tz=dt.timezone.utc),
    )


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


# def test_positions_and_stop_losses_stop_wrong_amount(
#     trades_and_positions, controller
# ):
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
async def test_sync_disconnected_does_not_capture_broker_state(controller, monkeypatch):
    def fail_position_read():
        raise AssertionError("broker snapshot should not be captured")

    async def fail_requested_positions():
        raise AssertionError("broker snapshot should not be captured")

    monkeypatch.setattr(controller.ib, "isConnected", lambda: False)
    monkeypatch.setattr(controller.ib, "positions", fail_position_read)
    monkeypatch.setattr(controller.ib, "reqPositionsAsync", fail_requested_positions)

    result = await controller.sync()

    assert result == SyncResult(False, "broker not connected")
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

    monkeypatch.setattr(controller, "cancel", fake_cancel)

    OrderSyncApplier(
        controller.ib,
        controller.sm,
        controller,
        True,
    ).handle_unknown_broker_orders([trade])

    assert cancelled == [trade]
    assert not controller._trading_disabled


def test_from_config_loads_controller_sync_options(Atom):
    controller = Controller.from_config(
        Trader(Atom.ib),
        top_config={
            "controller": {
                "broker_request_timeout": 3,
                "cancel_unknown_trades": True,
                "cancel_stray_orders": True,
                "handle_missing_brackets": "warn",
            },
        },
    )

    assert controller.broker_request_timeout == 3
    assert controller.cancel_unknown_trades
    assert controller.cancel_stray_orders
    assert controller.handle_missing_brackets == "warn"


def test_from_config_rejects_unknown_controller_config(Atom):
    with pytest.raises(ControllerError, match="Wrong parameter: invalid"):
        Controller.from_config(
            Trader(Atom.ib),
            top_config={"controller": {"invalid": True}},
        )


def test_unknown_broker_orders_can_be_left_active_by_config(
    controller, trade, monkeypatch
):
    cancelled = []

    def fake_cancel(received_trade):
        cancelled.append(received_trade)

    monkeypatch.setattr(controller, "cancel", fake_cancel)

    OrderSyncApplier(
        controller.ib,
        controller.sm,
        controller,
        False,
    ).handle_unknown_broker_orders([trade])

    assert cancelled == []
    assert not controller._trading_disabled


@pytest.mark.asyncio
async def test_order_sync_applier_returns_fresh_actions_per_report(controller, trade):
    applier = OrderSyncApplier(controller.ib, controller.sm, controller, False)
    controller.sm.order[trade.order.orderId] = OrderInfo(
        "coolstrategy", "OPEN", trade, {}
    )

    order_info = controller.sm.order[trade.order.orderId]
    first_actions = await applier.apply(
        OrderFindings(missing_active_orders=(order_info,)),
        make_broker_snapshot(),
    )

    second_actions = await applier.apply(OrderFindings(), make_broker_snapshot())

    assert [oi.trade for oi in first_actions.faulty_orders] == [trade]
    assert second_actions.faulty_orders == []


@pytest.mark.asyncio
async def test_unknown_broker_orders_skip_correction_trades(
    controller, trade, monkeypatch
):
    reconciled = []

    async def requested_positions():
        return []

    def fake_bracket_apply(self, findings):
        reconciled.append(findings)

    monkeypatch.setattr(controller.ib, "isConnected", lambda: True)
    monkeypatch.setattr(controller.ib, "positions", lambda: [])
    monkeypatch.setattr(controller.ib, "reqPositionsAsync", requested_positions)
    monkeypatch.setattr(controller.ib, "openTrades", lambda: [trade])
    monkeypatch.setattr(controller.ib, "trades", lambda: [])
    monkeypatch.setattr(controller.ib, "fills", lambda: [])
    monkeypatch.setattr(BracketActionExecutor, "apply", fake_bracket_apply)
    controller.cancel_unknown_trades = False

    result = await controller.sync()

    assert result == SyncResult(True, "recovery completed")
    assert reconciled == []


@pytest.mark.asyncio
async def test_open_trade_refresh_does_not_skip_correction_trades(
    controller, trade, monkeypatch
):
    old_trade = deepcopy(trade)
    old_trade.orderStatus = ibi.OrderStatus(status="Submitted", filled=0, remaining=1)
    old_trade.fills = []
    broker_trade = deepcopy(old_trade)
    controller.sm.order[old_trade.order.orderId] = OrderInfo(
        "coolstrategy", "OPEN", old_trade, {}
    )
    reconciled = []

    async def requested_positions():
        return []

    def fake_bracket_apply(self, findings):
        reconciled.append(findings)

    monkeypatch.setattr(controller.ib, "isConnected", lambda: True)
    monkeypatch.setattr(controller.ib, "positions", lambda: [])
    monkeypatch.setattr(controller.ib, "reqPositionsAsync", requested_positions)
    monkeypatch.setattr(controller.ib, "openTrades", lambda: [broker_trade])
    monkeypatch.setattr(controller.ib, "trades", lambda: [])
    monkeypatch.setattr(controller.ib, "fills", lambda: [])
    monkeypatch.setattr(BracketActionExecutor, "apply", fake_bracket_apply)

    result = await controller.sync()

    assert result == SyncResult(True)
    assert reconciled


def test_emergency_close_blocked_when_broker_position_protected(controller, trade):
    strategy = controller.sm.strategy["coolstrategy"]
    strategy.position = 1
    strategy.active_contract = trade.contract

    protective_trade = deepcopy(trade)
    protective_trade.order = ibi.Order(
        action="SELL", totalQuantity=1, orderType="TRAIL"
    )
    protective_trade.orderStatus = ibi.OrderStatus(status="Submitted")
    broker_position = ibi.Position(
        account="DU123",
        contract=trade.contract,
        position=1,
        avgCost=1,
    )
    snapshot = make_broker_snapshot(
        positions=(broker_position,),
        open_trades=(protective_trade,),
    )

    assert not EmergencyCloseGuard(controller.sm, snapshot, False).can_close(strategy)


def test_bracket_emergency_close_uses_broker_snapshot(controller, trade, monkeypatch):
    strategy = controller.sm.strategy["coolstrategy"]
    strategy.position = 1
    strategy.active_contract = trade.contract
    strategy.params = {"stop-loss": {"amount": 1}}

    broker_position = ibi.Position(
        account="DU123",
        contract=trade.contract,
        position=1,
        avgCost=1,
    )
    protective_trade = deepcopy(trade)
    protective_trade.order = ibi.Order(
        action="SELL", totalQuantity=1, orderType="TRAIL"
    )
    protective_trade.orderStatus = ibi.OrderStatus(status="Submitted")
    snapshot = make_broker_snapshot(
        positions=(broker_position,),
        open_trades=(protective_trade,),
    )
    closed = []

    def fail_live_position_read(contract):
        raise AssertionError("live broker position should not be read")

    monkeypatch.setattr(
        controller.trader, "position_for_contract", fail_live_position_read
    )
    monkeypatch.setattr(
        controller,
        "close_positions_for_strategy",
        lambda strategy_name, action: closed.append((strategy_name, action)),
    )

    BracketActionExecutor(controller, snapshot, False, "remove").apply(
        BracketFindings(
            missing_brackets=(BracketIssue(strategy, (), 1),),
        )
    )

    assert closed == []


# def test_StateMachine_lined_to_ib_orderStatusEvent(caplog):
#     """TODO: This doesnt test anythin yet."""
#     IB.orderStatusEvent.emit(456)
