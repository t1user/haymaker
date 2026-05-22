import asyncio
import random
from copy import deepcopy
from itertools import count

import ib_insync as ibi
import pytest
from helpers import wait_for_condition

from haymaker.controller.controller import Controller, ControllerError
from haymaker.controller.sync_brackets import BracketSyncResult, BracketSyncer
from haymaker.controller.sync_coordinator import (
    OrderFindings,
    OrderRecoveryResult,
    OrderSyncApplier,
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


def set_broker_state(
    controller: Controller,
    monkeypatch: pytest.MonkeyPatch,
    positions: tuple[ibi.Position, ...] = (),
    open_trades: tuple[ibi.Trade, ...] = (),
    trades: tuple[ibi.Trade, ...] = (),
    fills: tuple[ibi.Fill, ...] = (),
) -> None:
    """Set direct broker query results for focused controller tests."""
    monkeypatch.setattr(controller.ib, "positions", lambda: list(positions))
    monkeypatch.setattr(controller.ib, "openTrades", lambda: list(open_trades))
    monkeypatch.setattr(controller.ib, "trades", lambda: list(trades))
    monkeypatch.setattr(controller.ib, "fills", lambda: list(fills))


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

    assert not result
    assert controller._trading_disabled


@pytest.mark.asyncio
async def test_sync_disconnected_does_not_query_broker_state(controller, monkeypatch):
    def fail_position_read():
        raise AssertionError("broker state should not be queried")

    async def fail_requested_positions():
        raise AssertionError("broker state should not be queried")

    monkeypatch.setattr(controller.ib, "isConnected", lambda: False)
    monkeypatch.setattr(controller.ib, "positions", fail_position_read)
    monkeypatch.setattr(controller.ib, "reqPositionsAsync", fail_requested_positions)

    result = await controller.sync()

    assert not result
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

    assert not result
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

    result = OrderRecoveryResult(unknown_broker_trades=[trade])

    OrderSyncApplier(
        controller.ib,
        controller.sm,
        controller,
        True,
    ).handle_unknown_broker_orders(result)

    assert cancelled == [trade]
    assert result.cancelled_unknown_trades == [trade]
    assert not controller._trading_disabled


def test_from_config_loads_controller_sync_options(Atom):
    controller = Controller.from_config(
        Trader(Atom.ib),
        top_config={
            "controller": {
                "broker_request_timeout": 3,
                "sync_resync_delay": 0,
                "cancel_unknown_trades": True,
                "missing_brackets": "warn",
            },
        },
    )

    assert controller.broker_request_timeout == 3
    assert controller.sync_resync_delay == 0
    assert controller.cancel_unknown_trades
    assert controller.missing_brackets == "warn"


def test_from_config_rejects_unknown_controller_config(Atom):
    with pytest.raises(ControllerError, match="Wrong parameter: invalid"):
        Controller.from_config(
            Trader(Atom.ib),
            top_config={"controller": {"invalid": True}},
        )


def test_from_config_rejects_invalid_missing_brackets_value(Atom):
    with pytest.raises(ControllerError, match="missing_brackets"):
        Controller.from_config(
            Trader(Atom.ib),
            top_config={"controller": {"missing_brackets": "close"}},
        )


def test_unknown_broker_orders_can_be_left_active_by_config(
    controller, trade, monkeypatch
):
    cancelled = []

    def fake_cancel(received_trade):
        cancelled.append(received_trade)

    monkeypatch.setattr(controller, "cancel", fake_cancel)

    result = OrderRecoveryResult(unknown_broker_trades=[trade])

    OrderSyncApplier(
        controller.ib,
        controller.sm,
        controller,
        False,
    ).handle_unknown_broker_orders(result)

    assert cancelled == []
    assert result.cancelled_unknown_trades == []
    assert not controller._trading_disabled


@pytest.mark.asyncio
async def test_order_sync_applier_returns_fresh_actions_per_report(
    controller, trade, monkeypatch
):
    applier = OrderSyncApplier(controller.ib, controller.sm, controller, False)
    controller.sm.order[trade.order.orderId] = OrderInfo(
        "coolstrategy", "OPEN", trade, {}
    )
    set_broker_state(controller, monkeypatch)

    order_info = controller.sm.order[trade.order.orderId]
    first_actions = await applier.apply(
        OrderFindings(missing_active_orders=(order_info,)),
    )

    second_actions = await applier.apply(OrderFindings())

    assert [oi.trade for oi in first_actions.faulty_orders] == [trade]
    assert second_actions.faulty_orders == []


@pytest.mark.asyncio
async def test_unknown_broker_orders_skip_correction_trades(
    controller, trade, monkeypatch
):
    reconciled = []

    async def requested_positions():
        return []

    def fail_bracket_run(self):
        raise AssertionError("bracket sync should not run with unknown broker orders")

    monkeypatch.setattr(controller.ib, "isConnected", lambda: True)
    monkeypatch.setattr(controller.ib, "positions", lambda: [])
    monkeypatch.setattr(controller.ib, "reqPositionsAsync", requested_positions)
    monkeypatch.setattr(controller.ib, "openTrades", lambda: [trade])
    monkeypatch.setattr(controller.ib, "trades", lambda: [])
    monkeypatch.setattr(controller.ib, "fills", lambda: [])
    monkeypatch.setattr(BracketSyncer, "run", fail_bracket_run)
    controller.cancel_unknown_trades = False

    result = await controller.sync()

    assert result
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

    def fake_bracket_run(self):
        reconciled.append(self)
        return BracketSyncResult()

    controller.sync_resync_delay = 0
    monkeypatch.setattr(controller.ib, "isConnected", lambda: True)
    monkeypatch.setattr(controller.ib, "positions", lambda: [])
    monkeypatch.setattr(controller.ib, "reqPositionsAsync", requested_positions)
    monkeypatch.setattr(controller.ib, "openTrades", lambda: [broker_trade])
    monkeypatch.setattr(controller.ib, "trades", lambda: [])
    monkeypatch.setattr(controller.ib, "fills", lambda: [])
    monkeypatch.setattr(BracketSyncer, "run", fake_bracket_run)

    result = await controller.sync()

    assert result
    assert reconciled


def test_bracket_sync_does_not_close_protected_broker_position(
    controller, trade, monkeypatch
):
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
    set_broker_state(
        controller,
        monkeypatch,
        positions=(broker_position,),
        open_trades=(protective_trade,),
    )
    placed_orders = []

    def fail_live_position_read(contract):
        raise AssertionError("live broker position should not be read")

    monkeypatch.setattr(
        controller.trader, "position_for_contract", fail_live_position_read
    )
    monkeypatch.setattr(controller.trader, "trade", lambda contract, order: trade)
    monkeypatch.setattr(
        controller,
        "register_order",
        lambda strategy_name, action, trade, params: placed_orders.append(
            (strategy_name, action, trade, params)
        ),
    )

    result = BracketSyncer(controller, "remove").run()

    assert not result.changed_broker
    assert placed_orders == []


def test_bracket_sync_take_profit_does_not_protect_broker_position(
    controller, trade, monkeypatch
):
    strategy = controller.sm.strategy["coolstrategy"]
    strategy.position = 1
    strategy.active_contract = trade.contract

    broker_position = ibi.Position(
        account="DU123",
        contract=trade.contract,
        position=3,
        avgCost=1,
    )
    take_profit = deepcopy(trade)
    take_profit.order = ibi.Order(action="SELL", totalQuantity=3, orderType="LMT")
    take_profit.orderStatus = ibi.OrderStatus(status="Submitted")
    set_broker_state(
        controller,
        monkeypatch,
        positions=(broker_position,),
        open_trades=(take_profit,),
    )
    placed_orders = []

    monkeypatch.setattr(
        controller,
        "trade",
        lambda strategy_name, contract, order, action, params: placed_orders.append(
            (strategy_name, contract, order, action, params)
        )
        or trade,
    )

    result = BracketSyncer(controller, "remove").run()

    assert result.terminal_action
    assert result.closed_positions
    assert placed_orders[0][2].action == "SELL"
    assert placed_orders[0][2].totalQuantity == 3


def test_bracket_sync_blocks_ambiguous_broker_position_close(
    controller, trade, monkeypatch
):
    first = controller.sm.strategy["first"]
    first.position = 1
    first.active_contract = trade.contract
    second = controller.sm.strategy["second"]
    second.position = -1
    second.active_contract = trade.contract
    broker_position = ibi.Position(
        account="DU123",
        contract=trade.contract,
        position=1,
        avgCost=1,
    )
    set_broker_state(controller, monkeypatch, positions=(broker_position,))

    result = BracketSyncer(controller, "remove").run()

    assert result.blocked_reason
    assert not result.closed_positions


def test_bracket_sync_blocks_when_close_order_is_not_submitted(
    controller, trade, monkeypatch
):
    strategy = controller.sm.strategy["coolstrategy"]
    strategy.position = 1
    strategy.active_contract = trade.contract
    broker_position = ibi.Position(
        account="DU123",
        contract=trade.contract,
        position=1,
        avgCost=1,
    )
    set_broker_state(controller, monkeypatch, positions=(broker_position,))
    monkeypatch.setattr(controller, "trade", lambda *args: None)

    result = BracketSyncer(controller, "remove").run()

    assert result.blocked_reason
    assert not result.closed_positions


# def test_StateMachine_lined_to_ib_orderStatusEvent(caplog):
#     """TODO: This doesnt test anythin yet."""
#     IB.orderStatusEvent.emit(456)
