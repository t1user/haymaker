import asyncio
import logging
import random
from copy import deepcopy
from itertools import count
from types import SimpleNamespace

import ib_insync as ibi
import pytest
from helpers import wait_for_condition

from haymaker.controller.controller import Controller, ControllerError
from haymaker.controller.future_roller import FutureRoller
from haymaker.controller.sync_brackets import (
    BracketSync,
    BracketSyncAction,
    BracketSyncError,
)
from haymaker.controller.sync_coordinator import (
    SyncBrokenStateError,
    SyncCoordinator,
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
    disabled_reasons = []

    async def pending_positions():
        await asyncio.sleep(1)
        return []

    controller.broker_request_timeout = 0.01
    controller.sync_resync_delay = 0
    monkeypatch.setattr(
        controller, "disable_trading", lambda reason: disabled_reasons.append(reason)
    )
    monkeypatch.setattr(controller.ib, "isConnected", lambda: True)
    monkeypatch.setattr(controller.ib, "positions", lambda: [])
    monkeypatch.setattr(controller.ib, "reqPositionsAsync", pending_positions)

    result = await controller.sync()

    assert not result
    assert disabled_reasons == ["sync did not converge"]


@pytest.mark.asyncio
async def test_run_stops_before_sync_when_state_store_read_fails(
    controller, monkeypatch
):
    sync_called = False

    async def fail_read_from_store():
        raise RuntimeError("state store read failed")

    async def sync():
        nonlocal sync_called
        sync_called = True
        return True

    controller.cold_start = False
    monkeypatch.setattr(controller.sm, "read_from_store", fail_read_from_store)
    monkeypatch.setattr(controller, "sync", sync)

    result = await controller.run()

    assert not result
    assert not sync_called
    assert controller._trading_disabled


@pytest.mark.asyncio
async def test_sync_disconnected_does_not_query_broker_state(controller, monkeypatch):
    connection_attempts = 0
    disabled_reasons = []

    def disconnected():
        nonlocal connection_attempts
        connection_attempts += 1
        return False

    def fail_position_read():
        raise AssertionError("broker state should not be queried")

    async def fail_requested_positions():
        raise AssertionError("broker state should not be queried")

    controller.sync_resync_delay = 0
    monkeypatch.setattr(
        controller, "disable_trading", lambda reason: disabled_reasons.append(reason)
    )
    monkeypatch.setattr(controller.ib, "isConnected", disconnected)
    monkeypatch.setattr(controller.ib, "positions", fail_position_read)
    monkeypatch.setattr(controller.ib, "reqPositionsAsync", fail_requested_positions)

    result = await controller.sync()

    assert not result
    assert connection_attempts == 1
    assert disabled_reasons == []


@pytest.mark.asyncio
async def test_sync_disconnected_does_not_release_hold(controller, monkeypatch):
    def fail_position_read():
        raise AssertionError("broker state should not be queried")

    controller.set_hold()
    monkeypatch.setattr(controller.ib, "isConnected", lambda: False)
    monkeypatch.setattr(controller.ib, "positions", fail_position_read)

    result = await controller.sync()

    assert not result
    assert controller._hold
    assert not controller._trading_disabled


@pytest.mark.asyncio
async def test_sync_coordinator_returns_false_for_broker_state_timeout(
    controller, monkeypatch
):
    async def pending_positions():
        await asyncio.sleep(1)
        return []

    controller.broker_request_timeout = 0.01
    monkeypatch.setattr(controller.ib, "isConnected", lambda: True)
    monkeypatch.setattr(controller.ib, "positions", lambda: [])
    monkeypatch.setattr(controller.ib, "reqPositionsAsync", pending_positions)

    coordinator = SyncCoordinator(controller)
    result = await coordinator.run()

    assert not result
    assert not controller._trading_disabled


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

    controller.sync_resync_delay = 0
    monkeypatch.setattr(controller.ib, "isConnected", lambda: True)
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


@pytest.mark.asyncio
async def test_unknown_broker_orders_are_cancelled_without_disabling_trading(
    controller, trade, monkeypatch
):
    cancelled = []

    async def requested_positions():
        return []

    def fake_cancel(received_trade):
        cancelled.append(received_trade)
        return received_trade

    controller.cancel_unknown_trades = True
    monkeypatch.setattr(controller, "cancel", fake_cancel)
    monkeypatch.setattr(controller.ib, "isConnected", lambda: True)
    monkeypatch.setattr(controller.ib, "positions", lambda: [])
    monkeypatch.setattr(controller.ib, "reqPositionsAsync", requested_positions)
    monkeypatch.setattr(controller.ib, "openTrades", lambda: [trade])
    monkeypatch.setattr(controller.ib, "trades", lambda: [])
    monkeypatch.setattr(controller.ib, "fills", lambda: [])

    result = await SyncCoordinator(controller).run()

    assert not result
    assert cancelled == [trade]
    assert not controller._trading_disabled


def test_from_config_loads_controller_sync_options(Atom):
    controller = Controller.from_config(
        Trader(Atom.ib),
        top_config={
            "ignore_errors": [202, 321],
            "controller": {
                "broker_request_timeout": 3,
                "sync_max_attempts": 2,
                "sync_resync_delay": 0,
                "cancel_unknown_trades": True,
                "missing_brackets": "warn",
            },
        },
    )

    assert controller.broker_request_timeout == 3
    assert controller.sync_max_attempts == 2
    assert controller.sync_resync_delay == 0
    assert controller.cancel_unknown_trades
    assert controller.missing_brackets == "warn"
    assert controller.ignore_errors == [202, 321]


def test_routine_order_cancellation_is_logged_at_debug(controller, caplog):
    caplog.set_level(logging.DEBUG)

    controller.onBrokerMessage(123, 202, "Order cancelled", ibi.Contract())

    assert "Broker message 202: Order cancelled" in caplog.text


def test_ignored_broker_message_is_not_logged(controller, caplog):
    caplog.set_level(logging.DEBUG)
    controller.ignore_errors = [202]

    controller.onBrokerMessage(123, 202, "Order cancelled", ibi.Contract())

    assert "Order cancelled" not in caplog.text


def test_ignored_order_cancellation_does_not_hide_failed_order(controller, caplog):
    caplog.set_level(logging.ERROR)
    controller.ignore_errors = [202]

    controller.onBrokerMessage(123, 202, "YOUR ORDER IS NOT ACCEPTED", ibi.Contract())

    assert "ORDER NOT ACCEPTED" in caplog.text


def test_order_rejection_is_visible_and_registered(controller, caplog, monkeypatch):
    rejected = []
    caplog.set_level(logging.CRITICAL)
    monkeypatch.setattr(controller.sm, "register_rejected_order", rejected.append)

    controller.onBrokerMessage(123, 201, "Rejected", ibi.Contract())

    assert "ORDER REJECTED" in caplog.text
    assert rejected == [""]


def test_from_config_ignores_unknown_controller_config(Atom):
    controller = Controller.from_config(
        Trader(Atom.ib),
        top_config={
            "controller": {
                "invalid": True,
                "broker_request_timeout": 3,
            }
        },
    )

    assert controller.broker_request_timeout == 3
    assert not hasattr(controller, "invalid")


def test_from_config_rejects_invalid_missing_brackets_value(Atom):
    with pytest.raises(ControllerError, match="missing_brackets"):
        Controller.from_config(
            Trader(Atom.ib),
            top_config={"controller": {"missing_brackets": "close"}},
        )


@pytest.mark.asyncio
async def test_unknown_broker_orders_can_be_left_active_by_config(
    controller, trade, monkeypatch
):
    cancelled = []
    bracket_checked = []

    async def requested_positions():
        return []

    def fake_cancel(received_trade):
        cancelled.append(received_trade)

    monkeypatch.setattr(controller, "cancel", fake_cancel)
    monkeypatch.setattr(controller.ib, "isConnected", lambda: True)
    monkeypatch.setattr(controller.ib, "positions", lambda: [])
    monkeypatch.setattr(controller.ib, "reqPositionsAsync", requested_positions)
    monkeypatch.setattr(controller.ib, "openTrades", lambda: [trade])
    monkeypatch.setattr(controller.ib, "trades", lambda: [])
    monkeypatch.setattr(controller.ib, "fills", lambda: [])
    monkeypatch.setattr(
        BracketSyncAction,
        "from_policy",
        staticmethod(lambda policy, controller: bracket_checked.append(policy)),
    )
    controller.cancel_unknown_trades = False

    result = await SyncCoordinator(controller).run()

    assert result
    assert cancelled == []
    assert bracket_checked == []
    assert not controller._trading_disabled


@pytest.mark.asyncio
async def test_unknown_broker_orders_skip_correction_trades(
    controller, trade, monkeypatch
):
    bracket_checked = []

    async def requested_positions():
        return []

    monkeypatch.setattr(controller.ib, "isConnected", lambda: True)
    monkeypatch.setattr(controller.ib, "positions", lambda: [])
    monkeypatch.setattr(controller.ib, "reqPositionsAsync", requested_positions)
    monkeypatch.setattr(controller.ib, "openTrades", lambda: [trade])
    monkeypatch.setattr(controller.ib, "trades", lambda: [])
    monkeypatch.setattr(controller.ib, "fills", lambda: [])
    monkeypatch.setattr(
        BracketSyncAction,
        "from_policy",
        staticmethod(lambda policy, controller: bracket_checked.append(policy)),
    )
    controller.cancel_unknown_trades = False

    result = await controller.sync()

    assert result
    assert bracket_checked == []


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

    def fake_bracket_sync(policy, received_controller):
        reconciled.append((policy, received_controller))

    controller.sync_resync_delay = 0
    monkeypatch.setattr(controller.ib, "isConnected", lambda: True)
    monkeypatch.setattr(controller.ib, "positions", lambda: [])
    monkeypatch.setattr(controller.ib, "reqPositionsAsync", requested_positions)
    monkeypatch.setattr(controller.ib, "openTrades", lambda: [broker_trade])
    monkeypatch.setattr(controller.ib, "trades", lambda: [])
    monkeypatch.setattr(controller.ib, "fills", lambda: [])
    monkeypatch.setattr(
        BracketSyncAction,
        "from_policy",
        staticmethod(fake_bracket_sync),
    )

    result = await controller.sync()

    assert result
    assert controller.sm.order[old_trade.order.orderId].trade is broker_trade
    assert reconciled == [(controller.missing_brackets, controller)]


@pytest.mark.asyncio
async def test_sync_coordinator_relinks_open_trade_and_completes(
    controller, trade, monkeypatch
):
    old_trade = deepcopy(trade)
    old_trade.orderStatus = ibi.OrderStatus(status="Submitted", filled=0, remaining=1)
    old_trade.fills = []
    broker_trade = deepcopy(old_trade)
    controller.sm.order[old_trade.order.orderId] = OrderInfo(
        "coolstrategy", "OPEN", old_trade, {}
    )

    async def requested_positions():
        return []

    monkeypatch.setattr(controller.ib, "isConnected", lambda: True)
    monkeypatch.setattr(controller.ib, "positions", lambda: [])
    monkeypatch.setattr(controller.ib, "reqPositionsAsync", requested_positions)
    monkeypatch.setattr(controller.ib, "openTrades", lambda: [broker_trade])
    monkeypatch.setattr(controller.ib, "trades", lambda: [])
    monkeypatch.setattr(controller.ib, "fills", lambda: [])
    bracket_checked = []
    monkeypatch.setattr(
        BracketSyncAction,
        "from_policy",
        staticmethod(lambda policy, controller: bracket_checked.append(policy)),
    )

    result = await SyncCoordinator(controller).run()

    assert result
    assert controller.sm.order[old_trade.order.orderId].trade is broker_trade
    assert bracket_checked == [controller.missing_brackets]
    assert not controller._trading_disabled


@pytest.mark.asyncio
async def test_sync_coordinator_prunes_unmatched_local_order_and_retries(
    controller, trade, monkeypatch
):
    old_trade = deepcopy(trade)
    old_trade.orderStatus = ibi.OrderStatus(status="Submitted", filled=0, remaining=1)
    old_trade.fills = []
    controller.sm.order[old_trade.order.orderId] = OrderInfo(
        "coolstrategy", "OPEN", old_trade, {}
    )

    async def requested_positions():
        return []

    def fail_bracket_sync(policy, controller):
        raise AssertionError("bracket sync should not run after order recovery")

    monkeypatch.setattr(controller.ib, "isConnected", lambda: True)
    monkeypatch.setattr(controller.ib, "positions", lambda: [])
    monkeypatch.setattr(controller.ib, "reqPositionsAsync", requested_positions)
    monkeypatch.setattr(controller.ib, "openTrades", lambda: [])
    monkeypatch.setattr(controller.ib, "trades", lambda: [])
    monkeypatch.setattr(controller.ib, "fills", lambda: [])
    monkeypatch.setattr(
        BracketSyncAction,
        "from_policy",
        staticmethod(fail_bracket_sync),
    )

    result = await SyncCoordinator(controller).run()

    assert not result
    assert old_trade.order.orderId not in controller.sm.order
    assert not controller._trading_disabled


@pytest.mark.asyncio
async def test_sync_coordinator_raises_for_broken_state(controller, monkeypatch):
    async def requested_positions():
        return []

    def blocked_bracket_sync(policy, controller):
        raise BracketSyncError("broken bracket state")

    monkeypatch.setattr(controller.ib, "isConnected", lambda: True)
    monkeypatch.setattr(controller.ib, "positions", lambda: [])
    monkeypatch.setattr(controller.ib, "reqPositionsAsync", requested_positions)
    monkeypatch.setattr(controller.ib, "openTrades", lambda: [])
    monkeypatch.setattr(controller.ib, "trades", lambda: [])
    monkeypatch.setattr(controller.ib, "fills", lambda: [])
    monkeypatch.setattr(
        BracketSyncAction,
        "from_policy",
        staticmethod(blocked_bracket_sync),
    )

    coordinator = SyncCoordinator(controller)

    with pytest.raises(SyncBrokenStateError, match="bracket sync failed"):
        await coordinator.run()
    assert not controller._trading_disabled


@pytest.mark.asyncio
async def test_sync_disables_trading_when_recovery_does_not_converge(
    controller, monkeypatch
):
    attempts = []
    controller.sync_max_attempts = 2
    controller.sync_resync_delay = 0
    monkeypatch.setattr(controller.ib, "isConnected", lambda: True)

    async def retryable_sync_failure(self):
        attempts.append(self)
        return False

    monkeypatch.setattr(SyncCoordinator, "run", retryable_sync_failure)

    result = await controller.sync()

    assert not result
    assert len(attempts) == controller.sync_max_attempts
    assert controller._trading_disabled


def test_bracket_sync_does_not_report_protected_broker_position(
    controller, trade, monkeypatch
):
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

    result = BracketSync(controller)

    assert result.exposed_positions == []


def test_bracket_sync_take_profit_does_not_protect_broker_position(
    controller, trade, monkeypatch
):
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

    result = BracketSync(controller)

    assert len(result.exposed_positions) == 1
    assert result.exposed_positions[0].contract == trade.contract
    assert result.exposed_positions[0].broker_position == 3


def test_ignore_bracket_policy_does_not_query_broker_or_local_records(
    controller, monkeypatch
):
    def fail_broker_read():
        raise AssertionError("ignore policy should not read broker state")

    monkeypatch.setattr(controller.ib, "positions", fail_broker_read)

    result = BracketSyncAction.from_policy("ignore", controller)

    assert result.__class__.__name__ == "IgnoreBracketSyncAction"


def test_remove_bracket_policy_closes_strategy_with_missing_local_bracket(
    controller, trade, monkeypatch
):
    strategy = controller.sm.strategy["coolstrategy"]
    strategy.position = 1
    strategy.active_contract = trade.contract
    strategy.params = {"stop-loss": {"amount": 1}}
    closed = []
    set_broker_state(controller, monkeypatch)

    monkeypatch.setattr(
        controller,
        "close_positions_for_strategy",
        lambda strategy_name, action: closed.append((strategy_name, action)),
    )

    with pytest.raises(BracketSyncError):
        BracketSyncAction.from_policy("remove", controller)

    assert closed == [("coolstrategy", "MISSING BRACKET EMERGENCY CLOSE")]


def test_remove_bracket_policy_cancels_obsolete_local_bracket(
    controller, trade, monkeypatch
):
    controller.sm.strategy["coolstrategy"].position = 0
    obsolete_trade = deepcopy(trade)
    obsolete_trade.orderStatus = ibi.OrderStatus(
        status="Submitted",
        filled=0,
        remaining=1,
    )
    controller.sm.order[obsolete_trade.order.orderId] = OrderInfo(
        "coolstrategy",
        "STOP-LOSS",
        obsolete_trade,
        {},
    )
    cancelled = []
    set_broker_state(controller, monkeypatch)

    monkeypatch.setattr(
        controller.trader,
        "cancel",
        lambda received_trade: cancelled.append(received_trade),
    )

    with pytest.raises(BracketSyncError):
        BracketSyncAction.from_policy("remove", controller)

    assert cancelled == [obsolete_trade]


def test_future_roll_replacement_order_preserves_order_info_params():
    cancelled_trade = ibi.Trade(
        contract=ibi.Future(symbol="MBT", exchange="CMECRYPTO", localSymbol="MBTM6"),
        order=ibi.Order(
            orderId=73806,
            permId=123,
            action="SELL",
            totalQuantity=1,
            orderType="TRAIL",
            auxPrice=120.0,
            trailStopPrice=101000.0,
        ),
    )
    original_params = {
        "position_id": "MBT-position-1",
        "sl_points": 200.0,
        "min_tick": 5.0,
        "trail_multiple": 3.0,
    }
    oi = OrderInfo("dt_fast_MBT", "STOP-LOSS", cancelled_trade, original_params)
    new_contract = ibi.Future(symbol="MBT", exchange="CMECRYPTO", localSymbol="MBTU6")
    captured = {}

    def trade(strategy_str, contract, order, action, params):
        captured.update(
            {
                "strategy_str": strategy_str,
                "contract": contract,
                "order": order,
                "action": action,
                "params": params,
            }
        )
        return ibi.Trade(contract=contract, order=order)

    roller = object.__new__(FutureRoller)
    roller.controller = SimpleNamespace(trade=trade)

    roller.issue_new_order(
        cancelled_trade,
        oi,
        new_contract,
        "dt_fast_MBT",
        None,
        fill_price=250.0,
    )

    assert captured["params"] == original_params
    assert captured["params"] is not original_params
    assert captured["params"]["position_id"] == "MBT-position-1"
    assert captured["order"].trailStopPrice == 101250.0
    assert captured["order"].orderId == 0
    assert captured["order"].permId == 0


# def test_StateMachine_lined_to_ib_orderStatusEvent(caplog):
#     """TODO: This doesnt test anythin yet."""
#     IB.orderStatusEvent.emit(456)
