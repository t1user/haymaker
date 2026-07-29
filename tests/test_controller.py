import asyncio
import datetime as dt
import logging
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import ib_insync as ibi
import pytest
from helpers import wait_for_condition

from haymaker.book import OrderInfo, PositionState
from haymaker.components import PositionTarget, StandardOrderRole
from haymaker.controller import Controller
from haymaker.controller.controller import ControllerError, SyncOutcome
from haymaker.controller.sync_brackets import (
    BracketSync,
    BracketSyncAction,
    BracketSyncError,
)
from haymaker.controller.sync_coordinator import (
    SyncBrokenStateError,
    SyncCoordinator,
    verify_broker_position_source,
)
from haymaker.controller.terminator import Terminator
from haymaker.supervisor.codes import SUPERVISOR_OWNED_BROKER_CODES
from haymaker.trader import Trader


class FakeTrader:
    def __init__(self):
        self.trades = []
        self.cancelled = []
        self.broker_positions = {}

    def trade(self, contract, order):
        if not order.orderId:
            order.orderId = len(self.trades) + 1
        order.permId = order.permId or 100 + abs(order.orderId)
        trade = ibi.Trade(
            contract=contract,
            order=order,
            orderStatus=ibi.OrderStatus(
                orderId=order.orderId,
                status=ibi.OrderStatus.Submitted,
                remaining=order.totalQuantity,
            ),
        )
        self.trades.append(trade)
        return trade

    def cancel(self, trade):
        self.cancelled.append(trade)
        return trade

    def position_for_contract(self, contract):
        return self.broker_positions.get(contract, 0)

    def positions(self):
        return dict(self.broker_positions)


def contract():
    return ibi.Future(
        conId=1,
        symbol="ES",
        exchange="CME",
        localSymbol="ESM6",
    )


@pytest.fixture
def controller_runtime(atom_runtime):
    trader = FakeTrader()
    controller = Controller(trader=trader, missing_brackets="ignore")
    atom_runtime.bind_controller(controller)
    return atom_runtime, controller, trader


def fill(trade, exec_id="exec-1", quantity=1):
    return ibi.Fill(
        contract=trade.contract,
        execution=ibi.Execution(
            execId=exec_id,
            orderId=trade.order.orderId,
            permId=trade.order.permId,
            side="BOT" if trade.order.action == "BUY" else "SLD",
            shares=quantity,
            price=100,
            time=datetime.now(timezone.utc),
        ),
        commissionReport=ibi.CommissionReport(execId=exec_id),
        time=datetime.now(timezone.utc),
    )


def set_broker_state(
    controller: Controller,
    monkeypatch: pytest.MonkeyPatch,
    *,
    positions: tuple[ibi.Position, ...] = (),
    open_trades: tuple[ibi.Trade, ...] = (),
    trades: tuple[ibi.Trade, ...] = (),
    fills: tuple[ibi.Fill, ...] = (),
) -> None:
    """Set direct broker-query results for focused reconciliation tests."""

    monkeypatch.setattr(controller.ib, "positions", lambda: list(positions))
    monkeypatch.setattr(controller.ib, "openTrades", lambda: list(open_trades))
    monkeypatch.setattr(controller.ib, "trades", lambda: list(trades))
    monkeypatch.setattr(controller.ib, "fills", lambda: list(fills))


def save_active_order(
    controller: Controller,
    trade: ibi.Trade,
    *,
    source_key: str = "alpha",
    role: str = StandardOrderRole.OPEN,
) -> OrderInfo:
    """Persist an active copy of a broker Trade for sync tests."""

    trade.orderStatus = ibi.OrderStatus(
        orderId=trade.order.orderId,
        status=ibi.OrderStatus.Submitted,
        filled=0,
        remaining=trade.order.totalQuantity,
    )
    trade.fills = []
    return controller.register_order(
        trade,
        role=role,
        execution_model_name="brackets",
        source_key=source_key,
        position_id="episode",
    )


def test_from_mapping_constructs_nested_startup_config(atom_runtime):
    controller = Controller.from_mapping(
        {
            "startup": {"cold_start": False, "zero": True},
            "sync_frequency": 10,
        },
        trader=FakeTrader(),
    )

    assert controller.cold_start is False
    assert controller.zero is True
    assert controller.sync_frequency == 10


def test_from_mapping_rejects_unknown_controller_key(atom_runtime):
    with pytest.raises(TypeError, match="unknown"):
        Controller.from_mapping(
            {"unknown": True},
            trader=Trader(atom_runtime.ib),
        )


def test_from_mapping_rejects_non_mapping_startup(atom_runtime):
    with pytest.raises(TypeError, match="controller.startup"):
        Controller.from_mapping(
            {"startup": True},
            trader=Trader(atom_runtime.ib),
        )


def test_direct_construction_loads_controller_sync_options(atom_runtime):
    controller = Controller(
        Trader(atom_runtime.ib),
        ignore_errors=[202, 321, 10182, 1102],
        broker_request_timeout=3,
        sync_max_attempts=2,
        sync_resync_delay=0,
        cancel_unknown_trades=True,
        missing_brackets="warn",
    )

    assert controller.broker_request_timeout == 3
    assert controller.sync_max_attempts == 2
    assert controller.sync_resync_delay == 0
    assert controller.cancel_unknown_trades
    assert controller.missing_brackets == "warn"
    assert set(controller.ignore_errors) == (SUPERVISOR_OWNED_BROKER_CODES | {202, 321})


def test_direct_controller_does_not_schedule_future_roll(controller):
    assert controller._future_roll_timer is None


def test_controller_copies_future_roll_policies(controller):
    policies = {"automatic": True, "manual": False}

    controller.set_future_roll_policies(policies)
    policies["manual"] = True

    assert controller.future_roll_policies == {
        "automatic": True,
        "manual": False,
    }


def test_direct_construction_defers_future_roll_until_runtime_start(
    atom_runtime, monkeypatch
):
    timeranges = []

    class FakeTimerange:
        callback = None

        def __iadd__(self, callback):
            self.callback = callback
            return self

    def fake_timerange(*, start, step):
        timerange = FakeTimerange()
        timeranges.append((start, step, timerange))
        return timerange

    monkeypatch.setattr(
        "haymaker.controller.controller.ev.Event.timerange", fake_timerange
    )
    controller = Controller(Trader(atom_runtime.ib), future_roll_time=(14, 0))

    assert timeranges == []
    controller._ensure_runtime_timers_started()

    start, step, timerange = timeranges[0]
    assert start == dt.time(hour=14, minute=0, tzinfo=dt.UTC)
    assert step == dt.timedelta(days=1)
    assert timerange.callback == controller.roll_futures
    assert controller._future_roll_timer is timerange

    controller._ensure_runtime_timers_started()
    assert len(timeranges) == 1


def test_schedule_future_roll_ignores_duplicate_request(atom_runtime, monkeypatch):
    timeranges = []

    class FakeTimerange:
        def __iadd__(self, callback):
            return self

    def fake_timerange(*, start, step):
        timerange = FakeTimerange()
        timeranges.append(timerange)
        return timerange

    monkeypatch.setattr(
        "haymaker.controller.controller.ev.Event.timerange", fake_timerange
    )
    controller = Controller(Trader(atom_runtime.ib), future_roll_time=(14, 0))

    controller.schedule_future_roll()
    controller.schedule_future_roll()

    assert len(timeranges) == 1
    assert controller._future_roll_timer is timeranges[0]


def test_direct_construction_rejects_invalid_missing_brackets_value(
    atom_runtime,
):
    with pytest.raises(ControllerError, match="missing_brackets"):
        Controller(
            Trader(atom_runtime.ib),
            missing_brackets=cast(Any, "close"),
        )


@pytest.mark.asyncio
async def test_new_order_event_reports_unregistered_trade(caplog, atom_runtime):
    controller = Controller(Trader(atom_runtime.ib))
    atom_runtime.ib.newOrderEvent.emit(
        ibi.Trade(
            contract=ibi.Future(symbol="ES"),
            order=ibi.Order(orderId=123, permId=45678),
        )
    )

    assert await wait_for_condition(lambda: "123" in caplog.text)
    assert controller.book.order_by_id(123) is None


def test_routine_order_cancellation_is_logged_at_debug(controller, caplog):
    caplog.set_level(logging.DEBUG)

    controller.onErrEvent(123, 202, "Order cancelled", ibi.Contract())

    assert "Broker message 202: Order cancelled" in caplog.text
    assert caplog.records[-1].levelno == logging.DEBUG


def test_ignored_broker_message_is_not_logged(controller, caplog):
    caplog.set_level(logging.DEBUG)
    controller.ignore_errors = [202]

    controller.onErrEvent(123, 202, "Order cancelled", ibi.Contract())

    assert "Order cancelled" not in caplog.text


def test_supervisor_owned_broker_message_is_ignored_by_controller(controller, caplog):
    caplog.set_level(logging.DEBUG)

    controller.onErrEvent(
        -1,
        10182,
        "Failed to request live updates (disconnected).",
        ibi.Contract(),
    )

    assert "Failed to request live updates" not in caplog.text


def test_ignored_order_cancellation_does_not_hide_failed_order(controller, caplog):
    caplog.set_level(logging.ERROR)
    controller.ignore_errors = [202]

    controller.onErrEvent(123, 202, "YOUR ORDER IS NOT ACCEPTED", ibi.Contract())

    assert "ORDER NOT ACCEPTED" in caplog.text


def test_unknown_low_code_broker_message_is_visible(controller, caplog):
    caplog.set_level(logging.ERROR)

    controller.onErrEvent(123, 347, "Short sale slot validation failed", ibi.Contract())

    assert "Broker message 347: Short sale slot validation failed" in caplog.text


def test_known_request_validation_messages_remain_debug(controller, caplog):
    caplog.set_level(logging.DEBUG)

    controller.onErrEvent(123, 321, "Server validation message", ibi.Contract())

    assert "Broker message 321: Server validation message" in caplog.text
    assert caplog.records[-1].levelno == logging.DEBUG


def test_unknown_high_code_broker_message_remains_debug(controller, caplog):
    caplog.set_level(logging.DEBUG)

    controller.onErrEvent(123, 500, "Client side message", ibi.Contract())

    assert "Broker message 500: Client side message" in caplog.text
    assert caplog.records[-1].levelno == logging.DEBUG


def test_order_rejection_is_visible_and_registered(controller_runtime, caplog):
    runtime, controller, _ = controller_runtime
    trade = controller.trade(
        contract(),
        ibi.MarketOrder("BUY", 1),
        role=StandardOrderRole.OPEN,
        execution_model_name="brackets",
    )
    caplog.set_level(logging.CRITICAL)

    controller.onErrEvent(trade.order.orderId, 201, "Rejected", trade.contract)

    assert "ORDER REJECTED" in caplog.text
    assert runtime.book._rejected_orders["brackets"] == 1


def test_trade_registers_complete_attribution_immediately(controller_runtime):
    runtime, controller, trader = controller_runtime

    result = controller.trade(
        contract(),
        ibi.MarketOrder("BUY", 2),
        role=StandardOrderRole.OPEN,
        execution_model_name="brackets",
        source_key="alpha",
        position_id="episode",
        params={"atr": 5},
    )

    assert result is trader.trades[0]
    info = runtime.book.order_by_id(result.order.orderId)
    assert info.execution_model_name == "brackets"
    assert info.source_key == "alpha"
    assert info.position_id == "episode"
    assert info.role == StandardOrderRole.OPEN


def test_disabled_trading_does_not_submit_or_register(controller_runtime):
    runtime, controller, trader = controller_runtime
    controller.disable_trading("test")

    result = controller.trade(
        contract(),
        ibi.MarketOrder("BUY", 1),
        role=StandardOrderRole.OPEN,
        execution_model_name="brackets",
        source_key="alpha",
    )

    assert result is None
    assert trader.trades == []
    assert runtime.book.active_orders() == ()


@pytest.mark.asyncio
async def test_exec_details_applies_fill_once(controller_runtime):
    runtime, controller, _ = controller_runtime
    runtime.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=contract(),
        )
    )
    trade = controller.trade(
        contract(),
        ibi.MarketOrder("BUY", 1),
        role=StandardOrderRole.OPEN,
        execution_model_name="brackets",
        source_key="alpha",
        position_id="episode",
    )
    execution = fill(trade)

    await controller.onExecDetailsEvent(trade, execution)
    await controller.onExecDetailsEvent(trade, execution)

    assert runtime.book.position_state("alpha").quantity == 1
    assert len(runtime.book.order_by_id(trade.order.orderId).fills) == 1


@pytest.mark.asyncio
async def test_offline_zero_order_id_fill_rebinds_by_perm_id(controller_runtime):
    runtime, controller, _ = controller_runtime
    runtime.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=contract(),
        )
    )
    original = controller.trade(
        contract(),
        ibi.MarketOrder("BUY", 1),
        role=StandardOrderRole.OPEN,
        execution_model_name="brackets",
        source_key="alpha",
    )
    rebound = ibi.Trade(
        contract=contract(),
        order=ibi.Order(
            orderId=0,
            permId=original.order.permId,
            action="BUY",
            totalQuantity=1,
        ),
    )

    await controller.onExecDetailsEvent(rebound, fill(rebound))

    assert rebound.order.orderId == original.order.orderId
    assert runtime.book.position_state("alpha").quantity == 1


@pytest.mark.asyncio
async def test_unmatched_zero_order_id_fill_is_not_persisted(
    controller_runtime, caplog
):
    runtime, controller, _ = controller_runtime
    trade = ibi.Trade(
        contract=contract(),
        order=ibi.Order(
            orderId=0,
            permId=999001,
            action="BUY",
            totalQuantity=1,
        ),
    )
    execution = fill(trade, exec_id="zero-order-fill")

    with caplog.at_level(logging.ERROR):
        await controller.onExecDetailsEvent(trade, execution)

    assert runtime.book.order_by_id(0) is None
    assert runtime.book.order_by_perm_id(999001) is None
    assert "Cannot persist unknown orderId=0 trade" in caplog.text


@pytest.mark.asyncio
async def test_manual_trade_has_explicit_role_and_source_attribution(
    controller_runtime,
):
    runtime, controller, _ = controller_runtime
    runtime.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=contract(),
            quantity=1,
            position_id="episode",
        )
    )
    trade = ibi.Trade(
        contract=contract(),
        order=ibi.Order(
            orderId=-1,
            permId=999,
            action="SELL",
            totalQuantity=1,
        ),
    )

    await controller.onExecDetailsEvent(trade, fill(trade))

    info = runtime.book.order_by_id(-1)
    assert info.role == StandardOrderRole.MANUAL
    assert info.source_key == "alpha"
    assert info.position_id == "episode"


@pytest.mark.asyncio
async def test_commission_report_uses_book_owned_blotter(controller_runtime):
    runtime, controller, _ = controller_runtime

    class Blotter:
        def __init__(self):
            self.calls = []

        def log_commission(self, *args, **kwargs):
            self.calls.append((args, kwargs))

    blotter = Blotter()
    runtime.book.blotter = blotter
    controller.release_hold()
    trade = controller.trade(
        contract(),
        ibi.MarketOrder("BUY", 1),
        role=StandardOrderRole.OPEN,
        execution_model_name="brackets",
        source_key="alpha",
        position_id="episode",
    )
    report = ibi.CommissionReport(execId="exec-1", commission=1)
    execution = fill(trade)

    await controller.onCommissionReport(trade, execution, report)

    assert blotter.calls[0][1]["source_key"] == "alpha"
    assert blotter.calls[0][1]["position_id"] == "episode"
    assert blotter.calls[0][1]["role"] == StandardOrderRole.OPEN


@pytest.mark.asyncio
async def test_commission_report_persists_without_blotter(controller_runtime):
    """Commission evidence remains durable when reporting is disabled."""

    runtime, controller, _ = controller_runtime
    assert runtime.book.blotter is None
    assert len(controller.ib.commissionReportEvent) == 1
    controller.release_hold()
    trade = controller.trade(
        contract(),
        ibi.MarketOrder("BUY", 1),
        role=StandardOrderRole.OPEN,
        execution_model_name="brackets",
        source_key="alpha",
        position_id="episode",
    )
    execution = fill(trade)
    await controller.onExecDetailsEvent(trade, execution)
    report = ibi.CommissionReport(execId="exec-1", commission=1.25)

    await controller.onCommissionReport(trade, execution, report)

    info = runtime.book.order_by_id(trade.order.orderId)
    assert info.fills[0].commission_report == report


@pytest.mark.asyncio
async def test_commission_report_skips_unknown_zero_order_id(
    controller_runtime, caplog
):
    runtime, controller, _ = controller_runtime
    trade = ibi.Trade(
        contract=contract(),
        order=ibi.Order(orderId=0, totalQuantity=1),
    )
    report = ibi.CommissionReport(execId="exec-1")
    execution = fill(trade)
    controller.release_hold()

    with caplog.at_level(logging.DEBUG):
        await controller.onCommissionReport(trade, execution, report)

    assert runtime.book.order_by_id(0) is None
    assert runtime.book.blotter is None


def test_close_position_for_source_logs_fill(controller_runtime, caplog):
    runtime, controller, trader = controller_runtime
    runtime.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=contract(),
            quantity=1,
            position_id="episode",
        )
    )

    with caplog.at_level(logging.INFO, logger="haymaker.controller.controller"):
        controller.close_position_for_source("alpha", role="test close")
        close_trade = trader.trades[-1]
        close_trade.orderStatus.filled = 1
        close_trade.filledEvent.emit(close_trade)

    assert "test close trade filled" in caplog.text
    assert "alpha" in caplog.text


@pytest.mark.parametrize(
    ("role", "missing"),
    [
        (StandardOrderRole.STOP_LOSS, False),
        (StandardOrderRole.TAKE_PROFIT, True),
    ],
)
def test_bracket_sync_requires_stop_but_not_take_profit(
    controller_runtime, role, missing
):
    """Only an absent stop is a critical local bracket-record issue."""

    runtime, controller, _ = controller_runtime
    runtime.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=contract(),
            quantity=1,
            target_quantity=1,
            position_id="episode",
            bracket_inputs={"atr": 5},
        )
    )
    controller.trade(
        contract(),
        ibi.Order(
            action="SELL",
            totalQuantity=1,
            orderType=("STP" if role == StandardOrderRole.STOP_LOSS else "LMT"),
        ),
        role=role,
        execution_model_name="brackets",
        source_key="alpha",
        position_id="episode",
    )

    sync = BracketSync(controller)

    assert bool(sync.missing_brackets) is missing


def test_bracket_sync_does_not_report_protected_broker_position(
    controller_runtime,
):
    _, controller, _ = controller_runtime
    protected = ibi.Trade(
        contract=contract(),
        order=ibi.StopOrder("SELL", 1, stopPrice=100),
        orderStatus=ibi.OrderStatus(
            status=ibi.OrderStatus.Submitted,
            remaining=1,
        ),
    )
    controller.ib.positions = Mock(
        return_value=[ibi.Position("DU123", contract(), 1, 100)]
    )
    controller.ib.openTrades = Mock(return_value=[protected])

    assert BracketSync(controller).exposed_positions == []


def test_bracket_sync_take_profit_does_not_protect_broker_position(
    controller_runtime,
):
    _, controller, _ = controller_runtime
    take_profit = ibi.Trade(
        contract=contract(),
        order=ibi.LimitOrder("SELL", 1, lmtPrice=110),
        orderStatus=ibi.OrderStatus(
            status=ibi.OrderStatus.Submitted,
            remaining=1,
        ),
    )
    controller.ib.positions = Mock(
        return_value=[ibi.Position("DU123", contract(), 1, 100)]
    )
    controller.ib.openTrades = Mock(return_value=[take_profit])

    assert len(BracketSync(controller).exposed_positions) == 1


def test_ignore_bracket_policy_does_not_query_broker_or_book(
    controller_runtime,
):
    runtime, controller, _ = controller_runtime
    controller.ib.positions = Mock(
        side_effect=AssertionError("broker must not be queried")
    )
    runtime.book.position_states = Mock(
        side_effect=AssertionError("Book must not be queried")
    )

    BracketSyncAction.from_policy("ignore", controller)


def test_remove_bracket_policy_closes_source_missing_required_stop(
    controller_runtime,
):
    runtime, controller, _ = controller_runtime
    runtime.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=contract(),
            quantity=1,
            target_quantity=1,
            position_id="episode",
            bracket_inputs={"atr": 5},
        )
    )
    controller.ib.positions = Mock(return_value=[])
    controller.close_position_for_source = Mock()

    with pytest.raises(BracketSyncError):
        BracketSyncAction.from_policy("remove", controller)

    controller.close_position_for_source.assert_called_once_with(
        "alpha",
        role="MISSING_BRACKET_EMERGENCY_CLOSE",
    )


def test_remove_bracket_policy_defers_missing_stop_during_active_open(
    controller_runtime,
):
    runtime, controller, _ = controller_runtime
    runtime.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=contract(),
            quantity=1,
            target_quantity=1,
            position_id="episode",
            bracket_inputs={"atr": 5},
        )
    )
    controller.trade(
        contract(),
        ibi.MarketOrder("BUY", 1),
        role=StandardOrderRole.OPEN,
        execution_model_name="brackets",
        source_key="alpha",
        position_id="episode",
    )
    controller.ib.positions = Mock(return_value=[])
    controller.close_position_for_source = Mock()

    BracketSyncAction.from_policy("remove", controller)

    controller.close_position_for_source.assert_not_called()


def test_remove_bracket_policy_cancels_obsolete_bracket(
    controller_runtime,
):
    runtime, controller, _ = controller_runtime
    runtime.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=contract(),
        )
    )
    bracket = controller.trade(
        contract(),
        ibi.StopOrder("SELL", 1, stopPrice=100),
        role=StandardOrderRole.STOP_LOSS,
        execution_model_name="brackets",
        source_key="alpha",
    )
    controller.ib.positions = Mock(return_value=[])
    controller.cancel = Mock()

    with pytest.raises(BracketSyncError):
        BracketSyncAction.from_policy("remove", controller)

    controller.cancel.assert_called_once_with(bracket)


def test_remove_bracket_policy_defers_obsolete_bracket_during_active_close(
    controller_runtime,
):
    runtime, controller, _ = controller_runtime
    runtime.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=contract(),
        )
    )
    controller.trade(
        contract(),
        ibi.StopOrder("SELL", 1, stopPrice=100),
        role=StandardOrderRole.STOP_LOSS,
        execution_model_name="brackets",
        source_key="alpha",
    )
    controller.trade(
        contract(),
        ibi.MarketOrder("SELL", 1),
        role=StandardOrderRole.CLOSE,
        execution_model_name="brackets",
        source_key="alpha",
    )
    controller.ib.positions = Mock(return_value=[])
    controller.cancel = Mock()

    BracketSyncAction.from_policy("remove", controller)

    controller.cancel.assert_not_called()


@pytest.mark.asyncio
async def test_terminator_waits_for_cancellation_before_logical_close(
    controller_runtime,
):
    """Reset never overlaps an attributed close with a working exit order."""

    runtime, controller, _ = controller_runtime
    runtime.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=contract(),
            quantity=1,
            position_id="episode",
        )
    )
    protective = ibi.Trade(
        contract=contract(),
        order=ibi.Order(
            orderId=77,
            action="SELL",
            totalQuantity=1,
            orderType="STP",
        ),
        orderStatus=ibi.OrderStatus(
            orderId=77,
            status=ibi.OrderStatus.Submitted,
            remaining=1,
        ),
    )
    controller.ib.openTrades = Mock(return_value=[protective])
    controller.ib.positions = Mock(return_value=[])
    loop = asyncio.get_running_loop()

    def cancel(_trade):
        """Complete cancellation after more than one event-loop turn."""

        loop.call_soon(
            lambda: loop.call_soon(
                setattr,
                protective.orderStatus,
                "status",
                ibi.OrderStatus.Cancelled,
            )
        )
        return protective

    def close(*args, **kwargs):
        """Record that the close was submitted only after cancellation."""

        assert protective.isDone()
        return ibi.Trade(
            contract=args[0],
            order=ibi.Order(
                orderId=78,
                action="SELL",
                totalQuantity=1,
                orderType="MKT",
            ),
            orderStatus=ibi.OrderStatus(
                orderId=78,
                status=ibi.OrderStatus.Filled,
                filled=1,
                remaining=0,
            ),
        )

    controller.cancel = Mock(side_effect=cancel)
    controller.trade = Mock(side_effect=close)

    completed = await Terminator(controller).run()

    assert completed is True
    controller.trade.assert_called_once()


@pytest.mark.asyncio
async def test_terminator_liquidates_when_cancellation_does_not_complete(
    controller_runtime,
):
    """Cancellation grace expiry never prevents an urgent flattening attempt."""

    runtime, controller, _ = controller_runtime
    runtime.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=contract(),
            quantity=1,
            position_id="episode",
        )
    )
    protective = ibi.Trade(
        contract=contract(),
        order=ibi.Order(
            orderId=77,
            action="SELL",
            totalQuantity=1,
            orderType="STP",
        ),
        orderStatus=ibi.OrderStatus(
            orderId=77,
            status=ibi.OrderStatus.Submitted,
            remaining=1,
        ),
    )
    controller.ib.openTrades = Mock(return_value=[protective])
    controller.ib.positions = Mock(return_value=[])
    liquidation = ibi.Trade(
        contract=contract(),
        order=ibi.MarketOrder("SELL", 1),
        orderStatus=ibi.OrderStatus(
            orderId=78,
            status=ibi.OrderStatus.Filled,
            filled=1,
            remaining=0,
        ),
    )
    controller.trade = Mock(return_value=liquidation)
    terminator = Terminator(controller)
    terminator.cancellation_timeout = 0

    completed = await terminator.run()

    assert completed is True
    controller.trade.assert_called_once()
    assert controller.trade.call_args.kwargs["role"] == (StandardOrderRole.LIQUIDATION)


@pytest.mark.asyncio
async def test_terminator_does_not_hide_residual_behind_flat_state(
    controller_runtime,
):
    """A stale flat PositionState does not suppress broker liquidation."""

    runtime, controller, _ = controller_runtime
    runtime.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=contract(),
            quantity=0,
        )
    )
    controller.ib.openTrades = Mock(return_value=[])
    controller.ib.positions = Mock(
        return_value=[
            ibi.Position(
                account="DU123",
                contract=contract(),
                position=2,
                avgCost=100,
            )
        ]
    )
    done_trade = ibi.Trade(
        contract=contract(),
        order=ibi.Order(
            orderId=79,
            action="SELL",
            totalQuantity=2,
            orderType="MKT",
        ),
        orderStatus=ibi.OrderStatus(
            orderId=79,
            status=ibi.OrderStatus.Filled,
            filled=2,
            remaining=0,
        ),
    )
    controller.trade = Mock(return_value=done_trade)

    completed = await Terminator(controller).run()

    assert completed is True
    assert controller.trade.call_args.kwargs["role"] == (StandardOrderRole.LIQUIDATION)


@pytest.mark.asyncio
async def test_run_keeps_book_when_explicit_reset_fails(controller_runtime):
    """Failed reset disables trading without discarding recovery state."""

    runtime, controller, _ = controller_runtime
    controller.cold_start = True
    controller.reset = True
    controller.sync = AsyncMock(return_value=SyncOutcome.OK)
    controller.execute_stops_and_close_positions = AsyncMock(return_value=False)
    runtime.book.clear_state = Mock()

    completed = await controller.run()

    assert completed is False
    assert controller._trading_disabled is True
    assert controller.reset is True
    runtime.book.clear_state.assert_not_called()


@pytest.mark.asyncio
async def test_target_verification_uses_absolute_quantity(controller_runtime, caplog):
    runtime, controller, _ = controller_runtime
    target = PositionTarget(
        contract=contract(),
        target_quantity=3,
        source_key="alpha",
    )
    runtime.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=contract(),
            quantity=2,
            target_quantity=target.target_quantity,
            target_created_at=target.created_at,
        )
    )

    verified = await controller.verify_target_integrity(target, "brackets")

    assert verified is True
    assert "target=3.0 actual=2" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role",
    [StandardOrderRole.STOP_LOSS, StandardOrderRole.TAKE_PROFIT],
)
async def test_target_verification_does_not_wait_for_protective_orders(
    controller_runtime, role
):
    runtime, controller, _ = controller_runtime
    target = PositionTarget(
        contract=contract(),
        target_quantity=1,
        source_key="alpha",
    )
    runtime.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=target.contract,
            quantity=1,
            target_quantity=1,
            target_created_at=target.created_at,
        )
    )
    controller.trade(
        target.contract,
        ibi.Order(action="SELL", totalQuantity=1, orderType="STP"),
        role=role,
        execution_model_name="brackets",
        source_key="alpha",
    )
    controller.execution_verification_delay = 60

    verified = await asyncio.wait_for(
        controller.verify_target_integrity(target, "brackets"),
        timeout=0.1,
    )

    assert verified is True


@pytest.mark.asyncio
async def test_superseded_target_is_not_verified_or_compared_with_broker(
    controller_runtime, caplog
):
    runtime, controller, _ = controller_runtime
    old = PositionTarget(
        contract=contract(),
        target_quantity=1,
        source_key="alpha",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    runtime.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=old.contract,
            quantity=2,
            target_quantity=2,
            target_created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        )
    )
    controller.verify_position_with_broker = Mock()

    await controller.onData(old, "brackets")

    controller.verify_position_with_broker.assert_not_called()
    assert "Target not achieved" not in caplog.text


@pytest.mark.asyncio
async def test_nuke_liquidation_is_registered_with_episode_attribution(
    controller_runtime, monkeypatch
):
    runtime, controller, trader = controller_runtime
    runtime.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=contract(),
            quantity=2,
            position_id="episode",
        )
    )
    position = ibi.Position("account", contract(), 2, 100)
    monkeypatch.setattr(runtime.ib, "positions", lambda: [position])
    monkeypatch.setattr(
        runtime.ib,
        "qualifyContractsAsync",
        AsyncMock(return_value=[position.contract]),
    )

    await controller.close_positions()

    trade = trader.trades[0]
    info = runtime.book.order_by_id(trade.order.orderId)
    assert trade.order.action == "SELL"
    assert info.role == StandardOrderRole.LIQUIDATION
    assert info.execution_model_name == "brackets"
    assert info.source_key == "alpha"
    assert info.position_id == "episode"


@pytest.mark.asyncio
async def test_sync_timeout_disables_trading(controller, monkeypatch):
    disabled_reasons = []

    async def pending_positions():
        await asyncio.sleep(1)
        return []

    controller.broker_request_timeout = 0.01
    controller.sync_resync_delay = 0
    monkeypatch.setattr(
        controller,
        "disable_trading",
        lambda reason: disabled_reasons.append(reason),
    )
    monkeypatch.setattr(controller.ib, "isConnected", lambda: True)
    monkeypatch.setattr(controller.ib, "positions", lambda: [])
    monkeypatch.setattr(controller.ib, "reqPositionsAsync", pending_positions)

    result = await controller.sync()

    assert result is SyncOutcome.FAILED
    assert disabled_reasons == ["sync did not converge"]


@pytest.mark.asyncio
async def test_sync_skips_when_connection_unavailable(controller, monkeypatch):
    abort_event = asyncio.Event()
    abort_event.set()
    controller.set_sync_abort_event(abort_event)
    sync_body = AsyncMock(return_value=SyncOutcome.OK)
    monkeypatch.setattr(controller, "_sync", sync_body)

    result = await controller.sync()

    assert result is SyncOutcome.ABORTED
    sync_body.assert_not_awaited()
    assert not controller._trading_disabled


@pytest.mark.asyncio
async def test_run_treats_connection_unavailable_sync_as_abort(
    controller, monkeypatch, caplog
):
    abort_event = asyncio.Event()
    abort_event.set()
    controller.set_sync_abort_event(abort_event)
    monkeypatch.setattr(controller, "_sync", AsyncMock(return_value=SyncOutcome.OK))

    with caplog.at_level(logging.DEBUG):
        result = await controller.run()

    assert not result
    assert "Controller startup sync failed" not in caplog.text
    assert not controller._trading_disabled


@pytest.mark.asyncio
async def test_sync_aborts_in_flight_position_request(controller, monkeypatch):
    abort_event = asyncio.Event()
    request_started = asyncio.Event()
    disabled_reasons = []
    controller.set_sync_abort_event(abort_event)
    controller.broker_request_timeout = 10
    controller.sync_resync_delay = 0

    async def pending_positions():
        request_started.set()
        await asyncio.sleep(10)
        return []

    monkeypatch.setattr(
        controller,
        "disable_trading",
        lambda reason: disabled_reasons.append(reason),
    )
    monkeypatch.setattr(controller.ib, "isConnected", lambda: True)
    monkeypatch.setattr(controller.ib, "positions", lambda: [])
    monkeypatch.setattr(controller.ib, "reqPositionsAsync", pending_positions)

    sync_task = asyncio.create_task(controller.sync())
    await asyncio.wait_for(request_started.wait(), timeout=1)
    abort_event.set()

    result = await asyncio.wait_for(sync_task, timeout=1)

    assert result is SyncOutcome.ABORTED
    assert disabled_reasons == []


@pytest.mark.asyncio
async def test_sync_cancellation_cancels_inner_sync(controller, monkeypatch):
    inner_started = asyncio.Event()
    inner_cancelled = asyncio.Event()

    async def sync_body(*args):
        inner_started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            inner_cancelled.set()
            raise

    controller.set_sync_abort_event(asyncio.Event())
    monkeypatch.setattr(controller, "_sync", sync_body)

    sync_task = asyncio.create_task(controller.sync())
    await asyncio.wait_for(inner_started.wait(), timeout=1)
    sync_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await sync_task
    assert inner_cancelled.is_set()


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

    monkeypatch.setattr(
        controller,
        "disable_trading",
        lambda reason: disabled_reasons.append(reason),
    )
    monkeypatch.setattr(controller.ib, "isConnected", disconnected)
    monkeypatch.setattr(controller.ib, "positions", fail_position_read)
    monkeypatch.setattr(controller.ib, "reqPositionsAsync", fail_requested_positions)

    result = await controller.sync()

    assert result is SyncOutcome.FAILED
    assert connection_attempts == 1
    assert disabled_reasons == []


@pytest.mark.asyncio
async def test_sync_disconnected_does_not_release_hold(controller, monkeypatch):
    controller.set_hold()
    monkeypatch.setattr(controller.ib, "isConnected", lambda: False)
    monkeypatch.setattr(
        controller.ib,
        "positions",
        lambda: (_ for _ in ()).throw(
            AssertionError("broker state should not be queried")
        ),
    )

    result = await controller.sync()

    assert result is SyncOutcome.FAILED
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
    monkeypatch.setattr(controller.ib, "positions", lambda: [])
    monkeypatch.setattr(controller.ib, "reqPositionsAsync", pending_positions)

    coordinator = SyncCoordinator(controller)
    result = await coordinator.run()

    assert not result
    assert coordinator.request_restart
    assert not controller._trading_disabled


@pytest.mark.asyncio
async def test_broker_position_source_disagreement_requests_restart(
    controller, monkeypatch
):
    position = ibi.Position(
        account="DU123",
        contract=contract(),
        position=1,
        avgCost=1,
    )
    monkeypatch.setattr(controller.ib, "positions", lambda: [position])
    monkeypatch.setattr(controller.ib, "reqPositionsAsync", AsyncMock(return_value=[]))

    coordinator = SyncCoordinator(controller)
    result = await coordinator.run()

    assert not result
    assert coordinator.request_restart
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

    assert result is SyncOutcome.FAILED
    assert len(attempts) == controller.sync_max_attempts
    assert controller._trading_disabled


@pytest.mark.asyncio
async def test_sync_success_clears_restart_before_correction(controller, monkeypatch):
    restart_flags = []
    controller._restart_before_correction = True
    monkeypatch.setattr(controller.ib, "isConnected", lambda: True)

    async def successful_sync(self):
        restart_flags.append(self._restart_before_correction)
        return True

    monkeypatch.setattr(SyncCoordinator, "run", successful_sync)

    result = await controller.sync()

    assert result is SyncOutcome.OK
    assert restart_flags == [True]
    assert not controller._restart_before_correction


@pytest.mark.asyncio
async def test_completed_run_arms_restart_before_future_correction(
    controller, monkeypatch
):
    restart_flags = []
    controller._restart_before_correction = False
    monkeypatch.setattr(controller.ib, "isConnected", lambda: True)

    async def successful_sync(self):
        restart_flags.append(self._restart_before_correction)
        return True

    monkeypatch.setattr(SyncCoordinator, "run", successful_sync)

    result = await controller.run()

    assert result
    assert restart_flags == [False]
    assert controller._restart_before_correction


@pytest.mark.asyncio
async def test_unknown_broker_orders_are_cancelled_without_disabling_trading(
    controller, trade, monkeypatch
):
    cancelled = []
    set_broker_state(controller, monkeypatch, open_trades=(trade,))
    monkeypatch.setattr(controller.ib, "reqPositionsAsync", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        controller, "cancel", lambda broker_trade: cancelled.append(broker_trade)
    )
    controller.cancel_unknown_trades = True

    result = await SyncCoordinator(controller).run()

    assert not result
    assert cancelled == [trade]
    assert not controller._trading_disabled


@pytest.mark.asyncio
async def test_sync_coordinator_requests_restart_before_unknown_order_correction(
    controller, trade, monkeypatch
):
    cancelled = []
    set_broker_state(controller, monkeypatch, open_trades=(trade,))
    monkeypatch.setattr(controller.ib, "reqPositionsAsync", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        controller, "cancel", lambda broker_trade: cancelled.append(broker_trade)
    )
    controller.cancel_unknown_trades = True

    coordinator = SyncCoordinator(controller, restart_before_correction=True)
    result = await coordinator.run()

    assert not result
    assert coordinator.request_restart
    assert cancelled == []


@pytest.mark.asyncio
async def test_unknown_broker_orders_can_be_left_active_by_config(
    controller, trade, monkeypatch
):
    cancelled = []
    bracket_checked = []
    set_broker_state(controller, monkeypatch, open_trades=(trade,))
    monkeypatch.setattr(controller.ib, "reqPositionsAsync", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        controller, "cancel", lambda broker_trade: cancelled.append(broker_trade)
    )
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


@pytest.mark.asyncio
async def test_unknown_broker_orders_skip_bracket_correction(
    controller, trade, monkeypatch
):
    bracket_checked = []
    set_broker_state(controller, monkeypatch, open_trades=(trade,))
    monkeypatch.setattr(controller.ib, "isConnected", lambda: True)
    monkeypatch.setattr(controller.ib, "reqPositionsAsync", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        BracketSyncAction,
        "from_policy",
        staticmethod(lambda policy, controller: bracket_checked.append(policy)),
    )
    controller.cancel_unknown_trades = False
    controller._restart_before_correction = False

    result = await controller.sync()

    assert result is SyncOutcome.OK
    assert bracket_checked == []


@pytest.mark.asyncio
async def test_open_trade_refresh_does_not_skip_bracket_sync(
    controller, trade, monkeypatch
):
    old_trade = deepcopy(trade)
    info = save_active_order(controller, old_trade)
    broker_trade = deepcopy(old_trade)
    reconciled = []
    set_broker_state(controller, monkeypatch, open_trades=(broker_trade,))
    monkeypatch.setattr(controller.ib, "reqPositionsAsync", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        BracketSyncAction,
        "from_policy",
        staticmethod(lambda policy, received: reconciled.append((policy, received))),
    )

    result = await SyncCoordinator(controller).run()

    assert result
    assert info.trade is broker_trade
    assert reconciled == [(controller.missing_brackets, controller)]


@pytest.mark.asyncio
async def test_sync_coordinator_back_reports_done_trade_before_restart_gate(
    controller, trade, monkeypatch
):
    old_trade = deepcopy(trade)
    info = save_active_order(controller, old_trade)
    controller.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=old_trade.contract,
            position_id="episode",
        )
    )
    done_trade = deepcopy(trade)
    done_trade.order.orderId = 0
    done_trade.orderStatus.orderId = 0
    broker_position = ibi.Position(
        account="DU123",
        contract=old_trade.contract,
        position=1,
        avgCost=1,
    )
    set_broker_state(
        controller,
        monkeypatch,
        positions=(broker_position,),
        trades=(done_trade,),
    )
    monkeypatch.setattr(
        controller.ib,
        "reqPositionsAsync",
        AsyncMock(return_value=[broker_position]),
    )

    coordinator = SyncCoordinator(controller, restart_before_correction=True)
    result = await coordinator.run()

    assert not result
    assert not coordinator.request_restart
    assert await wait_for_condition(
        lambda: controller.book.position_state("alpha").quantity == 1
    )
    assert info.trade is done_trade


@pytest.mark.asyncio
async def test_sync_coordinator_prunes_unmatched_local_order_and_retries(
    controller, trade, monkeypatch
):
    old_trade = deepcopy(trade)
    info = save_active_order(controller, old_trade)
    set_broker_state(controller, monkeypatch)
    monkeypatch.setattr(controller.ib, "reqPositionsAsync", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        BracketSyncAction,
        "from_policy",
        staticmethod(
            lambda policy, controller: (_ for _ in ()).throw(
                AssertionError("bracket sync should not run")
            )
        ),
    )

    result = await SyncCoordinator(controller).run()

    assert not result
    assert controller.book.order_by_id(info.orderId) is info
    assert not info.active


@pytest.mark.asyncio
async def test_sync_coordinator_requests_restart_before_position_correction(
    controller, monkeypatch
):
    corrected = []
    controller.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=contract(),
            quantity=1,
        )
    )
    set_broker_state(controller, monkeypatch)
    monkeypatch.setattr(controller.ib, "reqPositionsAsync", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        SyncCoordinator,
        "handle_error_positions",
        lambda self, errors: corrected.append(errors),
    )

    coordinator = SyncCoordinator(controller, restart_before_correction=True)
    result = await coordinator.run()

    assert not result
    assert coordinator.request_restart
    assert corrected == []
    assert controller.book.position_state("alpha").quantity == 1


@pytest.mark.asyncio
async def test_sync_coordinator_allows_position_correction_after_restart(
    controller, monkeypatch
):
    corrected = []
    controller.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=contract(),
            quantity=1,
        )
    )
    set_broker_state(controller, monkeypatch)
    monkeypatch.setattr(controller.ib, "reqPositionsAsync", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        SyncCoordinator,
        "handle_error_positions",
        lambda self, errors: corrected.append(errors),
    )

    coordinator = SyncCoordinator(controller, restart_before_correction=False)
    result = await coordinator.run()

    assert not result
    assert not coordinator.request_restart
    assert corrected == [{contract(): 1.0}]


@pytest.mark.asyncio
async def test_sync_coordinator_raises_for_broken_bracket_state(
    controller, monkeypatch
):
    set_broker_state(controller, monkeypatch)
    monkeypatch.setattr(controller.ib, "reqPositionsAsync", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        BracketSyncAction,
        "from_policy",
        staticmethod(
            lambda policy, controller: (_ for _ in ()).throw(
                BracketSyncError("broken bracket state")
            )
        ),
    )

    with pytest.raises(SyncBrokenStateError, match="bracket sync failed"):
        await SyncCoordinator(controller).run()
    assert not controller._trading_disabled


@pytest.mark.asyncio
async def test_broker_position_request_timeout_is_unavailable(monkeypatch):
    ib = ibi.IB()
    monkeypatch.setattr(ib, "positions", lambda: [])
    monkeypatch.setattr(
        ib,
        "reqPositionsAsync",
        AsyncMock(side_effect=asyncio.TimeoutError),
    )

    assert not await verify_broker_position_source(ib, 0.01)


@pytest.mark.asyncio
async def test_clean_sync_releases_hold(controller_runtime, monkeypatch):
    runtime, controller, _ = controller_runtime
    monkeypatch.setattr(runtime.ib, "isConnected", lambda: True)
    monkeypatch.setattr(runtime.ib, "positions", lambda: [])
    monkeypatch.setattr(runtime.ib, "openTrades", lambda: [])
    monkeypatch.setattr(runtime.ib, "trades", lambda: [])
    monkeypatch.setattr(runtime.ib, "fills", lambda: [])
    monkeypatch.setattr(runtime.ib, "reqPositionsAsync", AsyncMock(return_value=[]))

    outcome = await controller.sync()

    assert outcome is SyncOutcome.OK
    assert controller._hold is False


@pytest.mark.asyncio
async def test_run_disables_trading_when_book_restore_fails(
    controller_runtime, monkeypatch
):
    runtime, controller, _ = controller_runtime
    controller.cold_start = False
    monkeypatch.setattr(
        runtime.book,
        "read_from_store",
        AsyncMock(side_effect=RuntimeError("store failed")),
    )
    sync = AsyncMock()
    monkeypatch.setattr(controller, "sync", sync)

    assert not await controller.run()
    assert controller._trading_disabled
    sync.assert_not_awaited()


def test_rejections_are_scoped_by_execution_model(controller_runtime):
    runtime, controller, _ = controller_runtime
    trade = controller.trade(
        contract(),
        ibi.MarketOrder("BUY", 1),
        role=StandardOrderRole.OPEN,
        execution_model_name="brackets",
    )

    controller.onErrEvent(
        trade.order.orderId,
        201,
        "rejected",
        contract(),
    )

    assert runtime.book._rejected_orders["brackets"] == 1
