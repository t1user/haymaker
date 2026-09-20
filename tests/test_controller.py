import asyncio
import datetime as dt
import logging
from functools import partial
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import ib_insync as ibi
import pytest
from helpers import wait_for_condition

from haymaker.async_wrappers import QueueProcessingError
from haymaker.book import Book, OrderInfo, PositionState
from haymaker.components import (
    BracketExecutionModel,
    FixedStop,
    PositionTarget,
    StandardOrderRole,
)
from haymaker.controller import Controller
from haymaker.controller.controller import ControllerError, SyncOutcome
from haymaker.controller.reset import Reset
from haymaker.controller.sync_brackets import (
    BracketSync,
    BracketSyncAction,
    BracketSyncError,
)
from haymaker.controller.sync_coordinator import (
    BrokerPositionStatus,
    SyncBrokenStateError,
    SyncCoordinator,
    verify_broker_position_source,
)
from haymaker.supervisor.codes import SUPERVISOR_OWNED_BROKER_CODES
from haymaker.trader import Trader


@pytest.mark.parametrize(
    "metadata",
    [
        {"execution_model_name": ""},
        {"execution_model_name": 5},
        {"role": ""},
        {"role": None},
        {"source_key": ""},
        {"source_key": 5},
        {"position_id": ""},
        {"position_id": 5},
        {"params": []},
        {"params": [1]},
    ],
)
def test_invalid_submission_metadata_never_reaches_broker(controller_runtime, metadata):
    """Both ordinary and emergency registered submissions validate before sending."""
    runtime, controller, trader = controller_runtime
    options = {"role": "OPEN", "execution_model_name": "brackets", **metadata}
    for submit in (controller.trade, controller._submit_registered_trade):
        with pytest.raises((TypeError, ValueError)):
            submit(contract(), ibi.MarketOrder("BUY", 1), **options)
    assert trader.trades == []
    assert runtime.book.orders.query() == ()


def test_health_checks_isolate_callables_and_reset_after_recovery(controller, caplog):
    """Partials, instances, duplicate names and exceptions report independently."""
    healthy = False

    class Checker:
        __hash__ = None

        def __call__(self):
            return healthy

    def raises():
        raise RuntimeError("broken checker")

    later = Mock(return_value=True)
    checks = [
        partial(bool, False),
        Checker(),
        lambda: healthy,
        lambda: healthy,
        raises,
        later,
    ]
    for check in checks:
        controller.set_health_check(check)
    controller.run_health_check()
    assert caplog.text.count("Health check failure for checker") == 5
    assert "Health checker raised" in caplog.text
    later.assert_called_once()
    caplog.clear()
    controller.run_health_check()
    assert caplog.text == ""
    healthy = True
    controller.run_health_check()
    healthy = False
    controller.run_health_check()
    assert caplog.text.count("Health check failure for checker") == 3


def test_account_identity_cannot_change_across_reconnects(controller, monkeypatch):
    monkeypatch.setattr(controller.ib, "managedAccounts", lambda: ["a"])
    controller.verify_broker_account(())
    controller.suspend_broker_work()
    monkeypatch.setattr(controller.ib, "managedAccounts", lambda: ["b"])
    with pytest.raises(SyncBrokenStateError, match="one account/subaccount"):
        controller.verify_broker_account(())


async def test_registered_unknown_order_still_prevents_startup(
    controller, trade, monkeypatch
):
    """Retaining unknown fill evidence must not turn it into an owned live order."""
    save_active_order(controller, trade, role=StandardOrderRole.UNKNOWN)
    set_broker_state(controller, monkeypatch, open_trades=(trade,))
    monkeypatch.setattr(controller.ib, "reqPositionsAsync", AsyncMock(return_value=[]))
    with pytest.raises(SyncBrokenStateError, match="Unknown broker orders"):
        await SyncCoordinator(controller).run()


@pytest.mark.parametrize("failure", [asyncio.TimeoutError, RuntimeError])
@pytest.mark.parametrize("attempts", [1, 3])
async def test_final_request_failure_always_requests_supervisor(
    controller, atom_runtime, monkeypatch, failure, attempts
):
    """Local retry exhaustion never turns unavailable broker state into a latch."""
    controller.sync_max_attempts = attempts
    controller.sync_resync_delay = 0
    set_broker_state(controller, monkeypatch)
    monkeypatch.setattr(controller.ib, "isConnected", lambda: True)
    requested = [ibi.Position("test", contract(), 1, 100)]
    monkeypatch.setattr(
        controller.ib,
        "reqPositionsAsync",
        AsyncMock(side_effect=[requested] * (attempts - 1) + [failure()]),
    )
    assert await controller.sync() is SyncOutcome.ABORTED
    assert len(atom_runtime.restart_requests) == 1
    assert not controller._trading_disabled


async def test_sync_serializes_requests_and_cancelled_waiter(controller, monkeypatch):
    """A cancelled timer waiter cannot orphan another caller's IB request."""
    set_broker_state(controller, monkeypatch)
    monkeypatch.setattr(controller.ib, "isConnected", lambda: True)
    started, release = asyncio.Event(), asyncio.Event()
    requests = 0

    async def positions():
        nonlocal requests
        requests += 1
        started.set()
        await release.wait()
        return []

    monkeypatch.setattr(controller.ib, "reqPositionsAsync", positions)
    first = asyncio.create_task(controller.sync())
    await started.wait()
    cancelled = asyncio.create_task(controller.sync())
    last = asyncio.create_task(controller.sync())
    await asyncio.sleep(0)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    assert requests == 1
    release.set()
    assert await first is SyncOutcome.OK
    assert await last is SyncOutcome.OK
    assert requests == 2


@pytest.mark.parametrize("completed_history", [False, True])
async def test_missing_partial_order_requires_terminal_evidence(
    controller, trade, monkeypatch, completed_history
):
    """Executions are accounted even when the remainder's status is unresolved."""
    trade.order.totalQuantity = 3
    info = save_active_order(controller, trade)
    execution = fill(trade)
    terminal = deepcopy(trade)
    terminal.orderStatus.status = ibi.OrderStatus.Cancelled
    terminal.fills = [execution]
    set_broker_state(
        controller,
        monkeypatch,
        positions=(ibi.Position("test", trade.contract, 1, 100),),
        fills=(execution,),
    )
    monkeypatch.setattr(controller.ib, "isConnected", lambda: True)
    monkeypatch.setattr(
        controller.ib,
        "reqPositionsAsync",
        AsyncMock(return_value=controller.ib.positions()),
    )
    monkeypatch.setattr(
        controller.ib,
        "reqCompletedOrdersAsync",
        AsyncMock(return_value=[terminal] if completed_history else []),
    )
    controller.sync_resync_delay = 0
    assert await controller.sync() is (
        SyncOutcome.OK if completed_history else SyncOutcome.FAILED
    )
    assert controller.book.positions.quantity(trade.contract) == 1
    assert len(info.fills) == 1
    assert info.trade.isDone() == completed_history
    assert controller.ib.reqCompletedOrdersAsync.await_count == 1


async def test_cancelled_session_trade_recovers_separate_execution_history(
    controller, trade, monkeypatch
):
    """Session cancellation and execution history are independent evidence."""
    trade.order.totalQuantity = 3
    info = save_active_order(controller, trade)
    execution = fill(trade)
    terminal = deepcopy(trade)
    terminal.orderStatus.status = ibi.OrderStatus.Cancelled
    set_broker_state(
        controller,
        monkeypatch,
        positions=(ibi.Position("test", trade.contract, 1, 100),),
        trades=(terminal,),
        fills=(execution,),
    )
    monkeypatch.setattr(
        controller.ib,
        "reqPositionsAsync",
        AsyncMock(return_value=controller.ib.positions()),
    )
    assert not await SyncCoordinator(controller).run()
    assert len(info.fills) == 1
    assert info.trade.isDone()
    assert controller.book.positions.quantity(trade.contract) == 1


async def test_snapshot_compares_concrete_identity_and_rejects_multiple_accounts(
    controller, monkeypatch
):
    """Equal display symbols cannot conceal different contracts or accounts."""
    left, right = contract(), contract()
    right.conId += 1
    monkeypatch.setattr(
        controller.ib, "positions", lambda: [ibi.Position("a", left, 1, 100)]
    )
    request = AsyncMock(return_value=[ibi.Position("a", right, 1, 100)])
    monkeypatch.setattr(controller.ib, "reqPositionsAsync", request)
    assert (
        await verify_broker_position_source(controller.ib, 1)
    ).status is BrokerPositionStatus.SNAPSHOT_DISAGREEMENT
    request.return_value = [ibi.Position("b", left, 1, 100)]
    with pytest.raises(SyncBrokenStateError, match="one account/subaccount"):
        await verify_broker_position_source(controller.ib, 1)


async def test_multisource_correction_fails_explicitly_after_pruning(
    controller, trade, monkeypatch, caplog
):
    """Real retry orchestration cannot infer ownership from a vanished close."""
    for source, quantity in (("alpha", 3), ("beta", 2)):
        controller.book.update_position(
            PositionState(
                source_key=source,
                execution_model_name="brackets",
                contract=trade.contract,
                quantity=quantity,
                target_quantity=quantity,
            )
        )
    info = save_active_order(controller, trade, role=StandardOrderRole.CLOSE)
    snapshot = [ibi.Position("test", trade.contract, 2, 100)]
    set_broker_state(controller, monkeypatch, positions=tuple(snapshot))
    monkeypatch.setattr(controller.ib, "isConnected", lambda: True)
    requested = AsyncMock(return_value=snapshot)
    monkeypatch.setattr(controller.ib, "reqPositionsAsync", requested)
    controller.position_mismatch_policy = "correct"
    controller._restart_before_correction = False
    controller.sync_resync_delay = 0
    assert await controller.sync() is SyncOutcome.FAILED
    assert requested.await_count == 2
    assert info.trade.isDone()
    assert controller.book.positions.for_source("alpha").quantity == 3
    assert controller.book.positions.for_source("beta").quantity == 2
    assert "Ambiguous source attribution" in caplog.text


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
            "startup": {"zero": True},
            "sync_frequency": 10,
        },
        trader=FakeTrader(),
    )

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


@pytest.mark.parametrize("policy", ["fail", "correct"])
def test_position_mismatch_policy_from_mapping(atom_runtime, policy):
    controller = Controller.from_mapping(
        {"position_mismatch_policy": policy}, trader=FakeTrader()
    )
    assert controller.position_mismatch_policy == policy


@pytest.mark.parametrize("policy", ["warn", "ignore", "", True, None])
def test_invalid_position_mismatch_policy_is_rejected(atom_runtime, policy):
    with pytest.raises(ControllerError, match="position_mismatch_policy"):
        Controller.from_mapping(
            {"position_mismatch_policy": policy}, trader=FakeTrader()
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
    assert controller.book.orders.by_id(123) is None


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


def test_unaccepted_paper_trading_disclaimer_is_logged_as_error(controller, caplog):
    caplog.set_level(logging.ERROR)

    controller.onErrEvent(
        -1,
        10141,
        "Paper trading disclaimer must first be accepted for API connection.",
        ibi.Contract(),
    )

    assert "Broker message 10141: Paper trading disclaimer" in caplog.text
    assert caplog.records[-1].levelno == logging.ERROR


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
    assert runtime.book.orders.rejection_count("brackets") == 1


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
    info = runtime.book.orders.by_id(result.order.orderId)
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
    assert runtime.book.orders.active() == ()


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

    assert runtime.book.positions.for_source("alpha").quantity == 1
    assert len(runtime.book.orders.by_id(trade.order.orderId).fills) == 1


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
    assert runtime.book.positions.for_source("alpha").quantity == 1


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

    assert runtime.book.orders.by_id(0) is None
    assert runtime.book.orders.by_perm_id(999001) is None
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

    info = runtime.book.orders.by_id(-1)
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

    info = runtime.book.orders.by_id(trade.order.orderId)
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

    with caplog.at_level(logging.DEBUG):
        await controller.onCommissionReport(trade, execution, report)

    assert runtime.book.orders.by_id(0) is None
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

    BracketSyncAction.from_policy("ignore", controller).sync()


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
        BracketSyncAction.from_policy("remove", controller).sync()

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

    BracketSyncAction.from_policy("remove", controller).sync()

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
        BracketSyncAction.from_policy("remove", controller).sync()

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

    BracketSyncAction.from_policy("remove", controller).sync()

    controller.cancel.assert_not_called()


@pytest.mark.asyncio
async def test_reset_waits_for_cancellation_before_logical_close(
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
    controller.ib.reqPositionsAsync = AsyncMock(return_value=[])
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

    completed = await Reset(controller).run()

    assert completed is True
    controller.trade.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_during_liquidation", [False, True])
async def test_reset_liquidates_when_cancellation_does_not_complete(
    controller_runtime, monkeypatch, cancel_during_liquidation, caplog
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
    controller.ib.reqPositionsAsync = AsyncMock(return_value=[])
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

    def close(*args, **kwargs):
        """Cancellation can finish after its grace period, while closing."""
        assert not protective.isDone()
        if cancel_during_liquidation:
            protective.orderStatus.status = ibi.OrderStatus.Cancelled
        return liquidation

    controller.trade = Mock(side_effect=close)
    controller.reset = True
    controller.sync = AsyncMock(return_value=SyncOutcome.OK)
    runtime.book.clear_state = Mock(wraps=runtime.book.clear_state)
    monkeypatch.setattr(Reset, "cancellation_timeout", 0)

    if cancel_during_liquidation:
        assert await controller.run() is SyncOutcome.OK
        runtime.book.clear_state.assert_called_once()
        assert controller.reset is False
    else:
        await assert_reset_fails(controller)
        assert "orderId=77 status=Submitted filled=0.0 remaining=1" in caplog.text
    controller.trade.assert_called_once()
    assert controller.trade.call_args.kwargs["role"] == (StandardOrderRole.LIQUIDATION)


@pytest.mark.asyncio
async def test_reset_does_not_hide_residual_behind_flat_state(
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
    controller.ib.reqPositionsAsync = AsyncMock(
        side_effect=[controller.ib.positions(), []]
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

    completed = await Reset(controller).run()

    assert completed is True
    assert controller.trade.call_args.kwargs["role"] == (StandardOrderRole.LIQUIDATION)


@pytest.mark.asyncio
async def test_run_keeps_book_when_explicit_reset_fails(controller_runtime):
    """Failed reset disables trading without discarding recovery state."""

    runtime, controller, _ = controller_runtime
    controller.reset = True
    controller.sync = AsyncMock(return_value=SyncOutcome.OK)
    controller.execute_reset = AsyncMock(return_value=False)
    runtime.book.clear_state = Mock()

    completed = await controller.run()

    assert completed is SyncOutcome.FAILED
    assert controller._trading_disabled is True
    assert controller.reset is True
    runtime.book.clear_state.assert_not_called()


@pytest.fixture
def reset_runtime(controller_runtime):
    """Seed recovery state and isolate startup sync from the real reset path."""
    runtime, controller, trader = controller_runtime
    runtime.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=contract(),
            quantity=2,
            position_id="episode",
            target_contract=contract(),
            target_quantity=2,
            bracket_inputs={"atr": 5},
            blocked_direction=-1,
        )
    )
    controller.reset = True
    controller.sync = AsyncMock(return_value=SyncOutcome.OK)
    runtime.book.clear_state = Mock(wraps=runtime.book.clear_state)
    controller.ib.openTrades = Mock(return_value=[])
    controller.ib.positions = Mock(return_value=[])
    controller.ib.reqPositionsAsync = AsyncMock(return_value=[])
    controller.trade = Mock(
        return_value=ibi.Trade(
            contract=contract(),
            order=ibi.MarketOrder("SELL", 2, orderId=78),
            orderStatus=ibi.OrderStatus(
                orderId=78, status=ibi.OrderStatus.Filled, filled=2, remaining=0
            ),
        )
    )
    return runtime, controller, trader


async def assert_reset_fails(controller: Controller) -> None:
    """Verify a failed reset preserves recovery state and the one-run flag."""
    before = dict(controller.book.positions.source_states())

    assert await controller.run() is SyncOutcome.FAILED

    assert controller._trading_disabled is True
    assert controller.reset is True
    cast(Mock, controller.book.clear_state).assert_not_called()
    assert controller.book.positions.source_states() == before


@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize(
    "status, filled, remaining",
    [
        (ibi.OrderStatus.Cancelled, 0, 2),
        (ibi.OrderStatus.ApiCancelled, 0, 2),
        (ibi.OrderStatus.Inactive, 0, 2),
        (ibi.OrderStatus.Cancelled, 1, 1),
        (ibi.OrderStatus.Submitted, 1, 1),
        (ibi.OrderStatus.Submitted, 0, 2),
        (ibi.OrderStatus.Filled, 1, 1),
        (ibi.OrderStatus.Filled, 1, 0),
        (ibi.OrderStatus.Filled, 2, 1),
    ],
)
async def test_reset_requires_full_liquidation_fill(
    reset_runtime, monkeypatch, caplog, residual, status, filled, remaining
):
    """Terminal status, partial fills and timeouts never authorize state clearing."""
    runtime, controller, _ = reset_runtime
    liquidation = controller.trade.return_value
    liquidation.orderStatus.status = status
    liquidation.orderStatus.filled = filled
    liquidation.orderStatus.remaining = remaining
    if residual:
        state = runtime.book.positions.for_source("alpha")
        runtime.book.update_position(replace(state, quantity=0))
        controller.ib.reqPositionsAsync.side_effect = [
            [ibi.Position("test", contract(), 2, 100)],
            [],
        ]
    monkeypatch.setattr("haymaker.controller.reset.asyncio.sleep", AsyncMock())

    await assert_reset_fails(controller)

    controller.trade.assert_called_once()
    assert "orderId=78" in caplog.text
    assert f"status={status} filled={filled} remaining={remaining}" in caplog.text


@pytest.mark.parametrize("residual", [False, True])
async def test_reset_fails_on_suppressed_liquidation(reset_runtime, residual, caplog):
    """An empty list of submitted trades cannot mask a required suppression."""
    runtime, controller, _ = reset_runtime
    controller.trade.return_value = None
    if residual:
        state = runtime.book.positions.for_source("alpha")
        runtime.book.update_position(replace(state, quantity=0))
        controller.ib.reqPositionsAsync.side_effect = [
            [ibi.Position("test", contract(), 2, 100)],
            [],
        ]

    await assert_reset_fails(controller)

    controller.trade.assert_called_once()
    assert "liquidation suppressed: contract=" in caplog.text
    assert "ESM6" in caplog.text
    if not residual:
        assert "source=alpha" in caplog.text


@pytest.mark.parametrize("logical_filled", [False, True])
async def test_reset_discovers_residuals_from_requested_positions(
    reset_runtime, logical_filled
):
    """Fresh broker-only holdings are closed even when the cache is empty."""
    runtime, controller, _ = reset_runtime
    residual_contract = ibi.Future(conId=2, symbol="NQ", localSymbol="NQM6")
    controller.ib.reqPositionsAsync.side_effect = [
        [ibi.Position("test", residual_contract, -3, 100)],
        [],
    ]
    logical_trade = controller.trade.return_value
    if not logical_filled:
        logical_trade.orderStatus.status = ibi.OrderStatus.Cancelled
        logical_trade.orderStatus.filled = 0
        logical_trade.orderStatus.remaining = 2
    residual_trade = ibi.Trade(
        contract=residual_contract,
        order=ibi.MarketOrder("BUY", 3, orderId=79),
        orderStatus=ibi.OrderStatus(
            orderId=79, status=ibi.OrderStatus.Filled, filled=3, remaining=0
        ),
    )
    controller.trade.side_effect = [logical_trade, residual_trade]

    if logical_filled:
        assert await controller.run() is SyncOutcome.OK
        runtime.book.clear_state.assert_called_once()
        assert runtime.book.positions.source_states() == {}
        assert controller.reset is False
        assert controller._trading_disabled is False
    else:
        await assert_reset_fails(controller)

    assert controller.trade.call_count == 2
    args, kwargs = controller.trade.call_args
    assert args[0] == residual_contract
    assert args[1].action == "BUY"
    assert args[1].totalQuantity == 3
    assert kwargs["role"] == StandardOrderRole.LIQUIDATION
    controller.ib.reqPositionsAsync.assert_awaited()
    assert controller.ib.reqPositionsAsync.await_count == 2


async def test_reset_ignores_stale_cached_residuals_when_request_succeeds(
    reset_runtime,
):
    """Successful empty discovery prevents a stale cache from opening new exposure."""
    runtime, controller, _ = reset_runtime
    controller.ib.positions.return_value = [
        ibi.Position("test", ibi.Future(conId=2, symbol="NQ"), 3, 100)
    ]

    assert await controller.run() is SyncOutcome.OK

    controller.trade.assert_called_once()
    runtime.book.clear_state.assert_called_once()
    assert controller.reset is False


@pytest.mark.parametrize("final_flat", [False, True])
async def test_reset_uses_fresh_snapshot_without_double_liquidation(
    reset_runtime, final_flat, caplog
):
    """An accepted logical close owns its contract even if a snapshot is non-flat."""
    runtime, controller, _ = reset_runtime
    positions = [ibi.Position("test", contract(), 2, 100)]
    controller.ib.reqPositionsAsync.side_effect = [
        positions,
        [ibi.Position("test", contract(), 0, 100)] if final_flat else positions,
    ]
    # Exercise both disagreement directions: stale non-flat and stale empty caches.
    controller.ib.positions.return_value = positions if final_flat else []

    if final_flat:
        assert await controller.run() is SyncOutcome.OK
        runtime.book.clear_state.assert_called_once()
        assert controller.reset is False
    else:
        await assert_reset_fails(controller)
        assert "broker positions remain non-flat" in caplog.text
        assert "position=2" in caplog.text
    controller.trade.assert_called_once()


@pytest.mark.parametrize("failure", ["timeout", "exception"])
async def test_reset_fails_when_final_position_request_is_unavailable(
    reset_runtime, failure, caplog
):
    """Even full fills and an empty cache need successful final broker authority."""
    _, controller, _ = reset_runtime
    controller.broker_request_timeout = 0.01
    calls = 0

    async def positions():
        """Allow discovery, then fail or stall the verification request."""
        nonlocal calls
        calls += 1
        if calls == 1:
            return []
        if failure == "exception":
            raise ConnectionError("positions unavailable")
        await asyncio.Future()

    controller.ib.reqPositionsAsync = positions

    await assert_reset_fails(controller)

    assert calls == 2
    assert "Reset position request" in caplog.text
    assert "final flat verification" in caplog.text


@pytest.mark.parametrize("final_available", [False, True])
async def test_reset_uses_cache_only_for_best_effort_residual_close(
    reset_runtime, final_available
):
    """Cached residuals can be attempted, but only a later request proves flatness."""
    runtime, controller, _ = reset_runtime
    state = runtime.book.positions.for_source("alpha")
    runtime.book.update_position(replace(state, quantity=0))
    controller.ib.positions.return_value = [ibi.Position("test", contract(), 2, 100)]
    controller.ib.reqPositionsAsync.side_effect = [
        ConnectionError("discovery unavailable"),
        [] if final_available else ConnectionError("verification unavailable"),
    ]

    if final_available:
        assert await controller.run() is SyncOutcome.OK
        runtime.book.clear_state.assert_called_once()
        assert controller.reset is False
    else:
        await assert_reset_fails(controller)
    controller.trade.assert_called_once()


async def test_reset_keeps_suppression_failure_after_residual_fills(reset_runtime):
    """A best-effort residual fill does not erase a failed logical submission."""
    _, controller, _ = reset_runtime
    controller.trade.side_effect = [None, controller.trade.return_value]
    controller.ib.reqPositionsAsync.side_effect = [
        [ibi.Position("test", contract(), 2, 100)],
        [],
    ]

    await assert_reset_fails(controller)

    assert controller.trade.call_count == 2


@pytest.mark.parametrize("exception", [ConnectionError, QueueProcessingError])
async def test_reset_does_not_swallow_submission_failure(reset_runtime, exception):
    """Broker and persistence submission exceptions remain fail-stop exceptions."""
    runtime, controller, _ = reset_runtime
    before = dict(runtime.book.positions.source_states())
    controller.trade.side_effect = exception("submission unavailable")

    with pytest.raises(exception, match="submission unavailable"):
        await controller.run()

    runtime.book.clear_state.assert_not_called()
    assert runtime.book.positions.source_states() == before
    assert controller.reset is True


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
            target_contract=target.contract,
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
            target_contract=target.contract,
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


@pytest.mark.parametrize("quantity", [-2, 2])
@pytest.mark.parametrize(
    "policy", ["normal", "disabled", "closed_market", "rejections"]
)
def test_emergency_reset_liquidation_is_registered_with_episode_attribution(
    controller_runtime, monkeypatch, quantity, policy
):
    """Emergency reset bypasses submission policy but retains Book attribution."""
    runtime, controller, trader = controller_runtime
    runtime.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=contract(),
            quantity=quantity,
            position_id="episode",
        )
    )
    position = ibi.Position("account", contract(), quantity, 100)
    monkeypatch.setattr(runtime.ib, "positions", lambda: [position])
    if policy == "disabled":
        controller.disable_trading("already disabled")
    elif policy == "closed_market":
        monkeypatch.setattr(controller, "verify_market_open", lambda contract: False)
    elif policy == "rejections":
        for _ in range(runtime.book.max_rejected_orders):
            runtime.book.orders.register_rejection("brackets")

    def cancel_orders():
        """Confirm trading is disabled before the first broker request."""
        assert controller._trading_disabled is True

    cancel = Mock(side_effect=cancel_orders)
    monkeypatch.setattr(runtime.ib, "reqGlobalCancel", cancel)
    clear_state = Mock(wraps=runtime.book.clear_state)
    monkeypatch.setattr(runtime.book, "clear_state", clear_state)

    controller.execute_emergency_reset()

    cancel.assert_called_once_with()
    clear_state.assert_not_called()
    assert controller._trading_disabled is True
    assert len(trader.trades) == 1
    trade = trader.trades[0]
    info = runtime.book.orders.by_id(trade.order.orderId)
    assert trade.order.action == ("BUY" if quantity < 0 else "SELL")
    assert trade.order.totalQuantity == abs(quantity)
    assert info.role == StandardOrderRole.LIQUIDATION
    assert info.execution_model_name == "brackets"
    assert info.source_key == "alpha"
    assert info.position_id == "episode"


@pytest.mark.parametrize("entrypoint", ["direct", "startup"])
@pytest.mark.parametrize("failure", ["cancel", "submission", "persistence"])
async def test_emergency_reset_failure_leaves_trading_disabled(
    controller_runtime, monkeypatch, entrypoint, failure
):
    """Every entry point disables trading even when broker or persistence work fails."""
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
    before = dict(runtime.book.positions.source_states())
    monkeypatch.setattr(
        runtime.ib, "positions", lambda: [ibi.Position("test", contract(), 2, 100)]
    )
    cancel = Mock()
    monkeypatch.setattr(runtime.ib, "reqGlobalCancel", cancel)
    clear_state = Mock(wraps=runtime.book.clear_state)
    monkeypatch.setattr(runtime.book, "clear_state", clear_state)
    if failure == "cancel":
        cancel.side_effect = RuntimeError("reset request failed")
    elif failure == "submission":
        monkeypatch.setattr(
            trader, "trade", Mock(side_effect=RuntimeError("reset request failed"))
        )
    else:
        monkeypatch.setattr(
            runtime.book,
            "check_writable",
            Mock(side_effect=QueueProcessingError("reset request failed")),
        )

    with pytest.raises(RuntimeError, match="reset request failed"):
        if entrypoint == "startup":
            controller.nuke = True
            await controller.run()
        else:
            controller.execute_emergency_reset()

    assert controller._trading_disabled is True
    assert trader.trades == []
    clear_state.assert_not_called()
    assert runtime.book.positions.source_states() == before


@pytest.mark.asyncio
async def test_sync_timeout_requests_supervisor_restart(
    controller, atom_runtime, monkeypatch
):
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

    assert result is SyncOutcome.ABORTED
    assert disabled_reasons == []
    assert atom_runtime.restart_requests == [
        "controller reconciliation requires fresh broker state"
    ]


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

    assert result is SyncOutcome.ABORTED
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
async def test_sync_disconnected_leaves_broker_work_suspended(controller, monkeypatch):
    controller.suspend_broker_work()
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
    assert not controller.broker_ready
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
async def test_broker_position_source_disagreement_retries_without_restart(
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
    assert not coordinator.request_restart
    assert not controller._trading_disabled


@pytest.mark.asyncio
async def test_fill_between_position_snapshots_retries_and_converges(
    controller, atom_runtime, monkeypatch
):
    target_contract = contract()
    initial_state = PositionState(
        source_key="alpha",
        execution_model_name="brackets",
        contract=target_contract,
        quantity=0,
        target_quantity=1,
        target_created_at=datetime.now(timezone.utc),
        position_id="episode",
        bracket_inputs={"atr": 2},
    )
    controller.book.update_position(initial_state)
    broker_position = ibi.Position(
        account="DU123",
        contract=target_contract,
        position=1,
        avgCost=1,
    )
    cached_positions: list[ibi.Position] = []
    request_count = 0

    set_broker_state(controller, monkeypatch)
    monkeypatch.setattr(controller.ib, "isConnected", lambda: True)
    monkeypatch.setattr(
        controller.ib,
        "positions",
        lambda: list(cached_positions),
    )

    async def requested_positions():
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            cached_positions.append(broker_position)
            controller.book.update_position(replace(initial_state, quantity=1))
        return [broker_position]

    monkeypatch.setattr(controller.ib, "reqPositionsAsync", requested_positions)
    controller.sync_max_attempts = 2
    controller.sync_resync_delay = 0

    outcome = await controller.sync()

    assert outcome is SyncOutcome.OK
    assert request_count == 2
    assert atom_runtime.restart_requests == []
    assert not controller._trading_disabled
    assert controller.book.positions.for_source("alpha").quantity == 1


@pytest.mark.asyncio
async def test_persistent_position_snapshot_disagreement_fails_without_restart(
    controller, atom_runtime, monkeypatch
):
    position = ibi.Position(
        account="DU123",
        contract=contract(),
        position=1,
        avgCost=1,
    )
    set_broker_state(controller, monkeypatch, positions=(position,))
    monkeypatch.setattr(controller.ib, "isConnected", lambda: True)
    monkeypatch.setattr(
        controller.ib,
        "reqPositionsAsync",
        AsyncMock(return_value=[]),
    )
    controller.sync_max_attempts = 2
    controller.sync_resync_delay = 0

    outcome = await controller.sync()

    assert outcome is SyncOutcome.FAILED
    assert atom_runtime.restart_requests == []
    assert controller._trading_disabled


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
async def test_sync_logs_every_attempt(controller, monkeypatch, caplog):
    controller.sync_max_attempts = 3
    controller.sync_resync_delay = 0
    monkeypatch.setattr(controller.ib, "isConnected", lambda: True)

    async def retryable_sync_failure(self):
        return False

    monkeypatch.setattr(SyncCoordinator, "run", retryable_sync_failure)
    caplog.set_level(logging.DEBUG, logger="haymaker.controller.controller")

    await controller.sync()

    assert [
        record.getMessage()
        for record in caplog.records
        if record.name == "haymaker.controller.controller"
        and record.getMessage().startswith("Sync attempt ")
    ] == ["Sync attempt 1/3", "Sync attempt 2/3", "Sync attempt 3/3"]


@pytest.mark.asyncio
async def test_sync_routes_restart_through_runtime_callback(
    controller, atom_runtime, monkeypatch
):
    async def restart_required(self):
        self.request_restart = True
        return False

    def fail_direct_disconnect():
        raise AssertionError("Controller must not disconnect the broker socket")

    controller.sync_max_attempts = 2
    controller.sync_resync_delay = 0
    monkeypatch.setattr(controller.ib, "isConnected", lambda: True)
    monkeypatch.setattr(controller.ib, "disconnect", fail_direct_disconnect)
    monkeypatch.setattr(SyncCoordinator, "run", restart_required)

    result = await controller.sync()

    assert result is SyncOutcome.ABORTED
    assert atom_runtime.restart_requests == [
        "controller reconciliation requires fresh broker state"
    ]
    assert not controller._restart_before_correction
    assert not controller._trading_disabled


@pytest.mark.asyncio
async def test_sync_fails_closed_when_supervisor_rejects_restart(
    controller, atom_runtime, monkeypatch
):
    disabled_reasons = []

    async def restart_required(self):
        self.request_restart = True
        return False

    controller.sync_max_attempts = 2
    controller.sync_resync_delay = 0
    monkeypatch.setattr(controller.ib, "isConnected", lambda: True)
    monkeypatch.setattr(SyncCoordinator, "run", restart_required)
    monkeypatch.setattr(atom_runtime, "request_restart", lambda reason: False)
    monkeypatch.setattr(
        controller, "disable_trading", lambda reason: disabled_reasons.append(reason)
    )

    result = await controller.sync()

    assert result is SyncOutcome.FAILED
    assert disabled_reasons == ["supervisor restart request rejected"]
    assert controller._restart_before_correction


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
    assert controller.book.orders.by_id(trade.order.orderId) is None
    assert controller.book.positions.quantity(trade.contract) == 0


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

    with pytest.raises(SyncBrokenStateError, match="Unknown broker orders"):
        await SyncCoordinator(controller).run()
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

    assert result is SyncOutcome.FAILED
    assert bracket_checked == []


@pytest.mark.asyncio
async def test_open_trade_refresh_does_not_skip_bracket_sync(
    controller, trade, monkeypatch
):
    old_trade = deepcopy(trade)
    info = save_active_order(controller, old_trade)
    broker_trade = deepcopy(old_trade)
    selected = Mock(return_value=Mock(run=AsyncMock()))
    set_broker_state(controller, monkeypatch, open_trades=(broker_trade,))
    monkeypatch.setattr(controller.ib, "reqPositionsAsync", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        BracketSyncAction,
        "from_policy",
        selected,
    )

    result = await SyncCoordinator(controller).run()

    assert result
    assert info.trade is broker_trade
    selected.assert_called_once_with(controller.missing_brackets, controller)
    selected.return_value.run.assert_awaited_once()


@pytest.mark.parametrize(
    ("role", "initial_quantity", "broker_quantity", "source_key"),
    [
        (StandardOrderRole.OPEN, 0, 1, "alpha"),
        (StandardOrderRole.CLOSE, 3, 2, "alpha"),
        (StandardOrderRole.TARGET_ADJUSTMENT, 0, 1, None),
    ],
)
@pytest.mark.asyncio
async def test_sync_back_reports_offline_partial_fill_for_open_order(
    controller,
    monkeypatch,
    role,
    initial_quantity,
    broker_quantity,
    source_key,
):
    """Reconnect accounts partial executions before comparing positions."""

    target_contract = contract()
    if source_key is not None:
        controller.book.update_position(
            PositionState(
                source_key=source_key,
                execution_model_name="brackets",
                contract=target_contract,
                quantity=initial_quantity,
                target_quantity=3 if role == StandardOrderRole.OPEN else 0,
                target_created_at=datetime.now(timezone.utc),
                position_id="episode",
            )
        )
    action = "SELL" if role == StandardOrderRole.CLOSE else "BUY"
    saved_trade = ibi.Trade(
        contract=target_contract,
        order=ibi.MarketOrder(
            action,
            3,
            orderId=77,
            permId=177,
        ),
        orderStatus=ibi.OrderStatus(
            orderId=77,
            status=ibi.OrderStatus.Submitted,
            remaining=3,
        ),
    )
    info = controller.register_order(
        saved_trade,
        role=role,
        execution_model_name="brackets" if source_key is not None else "serial",
        source_key=source_key,
        position_id="episode" if source_key is not None else None,
    )
    broker_trade = deepcopy(saved_trade)
    offline_fill = fill(broker_trade, exec_id=f"{role}-offline", quantity=1)
    broker_trade.fills.append(offline_fill)
    broker_trade.orderStatus.filled = 1
    broker_trade.orderStatus.remaining = 2
    broker_position = ibi.Position(
        account="DU123",
        contract=target_contract,
        position=broker_quantity,
        avgCost=100,
    )
    set_broker_state(
        controller,
        monkeypatch,
        positions=(broker_position,),
        open_trades=(broker_trade,),
        fills=(offline_fill,),
    )
    monkeypatch.setattr(
        controller.ib,
        "reqPositionsAsync",
        AsyncMock(return_value=[broker_position]),
    )

    assert await SyncCoordinator(controller).run()
    assert info.trade is broker_trade
    assert len(info.fills) == 1
    assert controller.book.positions.quantity(target_contract) == broker_quantity

    assert await SyncCoordinator(controller).run()
    assert len(info.fills) == 1
    assert controller.book.positions.quantity(target_contract) == broker_quantity

    live_fill = fill(broker_trade, exec_id=f"{role}-live", quantity=2)
    broker_trade.fills.append(live_fill)
    broker_trade.orderStatus.status = ibi.OrderStatus.Filled
    broker_trade.orderStatus.filled = 3
    broker_trade.orderStatus.remaining = 0
    await controller.onExecDetailsEvent(broker_trade, live_fill)
    assert len(info.fills) == 2
    assert controller.book.positions.quantity(target_contract) == (
        0 if role == StandardOrderRole.CLOSE else 3
    )


@pytest.mark.asyncio
async def test_sync_rejects_conflicting_fill_history(controller, monkeypatch):
    """A repeated execution identity cannot silently change accounted quantity."""

    target_contract = contract()
    controller.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=target_contract,
            position_id="episode",
        )
    )
    saved_trade = ibi.Trade(
        contract=target_contract,
        order=ibi.MarketOrder("BUY", 3, orderId=77, permId=177),
        orderStatus=ibi.OrderStatus(
            orderId=77,
            status=ibi.OrderStatus.Submitted,
            remaining=3,
        ),
    )
    controller.register_order(
        saved_trade,
        role=StandardOrderRole.OPEN,
        execution_model_name="brackets",
        source_key="alpha",
        position_id="episode",
    )
    await controller.onExecDetailsEvent(
        saved_trade,
        fill(saved_trade, exec_id="offline-1", quantity=1),
    )
    broker_trade = deepcopy(saved_trade)
    conflicting = fill(broker_trade, exec_id="offline-1", quantity=2)
    broker_trade.fills.append(conflicting)
    broker_trade.orderStatus.filled = 2
    broker_trade.orderStatus.remaining = 1
    broker_position = ibi.Position(
        account="DU123",
        contract=target_contract,
        position=2,
        avgCost=100,
    )
    set_broker_state(
        controller,
        monkeypatch,
        positions=(broker_position,),
        open_trades=(broker_trade,),
        fills=(conflicting,),
    )
    monkeypatch.setattr(
        controller.ib,
        "reqPositionsAsync",
        AsyncMock(return_value=[broker_position]),
    )

    with pytest.raises(SyncBrokenStateError, match="execution evidence"):
        await SyncCoordinator(controller).run()


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
        lambda: controller.book.positions.for_source("alpha").quantity == 1
    )
    assert info.trade is done_trade


@pytest.mark.asyncio
async def test_sync_coordinator_prunes_unmatched_local_order_and_retries(
    controller, trade, monkeypatch, caplog
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

    caplog.clear()
    with caplog.at_level(
        logging.WARNING, logger="haymaker.controller.sync_coordinator"
    ):
        result = await SyncCoordinator(controller).run()

    assert not result
    assert controller.book.orders.by_id(info.orderId) is info
    assert not info.active
    assert caplog.messages == [
        f"Pruned stale local order {info.orderId}; order was absent at broker."
    ]


@pytest.mark.asyncio
async def test_sync_coordinator_requests_restart_before_position_correction(
    controller, monkeypatch
):
    controller.position_mismatch_policy = "correct"
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
        lambda self, errors, broker_positions: corrected.append(errors),
    )

    coordinator = SyncCoordinator(controller, restart_before_correction=True)
    result = await coordinator.run()

    assert not result
    assert coordinator.request_restart
    assert corrected == []
    assert controller.book.positions.for_source("alpha").quantity == 1


@pytest.mark.asyncio
async def test_sync_coordinator_allows_position_correction_after_restart(
    controller, monkeypatch
):
    controller.position_mismatch_policy = "correct"
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
        lambda self, errors, broker_positions: corrected.append(errors),
    )

    coordinator = SyncCoordinator(controller, restart_before_correction=False)
    result = await coordinator.run()

    assert not result
    assert not coordinator.request_restart
    assert corrected == [{contract(): 1.0}]


@pytest.mark.asyncio
async def test_sync_defers_position_correction_while_open_order_is_active(
    controller, monkeypatch
):
    target_contract = contract()
    controller.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=target_contract,
            quantity=1,
            target_quantity=1,
            target_created_at=datetime.now(timezone.utc),
            position_id="episode",
        )
    )
    open_trade = ibi.Trade(
        contract=target_contract,
        order=ibi.MarketOrder("BUY", 1, orderId=77, permId=177),
    )
    save_active_order(
        controller,
        open_trade,
        role=StandardOrderRole.OPEN,
    )
    set_broker_state(
        controller,
        monkeypatch,
        open_trades=(open_trade,),
    )
    monkeypatch.setattr(
        controller.ib,
        "reqPositionsAsync",
        AsyncMock(return_value=[]),
    )

    coordinator = SyncCoordinator(controller, restart_before_correction=True)
    result = await coordinator.run()

    assert result
    assert not coordinator.request_restart
    state = controller.book.positions.for_source("alpha")
    assert state is not None
    assert state.quantity == 1
    assert state.target_quantity == 1


@pytest.mark.asyncio
async def test_broker_flat_correction_supersedes_target_before_recovery(
    controller, monkeypatch
):
    controller.position_mismatch_policy = "correct"
    old_target_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    controller.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=contract(),
            quantity=1,
            target_quantity=1,
            target_created_at=old_target_time,
            position_id="episode",
            blocked_direction=1,
            bracket_inputs={"atr": 2},
        )
    )
    set_broker_state(controller, monkeypatch)
    monkeypatch.setattr(
        controller.ib,
        "reqPositionsAsync",
        AsyncMock(return_value=[]),
    )

    result = await SyncCoordinator(
        controller,
        restart_before_correction=False,
    ).run()

    assert not result
    state = controller.book.positions.for_source("alpha")
    assert state is not None
    assert state.quantity == 0
    assert state.target_quantity == 0
    assert state.target_created_at is not None
    assert state.target_created_at > old_target_time
    assert state.position_id is None
    assert state.bracket_inputs == {}
    assert state.blocked_direction == 1

    submissions = []
    monkeypatch.setattr(
        controller,
        "trade",
        lambda *args, **kwargs: submissions.append((args, kwargs)),
    )
    BracketExecutionModel(
        "alpha",
        name="brackets",
        stop=FixedStop(2),
    ).recover()

    assert submissions == []


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

    snapshot = await verify_broker_position_source(ib, 0.01)

    assert snapshot.status is BrokerPositionStatus.REQUEST_UNAVAILABLE
    assert snapshot.positions == ()


@pytest.mark.asyncio
async def test_clean_sync_enables_broker_work(controller_runtime, monkeypatch):
    runtime, controller, _ = controller_runtime
    monkeypatch.setattr(runtime.ib, "isConnected", lambda: True)
    monkeypatch.setattr(runtime.ib, "positions", lambda: [])
    monkeypatch.setattr(runtime.ib, "openTrades", lambda: [])
    monkeypatch.setattr(runtime.ib, "trades", lambda: [])
    monkeypatch.setattr(runtime.ib, "fills", lambda: [])
    monkeypatch.setattr(runtime.ib, "reqPositionsAsync", AsyncMock(return_value=[]))

    outcome = await controller.sync()

    assert outcome is SyncOutcome.OK
    assert controller.broker_ready


@pytest.mark.asyncio
async def test_controller_run_does_not_restore_book(controller_runtime, monkeypatch):
    runtime, controller, _ = controller_runtime
    restore = Mock(side_effect=AssertionError("Controller must not restore Book"))
    monkeypatch.setattr(runtime.book, "_restore_documents", restore)
    sync = AsyncMock(return_value=SyncOutcome.OK)
    monkeypatch.setattr(controller, "sync", sync)

    assert await controller.run() is SyncOutcome.OK
    restore.assert_not_called()


@pytest.mark.parametrize("event", ["status", "fill", "commission"])
async def test_callbacks_rebind_accounting_before_recovery(
    controller_runtime, order_saver, state_saver, event
):
    """Every broker callback preserves one order identity across ID changes."""
    runtime, controller, _ = controller_runtime
    opening = controller.trade(
        contract(),
        ibi.MarketOrder("BUY", 2),
        role=StandardOrderRole.TARGET_ADJUSTMENT,
        execution_model_name="serial",
    )
    execution = fill(opening)
    await controller.onExecDetailsEvent(opening, execution)
    previous_id = opening.order.orderId
    rebound = ibi.Trade(
        contract=opening.contract,
        order=deepcopy(opening.order),
        orderStatus=replace(opening.orderStatus, orderId=55),
        fills=[execution],
    )
    rebound.order.orderId = 55
    report = ibi.CommissionReport(execId=execution.execution.execId, commission=2.5)
    if event == "status":
        controller.onOrderStatusEvent(rebound)
    elif event == "fill":
        await controller.onExecDetailsEvent(rebound, execution)
    else:
        await controller.onCommissionReport(rebound, execution, report)
    assert runtime.book.orders.by_id(previous_id) is None
    assert runtime.book.orders.by_id(55).trade is rebound
    for _ in range(2):
        recovered = Book(
            order_saver=order_saver,
            state_saver=state_saver,
            save_async=False,
            restore=True,
        )
        assert len(recovered.orders.query()) == 1
        assert recovered.positions.quantity(contract()) == 1
        if event == "commission":
            assert recovered.orders.by_id(55).fills[0].commission_report == report


def test_controller_refuses_submission_after_persistence_halts(
    controller_runtime, order_saver, monkeypatch
):
    """No second broker order may precede a known failed accounting write."""
    _, controller, trader = controller_runtime
    monkeypatch.setattr(order_saver, "save", Mock(side_effect=RuntimeError("offline")))
    args = (contract(), ibi.MarketOrder("BUY", 1))
    kwargs = dict(role=StandardOrderRole.OPEN, execution_model_name="brackets")
    with pytest.raises(RuntimeError, match="offline"):
        controller.trade(*args, **kwargs)
    assert len(trader.trades) == 1
    with pytest.raises(QueueProcessingError):
        controller.trade(*args, **kwargs)
    assert len(trader.trades) == 1


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

    assert runtime.book.orders.rejection_count("brackets") == 1
