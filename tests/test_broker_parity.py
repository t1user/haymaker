"""Broker quirks must not corrupt accounting or prematurely finish reporting."""

import asyncio
from copy import deepcopy
from datetime import datetime, timezone
from unittest.mock import Mock

import ib_insync as ibi
import pytest

from runtime_helpers import signal_execution_state
from test_book import fill, order_info, trade
from test_controller import controller_runtime  # noqa: F401

from haymaker.blotter import Blotter
from haymaker.book import Book, OrderInfo
from haymaker.controller.sync_coordinator import SyncCoordinator
from haymaker.controller.sync_routines import OrderSync


def reporting(runtime):
    blotter = Blotter(save_immediately=False, saver=Mock())
    runtime.book.blotter = blotter
    return blotter


async def execution(controller, current, exec_id, quantity=1):
    item = fill(current, exec_id=exec_id, quantity=quantity)
    item.commissionReport.execId = ""
    item.commissionReport.commission = 0
    item.commissionReport.realizedPNL = 0
    current.fills.append(item)
    await controller.onExecDetailsEvent(current, item)
    return item


@pytest.mark.parametrize("terminal_first", [False, True])
async def test_all_commissions_and_terminal_status_required(
    controller_runtime, terminal_first
):
    runtime, controller, _ = controller_runtime
    blotter = reporting(runtime)
    current = trade()
    runtime.book.save_order(order_info(current))
    first = await execution(controller, current, "first")
    second = await execution(controller, current, "second")
    if terminal_first:
        current.orderStatus.status = ibi.OrderStatus.Filled
        controller.onOrderStatusEvent(current)
    await controller.onCommissionReport(
        current, first, ibi.CommissionReport(execId="first", commission=1)
    )
    assert blotter.blotter == []
    await controller.onCommissionReport(
        current, second, ibi.CommissionReport(execId="second", commission=2)
    )
    if not terminal_first:
        assert blotter.blotter == []
        current.orderStatus.status = ibi.OrderStatus.Filled
        controller.onOrderStatusEvent(current)
    controller.onOrderStatusEvent(current)
    await controller.onCommissionReport(current, first, first.commissionReport)
    assert len(blotter.blotter) == 1
    assert blotter.blotter[0]["commission"] == 3
    assert blotter.blotter[0]["amount"] == 2


async def test_timeout_logs_once_and_late_report_still_completes(
    controller_runtime, caplog
):
    runtime, controller, _ = controller_runtime
    controller.commission_timeout = 0.001
    blotter = reporting(runtime)
    current = trade(quantity=1)
    runtime.book.save_order(order_info(current))
    item = await execution(controller, current, "late")
    current.orderStatus.status = ibi.OrderStatus.Filled
    controller.onOrderStatusEvent(current)
    controller.onOrderStatusEvent(current)
    await asyncio.sleep(0.01)
    controller.onOrderStatusEvent(current)
    await asyncio.sleep(0.01)
    assert caplog.text.count("Incomplete trade after commission timeout") == 1
    assert blotter.blotter == []
    assert not controller._trading_disabled
    assert runtime.book.positions.filled_quantity(current.contract) == 1
    await controller.onCommissionReport(
        current, item, ibi.CommissionReport(execId="late", commission=1.25)
    )
    assert len(blotter.blotter) == 1
    assert not controller._commission_waits


async def test_report_recovery_uses_durable_commissions_without_new_ib_reports(
    controller_runtime, order_saver, state_saver, monkeypatch
):
    runtime, controller, _ = controller_runtime
    current = trade(quantity=1)
    runtime.book.save_order(order_info(current))
    item = await execution(controller, current, "saved")
    await controller.onCommissionReport(
        current, item, ibi.CommissionReport(execId="saved", commission=1.25)
    )
    current.orderStatus.status = ibi.OrderStatus.Filled
    runtime.book.rebind_trade(current)
    runtime.book = Book(
        order_saver=order_saver,
        state_saver=state_saver,
        save_async=False,
        restore=True,
    )
    blotter = reporting(runtime)
    monkeypatch.setattr(controller.ib, "fills", lambda: [])
    coordinator = SyncCoordinator(controller)
    await coordinator.recover_commissions()
    await coordinator.recover_commissions()
    assert len(blotter.blotter) == 1
    assert blotter.blotter[0]["commission"] == 1.25


@pytest.mark.parametrize(
    "bad_report",
    [
        ibi.CommissionReport(),
        ibi.CommissionReport(execId="wrong"),
        ibi.CommissionReport(execId="exec-1", commission=float("nan")),
    ],
)
async def test_invalid_report_does_not_replace_valid_commission(
    controller_runtime, caplog, bad_report
):
    runtime, controller, _ = controller_runtime
    current = trade(quantity=1)
    runtime.book.save_order(order_info(current))
    item = fill(current)
    await controller.onExecDetailsEvent(current, item)
    await controller.onCommissionReport(current, item, bad_report)
    assert "Invalid commission report" in caplog.text
    assert runtime.book.orders.by_id(1).fills[0].commission_report.execId == "exec-1"


@pytest.mark.parametrize("reconstruct", [False, True])
async def test_mutated_ib_report_does_not_change_accepted_commission(
    controller_runtime,
    reconstruct,
):
    runtime, controller, _ = controller_runtime
    current = trade(quantity=1)
    runtime.book.save_order(order_info(current))
    item = fill(current)
    await controller.onExecDetailsEvent(current, item)
    await controller.onCommissionReport(current, item, item.commissionReport)
    if reconstruct:
        item = runtime.book.orders.by_id(1).execution_trade().fills[0]
    item.commissionReport.execId = ""
    item.commissionReport.commission = 999
    await controller.onCommissionReport(current, item, item.commissionReport)
    saved = runtime.book.orders.by_id(1).fills[0].commission_report
    assert saved.execId == "exec-1"
    assert saved.commission == 1


async def test_identity_conflict_event_disables_new_submissions(controller_runtime):
    runtime, controller, trader = controller_runtime
    current = trade(perm_id=111)
    runtime.book.save_order(order_info(current))
    current.order.permId = 222
    runtime.ib.orderStatusEvent.emit(current)
    assert controller._trading_disabled
    assert trader.trades == []


@pytest.mark.parametrize("event", ["new", "status", "fill", "commission"])
@pytest.mark.parametrize("perm_id", [0, 111])
async def test_unmatched_zero_id_is_logged_and_never_stored(
    controller_runtime, caplog, event, perm_id
):
    runtime, controller, _ = controller_runtime
    current = trade(order_id=0, perm_id=perm_id)
    item = fill(current)
    if event == "new":
        await controller.onNewOrderEvent(current)
    elif event == "status":
        controller.onOrderStatusEvent(current)
    elif event == "fill":
        await controller.onExecDetailsEvent(current, item)
    else:
        await controller.onCommissionReport(current, item, item.commissionReport)
    assert runtime.book.orders.query() == ()
    assert any(record.levelname == "ERROR" for record in caplog.records)
    assert "orderId=0" in caplog.text


async def test_real_ib_wrapper_fallback_cannot_account_foreign_execution(
    controller_runtime, caplog
):
    runtime, controller, _ = controller_runtime
    current = trade(perm_id=222)
    current.order.clientId = 3
    runtime.book.save_order(order_info(current))
    wrapper = runtime.ib.wrapper
    wrapper.trades[(3, 1)] = current
    wrapper.permId2Trade[222] = current
    incoming = ibi.Execution(
        execId="old-trade",
        orderId=1,
        clientId=3,
        permId=111,
        time=datetime.now(timezone.utc),
        side="BOT",
        shares=1,
        price=100,
    )
    wrapper.execDetails(-1, current.contract, incoming)
    await asyncio.sleep(0)
    # This is ib_insync's real fallback, not a simulated association in Haymaker.
    assert current.fills[0].execution.permId == 111
    assert runtime.book.orders.by_id(1).fills == ()
    assert runtime.book.positions.by_contract() == {}
    assert "foreign execution" in caplog.text
    recovered = OrderSync(runtime.ib, runtime.book).run()
    assert recovered.recovered_fills == []
    assert runtime.book.orders.by_id(1).execution_trade(current.fills).filled() == 0


@pytest.mark.parametrize("in_place", [False, True])
def test_reset_numbering_cannot_overwrite_persisted_identity(
    book, order_saver, state_saver, in_place
):
    original = trade(perm_id=111)
    book.save_order(order_info(original))
    book.apply_fill(original, fill(original))
    replacement = original if in_place else trade(perm_id=222)
    replacement.order.permId = 222
    with pytest.raises(ValueError, match="identify different orders"):
        book.rebind_trade(replacement)
    with pytest.raises(ValueError, match="identify different orders"):
        book.save_order(order_info(replacement))
    restored = Book(
        order_saver=order_saver,
        state_saver=state_saver,
        save_async=False,
        restore=True,
    )
    assert restored.orders.by_id(1).permId == 111
    assert restored.positions.filled_quantity(original.contract) == 1


def test_blotter_filters_placeholder_and_foreign_fill_from_all_calculations(caplog):
    blotter = Blotter(save_immediately=False, saver=Mock())
    current = trade(quantity=1)
    own = fill(current)
    foreign = fill(trade(perm_id=999), exec_id="foreign", quantity=100)
    empty = ibi.Fill(
        current.contract, ibi.Execution(), ibi.CommissionReport(), own.time
    )
    current.fills = [own, foreign, empty]
    current.orderStatus.status = ibi.OrderStatus.Filled
    current.orderStatus.avgFillPrice = 999  # Also untrustworthy after contamination.
    blotter.log_commission(current, own, own.commissionReport)
    row = blotter.blotter[0]
    assert row["amount"] == 1
    assert row["commission"] == 1
    assert row["price"] == own.execution.price
    assert len(row["fills"]) == 1
    assert "foreign execution" in caplog.text
    assert "empty execution" in caplog.text


@pytest.mark.parametrize("include_aggregate", [False, True])
def test_bag_waits_for_each_leg_commission_and_ignores_empty_aggregate_report(
    include_aggregate,
):
    blotter = Blotter(save_immediately=False, saver=Mock())
    current = trade(quantity=1)
    aggregate = fill(current, quantity=1)
    bag = ibi.Bag(
        symbol="ES",
        comboLegs=[
            ibi.ComboLeg(conId=1, ratio=1, action="BUY"),
            ibi.ComboLeg(conId=2, ratio=1, action="SELL"),
        ],
    )
    current.contract = bag
    aggregate = aggregate._replace(
        contract=bag, commissionReport=ibi.CommissionReport()
    )
    first = fill(trade(con_id=1), exec_id="leg-1")
    second = fill(trade(con_id=2), exec_id="leg-2")
    second.commissionReport.execId = ""
    current.orderStatus.status = ibi.OrderStatus.Filled
    current.fills = [aggregate, first] if include_aggregate else [first]
    blotter.log_commission(current, first, first.commissionReport)
    assert blotter.blotter == []  # Second leg has not arrived at all.
    current.fills.append(second)
    blotter.log_commission(current, first, first.commissionReport)
    assert blotter.blotter == []  # Second execution exists, but has no commission.
    second.commissionReport.execId = "leg-2"
    blotter.log_commission(current, second, second.commissionReport)
    assert len(blotter.blotter) == 1
    assert blotter.blotter[0]["amount"] == 1
    assert blotter.blotter[0]["commission"] == 2


async def test_foreign_commission_cannot_modify_normalized_evidence(
    controller_runtime, caplog
):
    runtime, controller, _ = controller_runtime
    current = trade()
    runtime.book.save_order(order_info(current))
    own = await execution(controller, current, "shared-id")
    foreign = fill(trade(perm_id=999), exec_id="shared-id")
    await controller.onCommissionReport(current, foreign, foreign.commissionReport)
    assert runtime.book.orders.by_id(1).fills[0].commission_report is None
    assert not own.commissionReport.execId
    assert "foreign execution" in caplog.text


async def test_ambiguous_manual_close_does_not_guess_a_signal(controller_runtime):
    runtime, controller, trader = controller_runtime
    current = trade(order_id=-1, perm_id=999, side="SELL", quantity=2)
    for signal_key in ("alpha", "beta"):
        runtime.book._update_signal_state(
            signal_execution_state(
                signal_key=signal_key,
                execution_model_name="brackets",
                contract=current.contract,
                filled_quantity=1,
                target_quantity=1,
            )
        )
    await controller.onExecDetailsEvent(current, fill(current, quantity=2))
    assert runtime.book.orders.by_id(-1).signal_key is None
    assert trader.cancelled == []
    for signal_key in ("alpha", "beta"):
        assert runtime.book.positions._for_signal(signal_key).filled_quantity == 1
        assert runtime.book.targets.for_signal(signal_key).target_quantity == 1


def test_blotter_uses_permanent_identity_across_restart_rebind_and_reused_number():
    saver = Mock()
    saver.read.return_value = [{"order_id": 1, "perm_id": 111}]
    blotter = Blotter(save_immediately=False, saver=saver)
    for order_id, perm_id in ((22, 111), (1, 222), (1, 222)):
        current = trade(order_id=order_id, perm_id=perm_id, quantity=1)
        item = fill(current)
        current.fills = [item]
        current.orderStatus.status = ibi.OrderStatus.Filled
        blotter.log_commission(current, item, item.commissionReport)
    assert len(blotter.blotter) == 1
    assert blotter.blotter[0]["perm_id"] == 222


@pytest.mark.parametrize(
    "terminal", [ibi.OrderStatus.Filled, ibi.OrderStatus.Cancelled]
)
async def test_successive_snapshots_retain_partial_final_and_commission_evidence(
    controller_runtime, order_saver, state_saver, terminal
):
    runtime, controller, _ = controller_runtime
    current = trade()
    runtime.book.save_order(order_info(current))
    snapshots = [deepcopy(order_saver.read({})[0])]
    current.orderStatus.status = ibi.OrderStatus.PreSubmitted
    controller.onOrderStatusEvent(current)
    snapshots.append(deepcopy(order_saver.read({})[0]))
    first = await execution(controller, current, "first")
    current.orderStatus.filled = 1
    current.orderStatus.remaining = 1
    controller.onOrderStatusEvent(current)
    snapshots.append(deepcopy(order_saver.read({})[0]))
    recovered = Book(
        order_saver=order_saver,
        state_saver=state_saver,
        save_async=False,
        restore=True,
    )
    assert recovered.positions.filled_quantity(current.contract) == 1
    assert not recovered.apply_fill(current, first)
    if terminal == ibi.OrderStatus.Filled:
        await execution(controller, current, "second")
        current.orderStatus.filled = 2
        current.orderStatus.remaining = 0
    current.orderStatus.status = terminal
    controller.onOrderStatusEvent(current)
    for item in current.fills:
        await controller.onCommissionReport(
            current,
            item,
            ibi.CommissionReport(execId=item.execution.execId, commission=1.25),
        )
    snapshots.append(deepcopy(order_saver.read({})[0]))
    orders = [OrderInfo.decode(snapshot) for snapshot in snapshots]
    assert [item.trade.orderStatus.status for item in orders] == [
        ibi.OrderStatus.Submitted,
        ibi.OrderStatus.PreSubmitted,
        ibi.OrderStatus.PreSubmitted,
        terminal,
    ]
    assert [len(item.fills) for item in orders] == [0, 0, 1, len(current.fills)]
    assert len(orders[-1].trade.fills) == len(current.fills)
    assert all(
        record.commission_report.commission == 1.25 for record in orders[-1].fills
    )


@pytest.mark.parametrize("quantity", [1, 2])
async def test_manual_full_close_clears_target_but_partial_close_does_not(
    controller_runtime, order_saver, state_saver, quantity
):
    runtime, controller, trader = controller_runtime
    current = trade(order_id=-1, perm_id=999, side="SELL", quantity=quantity)
    runtime.book._update_signal_state(
        signal_execution_state(
            signal_key="alpha",
            execution_model_name="brackets",
            contract=current.contract,
            filled_quantity=2,
            target_quantity=2,
            position_id="position-1",
            protection_inputs={"atr": 5},
        )
    )
    protective = trade(order_id=2, perm_id=102, side="SELL")
    runtime.book.save_order(order_info(protective, role="STOP_LOSS"))
    item = fill(current, quantity=quantity)
    await controller.onExecDetailsEvent(current, item)
    state = runtime.book.positions._for_signal("alpha")
    assert state.filled_quantity == 2 - quantity
    if quantity == 1:
        assert state.target_quantity == 2
        assert trader.cancelled == []
        return
    assert state.target_quantity == 0
    assert state.position_id is None
    assert state.protection_inputs == {}
    assert trader.cancelled == [protective]
    runtime.book = Book(
        order_saver=order_saver,
        state_saver=state_saver,
        save_async=False,
        restore=True,
    )
    await controller.onExecDetailsEvent(current, item)
    assert runtime.book.positions._for_signal("alpha").target_quantity == 0
    assert runtime.book.positions.filled_quantity(current.contract) == 0
