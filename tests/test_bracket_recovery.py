"""Independent broker-boundary coverage for missed initial protection."""

import asyncio
from dataclasses import replace
from unittest.mock import Mock

import ib_insync as ibi
import pytest
from episode_harness import (
    EpisodeBroker,
    EpisodeSignalModel,
    observation,
    settle_events,
)

from haymaker.base import Pipe
from haymaker.book import Book
from haymaker.blotter import Blotter
from haymaker.components import (
    BinarySignalProcessor,
    BracketExecutionModel,
    FixedSizeAllocator,
    FixedStop,
    PortfolioWrapper,
    SignalType,
    StandardOrderRole,
    TakeProfitAsStopMultiple,
)
from haymaker.controller import Controller
from haymaker.controller.controller import SyncOutcome


@pytest.fixture
def entry_path(atom_runtime_factory):
    """Build the complete one-to-one pipeline with real Controller accounting."""
    broker = EpisodeBroker()
    runtime = atom_runtime_factory(ib=broker)
    controller = Controller(trader=runtime.trader)
    runtime.bind_controller(controller)
    contract = ibi.Future("ES", conId=101, exchange="CME", localSymbol="ESU6")
    runtime.contract_registry.details.data[contract] = Mock(
        minTick=0.25, is_open=Mock(return_value=True)
    )
    signal = EpisodeSignalModel("alpha", contract, SignalType.EVENT)
    model = BracketExecutionModel(
        "alpha",
        name="brackets",
        stop=FixedStop(2),
        take_profit=TakeProfitAsStopMultiple(2, 2),
    )
    pipe = Pipe(
        signal, BinarySignalProcessor(), PortfolioWrapper(FixedSizeAllocator(3)), model
    )
    signal.onData(observation(1))
    return runtime, broker, model, pipe


@pytest.fixture
def restart_entry(atom_runtime_factory, order_saver, state_saver, monkeypatch):
    """Restore a fresh graph, using executions rather than completed Trade history."""

    def restart(runtime, broker):
        """Detach all old model callbacks and restore the actual saved documents."""
        book = Book(
            order_saver=order_saver,
            state_saver=state_saver,
            save_async=False,
            restore=True,
        )
        replacement = broker.restarted()
        monkeypatch.setattr(replacement, "isConnected", lambda: True)
        monkeypatch.setattr(replacement, "trades", lambda: [])
        fresh = atom_runtime_factory(
            ib=replacement, book_=book, contract_registry=runtime.contract_registry
        )
        controller = Controller(
            trader=fresh.trader, missing_brackets="remove", sync_resync_delay=0
        )
        fresh.bind_controller(controller)
        model = BracketExecutionModel(
            "alpha",
            name="brackets",
            stop=FixedStop(2),
            take_profit=TakeProfitAsStopMultiple(2, 2),
        )
        return fresh, replacement, controller, model

    return restart


@pytest.mark.parametrize("live_first_fill", [False, True])
async def test_offline_entry_recovers_weighted_price_before_remediation(
    entry_path, restart_entry, monkeypatch, live_first_fill
):
    """Offline executions, persisted partial fills and repeated recovery share one path."""
    runtime, broker, _, _ = entry_path
    entry = broker.submitted[0]
    await broker.fill(entry, 2, price=100, notify_events=live_first_fill)
    last = await broker.fill(entry, 1, price=103, notify_events=False)
    assert len(broker.submitted) == 1
    fresh, replacement, controller, model = restart_entry(runtime, broker)
    if live_first_fill:
        # Earlier fills can have aged out of broker history but remain in Book.
        monkeypatch.setattr(replacement, "fills", lambda: [last, last])
    assert await controller.run() is SyncOutcome.OK
    assert fresh.book.positions.for_source("alpha").quantity == 3
    assert len(replacement.submitted) == 3
    stop, take_profit = replacement.submitted[1:]
    assert stop.order.auxPrice == 91  # weighted entry 101 minus 2 * saved ATR 5
    assert take_profit.order.lmtPrice == 121
    assert stop.order.totalQuantity == take_profit.order.totalQuantity == 3
    assert stop.order.ocaGroup == take_profit.order.ocaGroup
    recovered = fresh.book.orders.by_id(1)
    assert recovered.trade.orderStatus.avgFillPrice == 101
    assert len(recovered.fills) == 2
    assert recovered.position_id == fresh.book.positions.for_source("alpha").position_id
    assert fresh.book.blotter is None
    assert await controller.sync() is SyncOutcome.OK
    model.recover()
    model.ensure_entry_brackets(recovered.trade)
    await settle_events()
    assert len(replacement.submitted) == 3


async def test_recover_after_fill_accounted_but_callback_missed(
    entry_path, restart_entry
):
    """Even an already terminal persisted entry is inspected without event replay."""
    runtime, broker, _, _ = entry_path
    await broker.fill(broker.submitted[0], price=103, notify_filled=False)
    fresh, replacement, controller, _ = restart_entry(runtime, broker)
    assert await controller.run() is SyncOutcome.OK
    assert len(replacement.submitted) == 3
    assert replacement.submitted[1].order.auxPrice == 93
    assert fresh.book.positions.for_source("alpha").quantity == 3


async def test_live_duplicate_completion_uses_accounted_price(entry_path):
    """The normal path also ignores stale status prices and repeated callbacks."""
    runtime, broker, model, _ = entry_path
    entry = broker.submitted[0]
    await broker.fill(entry, 2, price=100)
    await broker.fill(entry, 1, price=103, notify_filled=False)
    entry.orderStatus.avgFillPrice = 0
    entry.filledEvent.emit(entry)
    entry.filledEvent.emit(entry)
    await settle_events()
    assert len(broker.submitted) == 3
    assert broker.submitted[1].order.auxPrice == 91
    assert not runtime.controller._trading_disabled


@pytest.mark.parametrize("terminal", [False, True])
async def test_commissions_survive_suspended_broker_work_and_restart(
    entry_path, restart_entry, terminal
):
    """Every fill's late report persists while rolling/recovery is suspended."""
    runtime, broker, _, _ = entry_path
    trade = broker.submitted[0]
    first = await broker.fill(trade, 1)
    second = await broker.fill(trade, 2 if terminal else 1)
    runtime.controller.suspend_broker_work()
    await broker.commission(trade, first)
    await broker.commission(trade, second)
    fresh, _, controller, _ = restart_entry(runtime, broker)
    assert await controller.run() is SyncOutcome.OK
    records = fresh.book.orders.by_id(trade.order.orderId).fills
    assert [record.commission_report.commission for record in records] == [1.25, 1.25]


async def test_terminal_commissions_recovered_from_history_without_callbacks(
    entry_path, restart_entry
):
    """A completed known order is still swept for late per-fill commissions."""
    runtime, broker, _, _ = entry_path
    trade = broker.submitted[0]
    await broker.fill(trade, 1)
    await broker.fill(trade, 2)
    for fill in trade.fills:
        fill.commissionReport.execId = fill.execution.execId
        fill.commissionReport.commission = 1.25
        fill.commissionReport.currency = "USD"
    fresh, _, controller, _ = restart_entry(runtime, broker)
    blotter = Blotter(save_immediately=False, saver=Mock())
    fresh.book.blotter = blotter
    assert await controller.run() is SyncOutcome.OK
    assert [
        record.commission_report.commission
        for record in fresh.book.orders.by_id(1).fills
    ] == [1.25, 1.25]
    assert await controller.sync() is SyncOutcome.OK
    assert len(blotter.blotter) == 1
    assert blotter.blotter[0]["commission"] == 2.5


@pytest.mark.parametrize("winner", ["close", "take_profit"])
async def test_missing_stop_exit_cannot_reverse_exposure(
    entry_path, restart_entry, winner
):
    """Either broker exit wins once; its OCA peer cannot reverse the account."""
    runtime, broker, _, _ = entry_path
    await broker.fill(broker.submitted[0])
    broker.cancelOrder(broker.submitted[1].order)
    fresh, replacement, controller, _ = restart_entry(runtime, broker)
    assert await controller.run() is SyncOutcome.FAILED
    take_profit, close = replacement.submitted[2:]
    assert close.order.ocaGroup == take_profit.order.ocaGroup
    winning, other = (close, take_profit) if winner == "close" else (take_profit, close)
    await replacement.fill(winning)
    assert other.orderStatus.status == ibi.OrderStatus.Cancelled
    assert replacement.positions() == []
    assert fresh.book.positions.for_source("alpha").quantity == 0


@pytest.mark.parametrize("defect", ["quantity", "side", "contract"])
async def test_inadequate_stop_is_removed_and_episode_closed(
    entry_path, restart_entry, defect
):
    """Malformed stop evidence must drive the configured remove policy."""
    runtime, broker, _, _ = entry_path
    await broker.fill(broker.submitted[0])
    stop = broker.submitted[1]
    if defect == "quantity":
        stop.order.totalQuantity = stop.orderStatus.remaining = 1
    elif defect == "side":
        stop.order.action = "BUY"
    else:
        stop.contract = ibi.Future("NQ", conId=102, exchange="CME")
    fresh, replacement, controller, _ = restart_entry(runtime, broker)
    assert await controller.run() is SyncOutcome.FAILED
    assert replacement.submitted[1].isDone()
    close = replacement.submitted[-1]
    assert close.order.orderType == "MKT"
    await replacement.fill(close)
    assert replacement.positions() == []
    assert replacement.openTrades() == []
    assert fresh.book.positions.for_source("alpha").quantity == 0


@pytest.mark.parametrize("completion", ["delayed", "fill", "timeout"])
async def test_incompatible_exit_cancellation_boundary(
    entry_path, restart_entry, monkeypatch, completion
):
    """Wait for broker cancellation, re-read fills, and never close on timeout."""
    runtime, broker, _, _ = entry_path
    await broker.fill(broker.submitted[0])
    broker.submitted[1].order.totalQuantity = 1
    broker.submitted[1].orderStatus.remaining = 1
    fresh, replacement, controller, _ = restart_entry(runtime, broker)
    controller.broker_request_timeout = 0.05
    cancel = replacement.cancelOrder
    callbacks = []

    def delayed_cancel(order, **kwargs):
        if completion == "timeout":
            return None
        if completion == "fill" and order is replacement.submitted[2].order:
            callbacks.append(
                asyncio.create_task(replacement.fill(replacement.submitted[2]))
            )
        else:
            asyncio.get_running_loop().call_soon(cancel, order)
        return None

    monkeypatch.setattr(replacement, "cancelOrder", delayed_cancel)
    outcome = await controller.run()
    await asyncio.gather(*callbacks)
    if completion == "fill":
        assert outcome is SyncOutcome.OK
        assert len(replacement.submitted) == 3
        assert fresh.book.positions.for_source("alpha").quantity == 0
    else:
        assert outcome is SyncOutcome.FAILED
        assert len(replacement.submitted) == (3 if completion == "timeout" else 4)
        if completion == "delayed":
            await replacement.fill(replacement.submitted[-1])
            assert replacement.positions() == []


@pytest.mark.parametrize(
    "surviving_role", [StandardOrderRole.STOP_LOSS, StandardOrderRole.TAKE_PROFIT]
)
async def test_recovery_between_leg_submissions(
    entry_path, restart_entry, surviving_role
):
    """A lone stop is sufficient; a lone take-profit lends its existing OCA group."""
    runtime, broker, _, _ = entry_path
    await broker.fill(broker.submitted[0], notify_filled=False)
    state = runtime.book.positions.for_source("alpha")
    runtime.controller.trade(
        state.contract,
        ibi.Order(
            action="SELL",
            totalQuantity=3,
            orderType="STP",
            auxPrice=90 if surviving_role == StandardOrderRole.STOP_LOSS else 120,
            ocaGroup="existing-group",
            ocaType=1,
        ),
        role=surviving_role,
        execution_model_name="brackets",
        source_key="alpha",
        position_id=state.position_id,
    )
    fresh, replacement, controller, _ = restart_entry(runtime, broker)
    assert await controller.run() is SyncOutcome.OK
    expected_count = 2 if surviving_role == StandardOrderRole.STOP_LOSS else 3
    assert len(replacement.submitted) == expected_count
    stops = fresh.book.orders.active(
        source_key="alpha", role=StandardOrderRole.STOP_LOSS
    )
    assert len(stops) == 1
    assert stops[0].trade.order.ocaGroup == "existing-group"
    assert await controller.sync() is SyncOutcome.OK
    assert len(replacement.submitted) == expected_count


async def test_partial_entry_recovers_binding_but_not_brackets(
    entry_path, restart_entry
):
    """A still-working entry gets protection only when its final fill arrives."""
    runtime, broker, _, _ = entry_path
    await broker.fill(broker.submitted[0], 1, notify_events=False)
    # Its live open Trade carries the offline partial execution on reconnect.
    fresh, replacement, controller, model = restart_entry(runtime, broker)
    trade = replacement.submitted[0]
    assert await controller.run() is SyncOutcome.OK
    assert fresh.book.positions.for_source("alpha").quantity == 1
    assert len(fresh.book.orders.by_id(1).fills) == 1
    assert await controller.sync() is SyncOutcome.OK
    assert fresh.book.positions.for_source("alpha").quantity == 1
    assert len(fresh.book.orders.by_id(1).fills) == 1
    model.recover()
    assert len(replacement.submitted) == 1
    await replacement.fill(trade)
    assert len(replacement.submitted) == 3
    assert fresh.book.positions.for_source("alpha").quantity == 3


async def test_closed_episode_is_not_protected_again(entry_path, restart_entry):
    """An entry and its close may both complete before protection recovery runs."""
    runtime, broker, _, _ = entry_path
    await broker.fill(broker.submitted[0], notify_filled=False)
    state = runtime.book.positions.for_source("alpha")
    close = runtime.controller.trade(
        state.contract,
        ibi.MarketOrder("SELL", 3),
        role=StandardOrderRole.CLOSE,
        execution_model_name="brackets",
        source_key="alpha",
        position_id=state.position_id,
    )
    await broker.fill(close, notify_events=False)
    fresh, replacement, controller, _ = restart_entry(runtime, broker)
    assert await controller.run() is SyncOutcome.OK
    assert fresh.book.positions.for_source("alpha").quantity == 0
    assert len(replacement.submitted) == 2


async def test_active_close_keeps_its_own_sequencing(entry_path, restart_entry):
    """Remove policy must not submit a second close during missed-entry recovery."""
    runtime, broker, _, _ = entry_path
    await broker.fill(broker.submitted[0], notify_filled=False)
    state = runtime.book.positions.for_source("alpha")
    runtime.controller.trade(
        state.contract,
        ibi.MarketOrder("SELL", 3),
        role=StandardOrderRole.CLOSE,
        execution_model_name="brackets",
        source_key="alpha",
        position_id=state.position_id,
    )
    _, replacement, controller, _ = restart_entry(runtime, broker)
    assert await controller.run() is SyncOutcome.OK
    assert len(replacement.submitted) == 2


async def test_previous_stop_is_not_recreated(entry_path, restart_entry):
    """Missing previously installed protection remains the configured policy's job."""
    runtime, broker, _, _ = entry_path
    await broker.fill(broker.submitted[0])
    broker.cancelOrder(broker.submitted[1].order)
    await settle_events()
    fresh, replacement, controller, model = restart_entry(runtime, broker)
    controller.missing_brackets = "warn"
    assert await controller.run() is SyncOutcome.OK
    model.recover()
    assert len(replacement.submitted) == 3
    assert not fresh.book.orders.active(
        source_key="alpha", role=StandardOrderRole.STOP_LOSS
    )


@pytest.mark.parametrize("missing", ["fills", "inputs"])
async def test_incomplete_evidence_fails_without_fabricated_stop(entry_path, missing):
    """Critical recovery failure is explicit even when missing_brackets is ignore."""
    runtime, broker, model, _ = entry_path
    await broker.fill(broker.submitted[0], notify_filled=False)
    if missing == "fills":
        runtime.book.orders.by_id(1).fills = ()
    else:
        state = runtime.book.positions.for_source("alpha")
        runtime.book.update_position(replace(state, bracket_inputs={}))
    with pytest.raises((RuntimeError, KeyError), match="evidence|bracket input"):
        model.recover_protection()
    assert len(broker.submitted) == 1


async def test_controller_disables_on_initial_protection_recovery_failure(
    entry_path, restart_entry
):
    """Ignore policy does not conceal a failed attempted initial installation."""
    runtime, broker, _, _ = entry_path
    await broker.fill(broker.submitted[0], notify_filled=False)
    state = runtime.book.positions.for_source("alpha")
    runtime.book.update_position(replace(state, bracket_inputs={}))
    fresh, replacement, controller, _ = restart_entry(runtime, broker)
    controller.missing_brackets = "ignore"
    assert await controller.run() is SyncOutcome.FAILED
    assert controller._trading_disabled
    assert fresh.book.positions.for_source("alpha").quantity == 3
    assert len(replacement.submitted) == 1


async def test_recovery_uses_held_inputs_not_pending_target(entry_path, restart_entry):
    """New target inputs cannot overwrite the original episode's protection basis."""
    runtime, broker, _, _ = entry_path
    await broker.fill(broker.submitted[0], notify_filled=False)
    state = runtime.book.positions.for_source("alpha")
    runtime.book.update_position(replace(state, target_bracket_inputs={"atr": 99}))
    _, replacement, controller, _ = restart_entry(runtime, broker)
    assert await controller.run() is SyncOutcome.OK
    assert replacement.submitted[1].order.auxPrice == 90


async def test_recovery_requires_real_contract_tick(entry_path):
    """Offline repair must not price orders using the bracket leg's fallback tick."""
    runtime, broker, model, _ = entry_path
    await broker.fill(broker.submitted[0], notify_filled=False)
    runtime.contract_registry.details.clear()
    with pytest.raises(RuntimeError, match="minimum tick"):
        model.recover_protection()
    assert len(broker.submitted) == 1


async def test_delayed_entry_callback_cannot_protect_replacement_episode(entry_path):
    """The source key alone is not enough to authorize installing old brackets."""
    runtime, broker, model, _ = entry_path
    await broker.fill(broker.submitted[0], notify_filled=False)
    state = runtime.book.positions.for_source("alpha")
    runtime.book.update_position(replace(state, position_id="new-episode"))
    model.ensure_entry_brackets(broker.submitted[0])
    assert len(broker.submitted) == 1


async def test_suppressed_stop_does_not_leave_only_take_profit(entry_path, monkeypatch):
    """A failed critical submission is surfaced before attempting an optional exit."""
    runtime, broker, model, _ = entry_path
    await broker.fill(broker.submitted[0], notify_filled=False)
    submission = Mock(return_value=None)
    monkeypatch.setattr(runtime.controller, "trade", submission)
    with pytest.raises(RuntimeError, match="stop submission was suppressed"):
        model.ensure_entry_brackets(broker.submitted[0])
    assert submission.call_count == 1
    assert submission.call_args.kwargs["role"] == StandardOrderRole.STOP_LOSS
