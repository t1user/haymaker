"""Independent real-Controller roll scenarios, driven only at the IB boundary."""

from datetime import datetime, timedelta, timezone
from dataclasses import replace

import ib_insync as ibi
import pytest
from episode_harness import EpisodeBroker, settle_events

from haymaker.book import FutureRollStage
from haymaker.components import (
    BracketExecutionModel,
    FixedStop,
    PositionIntent,
    PositionTarget,
    SerialTargetExecutionModel,
    StandardOrderRole,
    TakeProfitAsStopMultiple,
)
from haymaker.contract_registry import ContractRegistry
from haymaker.controller import Controller
from haymaker.controller.controller import SyncOutcome
from haymaker.details_processor import Details


@pytest.mark.parametrize(
    "unavailable", ["disconnected", "startup", "recovering", "syncing"]
)
async def test_daily_roll_defers_unavailable_state_without_persisting_block(
    rolling, monkeypatch, unavailable
):
    """Process timers cannot convert an empty outage cache into durable damage."""
    runtime, broker, controller, (old, _, _) = rolling
    model = SerialTargetExecutionModel(name="serial")
    model.onData(PositionTarget(contract=old, target_quantity=2))
    await broker.fill(broker.submitted[-1])
    original_positions = broker.positions
    monkeypatch.setattr(broker, "positions", lambda: [])
    if unavailable == "disconnected":
        monkeypatch.setattr(broker, "isConnected", lambda: False)
        broker.disconnectedEvent.emit()
    elif unavailable == "startup":
        controller.suspend_broker_work()
    elif unavailable == "recovering":
        import asyncio

        event = asyncio.Event()
        event.set()
        controller.set_sync_abort_event(event)
    else:
        await controller._sync_lock.acquire()
    controller.roll_futures()
    assert runtime.book.rolls.all() == ()
    assert len(broker.submitted) == 1
    monkeypatch.setattr(broker, "positions", original_positions)
    monkeypatch.setattr(broker, "isConnected", lambda: True)
    if unavailable == "recovering":
        event.clear()
    if unavailable == "syncing":
        controller._sync_lock.release()
    assert await controller.sync() is SyncOutcome.OK
    controller.roll_futures()
    assert isinstance(broker.submitted[-1].contract, ibi.Bag)


@pytest.mark.parametrize("close_first", [False, True])
async def test_bracket_roll_refreshes_after_pending_entry_or_close(
    rolling, close_first
):
    """Pending entry grows the roll; pending close removes it without a BAG."""
    runtime, broker, controller, (old, active, _) = rolling
    model = BracketExecutionModel("alpha", name="brackets", stop=FixedStop(2))
    model.onData(
        PositionTarget(
            source_key="alpha",
            contract=old,
            target_quantity=2,
            intent=PositionIntent.OPEN,
            metadata={"atr": 5},
        )
    )
    entry = broker.submitted[-1]
    await broker.fill(entry, 1)
    if close_first:
        await broker.fill(entry, 1)
        model.onData(
            PositionTarget(
                source_key="alpha",
                contract=old,
                target_quantity=0,
                intent=PositionIntent.CLOSE,
            )
        )
    pending = broker.submitted[-1]
    controller.future_roller.roll()
    key = runtime.contract_registry.series_key(old)
    assert (
        runtime.book.rolls.for_series(key).stage
        is FutureRollStage.WAITING_FOR_ACTIVE_WORK
    )
    await broker.fill(pending)
    await settle_events()
    if close_first:
        assert not any(isinstance(t.contract, ibi.Bag) for t in broker.submitted)
        assert runtime.book.rolls.for_series(key).stage is FutureRollStage.COMPLETE
        assert runtime.book.positions.for_source("alpha").quantity == 0
    else:
        combo = broker.submitted[-1]
        assert isinstance(combo.contract, ibi.Bag)
        assert combo.order.totalQuantity == 2
        await broker.fill(combo)
        assert (
            runtime.book.positions.for_source("alpha").quantity
            == broker.quantities[active]
            == 2
        )


async def test_bracket_roll_rejects_replaced_episode_before_submission(rolling):
    """Recovery cannot use an old episode's plan for a new episode."""
    runtime, broker, controller, (old, _, _) = rolling
    model = BracketExecutionModel("alpha", name="brackets", stop=FixedStop(2))
    model.onData(
        PositionTarget(
            source_key="alpha",
            contract=old,
            target_quantity=2,
            intent=PositionIntent.OPEN,
            metadata={"atr": 5},
        )
    )
    entry = broker.submitted[-1]
    await broker.fill(entry, 1)
    controller.future_roller.roll()
    key = runtime.contract_registry.series_key(old)
    # Simulate incompatible restored episode state, not a normal fill transition.
    runtime.book.update_position(
        replace(runtime.book.positions.for_source("alpha"), position_id="different")
    )
    await broker.fill(entry, 1)
    assert runtime.book.rolls.for_series(key).stage is FutureRollStage.BLOCKED
    assert not any(isinstance(t.contract, ibi.Bag) for t in broker.submitted)


@pytest.fixture
def rolling(atom_runtime_factory, monkeypatch):
    """Qualify a full real registry chain and install the independent broker."""
    now = datetime.now(timezone.utc)
    chain = [
        ibi.Future(
            "ES",
            conId=index,
            exchange="CME",
            currency="USD",
            multiplier="50",
            localSymbol=f"ES-{index}",
            lastTradeDateOrContractMonth=(now + timedelta(days=days)).strftime(
                "%Y%m%d"
            ),
        )
        for index, days in enumerate((-30, 60, 150), 1)
    ]
    registry = ContractRegistry()
    registry.register_blueprint(ibi.Future("ES", exchange="CME"))
    registry.reset_data(
        [
            [
                ibi.ContractDetails(contract=c, timeZoneId="UTC", minTick=0.25)
                for c in chain
            ]
        ]
    )
    monkeypatch.setattr(Details, "is_open", lambda self: True)
    broker = EpisodeBroker()
    broker.register_contracts(*chain)
    runtime = atom_runtime_factory(ib=broker, contract_registry=registry)
    controller = Controller(trader=runtime.trader)
    runtime.bind_controller(controller)
    controller._broker_ready = True
    monkeypatch.setattr(broker, "isConnected", lambda: True)
    return runtime, broker, controller, chain


@pytest.mark.parametrize("newer_target", [False, True])
async def test_direct_roll_transfers_once_and_respects_new_explicit_target(
    rolling, newer_target
):
    """Physical combo legs and concrete targets agree after resumed completion."""
    runtime, broker, controller, (old, active, next_) = rolling
    model = SerialTargetExecutionModel(name="serial")
    for contract, quantity in ((old, 2), (active, 1)):
        model.onData(PositionTarget(contract=contract, target_quantity=quantity))
        await broker.fill(broker.submitted[-1])
    notifications = []
    controller.future_roller.completedEvent += notifications.append
    controller.future_roller.roll()
    combo = broker.submitted[-1]
    assert isinstance(combo.contract, ibi.Bag)
    if newer_target:
        model.onData(PositionTarget(contract=active, target_quantity=5))
        assert broker.submitted[-1] is combo
    await broker.fill(combo, 1)
    assert runtime.book.positions.quantity(old) == broker.quantities[old] == 1
    assert notifications == []
    # Exercise recovery without the Trade's filledEvent callback.
    await broker.fill(combo, 1, notify_filled=False)
    controller.future_roller.recover()
    await settle_events()
    assert len(notifications) == 1
    assert runtime.book.targets.for_contract(old).target_quantity == 0
    assert runtime.book.targets.for_contract(active).target_quantity == (
        5 if newer_target else 3
    )
    if newer_target:
        adjustment = broker.submitted[-1]
        assert adjustment.contract == active
        assert adjustment.order.totalQuantity == 2
        await broker.fill(adjustment)
    assert runtime.book.positions.quantity(active) == broker.quantities[active]
    assert runtime.book.positions.quantity(old) == broker.quantities[old] == 0
    count = len(broker.submitted)
    controller.future_roller.recover()
    model.recover()
    assert len(broker.submitted) == count


@pytest.mark.parametrize("reverse_during_roll", [False, True])
async def test_bracket_roll_preserves_episode_and_pending_reversal(
    rolling, reverse_during_roll
):
    """Replacement protection precedes continuation; pending OPEN stays distinct."""
    runtime, broker, controller, (old, active, next_) = rolling
    model = BracketExecutionModel(
        "alpha",
        name="brackets",
        stop=FixedStop(2),
        take_profit=TakeProfitAsStopMultiple(2, 2),
    )
    model.onData(
        PositionTarget(
            source_key="alpha",
            contract=old,
            target_quantity=2,
            intent=PositionIntent.OPEN,
            metadata={"atr": 5},
        )
    )
    await broker.fill(broker.submitted[-1])
    episode_id = runtime.book.positions.for_source("alpha").position_id
    controller.future_roller.roll()
    combo = broker.submitted[-1]
    assert isinstance(combo.contract, ibi.Bag)
    if reverse_during_roll:
        model.onData(
            PositionTarget(
                source_key="alpha",
                contract=next_,
                target_quantity=-2,
                intent=PositionIntent.REVERSE,
                metadata={"atr": 9},
            )
        )
    await broker.fill(combo, notify_filled=False)
    controller.future_roller.recover()
    await settle_events()
    state = runtime.book.positions.for_source("alpha")
    assert state.contract == active
    assert state.position_id == episode_id
    assert state.target_contract == (next_ if reverse_during_roll else old)
    assert state.bracket_inputs == {"atr": 5}
    key = runtime.contract_registry.series_key(old)
    assert runtime.book.rolls.for_series(key).stage is FutureRollStage.COMPLETE
    stops = runtime.book.orders.active(
        source_key="alpha", role=StandardOrderRole.STOP_LOSS
    )
    assert len(stops) == 1
    assert stops[0].trade.contract == active
    assert stops[0].position_id == episode_id
    assert broker.quantities[old] == 0
    assert broker.quantities[active] == state.quantity == 2
    if reverse_during_roll:
        close = broker.submitted[-1]
        assert close.contract == active
        assert (
            runtime.book.orders.by_id(close.order.orderId).role
            == StandardOrderRole.CLOSE
        )
        await broker.fill(close)
        opening = broker.submitted[-1]
        assert opening.contract == next_
        await broker.fill(opening)
        assert runtime.book.positions.for_source("alpha").position_id != episode_id
        assert runtime.book.positions.for_source("alpha").quantity == -2
