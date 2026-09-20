"""Safety boundaries for inferred position corrections and offline recovery."""

import asyncio
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import ib_insync as ibi
import pytest
from test_controller import FakeTrader, contract, fill, set_broker_state

from haymaker.book import (
    Book,
    FutureRollMode,
    PositionState,
    RollParticipant,
    RollState,
    TargetState,
)
from haymaker.components import StandardOrderRole
from haymaker.controller import Controller
from haymaker.controller.controller import SyncOutcome
from haymaker.controller.sync_coordinator import SyncCoordinator


@pytest.mark.asyncio
@pytest.mark.parametrize("restart_before_correction", [False, True])
@pytest.mark.parametrize("unknown_order", [False, True])
@pytest.mark.parametrize("accounting", ["episode", "opposing", "direct"])
async def test_fail_preserves_accounting_and_blocks_recovery(
    atom_runtime,
    monkeypatch,
    state_saver,
    order_saver,
    caplog,
    restart_before_correction,
    unknown_order,
    accounting,
):
    """No inference, order shortcut, or reconnect can clear a disputed holding."""
    trader = FakeTrader()
    controller = Controller(trader)
    atom_runtime.bind_controller(controller)
    held = contract()
    now = datetime.now(timezone.utc)
    if accounting == "direct":
        opening = trader.trade(held, ibi.MarketOrder("BUY", 2))
        controller.register_order(
            opening,
            role=StandardOrderRole.TARGET_ADJUSTMENT,
            execution_model_name="serial",
        )
        controller.book.apply_fill(opening, fill(opening, quantity=2))
        controller.book.update_target(
            TargetState(
                execution_model_name="serial",
                contract=held,
                target_quantity=2,
                target_created_at=now,
            )
        )
    else:
        for source, quantity in (
            [("alpha", 2), ("beta", -1)] if accounting == "opposing" else [("alpha", 2)]
        ):
            controller.book.update_position(
                PositionState(
                    source_key=source,
                    execution_model_name="brackets",
                    contract=held,
                    quantity=quantity,
                    target_quantity=quantity,
                    target_created_at=now,
                    position_id=f"{source}-episode",
                    bracket_inputs={"atr": 2},
                    blocked_direction=1,
                )
            )
    before = deepcopy(state_saver.read({}))
    expected_positions = controller.book.positions.by_contract()
    unknown = ibi.Trade(
        contract=held,
        order=ibi.LimitOrder("SELL", 1, 100, orderId=99, permId=199),
        orderStatus=ibi.OrderStatus(status=ibi.OrderStatus.Submitted),
    )
    set_broker_state(
        controller, monkeypatch, open_trades=(unknown,) if unknown_order else ()
    )
    monkeypatch.setattr(controller.ib, "isConnected", lambda: True)
    monkeypatch.setattr(controller.ib, "reqPositionsAsync", AsyncMock(return_value=[]))
    recover_rolls = Mock()
    recover_protection = Mock()
    monkeypatch.setattr(controller.future_roller, "recover", recover_rolls)
    monkeypatch.setattr(controller, "recover_protection", recover_protection)
    atom_runtime.request_restart = Mock()
    controller._restart_before_correction = restart_before_correction

    assert await controller.sync() is SyncOutcome.FAILED

    assert controller._trading_disabled
    assert state_saver.read({}) == before
    assert controller.book.positions.by_contract() == expected_positions
    recover_rolls.assert_not_called()
    recover_protection.assert_not_called()
    atom_runtime.request_restart.assert_not_called()
    assert "policy=fail" in caplog.text
    assert "conId=1" in caplog.text
    assert "broker=0.0" in caplog.text
    assert "repair accounting offline" in caplog.text
    assert not trader.cancelled
    assert (
        controller.trade(
            held,
            ibi.MarketOrder("BUY", 1),
            role="OPEN",
            execution_model_name="brackets",
        )
        is None
    )

    # Even an apparently clean later snapshot cannot unlatch this process.
    positions = [
        ibi.Position("DU123", c, q, 100) for c, q in expected_positions.items()
    ]
    monkeypatch.setattr(controller.ib, "positions", lambda: positions)
    monkeypatch.setattr(
        controller.ib, "reqPositionsAsync", AsyncMock(return_value=positions)
    )
    assert await controller.run() is SyncOutcome.FAILED
    controller.ib.reqPositionsAsync.assert_not_awaited()

    restored = Book(
        order_saver=order_saver, state_saver=state_saver, save_async=False, restore=True
    )
    assert restored.positions.by_contract() == expected_positions


@pytest.mark.asyncio
async def test_correct_requires_a_fresh_clean_pass(controller, monkeypatch):
    """Opt-in correction supersedes a stale target, then verifies the result."""
    controller.position_mismatch_policy = "correct"
    controller._restart_before_correction = False
    controller.sync_resync_delay = 0
    controller.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=contract(),
            quantity=2,
            target_quantity=2,
            position_id="episode",
            bracket_inputs={"atr": 2},
        )
    )
    set_broker_state(controller, monkeypatch)
    monkeypatch.setattr(controller.ib, "isConnected", lambda: True)
    requested = AsyncMock(return_value=[])
    monkeypatch.setattr(controller.ib, "reqPositionsAsync", requested)

    assert await controller.sync() is SyncOutcome.OK
    assert requested.await_count == 2
    state = controller.book.positions.for_source("alpha")
    assert state.quantity == state.target_quantity == 0
    assert state.position_id is None
    assert state.bracket_inputs == {}
    assert not controller._trading_disabled


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["OPEN", "CLOSE", "TARGET_ADJUSTMENT", "ROLL"])
async def test_active_work_defers_decision_until_it_settles(
    controller, monkeypatch, role
):
    """Working adjustments/roll endpoints defer; a pending roll alone does not."""
    held = contract()
    controller.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=held,
            quantity=2,
            target_quantity=2,
        )
    )
    params = None
    if role == "ROLL":
        destination = ibi.Future(
            conId=2, symbol="ES", exchange="CME", localSymbol="ESU6"
        )
        controller.book.update_roll(
            RollState(
                series_key="es",
                mode=FutureRollMode.BRACKET,
                executor_name="rolls",
                old_contract=held,
                new_contract=destination,
                participants=(
                    RollParticipant(
                        source_key="alpha",
                        execution_model_name="brackets",
                        quantity=2,
                    ),
                ),
            )
        )
        params = {
            "roll_state_key": "es",
            "old_contract": held,
            "new_contract": destination,
        }
    working = ibi.Trade(
        contract=ibi.Bag() if role == "ROLL" else held,
        order=ibi.MarketOrder("BUY", 2, orderId=77, permId=177),
        orderStatus=ibi.OrderStatus(status=ibi.OrderStatus.Submitted, remaining=2),
    )
    controller.register_order(
        working,
        role=role,
        execution_model_name="test",
        params=params,
        source_key=None if role == "TARGET_ADJUSTMENT" else "alpha",
    )
    set_broker_state(controller, monkeypatch, open_trades=(working,))
    monkeypatch.setattr(controller.ib, "isConnected", lambda: True)
    monkeypatch.setattr(controller.ib, "reqPositionsAsync", AsyncMock(return_value=[]))
    recover = Mock()
    monkeypatch.setattr(controller.future_roller, "recover", recover)
    assert await SyncCoordinator(controller).run()
    assert controller.book.positions.quantity(held) == 2

    working.orderStatus.status = ibi.OrderStatus.Cancelled
    set_broker_state(controller, monkeypatch)
    recover.reset_mock()
    assert await controller.sync() is SyncOutcome.FAILED
    recover.assert_not_called()
    assert controller.book.positions.quantity(held) == 2


@pytest.mark.asyncio
async def test_known_fills_still_account_after_failure(controller, monkeypatch):
    """The submission latch must not discard fills from existing broker orders."""
    working = ibi.Trade(
        contract=contract(),
        order=ibi.MarketOrder("BUY", 1, orderId=77, permId=177),
    )
    controller.register_order(
        working, role="OPEN", execution_model_name="brackets", source_key="alpha"
    )
    controller.disable_trading("position mismatch")
    execution = fill(working)
    await controller.onExecDetailsEvent(working, execution)
    await controller.onExecDetailsEvent(working, execution)
    report = ibi.CommissionReport(
        execId=execution.execution.execId, commission=1, currency="USD"
    )
    await controller.onCommissionReport(working, execution, report)
    assert controller.book.positions.quantity(contract()) == 1
    assert controller.book.orders.by_id(77).fills[0].commission_report.commission == 1


@pytest.mark.asyncio
async def test_offline_episode_repair_requires_a_new_controller(
    controller, atom_runtime, monkeypatch, order_saver, state_saver
):
    """A repaired durable checkpoint is accepted only after process restoration."""
    held = contract()
    original = PositionState(
        source_key="alpha",
        execution_model_name="brackets",
        contract=held,
        quantity=2,
        target_quantity=2,
        position_id="episode",
    )
    controller.book.update_position(original)
    set_broker_state(controller, monkeypatch)
    monkeypatch.setattr(controller.ib, "isConnected", lambda: True)
    monkeypatch.setattr(controller.ib, "reqPositionsAsync", AsyncMock(return_value=[]))
    assert await controller.sync() is SyncOutcome.FAILED

    # Only fake persistence is edited. Production repair is deliberately offline.
    repaired = replace(original, quantity=0, target_quantity=0, position_id=None)
    state_saver.save(repaired.encode())
    restored = Book(
        order_saver=order_saver, state_saver=state_saver, restore=True, save_async=False
    )
    atom_runtime.book = restored
    assert await controller.sync() is SyncOutcome.FAILED
    restarted = Controller(FakeTrader())
    atom_runtime.bind_controller(restarted)
    assert await restarted.run() is SyncOutcome.OK
    assert restored.positions.by_contract() == {}


@pytest.mark.asyncio
async def test_new_broker_contract_fails_without_partial_repair(
    controller, monkeypatch, state_saver
):
    """A replacement/new holding cannot clear the old episode before failure."""
    controller.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=contract(),
            quantity=2,
            target_quantity=2,
            position_id="episode",
        )
    )
    before = deepcopy(state_saver.read({}))
    replacement = ibi.Stock("NEW", "SMART", "USD", conId=2)
    positions = (ibi.Position("DU123", replacement, 10, 100),)
    set_broker_state(controller, monkeypatch, positions=positions)
    monkeypatch.setattr(controller.ib, "isConnected", lambda: True)
    monkeypatch.setattr(
        controller.ib, "reqPositionsAsync", AsyncMock(return_value=list(positions))
    )
    assert await controller.sync() is SyncOutcome.FAILED
    assert state_saver.read({}) == before


@pytest.mark.asyncio
async def test_sync_cycles_serialize_broker_position_requests(controller, monkeypatch):
    """Concurrent periodic/startup checks cannot race IB's positions request."""
    set_broker_state(controller, monkeypatch)
    monkeypatch.setattr(controller.ib, "isConnected", lambda: True)
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def request():
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return []

    monkeypatch.setattr(controller.ib, "reqPositionsAsync", request)
    first = asyncio.create_task(controller.sync())
    await asyncio.wait_for(entered.wait(), 1)
    second = asyncio.create_task(controller.sync())
    await asyncio.sleep(0)
    assert calls == 1
    release.set()
    assert await asyncio.gather(first, second) == [SyncOutcome.OK, SyncOutcome.OK]
