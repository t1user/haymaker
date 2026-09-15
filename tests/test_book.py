from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any
from unittest.mock import Mock

import ib_insync as ibi
import pytest

from haymaker.blotter import Blotter
from haymaker.book import (
    Book,
    FillRecord,
    FutureRollMode,
    FutureRollStage,
    OrderInfo,
    PositionState,
    RollParticipant,
    RollState,
    TargetState,
)
from haymaker.saver import AbstractBaseSaver


def contract(con_id: int = 1) -> ibi.Future:
    return ibi.Future(
        conId=con_id,
        symbol="ES",
        exchange="CME",
        localSymbol=f"ES-{con_id}",
    )


@pytest.mark.parametrize("source_key", [None, "alpha"])
def test_position_queries_read_balances_without_history(book, monkeypatch, source_key):
    """Both modes read maintained balances even when history is unavailable."""
    for con_id, quantity in ((1, 2), (2, 3)):
        trade_ = trade(order_id=con_id, con_id=con_id, quantity=quantity)
        book.save_order(
            order_info(
                trade_, source_key=(f"{source_key}-{con_id}" if source_key else None)
            )
        )
        book.apply_fill(
            trade_, fill(trade_, exec_id=f"fill-{con_id}", quantity=quantity)
        )
    book.update_position(
        PositionState(
            source_key="offset",
            execution_model_name="model",
            contract=contract(1),
            quantity=-1,
        )
    )
    for method in (
        "_order_contribution",
        "_episode_contribution",
        "_roll_contribution",
        "_rebuild",
    ):
        monkeypatch.setattr(
            book.positions,
            method,
            Mock(side_effect=AssertionError("query rebuilt history")),
        )
    assert book.positions.by_contract() == {contract(1): 1, contract(2): 3}
    assert book.positions.quantity(contract(1)) == 1
    assert book.positions.quantity(contract(2)) == 3
    assert book.positions.quantity(contract(3)) == 0
    result = book.positions.by_contract()
    result.clear()
    assert book.positions.by_contract() == {contract(1): 1, contract(2): 3}


def trade(
    *,
    order_id: int = 1,
    perm_id: int = 101,
    side: str = "BUY",
    quantity: float = 2,
    con_id: int = 1,
) -> ibi.Trade:
    return ibi.Trade(
        contract=contract(con_id),
        order=ibi.Order(
            orderId=order_id,
            permId=perm_id,
            action=side,
            totalQuantity=quantity,
        ),
        orderStatus=ibi.OrderStatus(
            orderId=order_id,
            status=ibi.OrderStatus.Submitted,
            remaining=quantity,
        ),
    )


def fill(
    trade_: ibi.Trade,
    *,
    exec_id: str = "exec-1",
    quantity: float = 1,
) -> ibi.Fill:
    return ibi.Fill(
        contract=trade_.contract,
        execution=ibi.Execution(
            execId=exec_id,
            orderId=trade_.order.orderId,
            permId=trade_.order.permId,
            side="BOT" if trade_.order.action == "BUY" else "SLD",
            shares=quantity,
            price=5000,
            time=datetime.now(timezone.utc),
        ),
        commissionReport=ibi.CommissionReport(
            execId=exec_id,
            commission=1,
            realizedPNL=2,
        ),
        time=datetime.now(timezone.utc),
    )


def order_info(trade_: ibi.Trade, **kwargs) -> OrderInfo:
    return OrderInfo(
        trade=trade_,
        role=kwargs.pop("role", "OPEN"),
        submitted_at=datetime.now(timezone.utc),
        execution_model_name=kwargs.pop("execution_model_name", "brackets"),
        source_key=kwargs.pop("source_key", "alpha"),
        position_id=kwargs.pop("position_id", "episode-1"),
        params=kwargs.pop("params", {"atr": 10}),
        **kwargs,
    )


def test_order_info_persists_complete_trade_identifiers_and_fill():
    trade_ = trade()
    record = FillRecord.from_fill(trade_, fill(trade_))
    info = order_info(trade_, fills=(record,))

    encoded = info.encode()

    assert encoded["orderId"] == 1
    assert encoded["clientId"] == 0
    assert encoded["permId"] == 101
    assert encoded["fills"][0]["deduplication_key"] == "exec-1"
    assert encoded["trade"]["Trade"]["order"]["Order"]["permId"] == 101


def test_order_info_requires_real_order_id_when_saved(book):
    with pytest.raises(ValueError, match="orderId 0"):
        book.save_order(order_info(trade(order_id=0)))


def test_execution_trade_merges_saved_and_broker_fills_without_accounting():
    """Reconstruction preserves evidence and derives a quantity-weighted price."""
    entry = trade(quantity=3)
    first = fill(entry, quantity=2)
    first.execution.price = 100
    second = fill(entry, exec_id="exec-2")
    second.execution.price = 103
    info = order_info(entry, fills=(FillRecord.from_fill(entry, first),))
    reconstructed = info.execution_trade((first, second, second))
    assert reconstructed.orderStatus.avgFillPrice == 101
    assert reconstructed.orderStatus.filled == 3
    assert reconstructed.orderStatus.remaining == 0
    assert reconstructed.orderStatus.status == ibi.OrderStatus.Filled
    assert len(reconstructed.fills) == 2
    assert len(info.fills) == 1
    assert entry.orderStatus.filled == 0


def test_execution_trade_rejects_conflicting_duplicate_evidence():
    """A repeated execId cannot silently change the recovered entry price."""
    entry = trade()
    first = fill(entry)
    conflicting = fill(entry)
    conflicting.execution.price += 10
    info = order_info(entry, fills=(FillRecord.from_fill(entry, first),))
    with pytest.raises(ValueError, match="Conflicting execution"):
        info.execution_trade((conflicting,))


def test_execution_trade_does_not_count_combo_legs_as_extra_bag_fills():
    """Generic order reconstruction must preserve roll evidence without double counting."""
    entry = trade(quantity=1)
    entry.contract = ibi.Bag(symbol="ES", exchange="CME")
    combo = fill(entry)
    combo.execution.price = 5
    leg = fill(entry, exec_id="leg-exec")
    leg = ibi.Fill(contract(), leg.execution, leg.commissionReport, leg.time)
    reconstructed = order_info(entry).execution_trade((combo, leg))
    assert reconstructed.orderStatus.filled == 1
    assert reconstructed.orderStatus.avgFillPrice == 5
    assert len(reconstructed.fills) == 2


@pytest.mark.parametrize("quantity,price", [(0, 100), (3, 100), (1, float("nan"))])
def test_execution_trade_rejects_invalid_execution_values(quantity, price):
    """Incomplete or corrupt evidence must not become valid-looking price data."""
    entry = trade()
    execution = fill(entry, quantity=quantity)
    execution.execution.price = price
    with pytest.raises(ValueError):
        order_info(entry).execution_trade((execution,))


def test_book_does_not_restore_by_default(order_saver, state_saver, monkeypatch):
    """Bare Book construction should remain storage-free for focused callers."""

    order_read = Mock(wraps=order_saver.read)
    state_read = Mock(wraps=state_saver.read)
    monkeypatch.setattr(order_saver, "read", order_read)
    monkeypatch.setattr(state_saver, "read", state_read)

    Book(
        order_saver=order_saver,
        state_saver=state_saver,
        save_async=False,
    )

    order_read.assert_not_called()
    state_read.assert_not_called()


def test_book_restore_failure_stops_construction(order_saver, state_saver, monkeypatch):
    """A persistence failure should prevent a partially restored Book."""

    monkeypatch.setattr(
        order_saver,
        "read",
        Mock(side_effect=RuntimeError("store failed")),
    )

    with pytest.raises(RuntimeError, match="store failed"):
        Book(
            order_saver=order_saver,
            state_saver=state_saver,
            save_async=False,
            restore=True,
        )


def test_order_lookup_uses_order_id_then_perm_id(book):
    info = book.save_order(order_info(trade()))

    assert book.orders.by_id(1) is info
    assert book.orders.by_perm_id(101) is info


@pytest.mark.parametrize("source_key", [None, "alpha"])
def test_shared_balances_follow_fills_not_targets(book, source_key):
    """Partial fills, opposite executions and duplicate callbacks share accounting."""
    opening = trade(quantity=3)
    info = book.save_order(order_info(opening, source_key=source_key))
    book.update_target(
        TargetState(
            execution_model_name="model",
            contract=opening.contract,
            target_quantity=10,
            target_created_at=datetime.now(timezone.utc),
        )
    )
    first = fill(opening, quantity=1)
    assert book.apply_fill(opening, first)
    assert not book.apply_fill(opening, first)
    book.save_order(info)
    assert book.positions.quantity(opening.contract) == 1
    book.apply_fill(opening, fill(opening, exec_id="second", quantity=2))
    closing = trade(order_id=2, side="SELL", quantity=4)
    book.save_order(order_info(closing, source_key=source_key, role="CLOSE"))
    book.apply_fill(closing, fill(closing, exec_id="close", quantity=3))
    assert book.positions.quantity(opening.contract) == 0
    assert book.positions.by_contract() == {}
    # Same Contract, new execution; a zero balance is not an accounting cutoff.
    book.apply_fill(closing, fill(closing, exec_id="short", quantity=1))
    assert book.positions.quantity(opening.contract) == -1


@pytest.mark.parametrize("source_key", [None, "alpha"])
def test_balance_write_interruption_recovers_without_double_fill(
    book, order_saver, state_saver, monkeypatch, source_key
):
    """Order/episode evidence recovers a missing subsequent balance write."""
    opening = trade()
    book.save_order(order_info(opening, source_key=source_key))
    saved = state_saver.save

    def interrupted(document):
        """Simulate process loss after evidence but before the derived balance."""
        if document.get("state_type") == "balance":
            raise RuntimeError("interrupted balance write")
        saved(document)

    execution = fill(opening)
    monkeypatch.setattr(state_saver, "save", interrupted)
    with pytest.raises(RuntimeError, match="interrupted balance"):
        book.apply_fill(opening, execution)
    monkeypatch.setattr(state_saver, "save", saved)
    for _ in range(2):
        recovered = Book(
            order_saver=order_saver,
            state_saver=state_saver,
            save_async=False,
            restore=True,
        )
        assert recovered.positions.quantity(opening.contract) == 1
        assert not recovered.apply_fill(opening, execution)
        assert recovered.positions.quantity(opening.contract) == 1
    assert state_saver.read({"state_key": "balance:1"})[0]["quantity"] == 1


@pytest.mark.parametrize("source_key", [None, "alpha"])
@pytest.mark.parametrize("saved_quantity", [None, -10, 10])
def test_restore_verifies_missing_or_inconsistent_balance(
    book, order_saver, state_saver, source_key, saved_quantity
):
    """The recovery balance is derived from accounting evidence, not a stale total."""
    opening = trade()
    book.save_order(order_info(opening, source_key=source_key))
    book.apply_fill(opening, fill(opening))
    balance = state_saver.store["state"].pop("balance:1")
    if saved_quantity is not None:
        state_saver.save({**balance, "quantity": saved_quantity})
    recovered = Book(
        order_saver=order_saver, state_saver=state_saver, save_async=False, restore=True
    )
    assert recovered.positions.by_contract() == {opening.contract: 1}
    assert state_saver.read({"state_key": "balance:1"})[0]["quantity"] == 1


def test_balance_restore_keeps_reconciled_episode_quantity(
    book, order_saver, state_saver
):
    """A broker correction must not be undone by replaying old episode fills."""
    opening = trade()
    book.save_order(order_info(opening))
    book.apply_fill(opening, fill(opening))
    book.update_position(
        replace(book.positions.for_source("alpha"), quantity=0, target_quantity=0)
    )
    recovered = Book(
        order_saver=order_saver, state_saver=state_saver, save_async=False, restore=True
    )
    assert recovered.positions.by_contract() == {}


def test_balance_repair_is_passive_before_event_loop(book, order_saver, state_saver):
    """LiveRuntime can restore/repair Book before its asyncio loop starts."""
    opening = trade()
    book.save_order(order_info(opening, source_key=None))
    book.apply_fill(opening, fill(opening))
    state_saver.store["state"].pop("balance:1")
    recovered = Book(
        order_saver=order_saver, state_saver=state_saver, save_async=True, restore=True
    )
    assert recovered.positions.quantity(opening.contract) == 1
    assert state_saver.read({"state_key": "balance:1"})[0]["quantity"] == 1


def test_balance_recovery_repairs_torn_roll_endpoints(
    book, order_saver, state_saver, monkeypatch
):
    """A crash between the two roll balance writes cannot move only one leg."""
    opening = trade(quantity=2)
    book.save_order(order_info(opening, source_key=None))
    book.apply_fill(opening, fill(opening, quantity=2))
    roll = trade(order_id=2, quantity=2)
    roll.contract = ibi.Bag(symbol="ES")
    book.save_order(
        order_info(
            roll,
            source_key=None,
            role="ROLL",
            params={"old_contract": contract(1), "new_contract": contract(2)},
        )
    )
    save = state_saver.save

    def interrupt_second_endpoint(document):
        """Persist the old zero but lose the replacement balance."""
        if document["state_key"] == "balance:2":
            raise RuntimeError("second endpoint not saved")
        save(document)

    monkeypatch.setattr(state_saver, "save", interrupt_second_endpoint)
    with pytest.raises(RuntimeError, match="second endpoint"):
        book.apply_fill(roll, fill(roll, exec_id="roll-fill", quantity=2))
    monkeypatch.setattr(state_saver, "save", save)
    assert state_saver.read({"state_key": "balance:1"})[0]["quantity"] == 0
    recovered = Book(
        order_saver=order_saver, state_saver=state_saver, save_async=False, restore=True
    )
    assert recovered.positions.by_contract() == {contract(2): 2}


def test_explicit_fill_cutoff_updates_current_balance(book):
    """Installing a recovery cutoff is a mutation, not work deferred to queries."""
    opening = trade()
    book.save_order(order_info(opening, source_key=None))
    book.apply_fill(opening, fill(opening))
    now = datetime.now(timezone.utc)
    book.update_target(
        TargetState(
            execution_model_name="model",
            contract=opening.contract,
            target_quantity=0,
            target_created_at=now,
            fill_evidence_start_at=now,
        )
    )
    assert book.positions.by_contract() == {}


def test_rebinding_and_commissions_do_not_reapply_balance(
    book, state_saver, monkeypatch
):
    """A changed broker orderId and later commission retain accounted quantity."""
    opening = trade(order_id=7, perm_id=900)
    book.save_order(order_info(opening, source_key=None))
    execution = fill(opening)
    book.apply_fill(opening, execution)
    save = Mock(wraps=state_saver.save)
    monkeypatch.setattr(state_saver, "save", save)
    rebound = trade(order_id=8, perm_id=900)
    assert book.rebind_trade(rebound) is None
    assert book.orders.by_id(7) is None
    assert not book.apply_fill(rebound, execution)
    book.update_commission(rebound, execution, execution.commissionReport)
    book.save_order(book.orders.by_id(8))
    assert book.positions.quantity(opening.contract) == 1
    save.assert_not_called()
    book.apply_fill(rebound, fill(rebound, exec_id="new-id"))
    assert book.positions.quantity(opening.contract) == 2


@pytest.mark.parametrize("source_key", [None, "alpha"])
async def test_balance_saves_follow_evidence_on_critical_queue(
    order_saver, state_saver, monkeypatch, source_key
):
    """DRAIN persists the shared balance only after the mutation establishing it."""
    writes = []
    save_order, save_state = order_saver.save, state_saver.save

    def record_order(document):
        """Capture the serialized evidence, not the subsequently mutable Trade."""
        writes.append(("order", deepcopy(document)))
        save_order(document)

    def record_state(document):
        """Record the ordered projection write."""
        writes.append((document["state_type"], deepcopy(document)))
        save_state(document)

    monkeypatch.setattr(order_saver, "save", record_order)
    monkeypatch.setattr(state_saver, "save", record_state)
    book = Book(order_saver=order_saver, state_saver=state_saver, save_async=True)
    opening = trade()
    book.save_order(order_info(opening, source_key=source_key))
    book.apply_fill(opening, fill(opening))
    assert book.positions.quantity(opening.contract) == 1
    await book.close()
    kinds = [kind for kind, document in writes]
    assert kinds == (
        ["order", "order", "balance"]
        if source_key is None
        else ["order", "order", "position", "balance"]
    )
    assert writes[1][1]["fills"][0]["deduplication_key"] == "exec-1"
    assert writes[-1][1]["quantity"] == 1


def test_roll_leg_evidence_replaces_bag_contribution(book, order_saver, state_saver):
    """Leg details replacing a BAG fallback must not double-count movement."""
    opening = trade(quantity=2)
    book.save_order(order_info(opening, source_key=None))
    book.apply_fill(opening, fill(opening, quantity=2))
    roll = trade(order_id=2, quantity=1)
    roll.contract = ibi.Bag(symbol="ES")
    book.save_order(
        order_info(
            roll,
            source_key=None,
            role="ROLL",
            params={"old_contract": contract(1), "new_contract": contract(2)},
        )
    )
    book.apply_fill(roll, fill(roll, exec_id="bag"))
    assert book.positions.by_contract() == {contract(1): 1, contract(2): 1}
    old_leg = fill(roll, exec_id="old-leg")
    old_leg = old_leg._replace(
        contract=contract(1), execution=replace(old_leg.execution, side="SLD")
    )
    new_leg = fill(roll, exec_id="new-leg")._replace(contract=contract(2))
    book.apply_fill(roll, old_leg)
    book.apply_fill(roll, new_leg)
    assert book.positions.by_contract() == {contract(1): 1, contract(2): 1}
    recovered = Book(
        order_saver=order_saver, state_saver=state_saver, save_async=False, restore=True
    )
    assert recovered.positions.by_contract() == book.positions.by_contract()
    assert not recovered.apply_fill(roll, old_leg)


def test_active_order_filters_all_attribution_fields(book):
    info = book.save_order(order_info(trade()))

    assert book.orders.active(
        source_key="alpha",
        contract=contract(),
        role="OPEN",
        execution_model_name="brackets",
    ) == (info,)
    assert book.orders.active(source_key="other") == ()


def test_apply_fill_is_idempotent_and_updates_position(book):
    trade_ = trade()
    book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=trade_.contract,
            position_id="episode-1",
        )
    )
    info = book.save_order(order_info(trade_))
    execution = fill(trade_)

    assert book.apply_fill(trade_, execution)
    assert not book.apply_fill(trade_, execution)
    assert book.positions.for_source("alpha").quantity == 1
    assert len(info.fills) == 1


def test_late_commission_updates_normalized_fill_evidence(book):
    trade_ = trade()
    book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=trade_.contract,
        )
    )
    info = book.save_order(order_info(trade_))
    execution = fill(trade_)._replace(commissionReport=ibi.CommissionReport())
    book.apply_fill(trade_, execution)
    report = ibi.CommissionReport(
        execId="exec-1",
        commission=1.25,
        realizedPNL=3.5,
    )

    assert book.update_commission(trade_, execution, report)
    assert info.fills[0].commission_report is report
    assert (
        info.encode()["fills"][0]["commission_report"]["CommissionReport"]["commission"]
        == 1.25
    )


def test_opposing_logical_positions_reconcile_to_broker_net(book):
    first = replace(
        PositionState(
            source_key="long",
            execution_model_name="one",
            contract=contract(),
        ),
        quantity=2,
    )
    second = replace(
        PositionState(
            source_key="short",
            execution_model_name="two",
            contract=contract(),
        ),
        quantity=-1,
    )
    book.update_position(first)
    book.update_position(second)

    assert book.positions.quantity(contract()) == 1
    assert book.positions.by_contract() == {contract(): 1}


def test_effective_quantity_includes_unfilled_working_order(book):
    trade_ = trade(quantity=2)
    book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=trade_.contract,
            quantity=1,
        )
    )
    book.save_order(order_info(trade_))

    assert book.effective_quantity("alpha") == 3


def test_stop_fill_persists_blocked_direction(book):
    trade_ = trade(side="SELL", quantity=1)
    book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=trade_.contract,
            quantity=1,
            position_id="episode-1",
        )
    )
    book.save_order(order_info(trade_, role="STOP_LOSS"))

    book.apply_fill(trade_, fill(trade_))

    state = book.positions.for_source("alpha")
    assert state.quantity == 0
    assert state.blocked_direction == 1


def test_close_fill_closes_episode_without_changing_block(book):
    trade_ = trade(side="SELL", quantity=1)
    book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=trade_.contract,
            quantity=1,
            target_quantity=0,
            position_id="episode-1",
            blocked_direction=1,
            bracket_inputs={"atr": 10},
        )
    )
    book.save_order(order_info(trade_, role="CLOSE"))

    book.apply_fill(trade_, fill(trade_))

    state = book.positions.for_source("alpha")
    assert state.position_id is None
    assert state.blocked_direction == 1
    assert state.bracket_inputs == {}


def test_protective_fill_closes_episode_and_latest_target(book):
    trade_ = trade(side="SELL", quantity=1)
    book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=trade_.contract,
            quantity=1,
            target_quantity=1,
            position_id="episode-1",
            bracket_inputs={"atr": 10},
        )
    )
    book.save_order(order_info(trade_, role="TAKE_PROFIT"))

    book.apply_fill(trade_, fill(trade_))

    state = book.positions.for_source("alpha")
    assert state.quantity == 0
    assert state.target_quantity == 0
    assert state.position_id is None
    assert state.blocked_direction == 1
    assert state.bracket_inputs == {}


@pytest.mark.parametrize("role", ["STOP_LOSS", "TAKE_PROFIT"])
def test_partial_protective_fill_sets_block_only_when_position_is_flat(book, role):
    trade_ = trade(side="SELL", quantity=2)
    book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=trade_.contract,
            quantity=2,
            position_id="episode-1",
        )
    )
    book.save_order(order_info(trade_, role=role))

    book.apply_fill(trade_, fill(trade_, exec_id="partial", quantity=1))

    state = book.positions.for_source("alpha")
    assert state.quantity == 1
    assert state.blocked_direction is None
    assert state.position_id == "episode-1"

    book.apply_fill(trade_, fill(trade_, exec_id="complete", quantity=1))

    state = book.positions.for_source("alpha")
    assert state.quantity == 0
    assert state.blocked_direction == 1
    assert state.position_id is None


def test_first_open_fill_clears_prior_block(book):
    trade_ = trade(side="BUY", quantity=2)
    book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=trade_.contract,
            blocked_direction=1,
            position_id="episode-2",
        )
    )
    book.save_order(order_info(trade_, role="OPEN", position_id="episode-2"))

    book.apply_fill(trade_, fill(trade_, quantity=0.5))

    state = book.positions.for_source("alpha")
    assert state.quantity == 0.5
    assert state.blocked_direction is None


def test_roll_fill_preserves_block(book):
    trade_ = trade(side="SELL", quantity=1)
    book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=trade_.contract,
            quantity=1,
            blocked_direction=1,
            position_id="episode-1",
        )
    )
    book.save_order(order_info(trade_, role="ROLL"))

    book.apply_fill(trade_, fill(trade_))

    assert book.positions.for_source("alpha").blocked_direction == 1


def test_contract_balance_recovers_from_completed_order_evidence(
    order_saver, state_saver
):
    first = Book(
        order_saver=order_saver,
        state_saver=state_saver,
        save_async=False,
    )
    trade_ = trade(quantity=2)
    info = order_info(
        trade_,
        source_key=None,
        position_id=None,
        execution_model_name="serial",
        role="TARGET_ADJUSTMENT",
    )
    first.save_order(info)
    first.apply_fill(trade_, fill(trade_, quantity=2))
    trade_.orderStatus.status = ibi.OrderStatus.Filled
    first.save_order(info)

    recovered = Book(
        order_saver=order_saver,
        state_saver=state_saver,
        save_async=False,
        restore=True,
    )

    assert recovered.orders.active() == ()
    assert recovered.positions.quantity(contract()) == 2


def test_target_state_rejects_stale_target(book):
    newer = TargetState(
        execution_model_name="serial",
        contract=contract(),
        target_quantity=2,
        target_created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    older = replace(
        newer,
        target_quantity=1,
        target_created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    book.update_target(newer)

    assert book.update_target(older) is newer
    assert book.targets.for_contract(contract()).target_quantity == 2


def test_trade_rebinding_matches_perm_id(book):
    original = trade(order_id=7, perm_id=900)
    book.save_order(order_info(original))
    rebound = trade(order_id=0, perm_id=900)

    assert book.rebind_trade(rebound) is None
    assert rebound.order.orderId == 7
    assert book.orders.by_id(7).trade is rebound


def test_portfolio_state_is_copied_and_read_only(book):
    original = {"weights": {"ES": 1}}
    book.portfolios.save("allocation", original)
    original["other"] = 2

    state = book.portfolios.load("allocation")

    assert "other" not in state
    with pytest.raises(TypeError):
        state["new"] = 1


def test_position_episode_ids_change_between_episodes(book):
    first = book.create_position_episode(
        "alpha",
        "brackets",
        contract(),
        target_quantity=1,
        target_created_at=datetime.now(timezone.utc),
        bracket_inputs={"atr": 10},
    )
    book.close_position_episode("alpha")
    second = book.create_position_episode(
        "alpha",
        "brackets",
        contract(),
        target_quantity=-1,
        target_created_at=datetime.now(timezone.utc),
        bracket_inputs={"atr": 10},
    )

    assert first.position_id != second.position_id


def test_position_state_rejects_boolean_blocked_direction():
    with pytest.raises(ValueError, match="blocked_direction"):
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            blocked_direction=True,
        )


def test_blotter_lookup_groups_source_and_position_episode(book):
    class Blotter:
        def records(self):
            return (
                {"source_key": "alpha", "position_id": "one", "realizedPNL": 2},
                {"source_key": "alpha", "position_id": "two", "realizedPNL": 3},
                {"source_key": "beta", "position_id": "one", "realizedPNL": 5},
            )

    book.blotter = Blotter()

    rows = book.blotter_records("alpha", position_id="one")

    assert len(rows) == 1
    assert rows[0]["realizedPNL"] == 2


def test_blotter_lookup_merges_queued_and_persisted_rows(book):
    class Saver(AbstractBaseSaver):
        def __init__(self):
            self.rows = [
                {
                    "order_id": 1,
                    "perm_id": 101,
                    "source_key": "alpha",
                    "position_id": "one",
                    "realizedPNL": 1,
                },
                {
                    "order_id": 2,
                    "perm_id": 102,
                    "source_key": "alpha",
                    "position_id": "one",
                    "realizedPNL": 4,
                },
            ]

        def save(self, data: Any, /, *args: Any) -> None:
            self.rows.append(data)

        def read(self, key=None, /, *args: Any) -> Any:
            return list(self.rows)

    saver = Saver()
    blotter = Blotter(saver=saver)
    blotter.save = saver.save
    blotter.save_report(
        {
            "order_id": 1,
            "perm_id": 101,
            "source_key": "alpha",
            "position_id": "one",
            "realizedPNL": 2,
        }
    )
    book.blotter = blotter

    rows = book.blotter_records("alpha", position_id="one")

    assert len(rows) == 2
    assert {row["realizedPNL"] for row in rows} == {2, 4}


def test_clear_state_persists_flat_tombstones_before_restart(
    book, order_saver, state_saver
):
    book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=contract(),
            quantity=1,
            target_quantity=1,
            target_created_at=datetime.now(timezone.utc),
            position_id="episode",
            blocked_direction=1,
            bracket_inputs={"atr": 10},
        )
    )
    book.update_target(
        TargetState(
            execution_model_name="serial",
            contract=contract(2),
            target_quantity=3,
            target_created_at=datetime.now(timezone.utc),
        )
    )
    book.portfolios.save("allocation", {"weights": {"ES": 1}})

    book.clear_state()

    assert book.positions.for_source("alpha") is None
    assert book.targets.for_contract(contract()) is None
    assert book.portfolios.load("allocation") is None

    recovered = Book(
        order_saver=order_saver,
        state_saver=state_saver,
        save_async=False,
        restore=True,
    )

    position = recovered.positions.for_source("alpha")
    assert position.quantity == 0
    assert position.target_quantity == 0
    assert position.position_id is None
    assert position.blocked_direction is None
    assert position.bracket_inputs == {}
    assert recovered.targets.for_contract(contract()) is None
    assert recovered.portfolios.load("allocation") == {}


@pytest.mark.parametrize("with_target", [False, True])
def test_clear_state_durably_resets_direct_fill_projection(
    book, order_saver, state_saver, with_target
):
    trade_ = trade(quantity=2)
    if with_target:
        book.update_target(
            TargetState(
                execution_model_name="serial",
                contract=trade_.contract,
                target_quantity=2,
                target_created_at=datetime.now(timezone.utc),
            )
        )
    book.save_order(
        order_info(
            trade_,
            role="TARGET_ADJUSTMENT",
            execution_model_name="serial",
            source_key=None,
            position_id=None,
        )
    )
    execution = fill(trade_, quantity=2)
    book.apply_fill(trade_, execution)
    assert book.positions.quantity(contract()) == 2

    book.clear_state()

    assert book.positions.quantity(contract()) == 0
    recovered = Book(
        order_saver=order_saver,
        state_saver=state_saver,
        save_async=False,
        restore=True,
    )
    assert recovered.targets.for_contract(contract()) is None
    assert recovered.positions.quantity(contract()) == 0

    recovered.update_target(
        TargetState(
            execution_model_name="serial",
            contract=trade_.contract,
            target_quantity=1,
            target_created_at=datetime.now(timezone.utc),
        )
    )
    assert recovered.positions.quantity(contract()) == 0


def test_direct_fill_after_clear_cutoff_is_accounted(book):
    trade_ = trade(quantity=2)
    book.update_target(
        TargetState(
            execution_model_name="serial",
            contract=trade_.contract,
            target_quantity=2,
            target_created_at=datetime.now(timezone.utc),
        )
    )
    book.save_order(
        order_info(
            trade_,
            role="TARGET_ADJUSTMENT",
            execution_model_name="serial",
            source_key=None,
            position_id=None,
        )
    )
    book.clear_state()

    late_fill = fill(trade_, exec_id="late-after-clear", quantity=1)
    book.apply_fill(trade_, late_fill)

    assert book.positions.quantity(contract()) == 1


def test_roll_state_round_trips_through_book_persistence(
    book, order_saver, state_saver
):
    created_at = datetime.now(timezone.utc)
    state = RollState(
        series_key="es-series",
        mode=FutureRollMode.DIRECT,
        executor_name="direct-roll",
        old_contract=contract(1),
        new_contract=contract(2),
        participants=(
            RollParticipant(
                execution_model_name="serial",
                quantity=2,
            ),
        ),
        stage=FutureRollStage.ROLL_ORDER_ACTIVE,
        roll_order_id=42,
        created_at=created_at,
        updated_at=created_at,
    )

    book.update_roll(state)
    recovered = Book(
        order_saver=order_saver,
        state_saver=state_saver,
        save_async=False,
        restore=True,
    )

    assert recovered.rolls.for_series("es-series") == state
    assert recovered.rolls.all(active_only=True) == (state,)
