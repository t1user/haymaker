"""Collection locality must retain coordinated accounting and recovery semantics."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import ib_insync as ibi
import mongomock
import pytest

from haymaker.book import (
    Book,
    FutureRollMode,
    PositionState,
    RollParticipant,
    RollState,
    TargetState,
)
from haymaker.saver import MongoSaver
from test_book import contract, fill, order_info, trade


def test_source_states_are_distinct_from_net_contract_balances(book):
    """Source episodes stay discoverable even when opposing holdings net to zero."""
    for source, quantity in (("long", 2), ("short", -2)):
        book.update_position(
            PositionState(
                source_key=source,
                execution_model_name="brackets",
                contract=contract(),
                quantity=quantity,
            )
        )
    sources = book.positions.source_states()
    assert set(sources) == {"long", "short"}
    assert book.positions.for_source("absent") is None
    assert len(book.positions.source_states_for_contract(contract())) == 2
    assert book.positions.quantity(contract()) == 0
    assert book.positions.by_contract() == {}
    with pytest.raises(TypeError):
        sources["intruder"] = sources["long"]


def test_restoration_reads_each_physical_collection_once(
    book, order_saver, state_saver, monkeypatch
):
    """Owners decode their documents without separate whole-collection queries."""
    book.update_position(PositionState(source_key="alpha", execution_model_name="b"))
    target = TargetState(
        contract=contract(),
        execution_model_name="serial",
        target_quantity=2,
        target_created_at=datetime.now(timezone.utc),
    )
    book.update_target(target)
    book.portfolios.save("allocation", {"weights": {"ES": 0.5}})
    order_read = Mock(wraps=order_saver.read)
    state_read = Mock(wraps=state_saver.read)
    monkeypatch.setattr(order_saver, "read", order_read)
    monkeypatch.setattr(state_saver, "read", state_read)
    recovered = Book(
        order_saver=order_saver,
        state_saver=state_saver,
        restore=True,
        save_async=False,
    )
    order_read.assert_called_once_with({})
    state_read.assert_called_once_with({})
    assert recovered.targets.for_contract(contract()) == target
    assert recovered.positions.for_source("alpha").quantity == 0
    assert recovered.portfolios.load("allocation") == {"weights": {"ES": 0.5}}


def test_stale_target_cannot_change_accounting_cutoff(book):
    """Recency rejection must happen before reset-boundary or balance changes."""
    opening = trade()
    book.save_order(order_info(opening, source_key=None))
    book.apply_fill(opening, fill(opening))
    timestamp = datetime.now(timezone.utc)
    target = TargetState(
        contract=opening.contract,
        execution_model_name="serial",
        target_quantity=3,
        target_created_at=timestamp,
    )
    book.update_target(target)
    stale = replace(
        target,
        target_created_at=timestamp - timedelta(days=1),
        fill_evidence_start_at=timestamp,
    )
    assert book.update_target(stale) is target
    assert book.targets.cutoffs == {}
    assert book.positions.quantity(opening.contract) == 1


async def test_all_owners_share_evidence_before_projection_order(
    order_saver, state_saver, monkeypatch
):
    """A single ordered writer preserves cross-collection call order."""
    writes = []
    save_order = order_saver.save
    save_state = state_saver.save

    def record_order(document):
        """Record the actual backend write, not its queue submission."""
        writes.append(("order", len(document["fills"])))
        save_order(document)

    def record_state(document):
        """Observe each state owner through the same physical backend."""
        writes.append((document["state_type"], document.get("quantity")))
        save_state(document)

    monkeypatch.setattr(order_saver, "save", record_order)
    monkeypatch.setattr(state_saver, "save", record_state)
    book = Book(order_saver=order_saver, state_saver=state_saver, save_async=True)
    opening = trade()
    book.save_order(order_info(opening))
    book.apply_fill(opening, fill(opening))
    book.update_target(
        TargetState(
            contract=contract(),
            execution_model_name="serial",
            target_quantity=3,
            target_created_at=datetime.now(timezone.utc),
        )
    )
    book.portfolios.save("allocation", {"value": 1})
    await book.close()
    assert writes == [
        ("order", 0),
        ("order", 1),
        ("position", 1),
        ("balance", 1),
        ("target", None),
        ("portfolio", None),
    ]


def test_recovered_shorter_trade_log_cannot_drop_new_fill_evidence(state_saver):
    """Mongo's status priority must not suppress newer callback evidence."""
    saver = MongoSaver(
        "orders", query_key="orderId", client=mongomock.MongoClient(), tz_aware=True
    )
    book = Book(order_saver=saver, state_saver=state_saver, save_async=False)
    opening = trade()
    later = datetime.now(timezone.utc) + timedelta(seconds=5)
    opening.log.append(ibi.TradeLogEntry(later, "Submitted", "acknowledged"))
    book.save_order(order_info(opening))
    book = Book(
        order_saver=saver, state_saver=state_saver, save_async=False, restore=True
    )
    rebound = trade()  # Broker can supply no old TradeLog entries on reconnect.
    execution = fill(rebound)
    assert book.apply_fill(rebound, execution)
    report = ibi.CommissionReport(execId=execution.execution.execId, commission=2)
    assert book.update_commission(rebound, execution, report)
    recovered = Book(
        order_saver=saver, state_saver=state_saver, save_async=False, restore=True
    )
    assert recovered.positions.quantity(contract()) == 1
    saved_fill = recovered.orders.by_id(rebound.order.orderId).fills[0]
    assert saved_fill.commission_report == report


def test_in_place_trade_id_change_rekeys_its_contribution(
    book, order_saver, state_saver
):
    """Mutable IB objects must not leave a second live or persisted order balance."""
    opening = trade()
    book.save_order(order_info(opening, source_key=None))
    book.apply_fill(opening, fill(opening))
    opening.order.orderId = 22
    book.rebind_trade(opening)
    assert book.orders.by_id(1) is None
    assert book.orders.by_id(22).previous_order_ids == (1,)
    recovered = Book(
        order_saver=order_saver, state_saver=state_saver, save_async=False, restore=True
    )
    assert len(recovered.orders.query()) == 1
    assert recovered.positions.quantity(contract()) == 1


def test_conflicting_broker_ids_fail_before_rebinding(book):
    """An orderId collision must not transfer evidence to another permanent order."""
    original = trade(perm_id=101)
    info = book.save_order(order_info(original))
    with pytest.raises(ValueError, match="identify different orders"):
        book.rebind_trade(trade(perm_id=202))
    assert book.orders.by_id(1) is info
    assert info.trade is original


def test_reused_superseded_id_fails_without_losing_evidence(
    book, order_saver, state_saver
):
    """A cyclic alias history must not silently remove every version of an order."""
    opening = trade()
    book.save_order(order_info(opening, source_key=None))
    book.apply_fill(opening, fill(opening))
    book.rebind_trade(trade(order_id=2))
    with pytest.raises(ValueError, match="superseded broker orderId"):
        book.rebind_trade(trade(order_id=1))
    recovered = Book(
        order_saver=order_saver, state_saver=state_saver, save_async=False, restore=True
    )
    assert recovered.positions.quantity(contract()) == 1
    assert len(recovered.orders.query()) == 1


@pytest.mark.parametrize("bag_first", [False, True])
def test_bracket_roll_accounts_leg_evidence_like_direct_rolls(book, bag_first):
    """BAG fallback and explicit legs describe one movement in both modes."""
    old, new = contract(1), contract(2)
    book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=old,
            quantity=2,
        )
    )
    book.update_roll(
        RollState(
            series_key="es",
            mode=FutureRollMode.BRACKET,
            executor_name="roller",
            old_contract=old,
            new_contract=new,
            participants=(
                RollParticipant(
                    source_key="alpha",
                    execution_model_name="brackets",
                    quantity=2,
                ),
            ),
        )
    )
    rolling = trade(order_id=2, perm_id=202)
    rolling.contract = ibi.Bag(symbol="ES")
    book.save_order(
        order_info(
            rolling,
            role="ROLL",
            source_key="alpha",
            params={"roll_state_key": "es", "old_contract": old, "new_contract": new},
        )
    )
    if bag_first:
        book.apply_fill(rolling, fill(rolling, exec_id="bag"))
        assert book.positions.by_contract() == {old: 1, new: 1}
    old_leg = fill(rolling, exec_id="old-leg")
    old_leg.execution.side = "SLD"
    book.apply_fill(rolling, old_leg._replace(contract=old))
    assert book.positions.by_contract() == {old: 1}
    new_leg = fill(rolling, exec_id="new-leg")
    book.apply_fill(rolling, new_leg._replace(contract=new))
    assert book.positions.by_contract() == {old: 1, new: 1}
