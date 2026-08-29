from dataclasses import replace
from datetime import datetime, timezone
from typing import Any
from unittest.mock import Mock

import ib_insync as ibi
import pytest

from haymaker.blotter import Blotter
from haymaker.book import Book, FillRecord, OrderInfo, PositionState, TargetState
from haymaker.saver import AbstractBaseSaver


def contract(con_id: int = 1) -> ibi.Future:
    return ibi.Future(
        conId=con_id,
        symbol="ES",
        exchange="CME",
        localSymbol=f"ES-{con_id}",
    )


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

    assert book.order_by_id(1) is info
    assert book.order_by_perm_id(101) is info


def test_active_order_filters_all_attribution_fields(book):
    info = book.save_order(order_info(trade()))

    assert book.active_orders(
        source_key="alpha",
        contract=contract(),
        role="OPEN",
        execution_model_name="brackets",
    ) == (info,)
    assert book.active_orders(source_key="other") == ()


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
    assert book.position_state("alpha").quantity == 1
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

    assert book.aggregate_quantity(contract()) == 1
    assert book.logical_positions() == {contract(): 1}


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

    state = book.position_state("alpha")
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

    state = book.position_state("alpha")
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

    state = book.position_state("alpha")
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

    state = book.position_state("alpha")
    assert state.quantity == 1
    assert state.blocked_direction is None
    assert state.position_id == "episode-1"

    book.apply_fill(trade_, fill(trade_, exec_id="complete", quantity=1))

    state = book.position_state("alpha")
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

    state = book.position_state("alpha")
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

    assert book.position_state("alpha").blocked_direction == 1


def test_direct_quantity_recovers_from_completed_order_evidence(
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

    assert recovered.active_orders() == ()
    assert recovered.aggregate_quantity(contract()) == 2


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
    assert book.target_state("serial", contract()).target_quantity == 2


def test_trade_rebinding_matches_perm_id(book):
    original = trade(order_id=7, perm_id=900)
    book.save_order(order_info(original))
    rebound = trade(order_id=0, perm_id=900)

    assert book.rebind_trade(rebound) is None
    assert rebound.order.orderId == 7
    assert book.order_by_id(7).trade is rebound


def test_portfolio_state_is_copied_and_read_only(book):
    original = {"weights": {"ES": 1}}
    book.save_portfolio_state("allocation", original)
    original["other"] = 2

    state = book.load_portfolio_state("allocation")

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
    book.save_portfolio_state("allocation", {"weights": {"ES": 1}})

    book.clear_state()

    assert book.position_state("alpha") is None
    assert book.target_state("serial", contract(2)) is None
    assert book.load_portfolio_state("allocation") is None

    recovered = Book(
        order_saver=order_saver,
        state_saver=state_saver,
        save_async=False,
        restore=True,
    )

    position = recovered.position_state("alpha")
    assert position.quantity == 0
    assert position.target_quantity == 0
    assert position.position_id is None
    assert position.blocked_direction is None
    assert position.bracket_inputs == {}
    assert recovered.target_state("serial", contract(2)).target_quantity == 0
    assert recovered.load_portfolio_state("allocation") == {}
