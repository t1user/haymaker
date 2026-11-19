"""Crash boundaries across order evidence, signal state and aggregate contract positions."""

from runtime_helpers import replace_signal_state, signal_execution_state

from datetime import datetime, timezone
from unittest.mock import Mock

import ib_insync as ibi
import pytest

from haymaker.async_wrappers import QueueProcessingError
from haymaker.book import Book
from test_book import fill, order_info, trade


def restore(order_saver, state_saver):
    """Construct a fresh Book using only the committed fake documents."""
    return Book(
        order_saver=order_saver, state_saver=state_saver, save_async=False, restore=True
    )


def test_interrupted_signal_projection_recovers_once(
    book, order_saver, state_saver, monkeypatch
):
    """Receipt of evidence must not imply that its position update committed."""
    opening = trade()
    book._update_signal_state(
        signal_execution_state(signal_key="alpha", execution_model_name="brackets")
    )
    book.save_order(order_info(opening))
    execution = fill(opening)
    save = state_saver.save

    def fail_position(document):
        """Lose the position write after committing the execution evidence."""
        if document.get("state_type") == "signal_execution":
            raise RuntimeError("interrupted position write")
        save(document)

    monkeypatch.setattr(state_saver, "save", fail_position)
    with pytest.raises(RuntimeError, match="interrupted position"):
        book.apply_fill(opening, execution)
    monkeypatch.setattr(state_saver, "save", save)
    for _ in range(2):
        recovered = restore(order_saver, state_saver)
        assert recovered.positions._for_signal("alpha").filled_quantity == 1
        assert recovered.positions.filled_quantity(opening.contract) == 1
        assert not recovered.apply_fill(opening, execution)


def test_manual_close_target_recovers_with_interrupted_position_write(
    book, order_saver, state_saver, monkeypatch
):
    """The same durable fill both flattens exposure and suppresses the old target."""
    closing = trade(order_id=-1, side="SELL")
    book._update_signal_state(
        signal_execution_state(
            signal_key="alpha",
            execution_model_name="brackets",
            contract=closing.contract,
            filled_quantity=2,
            target_quantity=2,
            position_id="position-1",
            protection_inputs={"atr": 5},
        )
    )
    book.save_order(order_info(closing, role="MANUAL"))
    save = state_saver.save
    monkeypatch.setattr(state_saver, "save", Mock(side_effect=RuntimeError("crash")))
    execution = fill(closing, quantity=2)
    with pytest.raises(RuntimeError, match="crash"):
        book.apply_fill(closing, execution)
    monkeypatch.setattr(state_saver, "save", save)
    for _ in range(2):
        recovered = restore(order_saver, state_saver)
        state = recovered.positions._for_signal("alpha")
        assert state.filled_quantity == state.target_quantity == 0
        assert state.position_id is None
        assert not state.protection_inputs
        assert not recovered.apply_fill(closing, execution)


@pytest.mark.parametrize("reset", [False, True])
def test_checkpoint_preserves_explicit_correction_and_reset(
    book, order_saver, state_saver, reset
):
    """Historical fills cannot undo a corrected or explicitly cleared position."""
    opening = trade()
    book.save_order(order_info(opening))
    book.apply_fill(opening, fill(opening))
    if reset:
        book.clear_state()
    else:
        book._update_signal_state(
            replace_signal_state(book.positions._for_signal("alpha"), filled_quantity=0)
        )
    recovered = restore(order_saver, state_saver)
    assert recovered.positions.filled_quantity(opening.contract) == 0
    assert not recovered.apply_fill(opening, fill(opening))


def test_rebound_order_is_counted_once_after_restart(book, order_saver, state_saver):
    """Old persisted broker IDs must not become extra execution evidence."""
    opening = trade(order_id=7, perm_id=900)
    book.save_order(order_info(opening, signal_key=None))
    execution = fill(opening)
    book.apply_fill(opening, execution)
    for order_id in (8, 3):
        rebound = trade(order_id=order_id, perm_id=900)
        book.rebind_trade(rebound)
        book = restore(order_saver, state_saver)
        assert book.positions.filled_quantity(opening.contract) == 1
        assert len(book.orders.query()) == 1
        assert book.orders.by_perm_id(900).orderId == order_id
        assert not book.apply_fill(rebound, execution)


def test_trade_log_survives_legacy_restore_rebind_and_overlap(
    book, order_saver, state_saver
):
    original = trade()
    time = datetime.now(timezone.utc)
    pending = ibi.TradeLogEntry(time, "PendingSubmit", "None")
    repeated = ibi.TradeLogEntry(time, "Submitted", "Modify")
    original.log.extend((pending, repeated, repeated))
    book.save_order(order_info(original))
    del order_saver.read({})[0]["trade_log"]

    book = restore(order_saver, state_saver)
    rebound = trade(order_id=2)
    rebound.log.append(repeated)
    book.rebind_trade(rebound)
    rebound.log.append(repeated)
    book.rebind_trade(rebound)
    book = restore(order_saver, state_saver)
    book.rebind_trade(trade(order_id=2))  # A fresh connection has no log.
    book = restore(order_saver, state_saver)
    info = book.orders.by_id(2)
    assert info.trade_log == (pending, repeated, repeated, repeated)
    assert info.trade.log == []
    assert rebound.log == [repeated, repeated]


def test_terminal_rebind_preserves_all_raw_fills_before_projection_failure(
    book, order_saver, state_saver, monkeypatch
):
    original = trade()
    original.orderStatus.status = "Filled"
    original.orderStatus.filled = 2
    original.fills.extend(
        fill(original, exec_id=exec_id) for exec_id in ("first", "second")
    )
    book.save_order(order_info(original))

    book = restore(order_saver, state_saver)
    with monkeypatch.context() as patch:
        patch.setattr(
            state_saver, "save", Mock(side_effect=RuntimeError("position write"))
        )
        with pytest.raises(RuntimeError, match="position write"):
            book.rebind_trade(trade())

    for _ in range(2):
        book = restore(order_saver, state_saver)
        records = book.orders.by_id(original.order.orderId).fills
        assert [record.execution.execId for record in records] == ["first", "second"]
        assert all(record.commission_report.commission == 1 for record in records)
        assert book.positions.filled_quantity(original.contract) == 2


def test_terminal_recovery_rejects_execution_before_correction(
    book, order_saver, state_saver
):
    original = trade()
    original.orderStatus.status = "Cancelled"
    original.orderStatus.filled = 1
    original.fills.append(fill(original))
    book.save_order(order_info(original))
    book._update_signal_state(
        signal_execution_state(
            signal_key="alpha",
            execution_model_name="brackets",
            contract=original.contract,
        )
    )
    book.correct_signal_position("alpha", 0, datetime.now(timezone.utc))
    book = restore(order_saver, state_saver)

    with pytest.raises(
        ValueError, match="predates a reset, correction or position change"
    ):
        book.recover_terminal_executions(book.orders.by_id(original.order.orderId))
    assert book.positions.filled_quantity(original.contract) == 0
    assert not book.orders.by_id(original.order.orderId).fills


def test_live_replay_and_new_fill_after_reset_keep_checkpoint(
    book, order_saver, state_saver
):
    """A cleared source cannot be resurrected by old callbacks, even after new work."""
    opening = trade()
    book.save_order(order_info(opening))
    old = fill(opening)
    book.apply_fill(opening, old)
    book.clear_state()
    assert not book.apply_fill(opening, old)
    assert book.positions._for_signal("alpha") is None
    new = fill(opening, exec_id="post-reset")
    assert book.apply_fill(opening, new)
    assert not book.apply_fill(opening, old)
    recovered = restore(order_saver, state_saver)
    assert recovered.positions._for_signal("alpha").filled_quantity == 1
    assert recovered.positions.filled_quantity(opening.contract) == 1


async def test_failed_evidence_write_stops_dependent_projections(
    order_saver, state_saver, monkeypatch
):
    """A critical queue must not skip evidence and commit a dependent contract_position."""
    save = order_saver.save

    def fail_evidence(document):
        """Allow submission evidence but reject the fill-bearing update."""
        if document["fills"]:
            raise RuntimeError("evidence unavailable")
        save(document)

    monkeypatch.setattr(order_saver, "save", fail_evidence)
    writes = Mock(wraps=state_saver.save)
    monkeypatch.setattr(state_saver, "save", writes)
    book = Book(order_saver=order_saver, state_saver=state_saver, save_async=True)
    opening = trade()
    book.save_order(order_info(opening))
    book.apply_fill(opening, fill(opening))
    with pytest.raises(QueueProcessingError):
        await book.close()
    writes.assert_not_called()
