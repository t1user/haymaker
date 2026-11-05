"""Crash boundaries across order evidence, signal state and aggregate contract positions."""

from runtime_helpers import replace_signal_state, signal_execution_state

from unittest.mock import Mock

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
