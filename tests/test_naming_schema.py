"""Public record scopes and conversion boundaries for the naming schema."""

from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone

import ib_insync as ibi
import pytest

import haymaker.book as book_api
import haymaker.components as components
from haymaker.base import Atom
from haymaker.enums import ExecutionMode
from haymaker.book import (
    Book,
    ContractPosition,
    ContractTarget,
    ProtectionState,
    SignalPosition,
    SignalTarget,
)
from runtime_helpers import signal_execution_state
from test_book import contract, fill, order_info, trade


def test_public_records_separate_filled_desired_and_protection(book, state_saver):
    now = datetime.now(timezone.utc)
    position = book.create_signal_position(
        "alpha",
        "model",
        contract(1),
        target_quantity=2,
        target_created_at=now,
        protection_inputs={"atr": 10},
    )
    book.update_position(replace(position, filled_quantity=1))
    position = book.positions.for_signal("alpha")
    protection = book.protection.for_signal("alpha")
    assert isinstance(position, SignalPosition)
    assert isinstance(protection, ProtectionState)
    assert not hasattr(position, "target_quantity")
    assert not hasattr(position, "protection_inputs")
    with pytest.raises(FrozenInstanceError):
        position.filled_quantity = 99
    with pytest.raises(TypeError):
        protection.protection_inputs["atr"] = 99

    target = SignalTarget(
        signal_key="alpha",
        execution_model_name="model",
        contract=contract(2),
        target_quantity=-3,
        target_created_at=now + timedelta(seconds=1),
        protection_inputs={"atr": 20},
    )
    book.update_signal_target(target)
    assert book.targets.for_signal("alpha") is target
    assert book.positions.for_signal("alpha").contract == contract(1)
    assert book.positions.for_signal("alpha").position_id == position.position_id
    assert book.protection.for_signal("alpha") is protection
    assert not hasattr(target, "filled_quantity")
    assert book.positions.by_signal()["alpha"] == book.positions.for_signal("alpha")
    assert book.positions.signal_positions_for_contract(contract(1)) == (
        book.positions.for_signal("alpha"),
    )
    assert book.targets.for_signal("missing") is None
    assert book.protection.for_signal("missing") is None

    document = state_saver.read({"state_key": "signal_execution:alpha"})[0]
    assert "schema_version" not in document
    assert document["filled_quantity"] == 1
    assert document["target_quantity"] == -3
    assert document["protection_inputs"] == {"atr": 10}
    assert document["target_protection_inputs"] == {"atr": 20}
    assert not {"position", "target", "protection"} & document.keys()


def test_contract_positions_keep_expiries_and_stocks_separate(book):
    stock = ibi.Stock("AAPL", "SMART", "USD", conId=3)
    for key, instrument, quantity in (
        ("early", contract(1), 2),
        ("late", contract(2), -1),
        ("stock", stock, 4),
    ):
        book.update_position(
            SignalPosition(
                signal_key=key,
                execution_model_name="model",
                contract=instrument,
                filled_quantity=quantity,
            )
        )
    assert book.positions.by_contract() == {contract(1): 2, contract(2): -1, stock: 4}
    for instrument, quantity in book.positions.by_contract().items():
        record = book.positions.for_contract(instrument)
        assert isinstance(record, ContractPosition)
        assert record.filled_quantity == quantity
        assert not hasattr(record, "target_quantity")
    assert book.positions.for_contract(contract(99)) is None
    assert book.positions.filled_quantity(contract(99)) == 0


def test_protective_exit_recovery_updates_all_views_once(
    book, order_saver, state_saver, monkeypatch
):
    opening = trade(quantity=1)
    book.create_signal_position(
        "alpha",
        "brackets",
        opening.contract,
        target_quantity=1,
        target_created_at=datetime.now(timezone.utc),
        protection_inputs={"atr": 10},
    )
    book.save_order(order_info(opening))
    book.apply_fill(opening, fill(opening))
    stop = trade(order_id=2, perm_id=102, side="SELL", quantity=1)
    book.save_order(order_info(stop, role="STOP_LOSS"))
    execution = fill(stop, exec_id="stop")
    save = state_saver.save

    def interrupted(document):
        if document.get("state_type") == "signal_execution":
            raise RuntimeError("lost signal-state write")
        save(document)

    monkeypatch.setattr(state_saver, "save", interrupted)
    with pytest.raises(RuntimeError, match="lost signal-state"):
        book.apply_fill(stop, execution)
    monkeypatch.setattr(state_saver, "save", save)
    for _ in range(2):
        recovered = Book(
            order_saver=order_saver,
            state_saver=state_saver,
            save_async=False,
            restore=True,
        )
        assert recovered.positions.for_signal("alpha").filled_quantity == 0
        assert recovered.positions.for_signal("alpha").position_id is None
        assert recovered.targets.for_signal("alpha").target_quantity == 0
        assert recovered.targets.for_signal("alpha").contract is None
        assert recovered.protection.for_signal("alpha").protection_inputs == {}
        assert recovered.positions.blocked_direction("alpha") == 1
        assert recovered.positions.filled_quantity(stop.contract) == 0
        assert not recovered.apply_fill(stop, execution)


def test_retired_interfaces_are_absent(atom_runtime):
    for name in (
        "PositionProposal",
        "PositionIntent",
        "BracketFutureRollExecutor",
        "DirectFutureRollExecutor",
    ):
        assert not hasattr(components, name)
    for name in (
        "PositionState",
        "TargetState",
        "SignalExecutionState",
        "ContractExecutionState",
    ):
        assert not hasattr(book_api, name)
    assert not hasattr(components, "FutureRollMode")
    assert not hasattr(book_api, "FutureRollMode")
    for name in ("BRACKET", "DIRECT"):
        with pytest.raises(ValueError):
            ExecutionMode(name)
    assert not hasattr(Atom, "contract_blueprint")
    for name in ("for_source", "source_states", "quantity"):
        assert not hasattr(atom_runtime.book.positions, name)
    with pytest.raises(TypeError, match="source_key"):
        components.Signal(**{"source_key": "alpha", "contract": contract(), "value": 1})

    class Portfolio(components.Portfolio):
        def process(self, signal):
            return ()

    with pytest.raises(TypeError, match="sources"):
        Portfolio(**{"sources": {"alpha"}})


def test_retired_order_schema_never_loses_attribution():
    document = order_info(trade()).encode()
    document["source_key"] = document.pop("signal_key")
    with pytest.raises(ValueError, match="conversion required"):
        book_api.OrderInfo.decode(document)
