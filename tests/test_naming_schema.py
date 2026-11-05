"""Public record scopes and conversion boundaries for the naming schema."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

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
    FutureRollStage,
    ProtectionState,
    RollParticipant,
    RollState,
    SignalPosition,
    SignalTarget,
)
from haymaker.misc import tree
from runtime_helpers import signal_execution_state
from test_book import contract, fill, order_info, trade
from test_components_migration import FakeClient, FakeCollection, migrate


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
    assert document["schema_version"] == 1
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


@pytest.mark.parametrize("collection", ["order", "state"])
@pytest.mark.parametrize("version", [None, 0, 2, True, "1"])
def test_schema_preflight_rejects_before_recovery_writes(
    order_saver, state_saver, collection, version
):
    opening = trade()
    info = order_info(opening)
    info.add_fill(book_api.FillRecord.from_fill(opening, fill(opening)))
    order_document = info.encode()
    state_document = signal_execution_state(
        signal_key="alpha", execution_model_name="brackets"
    ).encode()
    document = order_document if collection == "order" else state_document
    if version is None:
        document.pop("schema_version", None)
    else:
        document["schema_version"] = version
    order_saver.save(order_document)
    state_saver.save(state_document)
    order_saver.save = Mock(wraps=order_saver.save)
    state_saver.save = Mock(wraps=state_saver.save)
    with pytest.raises(ValueError, match="conversion required"):
        Book(
            order_saver=order_saver,
            state_saver=state_saver,
            save_async=False,
            restore=True,
        )
    order_saver.save.assert_not_called()
    state_saver.save.assert_not_called()


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


def legacy_component_document(document):
    """Explicit previous-schema fixture; never rename opaque payloads recursively."""
    document = deepcopy(document)
    document.pop("schema_version", None)
    for new, old in (
        ("signal_key", "source_key"),
        ("filled_quantity", "quantity"),
        ("protection_inputs", "bracket_inputs"),
        ("target_protection_inputs", "target_bracket_inputs"),
    ):
        if new in document:
            document[old] = document.pop(new)
    kind = document.get("state_type")
    old_kind = {
        "signal_execution": "position",
        "contract_position": "balance",
        "contract_execution": "target",
    }.get(kind, kind)
    if kind != old_kind:
        document["state_type"] = old_kind
        document["state_key"] = old_kind + ":" + document["state_key"].split(":", 1)[1]
    if kind == "roll":
        document["mode"] = {"SIGNAL": "BRACKET", "CONTRACT": "DIRECT"}[document["mode"]]
        document["participants"] = [
            legacy_component_document(p) for p in document["participants"]
        ]
        document["target_transfers"] = [
            legacy_component_document(p) for p in document["target_transfers"]
        ]
    return document


def test_previous_component_conversion_preserves_reset_and_opaque_payloads(
    book, order_saver, state_saver
):
    opening = trade(quantity=1)
    book.save_order(order_info(opening, signal_key=None))
    book.apply_fill(opening, fill(opening))
    opening.orderStatus.status = ibi.OrderStatus.Filled
    opening.orderStatus.remaining = 0
    opening.orderStatus.filled = 1
    book.save_order(book.orders.by_id(opening.order.orderId))
    book.clear_state()
    payload = {
        "source_key": "strategy-defined",
        "quantity": 123,
        "nested": {"bracket_inputs": [1, 2]},
    }
    book.portfolios.save("allocation", payload)
    orders = [legacy_component_document(row) for row in order_saver.read({})]
    states = [legacy_component_document(row) for row in state_saver.read({})]
    original = deepcopy((orders, states))
    client = FakeClient()
    client["old"]["orders"] = FakeCollection(orders)
    client["old"]["state"] = FakeCollection(states)
    report = migrate(
        client,
        source_database="old",
        target_database="new",
        apply=True,
        source_stopped=True,
    )
    assert report["target_verified"]
    assert (orders, states) == original
    converted = client["new"]["state"].documents
    target = next(row for row in converted if row["state_type"] == "contract_execution")
    assert target["fill_evidence_start_at"] is not None
    assert target["cleared"] is True
    assert (
        next(row for row in converted if row["state_type"] == "portfolio")["state"]
        == payload
    )
    assert all(row["schema_version"] == 1 for row in converted)


@pytest.mark.parametrize("mode", [ExecutionMode.SIGNAL, ExecutionMode.CONTRACT])
def test_completed_historical_roll_executor_allows_new_roll(
    book, order_saver, state_saver, atom_runtime_factory, mode
):
    timestamp = datetime.now(timezone.utc)
    completed = RollState(
        series_key="series",
        mode=mode,
        executor_name=(
            "BracketFutureRollExecutor"
            if mode is ExecutionMode.SIGNAL
            else "DirectFutureRollExecutor"
        ),
        old_contract=contract(1),
        new_contract=contract(2),
        participants=(
            RollParticipant(
                signal_key="alpha" if mode is ExecutionMode.SIGNAL else None,
                position_id="kept-id" if mode is ExecutionMode.SIGNAL else None,
                execution_model_name="brackets",
                filled_quantity=1,
            ),
        ),
        stage=FutureRollStage.COMPLETE,
        created_at=timestamp,
        updated_at=timestamp,
    )
    if mode is ExecutionMode.CONTRACT:
        completed = replace(
            completed,
            target_transfers=(
                ContractTarget(
                    execution_model_name="serial",
                    contract=contract(2),
                    target_quantity=1,
                    target_created_at=timestamp,
                    updated_at=timestamp,
                ),
            ),
        )
    state_saver.save(legacy_component_document(completed.encode()))
    client = FakeClient()
    client["old"]["state"] = FakeCollection(state_saver.read({}))
    migrate(
        client,
        source_database="old",
        target_database="new",
        apply=True,
        source_stopped=True,
    )
    state_saver.store["state"].clear()
    for row in client["new"]["state"].documents:
        state_saver.save(row)
    recovered = Book(
        order_saver=order_saver, state_saver=state_saver, save_async=False, restore=True
    )
    assert recovered.rolls.for_series("series").executor_name == completed.executor_name
    assert (
        recovered.rolls.for_series("series").target_transfers
        == completed.target_transfers
    )
    assert recovered.rolls.all(active_only=True) == ()
    runtime = atom_runtime_factory(book_=recovered)
    from haymaker.controller import Controller

    runtime.bind_controller(Controller(trader=runtime.trader))
    runtime.controller.register_execution_mode(mode)
    executor = runtime.controller.future_roller.register_executor()
    assert executor.name == (
        "SignalFutureRollExecutor"
        if mode is ExecutionMode.SIGNAL
        else "ContractFutureRollExecutor"
    )
    assert runtime.controller.future_roller.recover() is False
    next_roll = replace(
        completed,
        executor_name=executor.name,
        old_contract=contract(2),
        new_contract=contract(3),
        stage=FutureRollStage.PLANNED,
    )
    recovered.update_roll(next_roll)
    assert recovered.rolls.for_series("series") == next_roll


@pytest.mark.parametrize("version", [None, 1])
def test_retired_order_schema_never_loses_attribution(version):
    document = order_info(trade()).encode()
    document["source_key"] = document.pop("signal_key")
    if version is None:
        document.pop("schema_version")
    with pytest.raises(ValueError, match="conversion required"):
        book_api.OrderInfo.decode(document)


def test_previous_signal_correction_keeps_checkpoint_and_direction(
    book, order_saver, state_saver
):
    opening = trade(quantity=1)
    book.create_signal_position(
        "alpha",
        "brackets",
        opening.contract,
        target_quantity=1,
        target_created_at=datetime.now(timezone.utc),
        protection_inputs={"atr": 5},
    )
    book.save_order(order_info(opening))
    book.apply_fill(opening, fill(opening))
    opening.orderStatus.status = ibi.OrderStatus.Filled
    opening.orderStatus.filled = 1
    opening.orderStatus.remaining = 0
    book.save_order(book.orders.by_id(opening.order.orderId))
    book.update_blocked_direction("alpha", -1)
    book.correct_signal_position("alpha", 0, datetime.now(timezone.utc))
    expected = book.positions._for_signal("alpha").encode()
    client = FakeClient()
    client["old"]["orders"] = FakeCollection(
        [legacy_component_document(row) for row in order_saver.read({})]
    )
    client["old"]["state"] = FakeCollection(
        [legacy_component_document(row) for row in state_saver.read({})]
    )
    migrate(
        client,
        source_database="old",
        target_database="new",
        apply=True,
        source_stopped=True,
    )
    restored = next(
        row
        for row in client["new"]["state"].documents
        if row["state_type"] == "signal_execution"
    )
    assert all(restored[key] == value for key, value in expected.items())
    assert restored["applied_fill_keys"] == ["exec-1"]
    assert restored["blocked_direction"] == -1
    assert restored["filled_quantity"] == restored["target_quantity"] == 0
