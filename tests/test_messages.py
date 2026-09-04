"""Tests for the frozen built-in trading message boundaries."""

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from typing import Any

import ib_insync as ibi
import pytest

from haymaker.components import (
    PositionIntent,
    PositionProposal,
    PositionTarget,
    Signal,
    SignalPair,
    SignalType,
    StandardOrderRole,
)


def contract(con_id: int = 1) -> ibi.Future:
    return ibi.Future(conId=con_id, symbol="ES", exchange="CME")


def signal(**overrides) -> Signal:
    values: dict[str, Any] = {
        "source_key": "alpha",
        "contract": contract(),
        "value": 1,
        "signal_type": SignalType.STATE,
    }
    values.update(overrides)
    return Signal(**values)


def proposal(**overrides) -> PositionProposal:
    values: dict[str, Any] = {
        "signal": signal(),
        "target_direction": 1,
        "intent": PositionIntent.OPEN,
    }
    values.update(overrides)
    return PositionProposal(**values)


def target(**overrides) -> PositionTarget:
    values: dict[str, Any] = {"contract": contract(), "target_quantity": 1}
    values.update(overrides)
    return PositionTarget(**values)


def test_signal_is_frozen_keyword_only_and_copies_metadata():
    metadata = {"atr": 10}
    message = signal(metadata=metadata)
    metadata["atr"] = 20

    assert message.metadata["atr"] == 10
    with pytest.raises(TypeError):
        message.metadata["new"] = 1
    with pytest.raises(FrozenInstanceError):
        message.value = 2
    with pytest.raises(TypeError):
        Signal("alpha", contract(), 1, SignalType.STATE)


def test_signal_shares_contract_and_nested_metadata_by_design():
    signal_contract = contract()
    nested = {"values": [1]}
    message = signal(contract=signal_contract, metadata=nested)

    signal_contract.symbol = "NQ"
    nested["values"].append(2)

    assert message.contract.symbol == "NQ"
    assert message.metadata["values"] == [1, 2]


def test_signal_pair_is_frozen_hashable_and_preserved_as_signal_value():
    pair = SignalPair(entry=1, exit=-1)
    message = signal(value=pair)

    assert message.value is pair
    assert isinstance(hash(pair), int)
    with pytest.raises(FrozenInstanceError):
        pair.entry = -1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("field", ["entry", "exit"])
def test_signal_pair_rejects_non_finite_members(value, field):
    values = {"entry": 1, "exit": 0}
    values[field] = value

    with pytest.raises(ValueError, match="finite"):
        SignalPair(**values)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_messages_reject_non_finite_scalar_values(value):
    with pytest.raises(ValueError, match="finite"):
        signal(value=value)
    with pytest.raises(ValueError, match="finite"):
        target(target_quantity=value)


@pytest.mark.parametrize("value", [None, "2026-01-01"])
def test_required_creation_timestamps_reject_non_datetimes(value):
    for factory in (
        lambda: signal(created_at=value),
        lambda: proposal(created_at=value),
        lambda: target(created_at=value),
    ):
        with pytest.raises(TypeError, match="must be a datetime"):
            factory()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: signal(created_at=datetime(2026, 1, 1)),
        lambda: proposal(created_at=datetime(2026, 1, 1)),
        lambda: target(created_at=datetime(2026, 1, 1)),
    ],
)
def test_required_creation_timestamps_reject_naive_datetimes(factory):
    with pytest.raises(ValueError, match="timezone-aware"):
        factory()


def test_signal_as_of_is_optional_but_must_be_an_aware_datetime():
    assert signal(as_of=None).as_of is None

    with pytest.raises(TypeError, match="must be a datetime"):
        signal(as_of="2026-01-01")
    with pytest.raises(ValueError, match="timezone-aware"):
        signal(as_of=datetime(2026, 1, 1))


@pytest.mark.parametrize("source_key", [1, True])
def test_signal_rejects_non_string_source_key(source_key):
    with pytest.raises(TypeError, match="source_key must be a string"):
        signal(source_key=source_key)


def test_signal_rejects_empty_source_key():
    with pytest.raises(ValueError, match="source_key must not be empty"):
        signal(source_key="")


def test_signal_requires_signal_type_enum():
    with pytest.raises(TypeError, match="SignalType"):
        signal(signal_type="STATE")


@pytest.mark.parametrize("direction", [1.0, True, "1"])
def test_position_proposal_rejects_non_integer_direction(direction):
    with pytest.raises(TypeError, match="must be an integer"):
        proposal(target_direction=direction)


def test_position_proposal_rejects_integer_outside_binary_direction():
    with pytest.raises(ValueError, match="-1, 0, or 1"):
        proposal(target_direction=2)


def test_position_proposal_requires_typed_signal_and_intent():
    with pytest.raises(TypeError, match="signal must be a Signal"):
        proposal(signal="signal")
    with pytest.raises(TypeError, match="intent must be a PositionIntent"):
        proposal(intent="OPEN")


def test_signal_allows_blueprint_contract_but_target_requires_concrete_contract():
    assert signal(contract=contract(0)).contract.conId == 0

    with pytest.raises(ValueError, match="non-zero conId"):
        target(contract=contract(0))


@pytest.mark.parametrize("factory", [signal, target])
def test_messages_reject_non_contract(factory):
    with pytest.raises(TypeError, match="ib_insync.Contract"):
        factory(contract="ES")


@pytest.mark.parametrize("factory", [signal, target])
def test_messages_reject_non_mapping_metadata(factory):
    with pytest.raises(TypeError, match="metadata must be a mapping"):
        factory(metadata=[])


def test_position_target_has_optional_typed_intent_and_absolute_quantity():
    direct = target(target_quantity=-3)
    one_to_one = target(intent=PositionIntent.REVERSE)

    assert direct.target_quantity == -3
    assert direct.intent is None
    assert one_to_one.intent is PositionIntent.REVERSE
    with pytest.raises(TypeError, match="PositionIntent"):
        target(intent="OPEN")


@pytest.mark.parametrize(
    ("source_key", "exception", "message"),
    [
        (1, TypeError, "source_key must be a string"),
        ("", ValueError, "source_key must not be empty"),
    ],
)
def test_position_target_validates_optional_source_key(source_key, exception, message):
    with pytest.raises(exception, match=message):
        target(source_key=source_key)


@pytest.mark.parametrize(
    ("target_key", "exception", "message"),
    [
        (1, TypeError, "target_key must be a string"),
        ("", ValueError, "target_key must not be empty"),
    ],
)
def test_position_target_validates_optional_target_key(target_key, exception, message):
    with pytest.raises(exception, match=message):
        target(target_key=target_key)


def test_position_target_rejects_mixed_direct_and_one_to_one_identity():
    with pytest.raises(ValueError, match="both target_key and source_key"):
        target(target_key="direct", source_key="one-to-one")


@pytest.mark.parametrize("message", [signal(), proposal(), target()])
def test_envelope_messages_are_explicitly_unhashable(message):
    with pytest.raises(TypeError, match="unhashable"):
        hash(message)


def test_custom_order_roles_remain_valid():
    assert StandardOrderRole("ICEBERG_CHILD").value == "ICEBERG_CHILD"
