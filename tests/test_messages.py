"""Tests for the frozen built-in trading message boundaries."""

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from typing import Any

import ib_insync as ibi
import pytest

from haymaker.components import (
    TargetIntent,
    PositionRequest,
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
        "signal_key": "alpha",
        "contract": contract(),
        "value": 1,
        "signal_type": SignalType.STATE,
    }
    values.update(overrides)
    return Signal(**values)


def request(**overrides) -> PositionRequest:
    values: dict[str, Any] = {
        "signal": signal(),
        "target_direction": 1,
        "intent": TargetIntent.OPEN,
    }
    values.update(overrides)
    return PositionRequest(**values)


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
        lambda: request(created_at=value),
        lambda: target(created_at=value),
    ):
        with pytest.raises(TypeError, match="must be a datetime"):
            factory()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: signal(created_at=datetime(2026, 1, 1)),
        lambda: request(created_at=datetime(2026, 1, 1)),
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


@pytest.mark.parametrize("signal_key", [1, True])
def test_signal_rejects_non_string_signal_key(signal_key):
    with pytest.raises(TypeError, match="signal_key must be a string"):
        signal(signal_key=signal_key)


def test_signal_rejects_empty_signal_key():
    with pytest.raises(ValueError, match="signal_key must not be empty"):
        signal(signal_key="")


def test_signal_requires_signal_type_enum():
    with pytest.raises(TypeError, match="SignalType"):
        signal(signal_type="STATE")


@pytest.mark.parametrize("direction", [1.0, True, "1"])
def test_position_request_rejects_non_integer_direction(direction):
    with pytest.raises(TypeError, match="must be an integer"):
        request(target_direction=direction)


def test_position_request_rejects_integer_outside_binary_direction():
    with pytest.raises(ValueError, match="-1, 0, or 1"):
        request(target_direction=2)


def test_position_request_requires_typed_signal_and_intent():
    with pytest.raises(TypeError, match="signal must be a Signal"):
        request(signal="signal")
    with pytest.raises(TypeError, match="intent must be a TargetIntent"):
        request(intent="OPEN")


def test_signal_allows_contract_spec_contract_but_target_requires_concrete_contract():
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
    contract_execution = target(target_quantity=-3)
    position = target(intent=TargetIntent.REVERSE)

    assert contract_execution.target_quantity == -3
    assert contract_execution.intent is None
    assert position.intent is TargetIntent.REVERSE
    with pytest.raises(TypeError, match="TargetIntent"):
        target(intent="OPEN")


@pytest.mark.parametrize(
    ("signal_key", "exception", "message"),
    [
        (1, TypeError, "signal_key must be a string"),
        ("", ValueError, "signal_key must not be empty"),
    ],
)
def test_position_target_validates_optional_signal_key(signal_key, exception, message):
    with pytest.raises(exception, match=message):
        target(signal_key=signal_key)


def test_position_target_has_no_user_defined_execution_key():
    """Removed contract_execution identity is not silently accepted as a compatibility API."""
    with pytest.raises(TypeError, match="target_key"):
        target(target_key="removed")


@pytest.mark.parametrize("message", [signal(), request(), target()])
def test_envelope_messages_are_explicitly_unhashable(message):
    with pytest.raises(TypeError, match="unhashable"):
        hash(message)


def test_custom_order_roles_remain_valid():
    assert StandardOrderRole("ICEBERG_CHILD").value == "ICEBERG_CHILD"
