from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import ib_insync as ibi
import pytest

from haymaker.base import Atom
from haymaker.book import PositionState
from haymaker.components import (
    AlwaysOnBinarySignalProcessor,
    BinarySignalProcessor,
    LockableBinarySignalProcessor,
    PositionIntent,
    PositionProposal,
    PositionTarget,
    Signal,
    SignalType,
    StandardOrderRole,
)


def contract() -> ibi.Future:
    return ibi.Future(conId=1, symbol="ES", exchange="CME")


def signal(
    value: float,
    signal_type: SignalType = SignalType.STATE,
    metadata=None,
) -> Signal:
    return Signal(
        source_key="alpha",
        contract=contract(),
        value=value,
        signal_type=signal_type,
        metadata=metadata or {},
    )


class SignalSource(Atom):
    output_type = Signal

    def onData(self, data, *args):
        self.dataEvent.emit(data)


def capture(processor):
    result = []
    processor.dataEvent += result.append
    return result


def test_signal_is_frozen_keyword_only_and_copies_metadata():
    metadata = {"atr": 10}
    message = signal(1, metadata=metadata)
    metadata["atr"] = 20

    assert message.metadata["atr"] == 10
    with pytest.raises(TypeError):
        message.metadata["new"] = 1
    with pytest.raises(FrozenInstanceError):
        message.value = 2
    with pytest.raises(TypeError):
        Signal("alpha", contract(), 1, SignalType.STATE)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_signal_rejects_non_finite_value(value):
    with pytest.raises(ValueError, match="finite"):
        signal(value)


def test_messages_reject_naive_timestamps():
    with pytest.raises(ValueError, match="timezone-aware"):
        Signal(
            source_key="alpha",
            contract=contract(),
            value=1,
            signal_type=SignalType.STATE,
            created_at=datetime(2026, 1, 1),
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        PositionTarget(
            contract=contract(),
            target_quantity=1,
            created_at=datetime(2026, 1, 1),
        )


def test_position_proposal_requires_binary_direction():
    with pytest.raises(ValueError, match="-1, 0, or 1"):
        PositionProposal(
            signal=signal(1),
            target_direction=2,
            intent=PositionIntent.OPEN,
        )


def test_position_target_has_optional_intent_and_absolute_quantity():
    target = PositionTarget(
        contract=contract(),
        target_quantity=-3,
        created_at=datetime.now(timezone.utc),
    )

    assert target.target_quantity == -3
    assert target.intent is None


def test_custom_order_roles_remain_valid():
    assert StandardOrderRole("ICEBERG_CHILD").value == "ICEBERG_CHILD"


def test_state_zero_closes_existing_position(atom_runtime):
    atom_runtime.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=contract(),
            quantity=1,
        )
    )
    processor = BinarySignalProcessor()
    output = capture(processor)

    processor.onData(signal(0))

    assert output[0].target_direction == 0
    assert output[0].intent is PositionIntent.CLOSE


def test_event_zero_is_ignored_even_with_position(atom_runtime):
    atom_runtime.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=contract(),
            quantity=1,
        )
    )
    processor = BinarySignalProcessor()
    output = capture(processor)

    processor.onData(signal(0, SignalType.EVENT))

    assert output == []


def test_repeated_event_represents_another_open_after_flat(atom_runtime):
    processor = BinarySignalProcessor()
    output = capture(processor)

    processor.onData(signal(1, SignalType.EVENT))
    processor.onData(signal(1, SignalType.EVENT))

    assert [proposal.intent for proposal in output] == [
        PositionIntent.OPEN,
        PositionIntent.OPEN,
    ]


def test_ordinary_opposing_binary_signal_closes_first(atom_runtime):
    atom_runtime.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=contract(),
            quantity=1,
        )
    )
    processor = BinarySignalProcessor()
    output = capture(processor)

    processor.onData(signal(-1))

    assert output[0].target_direction == 0
    assert output[0].intent is PositionIntent.CLOSE


def test_always_on_opposing_signal_reverses(atom_runtime):
    atom_runtime.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=contract(),
            quantity=1,
        )
    )
    processor = AlwaysOnBinarySignalProcessor()
    output = capture(processor)

    processor.onData(signal(-1))

    assert output[0].target_direction == -1
    assert output[0].intent is PositionIntent.REVERSE


def test_lockable_processor_suppresses_blocked_open(atom_runtime):
    atom_runtime.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=contract(),
            blocked_direction=1,
        )
    )
    processor = LockableBinarySignalProcessor()
    output = capture(processor)

    processor.onData(signal(1))

    assert output == []


def test_connection_validation_happens_before_any_wiring(atom_runtime):
    source = SignalSource()
    valid = BinarySignalProcessor()
    invalid = BinarySignalProcessor()

    class WrongSource(Atom):
        output_type = dict

    with pytest.raises(TypeError):
        WrongSource().connect(valid, invalid)

    assert len(valid.startEvent) == 0
    assert len(invalid.startEvent) == 0


def test_processor_rejects_actual_wrong_message(atom_runtime):
    with pytest.raises(TypeError, match="only Signal"):
        BinarySignalProcessor().onData({"value": 1})
