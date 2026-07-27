from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from itertools import product

import ib_insync as ibi
import pytest

from haymaker.base import Atom
from haymaker.book import OrderInfo, PositionState
from haymaker.components import (
    BinaryEntryExitSignalProcessor,
    BinarySignalProcessor,
    OpposingSignalPolicy,
    PositionIntent,
    PositionProposal,
    PositionTarget,
    Signal,
    SignalPair,
    SignalType,
    StandardOrderRole,
)


def contract() -> ibi.Future:
    return ibi.Future(conId=1, symbol="ES", exchange="CME")


def signal(
    value: float | SignalPair,
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


def set_position(atom_runtime, quantity, *, blocked_direction=None):
    atom_runtime.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=contract(),
            quantity=quantity,
            blocked_direction=blocked_direction,
        )
    )


def outcome(proposal):
    if proposal is None:
        return None
    return proposal.target_direction, proposal.intent


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


def test_signal_pair_is_frozen_and_preserved_as_signal_value():
    pair = SignalPair(entry=1, exit=-1)
    message = signal(pair)

    assert message.value is pair
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
def test_signal_rejects_non_finite_scalar_value(value):
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


SCALAR_EXPECTED = {
    SignalType.STATE: {
        OpposingSignalPolicy.CLOSE: {
            -1: {
                -1: None,
                0: (0, PositionIntent.CLOSE),
                1: (0, PositionIntent.CLOSE),
            },
            0: {
                -1: (-1, PositionIntent.OPEN),
                0: None,
                1: (1, PositionIntent.OPEN),
            },
            1: {
                -1: (0, PositionIntent.CLOSE),
                0: (0, PositionIntent.CLOSE),
                1: None,
            },
        },
        OpposingSignalPolicy.REVERSE: {
            -1: {
                -1: None,
                0: (0, PositionIntent.CLOSE),
                1: (1, PositionIntent.REVERSE),
            },
            0: {
                -1: (-1, PositionIntent.OPEN),
                0: None,
                1: (1, PositionIntent.OPEN),
            },
            1: {
                -1: (-1, PositionIntent.REVERSE),
                0: (0, PositionIntent.CLOSE),
                1: None,
            },
        },
    },
    SignalType.EVENT: {
        OpposingSignalPolicy.CLOSE: {
            -1: {-1: None, 0: None, 1: (0, PositionIntent.CLOSE)},
            0: {
                -1: (-1, PositionIntent.OPEN),
                0: None,
                1: (1, PositionIntent.OPEN),
            },
            1: {-1: (0, PositionIntent.CLOSE), 0: None, 1: None},
        },
        OpposingSignalPolicy.REVERSE: {
            -1: {-1: None, 0: None, 1: (1, PositionIntent.REVERSE)},
            0: {
                -1: (-1, PositionIntent.OPEN),
                0: None,
                1: (1, PositionIntent.OPEN),
            },
            1: {-1: (-1, PositionIntent.REVERSE), 0: None, 1: None},
        },
    },
}


@pytest.mark.parametrize(
    ("signal_type", "opposing", "current", "value"),
    product(
        tuple(SignalType),
        tuple(OpposingSignalPolicy),
        (-1, 0, 1),
        (-1, 0, 1),
    ),
)
def test_scalar_binary_processor_full_transition_matrix(
    atom_runtime, signal_type, opposing, current, value
):
    set_position(atom_runtime, current)
    processor = BinarySignalProcessor(opposing=opposing)
    message = signal(value, signal_type)

    proposal = processor.process(message)

    assert outcome(proposal) == SCALAR_EXPECTED[signal_type][opposing][current][value]
    if proposal is not None:
        assert proposal.signal is message


PAIR_SELECTED_EXPECTED = {
    SignalType.STATE: {
        -1: {
            -1: None,
            0: (0, PositionIntent.CLOSE),
            1: (0, PositionIntent.CLOSE),
        },
        0: {
            -1: (-1, PositionIntent.OPEN),
            0: None,
            1: (1, PositionIntent.OPEN),
        },
        1: {
            -1: (0, PositionIntent.CLOSE),
            0: (0, PositionIntent.CLOSE),
            1: None,
        },
    },
    SignalType.EVENT: {
        -1: {-1: None, 0: None, 1: (0, PositionIntent.CLOSE)},
        0: {
            -1: (-1, PositionIntent.OPEN),
            0: None,
            1: (1, PositionIntent.OPEN),
        },
        1: {-1: (0, PositionIntent.CLOSE), 0: None, 1: None},
    },
}


@pytest.mark.parametrize(
    ("signal_type", "current", "entry", "exit"),
    product(tuple(SignalType), (-1, 0, 1), (-1, 0, 1), (-1, 0, 1)),
)
def test_entry_exit_processor_full_transition_matrix(
    atom_runtime, signal_type, current, entry, exit
):
    set_position(atom_runtime, current)
    processor = BinaryEntryExitSignalProcessor()
    message = signal(
        SignalPair(entry=entry, exit=exit),
        signal_type,
    )
    selected = entry if current == 0 else exit

    proposal = processor.process(message)

    assert outcome(proposal) == PAIR_SELECTED_EXPECTED[signal_type][current][selected]
    if proposal is not None:
        assert proposal.signal is message


@pytest.mark.parametrize(
    "processor_type", [BinarySignalProcessor, BinaryEntryExitSignalProcessor]
)
@pytest.mark.parametrize("blocked", [None, -1, 1])
@pytest.mark.parametrize("direction", [-1, 1])
def test_processors_enforce_only_matching_blocked_open(
    atom_runtime, processor_type, blocked, direction
):
    set_position(atom_runtime, 0, blocked_direction=blocked)
    processor = processor_type(respect_blocked_direction=True)
    value = (
        direction
        if processor_type is BinarySignalProcessor
        else SignalPair(entry=direction, exit=-direction)
    )

    proposal = processor.process(signal(value))

    assert (proposal is None) is (blocked == direction)
    if proposal is not None:
        assert outcome(proposal) == (direction, PositionIntent.OPEN)


@pytest.mark.parametrize(
    "processor",
    [
        pytest.param(BinarySignalProcessor, id="scalar"),
        pytest.param(BinaryEntryExitSignalProcessor, id="entry-exit"),
    ],
)
def test_processors_ignore_block_when_effectively_positioned(atom_runtime, processor):
    set_position(atom_runtime, 1, blocked_direction=-1)
    subject = processor(respect_blocked_direction=True)
    value = -1 if processor is BinarySignalProcessor else SignalPair(entry=1, exit=-1)

    proposal = subject.process(signal(value))

    assert outcome(proposal) == (0, PositionIntent.CLOSE)


def test_processor_uses_working_orders_in_effective_direction(atom_runtime):
    set_position(atom_runtime, 0)
    working_trade = ibi.Trade(
        contract=contract(),
        order=ibi.Order(
            orderId=10,
            permId=110,
            action="BUY",
            totalQuantity=2,
        ),
        orderStatus=ibi.OrderStatus(
            orderId=10,
            status=ibi.OrderStatus.Submitted,
            remaining=2,
        ),
    )
    atom_runtime.book.save_order(
        OrderInfo(
            trade=working_trade,
            role="OPEN",
            submitted_at=datetime.now(timezone.utc),
            execution_model_name="brackets",
            source_key="alpha",
        )
    )

    proposal = BinarySignalProcessor().process(signal(-1))

    assert outcome(proposal) == (0, PositionIntent.CLOSE)


@pytest.mark.parametrize("value", [-2, -0.5, 0.5, 2])
def test_scalar_processor_rejects_non_binary_value(atom_runtime, value):
    with pytest.raises(ValueError, match="-1, 0, or 1"):
        BinarySignalProcessor().process(signal(value))


@pytest.mark.parametrize(
    "pair",
    [
        SignalPair(entry=0.5, exit=0),
        SignalPair(entry=0, exit=-0.5),
    ],
)
@pytest.mark.parametrize("current", [0, 1])
def test_entry_exit_processor_validates_both_values(atom_runtime, pair, current):
    set_position(atom_runtime, current)

    with pytest.raises(ValueError, match="-1, 0, or 1"):
        BinaryEntryExitSignalProcessor().process(signal(pair))


def test_processors_reject_wrong_value_shape(atom_runtime):
    with pytest.raises(TypeError, match="scalar"):
        BinarySignalProcessor().process(signal(SignalPair(entry=1, exit=0)))
    with pytest.raises(TypeError, match="SignalPair"):
        BinaryEntryExitSignalProcessor().process(signal(1))


def test_on_data_emits_only_actionable_proposals(atom_runtime):
    processor = BinarySignalProcessor()
    output = capture(processor)

    processor.onData(signal(1))
    processor.onData(signal(0))

    assert len(output) == 1
    assert outcome(output[0]) == (1, PositionIntent.OPEN)


def test_connection_validation_happens_before_any_wiring(atom_runtime):
    valid = BinarySignalProcessor()
    invalid = BinarySignalProcessor()

    class WrongSource(Atom):
        output_type = dict

    source = WrongSource()
    with pytest.raises(TypeError):
        source.connect(valid, invalid)

    assert len(source.startEvent) == 0
    assert len(source.dataEvent) == 0


def test_processor_accepts_declared_signal_source(atom_runtime):
    source = SignalSource()
    processor = BinarySignalProcessor()

    source.connect(processor)

    assert len(source.startEvent) == 1


def test_processor_rejects_actual_wrong_message(atom_runtime):
    with pytest.raises(TypeError, match="only Signal"):
        BinarySignalProcessor().onData({"value": 1})


def test_constructor_arguments_require_declared_types(atom_runtime):
    with pytest.raises(TypeError, match="OpposingSignalPolicy"):
        BinarySignalProcessor(opposing="CLOSE")
    with pytest.raises(TypeError, match="bool"):
        BinarySignalProcessor(respect_blocked_direction=1)
