from __future__ import annotations

import pytest

from haymaker.base import Atom, Pipe
from haymaker.signals import (
    AlwaysOnBinarySignalProcessor,
    AlwaysOnLockableBinarySignalProcessor,
    BinarySignalProcessor,
    BlipBinarySignalProcessor,
    LockableBinarySignalProcessor,
    LockableBlipBinarySignalProcessor,
    binary_signal_processor_factory,
)
from haymaker.state_machine import StrategyContainer


def test_BinarySignalProcessor_instantiates():
    proc = BinarySignalProcessor()
    assert isinstance(proc, BinarySignalProcessor)


def test_repr():
    bp = BinarySignalProcessor()
    assert "BinarySignalProcessor" in bp.__repr__()


def test_repr_Lockable():
    bp = LockableBinarySignalProcessor()
    assert "LockableBinarySignalProcessor" in bp.__repr__()


def test_repr_AlwaysOnLockable():
    bp = AlwaysOnLockableBinarySignalProcessor()
    assert "AlwaysOnLockableBinarySignalProcessor" in bp.__repr__()


def test_repr_AlwaysOn():
    bp = AlwaysOnBinarySignalProcessor()
    assert "AlwaysOnBinarySignalProcessor" in bp.__repr__()


@pytest.fixture
def StateMachine(strategy_saver):
    class FakeStateMachine:
        """
        Simulate state machine always returning desired values for
        position and lock.
        """

        def __init__(self, position=0.0, lock=0):
            self.position = position
            self.lock = lock

        strategy = StrategyContainer(strategy_saver)

        def position_and_order_for_strategy(self, strategy_str: str) -> float:
            return self.position

        def locked(self, key: str) -> int:
            return self.lock

    return FakeStateMachine


@pytest.fixture
def pipe(StateMachine):
    class SourceAtom(Atom):
        def run(self):
            self.dataEvent.emit({"strategy": "eska_NQ", "signal": 1})

    source = SourceAtom()

    sm = StateMachine()

    processor = binary_signal_processor_factory(lockable=False, always_on=False)
    processor_instance = processor(state_machine=sm)

    class OutputAtom(Atom):
        def onData(self, data, *args):
            self.out = data

    out = OutputAtom()
    Pipe(source, processor_instance, out)
    source.run()
    return out.out


def test_signal_emits_dict(pipe):
    assert isinstance(pipe, dict)


def test_correct_action(pipe):
    assert pipe["action"] == "OPEN"


def test_signal_included_in_output(pipe):
    assert "signal" in pipe.keys()


def test_target_position_included_in_output(pipe):
    assert "target_position" in pipe.keys()


# =================================================
# Testing actions
# =================================================

# Some comments are nonsense


@pytest.mark.parametrize(
    "test_input,expected",
    [
        # pos, lock, signal, lockable, always_on
        ((0, 0, 0, False, False), None),  # No signal must generate no signal
        ((0, 1, 0, True, False), None),  # No signal must generate no signal
        ((1, 0, 0, False, False), "CLOSE"),  # Zero signal, zero position (lockable)
        ((-1, 0, 0, False, False), "CLOSE"),  # Zero signal, zero position (lockable)
        ((1, 0, 0, True, False), "CLOSE"),  # Zero signal, zero position (lockable)
        ((-1, 0, 0, True, False), "CLOSE"),  # Zero signal, zero position (lockable)
        # ---
        ((1, 0, 1, False, False), None),  # Same signal with existing position
        ((-1, 0, -1, False, False), None),  # Same, opposite direction
        ((1, 0, -1, False, True), "REVERSE"),  # reverse signal, position, always-on
        ((-1, 0, 1, False, True), "REVERSE"),  # Same, opposite direction
        ((0, 1, 1, True, False), None),  # No position, signal, lockable, with lock
        ((0, -1, -1, True, False), None),  # Same, opposite direction
        ((0, 1, 1, False, False), "OPEN"),  # No position, signal, not lockable, lock
        ((0, -1, -1, False, False), "OPEN"),  # Same, oppposite direction
        ((0, 0, 1, True, False), "OPEN"),  # No position, signal, lockable, no lock
        ((0, 0, -1, True, False), "OPEN"),  # Same, opposite direction
        ((0, 1, 1, True, False), None),  # No position, lock, signal
        ((0, 0, 1, True, False), "OPEN"),  # No position, signal, lockable, no lock
        ((0, -1, 1, True, True), "OPEN"),  # Lockable, signal, irrelevant lock
        ((-1, -1, 1, True, True), "REVERSE"),  # Same but with position
        ((-1, 0, 1, True, True), "REVERSE"),  # Always_on position with opposite signal
        ((1, 0, -1, True, True), "REVERSE"),  # Same, opposite direction
        ((1, 0, -1, False, False), "CLOSE"),  # Position, opposite, closing signal
        ((-1, 0, 1, False, False), "CLOSE"),  # Same, reverse direction
        ((-2, 0, 1, True, True), "REVERSE"),  # Always_on position with opposite signal
        ((3, 0, -1, True, True), "REVERSE"),  # Same, opposite direction
        ((2, 0, -1, False, False), "CLOSE"),  # Position, opposite, closing signal
        ((-4, 0, 1, False, False), "CLOSE"),  # Same, reverse direction
    ],
)
def test_signal_paths_actions(test_input, expected, StateMachine):
    position, lock, signal, lockable, always_on = test_input

    sm = StateMachine(position, lock)

    processor = binary_signal_processor_factory(lockable, always_on)
    processor_instance = processor(state_machine=sm)
    action = processor_instance.process_signal("x", signal, signal)

    assert action == expected


# =================================================
# Testing positions
# =================================================

# Some comments are nonsense


@pytest.mark.parametrize(
    "test_input,expected",
    [
        # pos, lock, signal, lockable, always_on
        ((0, 0, 0, False, False), None),  # No signal must generate no signal
        ((0, 1, 0, True, False), None),  # No signal must generate no signal
        ((1, 0, 0, False, False), 0),  # Zero signal means no position
        ((-1, 0, 0, False, False), 0),  # Zero signal means no position
        ((1, 0, 0, True, False), 0),  # Zero signal, (lockable)
        ((-1, 0, 0, True, False), 0),  # Zero signal (lockable)
        # ---
        ((1, 0, 1, False, False), None),  # Same signal with existing position
        ((-1, 0, -1, False, False), None),  # Same, opposite direction
        ((1, 0, -1, False, True), -1),  # Reverse signal, existing position, always-on
        ((-1, 0, 1, False, True), 1),  # Same, opposite direction
        ((0, 1, 1, True, False), None),  # No position, signal, lockable, with lock
        ((0, -1, -1, True, False), None),  # Same, opposite direction
        ((0, 1, 1, False, False), 1),  # No position, signal, not lockable, with lock
        ((0, -1, -1, False, False), -1),  # Same, oppposite direction
        ((0, 0, 1, True, False), 1),  # No position, signal, lockable, no lock
        ((0, 1, 1, True, False), None),  # Same, opposite direction
        ((0, 0, 1, True, False), 1),  # No position, signal, lockable, no lock
        ((0, -1, 1, True, True), 1),  # Lockable, signal, opposite (irrelevant) lock
        ((-1, 0, 1, True, True), 1),  # Always_on position with opposite signal
        ((1, 0, -1, True, True), -1),  # Same, opposite direction
        ((1, 0, -1, False, False), 0),  # Position, opposite, closing signal
        ((-1, 0, 1, False, False), 0),  # Same, reverse direction
        # same but with non-zero positions (which should be interpreted as np.sign)
        ((2, 0, -1, True, True), -1),  # Same, opposite direction
        ((2, 0, -1, False, False), 0),  # Position, opposite, closing signal
        ((-2, 0, 1, False, False), 0),  # Same, reverse direction
    ],
)
def test_signal_paths_positions(test_input, expected, StateMachine):
    position, lock, signal, lockable, always_on = test_input

    sm = StateMachine(position, lock)

    class OutputAtom(Atom):
        out = {}

        def onData(self, data, *args):
            self.out = data

    output = OutputAtom()

    processor = binary_signal_processor_factory(lockable, always_on)
    print(processor)
    processor_instance = processor(state_machine=sm)
    processor_instance += output
    processor_instance.onData({"strategy": "x", "signal": signal})

    target_position = output.out.get("target_position")

    assert target_position == expected


# =================================================
# Testing actions - object by object
# =================================================


@pytest.mark.parametrize(
    "test_input,expected",
    [  # pos, sig
        ((0, 0), None),
        ((0, 1), "OPEN"),
        ((0, -1), "OPEN"),
        ((1, 0), "CLOSE"),
        ((-1, 0), "CLOSE"),
        ((1, 1), None),
        ((1, -1), "CLOSE"),
        ((-1, 1), "CLOSE"),
        ((-1, -1), None),
    ],
)
def test_signal_paths_BinarySignalProcessor(test_input, expected, StateMachine):
    position, signal = test_input

    sm = StateMachine(position=position)

    processor_instance = BinarySignalProcessor(state_machine=sm)
    action = processor_instance.process_signal("x", signal, signal)

    assert action == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [  # pos, sig
        ((0, 0), None),
        ((0, 1), "OPEN"),
        ((0, -1), "OPEN"),
        ((1, 0), None),
        ((-1, 0), None),
        ((1, 1), None),
        ((1, -1), "CLOSE"),
        ((-1, 1), "CLOSE"),
        ((-1, -1), None),
    ],
)
def test_signal_paths_BlipBinarySignalProcessor(test_input, expected, StateMachine):
    position, signal = test_input

    sm = StateMachine(position=position)

    processor_instance = BlipBinarySignalProcessor(state_machine=sm)
    action = processor_instance.process_signal("x", signal, signal)

    assert action == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        # pos, signal, lock
        ((0, 0, 0), None),
        ((0, 1, 0), "OPEN"),
        ((0, -1, 0), "OPEN"),
        ((1, 0, 0), "CLOSE"),
        ((-1, 0, 0), "CLOSE"),
        ((1, 1, 0), None),
        ((1, -1, 0), "CLOSE"),
        ((-1, 1, 0), "CLOSE"),
        ((-1, -1, 0), None),
        ((0, 0, 1), None),
        ((0, 1, 1), None),
        ((0, -1, 1), "OPEN"),
        ((1, 0, 1), "CLOSE"),
        ((-1, 0, 1), "CLOSE"),
        ((1, 1, 1), None),
        ((1, -1, 1), "CLOSE"),
        # ((-1, 1, 1), "CLOSE"), IMPOSSIBLE
        ((-1, -1, 1), None),
        ((0, 0, -1), None),
        ((0, 1, -1), "OPEN"),
        ((0, -1, -1), None),
        ((1, 0, -1), "CLOSE"),
        ((-1, 0, -1), "CLOSE"),
        ((1, 1, -1), None),
        # ((1, -1, -1), "CLOSE"), IMPOSSIBLE
        ((-1, 1, -1), "CLOSE"),
        ((-1, -1, -1), None),
    ],
)
def test_signal_paths_LockableBinarySignalProcessor(test_input, expected, StateMachine):
    position, signal, lock = test_input

    sm = StateMachine(position, lock)

    processor_instance = LockableBinarySignalProcessor(state_machine=sm)
    action = processor_instance.process_signal("x", signal, signal)

    assert action == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        # pos, signal, lock
        ((0, 0, 0), None),
        ((0, 1, 0), "OPEN"),
        ((0, -1, 0), "OPEN"),
        ((1, 0, 0), None),
        ((-1, 0, 0), None),
        ((1, 1, 0), None),
        ((1, -1, 0), "CLOSE"),
        ((-1, 1, 0), "CLOSE"),
        ((-1, -1, 0), None),
        ((0, 0, 1), None),
        ((0, 1, 1), None),
        ((0, -1, 1), "OPEN"),
        ((1, 0, 1), None),
        ((-1, 0, 1), None),
        ((1, 1, 1), None),
        ((1, -1, 1), "CLOSE"),
        # ((-1, 1, 1), "CLOSE"), IMPOSSIBLE
        ((-1, -1, 1), None),
        ((0, 0, -1), None),
        ((0, 1, -1), "OPEN"),
        ((0, -1, -1), None),
        ((1, 0, -1), None),
        ((-1, 0, -1), None),
        ((1, 1, -1), None),
        # ((1, -1, -1), "CLOSE"), IMPOSSIBLE
        ((-1, 1, -1), "CLOSE"),
        ((-1, -1, -1), None),
    ],
)
def test_signal_paths_LockableBlipBinarySignalProcessor(
    test_input, expected, StateMachine
):
    position, signal, lock = test_input

    sm = StateMachine(position, lock)

    processor_instance = LockableBlipBinarySignalProcessor(state_machine=sm)
    action = processor_instance.process_signal("x", signal, signal)

    assert action == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        # pos, signal, lock
        ((0, 0, 0), None),
        ((0, 1, 0), "OPEN"),
        ((0, -1, 0), "OPEN"),
        ((1, 0, 0), "CLOSE"),
        ((-1, 0, 0), "CLOSE"),
        ((1, 1, 0), None),
        ((1, -1, 0), "REVERSE"),
        ((-1, 1, 0), "REVERSE"),
        ((-1, -1, 0), None),
        ((0, 0, 1), None),
        ((0, 1, 1), None),
        ((0, -1, 1), "OPEN"),
        ((1, 0, 1), "CLOSE"),
        ((-1, 0, 1), "CLOSE"),
        ((1, 1, 1), None),
        ((1, -1, 1), "REVERSE"),
        # ((-1, 1, 1), "CLOSE"), IMPOSSIBLE
        ((-1, -1, 1), None),
        ((0, 0, -1), None),
        ((0, 1, -1), "OPEN"),
        ((0, -1, -1), None),
        ((1, 0, -1), "CLOSE"),
        ((-1, 0, -1), "CLOSE"),
        ((1, 1, -1), None),
        # ((1, -1, -1), "CLOSE"), IMPOSSIBLE
        ((-1, 1, -1), "REVERSE"),
        ((-1, -1, -1), None),
    ],
)
def test_signal_paths_AlwaysOnLockableBinarySignalProcessor(
    test_input, expected, StateMachine
):
    position, signal, lock = test_input

    sm = StateMachine(position, lock)
    strategy = sm.strategy["x"]
    strategy.position = position

    processor_instance = AlwaysOnLockableBinarySignalProcessor(state_machine=sm)
    action = processor_instance.process_signal("x", signal, signal)

    assert action == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [  # pos, sig
        ((0, 0), None),
        ((0, 1), "OPEN"),
        ((0, -1), "OPEN"),
        ((1, 0), "CLOSE"),
        ((-1, 0), "CLOSE"),
        ((1, 1), None),
        ((1, -1), "REVERSE"),
        ((-1, 1), "REVERSE"),
        ((-1, -1), None),
    ],
)
def test_signal_paths_AlwaysOnBinarySignalProcessor(test_input, expected, StateMachine):
    position, signal = test_input

    sm = StateMachine(position=position)

    processor_instance = AlwaysOnBinarySignalProcessor(state_machine=sm)
    action = processor_instance.process_signal("x", signal, signal)

    assert action == expected


# =================================================
# Testing target positions - object by object
# =================================================


@pytest.mark.parametrize(
    "test_input,expected",
    [  # pos, sig
        ((0, 0), None),
        ((0, 1), 1),
        ((0, -1), -1),
        ((1, 0), 0),
        ((-1, 0), 0),
        ((1, 1), None),
        ((1, -1), 0),
        ((-1, 1), 0),
        ((-1, -1), None),
    ],
)
def test_signal_paths_positions_BinarySignalProcessor(
    test_input, expected, StateMachine
):
    position, signal = test_input

    class FakeStateMachine(StateMachine):

        def locked(self, key):
            raise TypeError("Shouldn't be here")

    sm = FakeStateMachine(position)
    strategy = sm.strategy["x"]
    strategy.position = position

    class OutputAtom(Atom):
        out = {}

        def onData(self, data, *args):
            self.out = data

    output = OutputAtom()
    processor_instance = BinarySignalProcessor(state_machine=sm)
    processor_instance += output
    processor_instance.onData({"strategy": "x", "signal": signal})

    target_position = output.out.get("target_position")

    assert target_position == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [  # pos, sig
        ((0, 0), None),
        ((0, 1), 1),
        ((0, -1), -1),
        ((1, 0), None),
        ((-1, 0), None),
        ((1, 1), None),
        ((1, -1), 0),
        ((-1, 1), 0),
        ((-1, -1), None),
    ],
)
def test_signal_paths_positions_BlipBinarySignalProcessor(
    test_input, expected, StateMachine
):
    position, signal = test_input

    class FakeStateMachine(StateMachine):

        def locked(self, key):
            raise TypeError("Shouldn't be here")

    sm = FakeStateMachine(position)
    strategy = sm.strategy["x"]
    strategy.position = position

    class OutputAtom(Atom):
        out = {}

        def onData(self, data, *args):
            self.out = data

    output = OutputAtom()
    processor_instance = BlipBinarySignalProcessor(state_machine=sm)
    processor_instance += output
    processor_instance.onData({"strategy": "x", "signal": signal})

    target_position = output.out.get("target_position")

    assert target_position == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        # pos, signal, lock
        ((0, 0, 0), None),
        ((0, 1, 0), 1),
        ((0, -1, 0), -1),
        ((1, 0, 0), 0),
        ((-1, 0, 0), 0),
        ((1, 1, 0), None),
        ((1, -1, 0), 0),
        ((-1, 1, 0), 0),
        ((-1, -1, 0), None),
        ((0, 0, 1), None),
        ((0, 1, 1), None),
        ((0, -1, 1), -1),
        ((1, 0, 1), 0),
        ((-1, 0, 1), 0),
        ((1, 1, 1), None),
        ((1, -1, 1), 0),
        # ((-1, 1, 1), "CLOSE"), IMPOSSIBLE
        ((-1, -1, 1), None),
        ((0, 0, -1), None),
        ((0, 1, -1), 1),
        ((0, -1, -1), None),
        ((1, 0, -1), 0),
        ((-1, 0, -1), 0),
        ((1, 1, -1), None),
        # ((1, -1, -1), "CLOSE"), IMPOSSIBLE
        ((-1, 1, -1), 0),
        ((-1, -1, -1), None),
    ],
)
def test_signal_paths_positions_LockableBinarySignalProcessor(
    test_input, expected, StateMachine
):
    position, signal, lock = test_input

    sm = StateMachine(position, lock)

    class OutputAtom(Atom):
        out = {}

        def onData(self, data, *args):
            self.out = data

    output = OutputAtom()

    processor_instance = LockableBinarySignalProcessor(state_machine=sm)

    processor_instance += output
    processor_instance.onData({"strategy": "x", "signal": signal})

    target_position = output.out.get("target_position")

    assert target_position == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        # pos, signal, lock
        ((0, 0, 0), None),
        ((0, 1, 0), 1),
        ((0, -1, 0), -1),
        ((1, 0, 0), 0),
        ((-1, 0, 0), 0),
        ((1, 1, 0), None),
        ((1, -1, 0), -1),
        ((-1, 1, 0), 1),
        ((-1, -1, 0), None),
        ((0, 0, 1), None),
        ((0, 1, 1), None),
        ((0, -1, 1), -1),
        ((1, 0, 1), 0),
        ((-1, 0, 1), 0),
        ((1, 1, 1), None),
        ((1, -1, 1), -1),
        # ((-1, 1, 1), "CLOSE"), IMPOSSIBLE
        ((-1, -1, 1), None),
        ((0, 0, -1), None),
        ((0, 1, -1), 1),
        ((0, -1, -1), None),
        ((1, 0, -1), 0),
        ((-1, 0, -1), 0),
        ((1, 1, -1), None),
        # ((1, -1, -1), "CLOSE"), IMPOSSIBLE
        ((-1, 1, -1), 1),
        ((-1, -1, -1), None),
    ],
)
def test_signal_paths_positions_AlwaysOnLockableBinarySignalProcessor(
    test_input, expected, StateMachine
):
    position, signal, lock = test_input

    sm = StateMachine(position, lock)

    class OutputAtom(Atom):
        out = {}

        def onData(self, data, *args):
            self.out = data

    output = OutputAtom()

    processor_instance = AlwaysOnLockableBinarySignalProcessor(state_machine=sm)
    processor_instance += output
    processor_instance.onData({"strategy": "x", "signal": signal})

    target_position = output.out.get("target_position")

    assert target_position == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [  # pos, sig
        ((0, 0), None),
        ((0, 1), 1),
        ((0, -1), -1),
        ((1, 0), 0),
        ((-1, 0), 0),
        ((1, 1), None),
        ((1, -1), -1),
        ((-1, 1), 1),
        ((-1, -1), None),
    ],
)
def test_signal_paths_positions_AlwaysOnBinarySignalProcessor(
    test_input, expected, StateMachine
):
    position, signal = test_input

    sm = StateMachine(position)

    class OutputAtom(Atom):
        out = {}

        def onData(self, data, *args):
            self.out = data

    output = OutputAtom()

    processor_instance = AlwaysOnBinarySignalProcessor(state_machine=sm)
    processor_instance += output
    processor_instance.onData({"strategy": "x", "signal": signal})

    target_position = output.out.get("target_position")

    assert target_position == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [  # pos, sig_in, sig_out
        ((0, 0, 1), None),
        ((0, 1, 0), 1),
        ((0, -1, 0), -1),
        ((0, 1, -1), 1),
        ((0, -1, 1), -1),
        ((1, 0, 1), None),  # out signals should be ignored if in the same direction
        ((-1, 0, -1), None),  # out signals should be ignored if in the same direction
        ((1, -1, 0), 0),  # in signals should be ignored if position exists
        ((-1, 1, 0), 0),  # in signals should be ignored if position exists
        ((1, -1, -1), 0),  # out signals should be acted on
        ((-1, 1, 1), 0),  # out signals should be acted on
        ((1, 0, -1), 0),  # out signals should be acted on
        ((-1, 0, 1), 0),  # out signals should be acted on
    ],
)
def test_double_signals_BinarySignalProcessor(test_input, expected, StateMachine):
    position, signal_in, signal_out = test_input

    sm = StateMachine(position)

    class OutputAtom(Atom):
        out = {}

        def onData(self, data, *args):
            self.out = data

    output = OutputAtom()

    processor_instance = BinarySignalProcessor(
        signal_fields=("signal_in", "signal_out"), state_machine=sm
    )
    processor_instance += output
    processor_instance.onData(
        {"strategy": "x", "signal_in": signal_in, "signal_out": signal_out}
    )

    target_position = output.out.get("target_position")

    assert target_position == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [  # pos, sig_in, sig_out
        ((0, 0, 1), None),
        ((0, 1, 0), 1),
        ((0, -1, 0), -1),
        ((0, 1, -1), 1),
        ((0, -1, 1), -1),
        ((1, 0, 1), None),  # out signals should be ignored if in the same direction
        ((-1, 0, -1), None),  # out signals should be ignored if in the same direction
        ((1, -1, 0), None),  # in signals should be ignored if position exists
        ((-1, 1, 0), None),  # in signals should be ignored if position exists
        ((1, -1, -1), 0),  # out signals should be acted on
        ((-1, 1, 1), 0),  # out signals should be acted on
        ((1, 0, -1), 0),  # out signals should be acted on
        ((-1, 0, 1), 0),  # out signals should be acted on
    ],
)
def test_double_signals_BlipBinarySignalProcessor(test_input, expected, StateMachine):
    position, signal_in, signal_out = test_input

    sm = StateMachine(position)

    class OutputAtom(Atom):
        out = {}

        def onData(self, data, *args):
            self.out = data

    output = OutputAtom()

    processor_instance = BlipBinarySignalProcessor(
        signal_fields=("signal_in", "signal_out"), state_machine=sm
    )
    processor_instance += output
    processor_instance.onData(
        {"strategy": "x", "signal_in": signal_in, "signal_out": signal_out}
    )

    target_position = output.out.get("target_position")

    assert target_position == expected
