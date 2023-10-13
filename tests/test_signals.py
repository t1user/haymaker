from __future__ import annotations

import pytest

from ib_tools.base import Atom, Pipe
from ib_tools.signals import (
    AlwaysOnBinarySignalProcessor,
    AlwaysOnLockableBinarySignalProcessor,
    BinarySignalProcessor,
    LockableBinarySignalProcessor,
    LockableBlipBinarySignalProcessor,
    binary_signal_processor_factory,
)


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
def pipe():
    class SourceAtom(Atom):
        def run(self):
            self.dataEvent.emit({"strategy": "eska_NQ", "signal": 1})

    source = SourceAtom()

    class FakeStateMachine:
        def position(self, key):
            return 0

        def locked(self, key):
            return 0

    sm = FakeStateMachine()

    processor = binary_signal_processor_factory(lockable=False, always_on=False)
    processor_instance = processor(sm)

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
        ((1, 0, 0, False, False), None),  # No signal no change (blip based)
        ((-1, 0, 0, False, False), None),  # No signal no change (blip based)
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
def test_signal_paths_actions(test_input, expected):
    position, lock, signal, lockable, always_on = test_input

    class FakeStateMachine:
        def position(self, key):
            return position

        def locked(self, key):
            return lock

    sm = FakeStateMachine()

    processor = binary_signal_processor_factory(lockable, always_on)
    processor_instance = processor(sm)
    action = processor_instance.process_signal("x", signal)

    assert action == expected


# =================================================
# Testing positions
# =================================================

# Some comments are nonsens


@pytest.mark.parametrize(
    "test_input,expected",
    [
        # pos, lock, signal, lockable, always_on
        ((0, 0, 0, False, False), None),  # No signal must generate no signal
        ((0, 1, 0, True, False), None),  # No signal must generate no signal
        ((1, 0, 0, False, False), None),  # No signal no change (blip based)
        ((-1, 0, 0, False, False), None),  # No signal no change (blip based)
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
def test_signal_paths_positions(test_input, expected):
    position, lock, signal, lockable, always_on = test_input

    class FakeStateMachine:
        def position(self, key):
            return position

        def locked(self, key):
            return lock

    sm = FakeStateMachine()

    class OutputAtom(Atom):
        out = {}

        def onData(self, data, *args):
            self.out = data

    output = OutputAtom()

    processor = binary_signal_processor_factory(lockable, always_on)
    processor_instance = processor(sm)
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
        ((1, 0), None),
        ((-1, 0), None),
        ((1, 1), None),
        ((1, -1), "CLOSE"),
        ((-1, 1), "CLOSE"),
        ((-1, -1), None),
    ],
)
def test_signal_paths_BinarySignalProcessor(test_input, expected):
    position, signal = test_input

    class FakeStateMachine:
        def position(self, key):
            return position

        def locked(self, key):
            raise TypeError("Shouldn't be here")

    sm = FakeStateMachine()

    processor_instance = BinarySignalProcessor(sm)
    action = processor_instance.process_signal("x", signal)

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
def test_signal_paths_LockableBinarySignalProcessor(test_input, expected):
    position, signal, lock = test_input

    class FakeStateMachine:
        def position(self, key):
            return position

        def locked(self, key):
            return lock

    sm = FakeStateMachine()

    processor_instance = LockableBinarySignalProcessor(sm)
    action = processor_instance.process_signal("x", signal)

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
def test_signal_paths_LockableBlipBinarySignalProcessor(test_input, expected):
    position, signal, lock = test_input

    class FakeStateMachine:
        def position(self, key):
            return position

        def locked(self, key):
            return lock

    sm = FakeStateMachine()

    processor_instance = LockableBlipBinarySignalProcessor(sm)
    action = processor_instance.process_signal("x", signal)

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
def test_signal_paths_AlwaysOnLockableBinarySignalProcessor(test_input, expected):
    position, signal, lock = test_input

    class FakeStateMachine:
        def position(self, key):
            return position

        def locked(self, key):
            return lock

    sm = FakeStateMachine()

    processor_instance = AlwaysOnLockableBinarySignalProcessor(sm)
    action = processor_instance.process_signal("x", signal)

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
        ((1, -1), "REVERSE"),
        ((-1, 1), "REVERSE"),
        ((-1, -1), None),
    ],
)
def test_signal_paths_AlwaysOnBinarySignalProcessor(test_input, expected):
    position, signal = test_input

    class FakeStateMachine:
        def position(self, key):
            return position

        def locked(self, key):
            raise TypeError("Shouldn't be here")

    sm = FakeStateMachine()

    processor_instance = AlwaysOnBinarySignalProcessor(sm)
    action = processor_instance.process_signal("x", signal)

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
        ((1, 0), None),
        ((-1, 0), None),
        ((1, 1), None),
        ((1, -1), 0),
        ((-1, 1), 0),
        ((-1, -1), None),
    ],
)
def test_signal_paths_positions_BinarySignalProcessor(test_input, expected):
    position, signal = test_input

    class FakeStateMachine:
        def position(self, key):
            return position

        def locked(self, key):
            raise TypeError("Shouldn't be here")

    sm = FakeStateMachine()

    class OutputAtom(Atom):
        out = {}

        def onData(self, data, *args):
            self.out = data

    output = OutputAtom()
    processor_instance = BinarySignalProcessor(sm)
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
def test_signal_paths_positions_LockableBinarySignalProcessor(test_input, expected):
    position, signal, lock = test_input

    class FakeStateMachine:
        def position(self, key):
            return position

        def locked(self, key):
            return lock

    sm = FakeStateMachine()

    class OutputAtom(Atom):
        out = {}

        def onData(self, data, *args):
            self.out = data

    output = OutputAtom()

    processor_instance = LockableBinarySignalProcessor(sm)

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
    test_input, expected
):
    position, signal, lock = test_input

    class FakeStateMachine:
        def position(self, key):
            return position

        def locked(self, key):
            return lock

    sm = FakeStateMachine()

    class OutputAtom(Atom):
        out = {}

        def onData(self, data, *args):
            self.out = data

    output = OutputAtom()

    processor_instance = AlwaysOnLockableBinarySignalProcessor(sm)
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
        ((1, -1), -1),
        ((-1, 1), 1),
        ((-1, -1), None),
    ],
)
def test_signal_paths_positions_AlwaysOnBinarySignalProcessor(test_input, expected):
    position, signal = test_input

    class FakeStateMachine:
        def position(self, key):
            return position

        def locked(self, key):
            raise TypeError("Shouldn't be here")

    sm = FakeStateMachine()

    class OutputAtom(Atom):
        out = {}

        def onData(self, data, *args):
            self.out = data

    output = OutputAtom()

    processor_instance = AlwaysOnBinarySignalProcessor(sm)
    processor_instance += output
    processor_instance.onData({"strategy": "x", "signal": signal})

    target_position = output.out.get("target_position")

    assert target_position == expected
