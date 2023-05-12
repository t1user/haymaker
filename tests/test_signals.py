import pytest

from ib_tools.base import Atom
from ib_tools.signals import BinarySignalProcessor, Signal, SignalProcessor


def test_interface_not_instantiable():
    with pytest.raises(TypeError):
        SignalProcessor()


def test_signal_processor_decorator_working():
    class FakeStateMachine:
        def position(self, key):
            return 1

        def locked(self, key):
            return False

    sm = FakeStateMachine()

    bproc = BinarySignalProcessor(sm)

    class Brick(Atom):
        @bproc
        def signal(self):
            return Signal("NQ", 1, {}, None)

    b = Brick()
    assert b.signal() == 1


def test_signal_processor_working_when_not_being_decorator():
    class FakeStateMachine:
        def position(self, key):
            return 1

        def locked(self, key):
            return False

    sm = FakeStateMachine()

    bproc = BinarySignalProcessor(sm)

    class Brick(Atom):
        def signal(self):
            return Signal("NQ", 1, {}, None)

    b = Brick()
    assert bproc.process_signal(b.signal()) == 1


def test_repr():
    bp = BinarySignalProcessor("xxx")
    assert bp.__repr__() == "BinarySignalProcessor"


@pytest.mark.parametrize(
    "test_input,expected",
    [
        # pos, lock, signal, lockable, always_on
        ((1, 0, 1, False, False), 1),  # Same signal with existing position
        ((-1, 0, -1, False, False), -1),  # Same, opposite direction
        ((1, 0, -1, False, True), -1),  # Reverse signal, existing position, always-on
        ((1, 0, 1, False, True), 1),  # Same, opposite direction
        ((0, 1, 1, True, False), 0),  # No position, signal, lockable, with lock
        ((0, -1, -1, True, False), 0),  # Same, opposite direction
        ((0, 1, 1, False, False), 1),  # No position, signal, not lockable, with lock
        ((0, -1, -1, False, False), -1),  # Same, oppposite direction
        ((0, 0, 1, True, False), 1),  # No position, signal, lockable, no lock
        ((0, 1, 1, True, False), 0),  # Same, opposite direction
        ((0, 0, 1, True, False), 1),  # No position, signal, lockable, no lock
        ((0, -1, 1, True, True), 1),  # Lockable, signal, opposite (irrelevant) lock
        ((-1, 0, 1, True, True), 1),  # Always_on position with opposite signal
        ((1, 0, -1, True, True), -1),  # Same, opposite direction
    ],
)
def test_signal_paths(test_input, expected):
    class FakeStateMachine:
        def position(self, key):
            return test_input[0]

        def locked(self, key):
            return test_input[1]

    sm = FakeStateMachine()
    bproc = BinarySignalProcessor(sm)

    class Brick(Atom):
        @bproc
        def signal(self):
            return Signal("NQ", test_input[2], {}, None, test_input[3], test_input[4])

    b = Brick()
    assert b.signal() == expected
