from __future__ import annotations

import pytest

from ib_tools.signals import (
    AlwaysOnBinarySignalProcessor,
    AlwaysOnLockableBinarySignalProcessor,
    BinarySignalProcessor,
    LockableBinarySignalProcessor,
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


# =================================================
# Testing actions
# =================================================


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


# # =================================================
# # Testing positions
# # =================================================


# @pytest.mark.parametrize(
#     "test_input,expected",
#     [
#         # pos, lock, signal, lockable, always_on
#         ((0, 0, 0, False, False), 0),  # No signal must generate no signal
#         ((0, 1, 0, True, False), 0),  # No signal must generate no signal
#         ((1, 0, 0, False, False), 1),  # No signal no change (blip based)
#         ((-1, 0, 0, False, False), -1),  # No signal no change (blip based)
#         ((1, 0, 0, True, False), 0),  # Zero signal, zero position (lockable)
#         ((-1, 0, 0, True, False), 0),  # Zero signal, zero position (lockable)
#         # ---
#         ((1, 0, 1, False, False), 1),  # Same signal with existing position
#         ((-1, 0, -1, False, False), -1),  # Same, opposite direction
#         ((1, 0, -1, False, True), -1),  # Reverse signal, existing position, always-on
#         ((1, 0, 1, False, True), 1),  # Same, opposite direction
#         ((0, 1, 1, True, False), 0),  # No position, signal, lockable, with lock
#         ((0, -1, -1, True, False), 0),  # Same, opposite direction
#         ((0, 1, 1, False, False), 1),  # No position, signal, not lockable, with lock
#         ((0, -1, -1, False, False), -1),  # Same, oppposite direction
#         ((0, 0, 1, True, False), 1),  # No position, signal, lockable, no lock
#         ((0, 1, 1, True, False), 0),  # Same, opposite direction
#         ((0, 0, 1, True, False), 1),  # No position, signal, lockable, no lock
#         ((0, -1, 1, True, True), 1),  # Lockable, signal, opposite (irrelevant) lock
#         ((-1, 0, 1, True, True), 1),  # Always_on position with opposite signal
#         ((1, 0, -1, True, True), -1),  # Same, opposite direction
#         ((1, 0, -1, False, False), 0),  # Position, opposite, closing signal
#         ((-1, 0, 1, False, False), 0),  # Same, reverse direction
#         # same but with non-zero positions
#         ((2, 0, -1, True, True), -1),  # Same, opposite direction
#         ((2, 0, -1, False, False), 0),  # Position, opposite, closing signal
#         ((-2, 0, 1, False, False), 0),  # Same, reverse direction
#     ],
# )
# def test_signal_paths(test_input, expected):
#     class FakeStateMachine:
#         def position(self, key):
#             return test_input[0]

#         def locked(self, key):
#             return test_input[1]

#     sm = FakeStateMachine()
#     bproc = BinarySignalProcessor(sm)

#     class Brick:
#         @bproc
#         def signal(self):
#             signal = test_input[2]
#             context = SignalContext(
#                 ("NQ", "some_strategy"), test_input[3], test_input[4]
#             )
#             return signal, context

#     b = Brick()
#     assert b.signal()[0] == expected


# # =================================================
# # Testing transactions
# # =================================================


# # @pytest.mark.skip("not implement yet")
# @pytest.mark.parametrize(
#     "test_input,expected",
#     [
#         # pos, lock, signal, lockable, always_on
#         ((0, 0, 0, False, False), 0),  # No signal must generate no signal
#         ((0, 1, 0, True, False), 0),  # No signal must generate no signal
#         ((1, 0, 0, False, False), 0),  # No signal no change (blip based)
#         ((-1, 0, 0, False, False), 0),  # No signal no change (blip based)
#         ((1, 0, 0, True, False), -1),  # Zero signal, zero position (lockable)
#         ((-1, 0, 0, True, False), 1),  # Zero signal, zero position (lockable)
#         # -----
#         ((1, 0, 1, False, False), 0),  # Same signal with existing position
#         ((-1, 0, -1, False, False), 0),  # Same, opposite direction
#         ((1, 0, -1, False, True), -2),  # Reverse signal, existing position, always-on
#         ((-1, 0, 1, False, True), 2),  # Same, opposite direction
#         ((0, 1, 1, True, False), 0),  # No position, signal, lockable, with lock
#         ((0, -1, -1, True, False), 0),  # Same, opposite direction
#         ((0, 1, 1, False, False), 1),  # No position, signal, not lockable, with lock
#         ((0, -1, -1, False, False), -1),  # Same, oppposite direction
#         ((0, 0, 1, True, False), 1),  # No position, signal, lockable, no lock
#         ((0, 1, 1, True, False), 0),  # Same, opposite direction
#         ((0, 0, 1, True, False), 1),  # No position, signal, lockable, no lock
#         ((0, -1, 1, True, True), 1),  # Lockable, signal, opposite (irrelevant) lock
#         ((-1, 0, 1, True, True), 2),  # Always_on position with opposite signal
#         ((1, 0, -1, True, True), -2),  # Same, opposite direction
#         ((1, 0, -1, False, False), -1),  # Position, opposite, closing signal
#         ((-1, 0, 1, False, False), 1),  # Same, reverse direction
#         ((1, 0, 1, False, True), 0),  # Irrelevant signal, always-on
#         # same but with non-zero position
#         ((2, 0, 1, False, False), 0),  # Same signal with existing position
#         ((-2, 0, -1, False, False), 0),  # Same, opposite direction
#         ((3, 0, -1, False, True), -2),  # Reverse signal, existing position, always-on
#         ((-3, 0, 1, False, True), 2),  # Same, opposite direction
#         ((-2, 0, 1, True, True), 2),  # Always_on position with opposite signal
#         ((3, 0, -1, True, True), -2),  # Same, opposite direction
#         ((3, 0, -1, False, False), -1),  # Position, opposite, closing signal
#         ((-2, 0, 1, False, False), 1),  # Same, reverse direction
#         ((4, 0, 1, False, True), 0),  # Irrelevant signal, always-on
#         ((-4, 0, -1, False, True), 0),  # Irrelevant signal, always-on
#     ],
# )
# def test_signal_paths_transactions(test_input, expected):
#     class FakeStateMachine:
#         def position(self, key):
#             return test_input[0]

#         def locked(self, key):
#             return test_input[1]

#     sm = FakeStateMachine()
#     bproc = BinarySignalProcessor(sm)

#     class Brick:
#         @bproc
#         def signal(self):
#             signal = test_input[2]
#             context = SignalContext(
#                 ("NQ", "some_strategy"), test_input[3], test_input[4]
#             )
#             return signal, context

#     b = Brick()
#     assert b.signal()[1] == expected
