"""One-to-one processors for structured binary trading signals."""

from __future__ import annotations

from abc import ABC
from typing import ClassVar, Literal, cast

from ..base import Atom
from ..misc import sign
from .messages import PositionIntent, PositionProposal, Signal, SignalType


def _emitted_type(source: Atom) -> type | None:
    """Return a source's declared output type, if it has one."""

    return getattr(source, "output_type", None)


class BinarySignalProcessor(Atom):
    """Convert binary Signals into one-to-one position proposals.

    Input values are interpreted by :class:`SignalType`. ``STATE`` zero means
    flat, while ``EVENT`` zero is ignored. A non-zero signal opens from flat,
    is suppressed when it matches the effective position, and closes first
    when it opposes an existing position.

    The processor reads working-order-aware quantity from ``Book`` and emits
    zero or one immutable :class:`PositionProposal`.
    """

    input_type: ClassVar[type] = Signal
    output_type: ClassVar[type] = PositionProposal

    def validate_source(self, source: Atom) -> None:
        """Require an upstream Atom that declares Signal output."""

        if _emitted_type(source) is not Signal:
            raise TypeError(
                f"{type(self).__name__} requires a source declaring "
                "output_type=Signal"
            )

    def onData(self, signal: Signal, *args: object) -> None:
        """Process one Signal and emit a proposal only when action is needed."""

        if not isinstance(signal, Signal):
            raise TypeError(f"{type(self).__name__} accepts only Signal")
        proposal = self.process(signal)
        if proposal is not None:
            self.dataEvent.emit(proposal)

    def process(self, signal: Signal) -> PositionProposal | None:
        """Return the one-to-one transition implied by ``signal``."""

        direction = sign(signal.value)
        current_direction = sign(self.book.effective_quantity(signal.source_key))
        if direction == 0:
            if signal.signal_type is SignalType.EVENT or current_direction == 0:
                return None
            return PositionProposal(
                signal=signal,
                target_direction=0,
                intent=PositionIntent.CLOSE,
            )
        if current_direction == 0:
            if self.opening_is_blocked(signal.source_key, direction):
                return None
            return PositionProposal(
                signal=signal,
                target_direction=direction,
                intent=PositionIntent.OPEN,
            )
        if current_direction == direction:
            return None
        return self.opposing_proposal(signal, direction)

    def opening_is_blocked(self, source_key: str, direction: int) -> bool:
        """Return whether a new position should be suppressed."""

        return False

    def opposing_proposal(
        self, signal: Signal, direction: int
    ) -> PositionProposal:
        """Close an existing position before an opposing ordinary signal."""

        return PositionProposal(
            signal=signal,
            target_direction=0,
            intent=PositionIntent.CLOSE,
        )


class LockableBinarySignalProcessor(BinarySignalProcessor):
    """Suppress opening in a source's persisted stopped-out direction.

    Use this processor in the one-to-one path when stop-loss handling should
    prevent immediate re-entry in the stopped direction. It accepts Signal,
    applies the ordinary close-first binary behavior, queries
    ``Book.blocked_direction(source_key)`` before opening from flat, and emits
    zero or one PositionProposal. Construct it without arguments after a
    RuntimeContext has been installed.
    """

    def opening_is_blocked(self, source_key: str, direction: int) -> bool:
        """Return whether Book blocks this source and direction."""

        return self.book.blocked_direction(source_key) == direction


class AlwaysOnBinarySignalProcessor(BinarySignalProcessor):
    """Reverse directly when a non-zero signal opposes an existing position.

    Use this processor for continuously invested one-to-one strategies. It
    accepts Signal and emits a REVERSE PositionProposal instead of first
    flattening on an opposing non-zero input. STATE zero still closes and
    EVENT zero remains ignored. Construct it without arguments after a
    RuntimeContext has been installed.
    """

    def opposing_proposal(
        self, signal: Signal, direction: int
    ) -> PositionProposal:
        """Return a reversal proposal for an opposing non-zero signal."""

        return PositionProposal(
            signal=signal,
            target_direction=cast(Literal[-1, 1], direction),
            intent=PositionIntent.REVERSE,
        )


class AlwaysOnLockableBinarySignalProcessor(
    LockableBinarySignalProcessor, AlwaysOnBinarySignalProcessor
):
    """Combine stop-out opening locks with always-on reversal behavior.

    This no-argument one-to-one processor accepts Signal, reverses directly on
    opposing non-zero input, and suppresses an opening from flat when Book
    blocks that direction. It emits zero or one PositionProposal and requires
    an installed RuntimeContext.
    """


def binary_signal_processor_factory(
    *, lockable: bool = False, always_on: bool = False
) -> type[BinarySignalProcessor]:
    """Select a built-in one-to-one processor class.

    Args:
        lockable: Suppress new positions in Book's blocked direction.
        always_on: Reverse rather than close on an opposing non-zero signal.

    Returns:
        Processor class matching the requested semantics.
    """

    if lockable and always_on:
        return AlwaysOnLockableBinarySignalProcessor
    if lockable:
        return LockableBinarySignalProcessor
    if always_on:
        return AlwaysOnBinarySignalProcessor
    return BinarySignalProcessor


__all__ = [
    "AlwaysOnBinarySignalProcessor",
    "AlwaysOnLockableBinarySignalProcessor",
    "BinarySignalProcessor",
    "LockableBinarySignalProcessor",
    "binary_signal_processor_factory",
]
