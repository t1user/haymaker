"""One-to-one processors for structured binary trading signals."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, cast

from ..base import Atom
from ..misc import sign
from .messages import (
    PositionIntent,
    PositionProposal,
    Signal,
    SignalPair,
    SignalType,
)

BinaryDirection = Literal[-1, 0, 1]


def _binary_direction(value: float, field_name: str) -> BinaryDirection:
    """Validate and return an exact binary direction."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number")
    if value not in (-1, 0, 1):
        raise ValueError(f"{field_name} must be -1, 0, or 1")
    return cast(BinaryDirection, int(value))


class OpposingSignalPolicy(StrEnum):
    """Choose how an opposing scalar binary signal is handled.

    ``CLOSE`` first targets flat and waits for a later input before opening
    the other direction. ``REVERSE`` targets the opposing direction
    immediately.
    """

    CLOSE = "CLOSE"
    REVERSE = "REVERSE"


class _BaseBinarySignalProcessor(Atom):
    """Share validation and proposal construction for binary processors."""

    def __init__(self, *, respect_blocked_direction: bool = False) -> None:
        super().__init__()
        if not isinstance(respect_blocked_direction, bool):
            raise TypeError("respect_blocked_direction must be a bool")
        self.respect_blocked_direction = respect_blocked_direction

    def onData(self, data: Signal, *args: object) -> None:
        """Process one Signal and emit a proposal only when action is needed."""

        signal = data
        if not isinstance(signal, Signal):
            raise TypeError(f"{type(self).__name__} accepts only Signal")
        proposal = self.process(signal)
        if proposal is not None:
            self.dataEvent.emit(proposal)

    def process(self, signal: Signal) -> PositionProposal | None:
        """Return the one-to-one transition implied by ``signal``."""

        raise NotImplementedError

    def _proposal(
        self,
        signal: Signal,
        value: float,
        *,
        current_direction: BinaryDirection,
        opposing: OpposingSignalPolicy,
    ) -> PositionProposal | None:
        """Build the transition for one selected binary value."""

        direction = _binary_direction(value, "signal value")
        if direction == 0:
            if signal.signal_type is SignalType.EVENT or current_direction == 0:
                return None
            return PositionProposal(
                signal=signal,
                target_direction=0,
                intent=PositionIntent.CLOSE,
            )
        if current_direction == 0:
            if (
                self.respect_blocked_direction
                and self.book.blocked_direction(signal.source_key) == direction
            ):
                return None
            return PositionProposal(
                signal=signal,
                target_direction=direction,
                intent=PositionIntent.OPEN,
            )
        if current_direction == direction:
            return None
        if opposing is OpposingSignalPolicy.CLOSE:
            return PositionProposal(
                signal=signal,
                target_direction=0,
                intent=PositionIntent.CLOSE,
            )
        return PositionProposal(
            signal=signal,
            target_direction=direction,
            intent=PositionIntent.REVERSE,
        )

    def _current_direction(self, source_key: str) -> BinaryDirection:
        """Return the effective signed direction for a source."""

        return cast(BinaryDirection, sign(self.book.effective_quantity(source_key)))


class BinarySignalProcessor(_BaseBinarySignalProcessor):
    """Convert scalar binary Signals into one-to-one position proposals.

    Use this processor when one field describes both entry and exit behavior.
    It accepts a :class:`Signal` whose value is exactly ``-1``, ``0``, or
    ``1`` and emits zero or one :class:`PositionProposal`. From flat, a
    non-zero value opens. A matching held direction is ignored. For ``STATE``
    signals, zero closes; for ``EVENT`` signals, zero is ignored.

    Args:
        opposing: Whether a non-zero value opposite an effective position
            closes first or reverses immediately. Defaults to ``CLOSE``.
        respect_blocked_direction: If true, suppress opening from flat in
            Book's persisted blocked direction. An opening in the opposite
            direction remains allowed.

    Raises:
        TypeError: If the input is not a Signal, has a SignalPair value, or
            constructor arguments have invalid types.
        ValueError: If the scalar value is outside ``-1``, ``0``, and ``1``.
    """

    def __init__(
        self,
        *,
        opposing: OpposingSignalPolicy = OpposingSignalPolicy.CLOSE,
        respect_blocked_direction: bool = False,
    ) -> None:
        if not isinstance(opposing, OpposingSignalPolicy):
            raise TypeError("opposing must be an OpposingSignalPolicy")
        self.opposing = opposing
        super().__init__(respect_blocked_direction=respect_blocked_direction)

    def process(self, signal: Signal) -> PositionProposal | None:
        """Return the transition implied by a scalar binary Signal."""

        if not isinstance(signal, Signal):
            raise TypeError("BinarySignalProcessor accepts only Signal")
        if isinstance(signal.value, SignalPair):
            raise TypeError("BinarySignalProcessor requires a scalar Signal value")
        return self._proposal(
            signal,
            signal.value,
            current_direction=self._current_direction(signal.source_key),
            opposing=self.opposing,
        )


class BinaryEntryExitSignalProcessor(_BaseBinarySignalProcessor):
    """Process separate binary entry and exit values for one source.

    Use this processor when a strategy calculates independent entry and exit
    conditions. It accepts a :class:`Signal` carrying :class:`SignalPair`.
    While effectively flat, only ``entry`` is consulted. While positioned,
    only ``exit`` is consulted: a matching value is ignored, an opposing value
    closes, ``STATE`` zero closes, and ``EVENT`` zero is ignored. This
    processor never reverses directly.

    Args:
        respect_blocked_direction: If true, suppress an entry from flat in
            Book's persisted blocked direction. An opposite entry is allowed.

    Raises:
        TypeError: If the input is not a Signal or does not carry SignalPair.
        ValueError: If either pair value is outside ``-1``, ``0``, and ``1``.
    """

    def process(self, signal: Signal) -> PositionProposal | None:
        """Return the transition implied by the applicable pair member."""

        if not isinstance(signal, Signal):
            raise TypeError("BinaryEntryExitSignalProcessor accepts only Signal")
        if not isinstance(signal.value, SignalPair):
            raise TypeError(
                "BinaryEntryExitSignalProcessor requires a SignalPair value"
            )
        _binary_direction(signal.value.entry, "entry signal value")
        _binary_direction(signal.value.exit, "exit signal value")
        current_direction = self._current_direction(signal.source_key)
        value = signal.value.entry if current_direction == 0 else signal.value.exit
        return self._proposal(
            signal,
            value,
            current_direction=current_direction,
            opposing=OpposingSignalPolicy.CLOSE,
        )


__all__ = [
    "BinaryEntryExitSignalProcessor",
    "BinarySignalProcessor",
    "OpposingSignalPolicy",
]
