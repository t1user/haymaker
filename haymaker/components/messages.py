"""Immutable messages shared by Haymaker's built-in trading components."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Literal

import ib_insync as ibi


def utc_now() -> datetime:
    """Return the current timezone-aware UTC time."""

    return datetime.now(timezone.utc)


def _validate_timestamp(value: datetime | None, field_name: str) -> None:
    """Validate that an optional timestamp contains timezone information."""

    if value is not None and (value.tzinfo is None or value.utcoffset() is None):
        raise ValueError(f"{field_name} must be timezone-aware")


def _finite_number(value: float, field_name: str) -> float:
    """Return a finite float or raise a field-specific validation error."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _contract(value: ibi.Contract) -> ibi.Contract:
    """Return an IB contract or raise a clear validation error."""

    if not isinstance(value, ibi.Contract):
        raise TypeError("contract must be an ib_insync.Contract")
    return value


def _metadata(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Copy metadata and expose the top-level mapping as read-only."""

    if not isinstance(value, Mapping):
        raise TypeError("metadata must be a mapping")
    return MappingProxyType(dict(value))


class SignalType(StrEnum):
    """Declare state-replacement or event-occurrence Signal semantics.

    ``STATE`` replaces a source's prior desired state. Every ``EVENT`` is a new
    event; repetition does not merely reaffirm earlier state.
    """

    STATE = "STATE"
    EVENT = "EVENT"


class PositionIntent(StrEnum):
    """Assert the initial transition for a one-to-one position target.

    Intent is mandatory on PositionProposal and optional on general
    PositionTarget. It does not replace the target's authoritative quantity.
    """

    OPEN = "OPEN"
    CLOSE = "CLOSE"
    REVERSE = "REVERSE"


class StandardOrderRole(StrEnum):
    """Provide standard order attribution while permitting custom strings.

    Constructing the enum with any non-empty custom string preserves that
    value, so persistence and blotter records are not limited to this standard
    vocabulary.
    """

    OPEN = "OPEN"
    CLOSE = "CLOSE"
    TARGET_ADJUSTMENT = "TARGET_ADJUSTMENT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    ROLL = "ROLL"
    LIQUIDATION = "LIQUIDATION"
    RECONCILIATION = "RECONCILIATION"
    MANUAL = "MANUAL"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def _missing_(cls, value: object) -> StandardOrderRole | None:
        """Create an unnamed member for a non-empty custom string role."""

        if not isinstance(value, str) or not value:
            return None
        member = str.__new__(cls, value)
        member._name_ = value
        member._value_ = value
        return member


@dataclass(frozen=True, kw_only=True)
class SignalPair:
    """Carry distinct entry and exit values in one Signal.

    Use this value with :class:`BinaryEntryExitSignalProcessor` when a
    one-to-one strategy calculates separate entry and exit conditions for the
    same observation. Both values must be finite numbers; the receiving
    processor may impose narrower constraints such as ``-1``, ``0``, and
    ``1``.

    Args:
        entry: Value consulted while the source is effectively flat.
        exit: Value consulted while the source has an effective position.

    Raises:
        TypeError: If either value is not a real number.
        ValueError: If either value is not finite.
    """

    entry: float
    exit: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "entry", _finite_number(self.entry, "entry"))
        object.__setattr__(self, "exit", _finite_number(self.exit, "exit"))


@dataclass(frozen=True, kw_only=True)
class Signal:
    """Represent one structured trading input.

    Args:
        source_key: Stable opaque identity of the logical input path.
        contract: Contract to which the signal applies.
        value: Finite scalar value or an entry/exit pair. Consumers decide
            which value shapes they support.
        signal_type: Whether the value replaces state or records an event.
        as_of: Optional effective time of the market observation.
        created_at: Local signal-generation time.
        metadata: Optional calculated or audit information. The mapping is
            copied and its top level is exposed read-only.

    Raises:
        TypeError: If a field has the wrong structural type.
        ValueError: If an identity, number, or timestamp is invalid.
    """

    source_key: str
    contract: ibi.Contract
    value: float | SignalPair
    signal_type: SignalType
    as_of: datetime | None = None
    created_at: datetime = field(default_factory=utc_now)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source_key, str) or not self.source_key:
            raise ValueError("source_key must be a non-empty string")
        object.__setattr__(self, "contract", _contract(self.contract))
        if not isinstance(self.value, SignalPair):
            object.__setattr__(self, "value", _finite_number(self.value, "value"))
        if not isinstance(self.signal_type, SignalType):
            raise TypeError("signal_type must be a SignalType")
        _validate_timestamp(self.as_of, "as_of")
        _validate_timestamp(self.created_at, "created_at")
        object.__setattr__(self, "metadata", _metadata(self.metadata))


@dataclass(frozen=True, kw_only=True)
class PositionProposal:
    """Carry one signal processor's one-to-one position decision.

    Args:
        signal: Original immutable signal.
        target_direction: Desired short, flat, or long direction.
        intent: Mandatory episode transition asserted at this boundary.
        created_at: Time at which the proposal was created.

    Raises:
        TypeError: If ``signal`` or ``intent`` has the wrong type.
        ValueError: If the direction or timestamp is invalid.
    """

    signal: Signal
    target_direction: Literal[-1, 0, 1]
    intent: PositionIntent
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not isinstance(self.signal, Signal):
            raise TypeError("signal must be a Signal")
        if isinstance(self.target_direction, bool) or self.target_direction not in (
            -1,
            0,
            1,
        ):
            raise ValueError("target_direction must be -1, 0, or 1")
        if not isinstance(self.intent, PositionIntent):
            raise TypeError("intent must be a PositionIntent")
        _validate_timestamp(self.created_at, "created_at")


@dataclass(frozen=True, kw_only=True)
class PositionTarget:
    """Represent an absolute signed execution setpoint.

    Args:
        contract: Concrete execution Contract.
        target_quantity: Absolute signed quantity desired after convergence.
        created_at: Time at which this target superseded an earlier target.
        source_key: Optional one-to-one input identity.
        intent: Optional initial one-to-one lifecycle assertion. The numeric
            target remains authoritative after acceptance.
        metadata: Optional execution inputs or audit references. The mapping is
            copied and its top level is exposed read-only.

    Raises:
        TypeError: If a field has the wrong structural type.
        ValueError: If quantity, identity, or timestamp is invalid.
    """

    contract: ibi.Contract
    target_quantity: float
    created_at: datetime = field(default_factory=utc_now)
    source_key: str | None = None
    intent: PositionIntent | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "contract", _contract(self.contract))
        object.__setattr__(
            self,
            "target_quantity",
            _finite_number(self.target_quantity, "target_quantity"),
        )
        _validate_timestamp(self.created_at, "created_at")
        if self.source_key is not None and (
            not isinstance(self.source_key, str) or not self.source_key
        ):
            raise ValueError("source_key must be None or a non-empty string")
        if self.intent is not None and not isinstance(self.intent, PositionIntent):
            raise TypeError("intent must be None or a PositionIntent")
        object.__setattr__(self, "metadata", _metadata(self.metadata))


__all__ = [
    "PositionIntent",
    "PositionProposal",
    "PositionTarget",
    "Signal",
    "SignalPair",
    "SignalType",
    "StandardOrderRole",
]
