"""Shared primitive normalization and Interactive Brokers field validation."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from datetime import datetime
from numbers import Real
from types import MappingProxyType
from typing import Any, Callable, TypeVar

import ib_insync as ibi

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


def aware_datetime(value: object, name: str) -> datetime:
    """Return a timezone-aware datetime.

    Args:
        value: Candidate datetime.
        name: Field name used in validation messages.

    Raises:
        TypeError: If ``value`` is not a datetime.
        ValueError: If ``value`` has no effective timezone offset.
    """

    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def optional_aware_datetime(value: object, name: str) -> datetime | None:
    """Return ``None`` or a timezone-aware datetime.

    Args:
        value: Candidate optional datetime.
        name: Field name used in validation messages.

    Raises:
        TypeError: If a non-``None`` value is not a datetime.
        ValueError: If a datetime has no effective timezone offset.
    """

    if value is None:
        return None
    return aware_datetime(value, name)


def finite_number(value: object, name: str) -> float:
    """Normalize a real finite number to ``float``.

    Args:
        value: Candidate numeric value. Booleans are rejected.
        name: Field name used in validation messages.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If ``value`` is NaN or infinite.
    """

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def readonly_mapping(value: Mapping[K, V], name: str) -> Mapping[K, V]:
    """Copy a mapping and return a read-only top-level view.

    Args:
        value: Mapping to copy.
        name: Field name used in validation messages.

    Raises:
        TypeError: If ``value`` is not a mapping.
    """

    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return MappingProxyType(dict(value))


def non_empty_string(value: object, name: str) -> str:
    """Return a non-empty string.

    Args:
        value: Candidate string.
        name: Field name used in validation messages.

    Raises:
        TypeError: If ``value`` is not a string.
        ValueError: If ``value`` is empty.
    """

    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def ib_contract(value: object, name: str = "contract") -> ibi.Contract:
    """Return an Interactive Brokers Contract.

    Args:
        value: Candidate Contract.
        name: Field name used in validation messages.

    Raises:
        TypeError: If ``value`` is not an ``ib_insync.Contract``.
    """

    if not isinstance(value, ibi.Contract):
        raise TypeError(f"{name} must be an ib_insync.Contract")
    return value


def qualified_contract(value: object, name: str = "contract") -> ibi.Contract:
    """Return an IB Contract with a concrete non-zero ``conId``.

    Args:
        value: Candidate Contract.
        name: Field name used in validation messages.

    Raises:
        TypeError: If ``value`` is not an ``ib_insync.Contract``.
        ValueError: If the Contract has no concrete ``conId``.
    """

    contract = ib_contract(value, name)
    if not contract.conId:
        raise ValueError(f"{name} must have a non-zero conId")
    return contract


class Validator:
    """
    Descriptor class to validate attributes of a class.

    Args:
        *validators: Callables that will be used to validate the attribute.

    """

    def __init__(self, *validators: Callable[[T], T]):
        self.validators = validators

    def __set_name__(self, owner, name) -> None:
        self.private_name = "_" + name

    def __get__(self, obj, objtype=None):
        return getattr(obj, self.private_name)

    def __set__(self, obj, value: Any) -> None:
        if self.validate(value):
            setattr(obj, self.private_name, value)

    def validate(self, value) -> bool:
        for validator in self.validators:
            try:
                validator(value)
            except ValueError as exc:
                raise ValueError(
                    f"Failed to validate attr: {self.private_name.strip('_')} {exc}"
                ) from exc
        return True


def bar_size_validator(s: str) -> str:
    """Verify if given string is a valid IB api bar size str"""
    ok_str = [
        "1 secs",
        "5 secs",
        "10 secs",
        "15 secs",
        "30 secs",
        "1 min",
        "2 mins",
        "3 mins",
        "5 mins",
        "10 mins",
        "15 mins",
        "20 mins",
        "30 mins",
        "1 hour",
        "2 hours",
        "3 hours",
        "4 hours",
        "8 hours",
        "1 day",
        "1 week",
        "1 month",
    ]
    if s not in ok_str:
        raise ValueError(f"bar size : {s} is invalid, must be one of {ok_str}")
    else:
        return s


def wts_validator(s: str) -> str:
    """Verify if given string is a valide IB api whatToShow str"""
    ok_str = [
        "TRADES",
        "MIDPOINT",
        "BID",
        "ASK",
        "BID_ASK",
        "ADJUSTED_LAST",
        "HISTORICAL_VOLATILITY",
        "OPTION_IMPLIED_VOLATILITY",
        "REBATE_RATE",
        "FEE_RATE",
        "YIELD_BID",
        "YIELD_ASK",
        "YIELD_BID_ASK",
        "YIELD_LAST",
    ]
    if s not in ok_str:
        raise ValueError(f"{s} is a wrong whatToShow value, must be one of {ok_str}")
    else:
        return s


def order_field_validator(value: dict[str, Any]) -> dict[str, Any]:
    """Validate if :class:`ibi.Order` is instantiated with correct args."""
    if diff := (set(value.keys()) - set(dataclasses.asdict(ibi.Order()).keys())):
        raise ValueError(f"Wrong order attrs: {diff}")
    else:
        return value
