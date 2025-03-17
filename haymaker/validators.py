from __future__ import annotations

import dataclasses
from typing import Any, Callable, TypeVar

import ib_insync as ibi

T = TypeVar("T")


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
