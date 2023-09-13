from __future__ import annotations

import itertools
import random
import string
from enum import Enum
from functools import wraps
from typing import Callable, Literal

import ib_insync as ibi


def action(signal: int) -> str:
    """
    Convert numerical trade direction signal (-1, or 1) to string
    ('BUY' or 'SELL').
    """
    assert signal in (-1, 1), "Invalid trade signal"
    return "BUY" if signal == 1 else "SELL"


def action_to_signal(action: str) -> Literal[-1, 1]:
    assert action.upper() in ("BUY", "SELL"), "Invalid trade signal"
    return 1 if action == "BUY" else -1


class Sigact(str, Enum):
    """
    Signal action.  Indication from a strategy what kind of action is
    required.
    """

    open = "OPEN"
    close = "CLOSE"
    reverse = "REVERSE"
    OPEN = "OPEN"
    CLOSE = "CLOSE"
    REVERSE = "REVERSE"


Lock = Literal[-1, 0, 1]
Signal = Literal[-1, 0, 1]
Action = Literal["OPEN", "CLOSE", "REVERSE"]
Callback = Callable[[ibi.Trade], None]


class Counter:
    """
    Generate a string of consecutive numbers preceded by a character
    string, which is very unlikely to be repeated after multiple class
    re-instantiations.
    """

    counter_seed = itertools.count(1, 1).__next__

    def __init__(self, number_of_letters=5, number_of_numbers=6):
        self.counter = itertools.count(
            100000 * self.counter_seed(), 1  # type: ignore
        ).__next__
        self.character_string = "".join(
            random.choices(string.ascii_letters, k=number_of_letters)
        )

    def __call__(self) -> str:
        return f"{self.character_string}{self.counter()}"


def round_tick(price: float, tick_size: float) -> float:
    floor = price // tick_size
    remainder = price % tick_size
    if remainder > (tick_size / 2):
        floor += 1
    return round(floor * tick_size, 4)


# =====================================================
# NOT IN USE
# =====================================================


def doublewrap(func):
    """
    A decorator decorator, allowing the decorator to be used as either:
    ``@decorator(*args, **kwargs)`` or ``@decorator``

    """

    @wraps(func)
    def new_dec(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            # actual decorated function
            return func(args[0])
        else:
            # decorator arguments
            return lambda realf: func(realf, *args, **kwargs)

    return new_dec


@doublewrap
def dataframe_signal_extractor(func, field: str = "signal"):
    """
    This is a decorator to apply to method returning dataframe.  The
    method should return dataframe, then decorator will extract signal
    from it.

    Signal is from the last row in the dataframe, column name
    determined by ``field``, if none given column ``signal`` will be
    used.

    Args:
    -----

    field : name of dataframe collumn with signals
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        df = func(*args, **kwargs)
        return df[field].iloc[-1]

    return wrapper
