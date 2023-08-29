from __future__ import annotations

from functools import wraps
from typing import Literal, Tuple


def action(signal: int) -> str:
    """
    Convert numerical trade direction signal (-1, or 1) to string
    ('BUY' or 'SELL').
    """
    assert signal in (-1, 1), "Invalid trade signal"
    return "BUY" if signal == 1 else "SELL"


P = Literal[-1, 0, 1]
S = Literal[-2, -1, 0, 1, 2]

Operation = Literal["entry", "close", "reverse", None]
# position, signal, operation
PS = Tuple[P, S, Operation]


Lock = Literal[-1, 0, 1]
Signal = Literal[-1, 0, 1]
Action = Literal["OPEN", "CLOSE", "REVERSE"]


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
