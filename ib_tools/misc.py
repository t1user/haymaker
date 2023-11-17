from __future__ import annotations

import itertools
import random
import string
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Literal, Optional

import ib_insync as ibi
import pytz


def action(signal: int) -> str:
    """
    Convert numerical trade direction signal (-1, or 1) to string
    ('BUY' or 'SELL').
    """
    assert signal in (-1, 1), f"Invalid trade signal {signal}"

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


def sign(x: float) -> Literal[-1, 0, 1]:
    return 0 if x == 0 else -1 if x < 0 else 1


def process_trading_hours(
    th: str, input_tz: str = "US/Central", *, output_tz: str = "UTC"
) -> list[tuple[datetime, datetime]]:
    """
    Given string from :attr:`ibi.ContractDetails.tradingHours` return
    active hours as a list of (from, to) tuples.

    Args:
    -----

    tzname: instrument's timezone

    output_tzname: output will be converted to this timezone (best if
    left at UTC); this param is really for testing
    """
    input_tz_ = pytz.timezone(input_tz)
    output_tz_ = pytz.timezone(output_tz)

    def datetime_tuples(s: str) -> tuple[Optional[datetime], Optional[datetime]]:
        def to_datetime(datetime_string: str) -> Optional[datetime]:
            if datetime_string[-6:] == "CLOSED":
                return None
            else:
                return input_tz_.localize(
                    datetime.strptime(datetime_string, "%Y%m%d:%H%M")
                ).astimezone(tz=output_tz_)

        try:
            f, t = s.split("-")
        except ValueError:
            return (None, None)

        return to_datetime(f), to_datetime(t)

    out = []
    for i in th.split(";"):
        tuples = datetime_tuples(i)
        if not tuples[0]:
            continue
        else:
            out.append(tuples)
    return out  # type: ignore


def is_active(
    time_tuples: Optional[list[tuple[datetime, datetime]]] = None,
    now: Optional[datetime] = None,
) -> bool:
    """
    Given list of trading hours tuples from `.process_trading_hours`
    check if the market is active at the moment.
    """
    if not time_tuples:
        return True

    if not now:
        now = datetime.now(tz=timezone.utc)

    def test_p(t):
        return t[0] < now < t[1]

    for t in time_tuples:
        if test_p(t):
            return True
    return False


def next_open(
    time_tuples: Optional[list[tuple[datetime, datetime]]] = None,
    now: Optional[datetime] = None,
) -> datetime:
    """
    Given list of trading hours tuples from `.process_trading_hours`
    return time of nearest market re-open (regardless if market is
    open now).  Should be used after it has been tested that
    `.is_active` is False.
    """

    if not now:
        now = datetime.now(tz=pytz.timezone("UTC"))

    if not time_tuples:
        return now

    left_edges = [e[0] for e in time_tuples if e[0] > now]
    # print(left_edges)
    return left_edges[0]
