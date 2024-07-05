from __future__ import annotations

import datetime as dt
import itertools
import random
import string
from collections import UserDict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union

import ib_insync as ibi
import pytz

from .config import CONFIG


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


def reverse_action(action: str) -> Literal["BUY", "SELL"]:
    assert action.upper() in ("BUY", "SELL"), "Invalid order action"
    return "BUY" if action == "SELL" else "SELL"


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


COUNTER = Counter()


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
) -> list[tuple[dt.datetime, dt.datetime]]:
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

    def datetime_tuples(s: str) -> tuple[Optional[dt.datetime], Optional[dt.datetime]]:
        def to_datetime(datetime_string: str) -> Optional[dt.datetime]:
            if datetime_string[-6:] == "CLOSED":
                return None
            else:
                return input_tz_.localize(
                    dt.datetime.strptime(datetime_string, "%Y%m%d:%H%M")
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
    time_tuples: Optional[list[tuple[dt.datetime, dt.datetime]]] = None,
    now: Optional[dt.datetime] = None,
) -> bool:
    """
    Given list of trading hours tuples from `.process_trading_hours`
    check if the market is active at the moment.
    """
    if not time_tuples:
        return True

    if not now:
        now = dt.datetime.now(tz=dt.timezone.utc)

    def test_p(t):
        return t[0] < now < t[1]

    for t in time_tuples:
        if test_p(t):
            return True
    return False


def next_active(
    time_tuples: Optional[list[tuple[dt.datetime, dt.datetime]]] = None,
    now: Optional[dt.datetime] = None,
) -> dt.datetime:
    """
    Given list of trading hours tuples from `.process_trading_hours`
    return time of nearest market re-open (regardless if market is
    open now).  Should be used after it has been tested that
    `.is_active` is False.
    """

    if not now:
        now = dt.datetime.now(tz=pytz.timezone("UTC"))

    if not time_tuples:
        return now

    left_edges = [e[0] for e in time_tuples if e[0] > now]
    # print(left_edges)
    return left_edges[0]


# ###### Serializer ########


def tree(obj):
    """
    Convert object to a tree of lists, dicts and simple values.
    The result can be serialized to JSON.
    """
    if isinstance(obj, (bool, float, int, str, bytes)):
        return obj
    elif isinstance(obj, (dt.date, dt.time)):
        return obj.isoformat()
    elif isinstance(obj, (dict, UserDict)):
        return {k: tree(v) for k, v in obj.items()}
    elif ibi.util.isnamedtupleinstance(obj):
        return {
            obj.__class__.__qualname__: {f: tree(getattr(obj, f)) for f in obj._fields}
        }
    elif isinstance(obj, (list, tuple, set)):
        return [tree(i) for i in obj]
    elif ibi.util.is_dataclass(obj):
        return {obj.__class__.__qualname__: tree(ibi.util.dataclassNonDefaults(obj))}
    else:
        return str(obj)


# ###### De-serializer ########


def decode_tree(obj: Any) -> Any:
    """
    Convert simple values created with :func:`.tree` back to orginal objects.

    """

    obj_list = ibi.__all__

    def process_value(
        value: Any,
    ) -> Union[bool, int, float, bytes, list, dt.datetime, str, None]:
        if isinstance(value, dict):
            v = process_dict(value)
            return v
        elif isinstance(value, (bool, int, float, bytes)):
            return value
        elif isinstance(value, list):
            return [process_value(i) for i in value]
        elif isinstance(value, str):
            if value == "None":
                return None
            try:
                return dt.datetime.fromisoformat(value)
            except ValueError:
                return value
        else:
            return value

    def process_key(k: str, v: Any) -> Any:
        if k in obj_list:
            # `Order` subclassess will get errors on some values
            # so making sure to use only the superclass
            if k.endswith("Order"):
                k = "Order"
            obj = eval(f"ibi.{k}")
            # `Contract` must be created using its `create` staticmethod
            obj = getattr(obj, "create", None) or obj
            return obj(**v)
        else:
            return k

    def process_dict(d: dict) -> Any:
        output = {}
        for k, v in d.items():
            value_ = process_value(v) if k != "lastTradeDateOrContractMonth" else v
            key_ = process_key(k, value_)
            if isinstance(key_, str):
                output[key_] = value_
            else:
                return key_
        return output

    return process_value(obj)


def default_path(*dirnames: str) -> str:
    """
    Return path created by joining  ~/ib_data/ and recursively all dirnames
    If the path doesn't exist create it.
    """
    home = Path.home()
    default_directory = CONFIG.get("data_folder", "ib_data")
    path = Path(home, default_directory, *dirnames)
    path.mkdir(exist_ok=True, parents=True)
    return str(path)


def trade_fill_price(trade: ibi.Trade) -> float:
    return weighted_average(
        *[(fill.execution.price, fill.execution.shares) for fill in trade.fills]
    )


def weighted_average(*values: tuple[float, float]) -> float:
    """
    Return weighted average of `values`.

    Args:
    values - tuple of (value, weight)

    """
    running_multiples = 0.0
    running_total = 0.0
    for i, k in values:
        running_multiples += i * k
        running_total += k
    try:
        return running_multiples / running_total
    except ZeroDivisionError:
        return 0
