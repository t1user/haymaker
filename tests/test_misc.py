from datetime import datetime, timezone
from typing import Literal
from zoneinfo import ZoneInfo

import pytest

from haymaker.misc import Counter, is_active, next_active, process_trading_hours, sign


def test_Counter():
    c = Counter()
    num = c()
    assert isinstance(num, str)
    assert num.endswith("00000")


def test_Counter_increments_by_one():
    c = Counter()
    num = c()
    num1 = c()
    assert int(num1[-1]) - int(num[-1]) == 1


def test_Counter_doesnt_duplicate_on_reinstantiation():
    c = Counter()
    num = c()
    d = Counter()
    num1 = d()
    assert num != num1


@pytest.mark.parametrize(
    "input,expected",
    [(0, 0), (5, 1), (-10, -1), (-3.4, -1), (2.234, 1), (-0, 0), (+0, 0), (-0.0000, 0)],
)
def test_sign_function(input: float | Literal[0] | Literal[5] | Literal[-10], expected):
    assert sign(input) == expected


trading_hours_string = "20231016:1700-20231017:1600;20231017:1700-20231018:1600;20231018:1700-20231019:1600;20231019:1700-20231020:1600;20231021:CLOSED;20231022:1700-20231023:1600"  # noqa


def test_process_trading_hours_returns_no_errors():
    assert isinstance(process_trading_hours(trading_hours_string), list)


def test_process_trading_hours():
    """
    Output generated by `proces_trading_hours` in Eastern compared to
    output given in UTC.
    """
    output = [
        (
            datetime(2023, 10, 16, 17, 00, tzinfo=timezone.utc),
            datetime(2023, 10, 17, 16, 00, tzinfo=timezone.utc),
        ),
        (
            datetime(2023, 10, 17, 17, 00, tzinfo=timezone.utc),
            datetime(2023, 10, 18, 16, 00, tzinfo=timezone.utc),
        ),
        (
            datetime(2023, 10, 18, 17, 00, tzinfo=timezone.utc),
            datetime(2023, 10, 19, 16, 00, tzinfo=timezone.utc),
        ),
        (
            datetime(2023, 10, 19, 17, 00, tzinfo=timezone.utc),
            datetime(2023, 10, 20, 16, 00, tzinfo=timezone.utc),
        ),
        # non trading day is just skipped
        (
            datetime(2023, 10, 22, 17, 00, tzinfo=timezone.utc),
            datetime(2023, 10, 23, 16, 00, tzinfo=timezone.utc),
        ),
    ]
    assert (
        process_trading_hours(
            trading_hours_string, input_tz="UTC", output_tz="US/Eastern"
        )
        == output
    )


@pytest.mark.parametrize(
    "datetimetuple,result",
    [
        ((2023, 10, 16, 17, 15), True),  # from first tuple
        ((2023, 10, 17, 20, 15), True),  # from non-first tuple
        ((2023, 10, 17, 16, 30), False),  # during daily off hours
        ((2023, 10, 21, 17, 15), False),  # during closed day]
    ],
)
def test_is_active(datetimetuple: tuple[int, int, int, int, int], result: bool):
    # hours output in UTC, which compares correctly with now given in US/Central
    hours = process_trading_hours(trading_hours_string, input_tz="US/Central")
    now = datetime(*datetimetuple, tzinfo=ZoneInfo("US/Central"))
    assert is_active(hours, now=now) == result


@pytest.mark.parametrize(
    "datetimetuple,result",
    [
        (
            (2023, 10, 16, 17, 15),
            (2023, 10, 17, 17, 00),
        ),  # from first tuple, market is active
        (
            (2023, 10, 17, 20, 15),
            (2023, 10, 18, 17, 00),
        ),  # from non-first tuple, market is active
        ((2023, 10, 17, 16, 30), (2023, 10, 17, 17, 00)),  # during daily off hours
        ((2023, 10, 21, 17, 15), (2023, 10, 22, 17, 00)),  # during closed day]
    ],
)
def test_next_active(
    datetimetuple: tuple[int, int, int, int, int],
    result: tuple[int, int, int, int, int],
):
    hours = process_trading_hours(
        trading_hours_string,
    )
    now = datetime(*datetimetuple, tzinfo=ZoneInfo("US/Central"))
    assert next_active(hours, now=now) == datetime(
        *result, tzinfo=ZoneInfo("US/Central")
    )
