from datetime import datetime, timedelta
from random import randint
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from haymaker.durationStr_converters import (
    barSizeSetting_to_timedelta,
    datapoints_to_durationStr,
    datapoints_to_timedelta,
    date_to_delta,
    date_to_delta_wrapper,
    delta_to_durationStr,
    durationStr_to_datapoints,
    durationStr_to_offset,
    offset_durationStr,
    us_business_days_offset,
)


@pytest.mark.parametrize(
    "barSizeSetting,delta",
    [
        ("30 secs", timedelta(seconds=30)),
        ("1 day", timedelta(days=1)),
        ("2 days", timedelta(days=2)),
        ("1 week", timedelta(days=5)),
        ("1 month", timedelta(days=22)),
    ],
)
def test_barSizeSetting_to_timedelta_adjusted(
    barSizeSetting: str, delta: timedelta
) -> None:
    assert barSizeSetting_to_timedelta(barSizeSetting, adjusted=True) == delta


@pytest.mark.parametrize(
    "barSizeSetting,delta",
    [
        ("30 secs", timedelta(seconds=30)),
        ("1 day", timedelta(days=1)),
        ("2 days", timedelta(days=2)),
        ("1 week", timedelta(weeks=1)),
        ("1 month", timedelta(days=30)),
    ],
)
def test_barSizeSetting_to_timedelta_unadjusted(
    barSizeSetting: str, delta: timedelta
) -> None:
    assert barSizeSetting_to_timedelta(barSizeSetting, adjusted=False) == delta


@pytest.mark.parametrize(
    "durationStr,delta",
    [
        ("34 S", timedelta(seconds=34)),
        ("2 W", timedelta(days=14)),
        ("1 W", timedelta(days=7)),
        ("10 D", us_business_days_offset * 10),
        ("11 D", us_business_days_offset * 11),
        ("6 D", us_business_days_offset * 6),
    ],
)
def test_durationStr_to_offset(durationStr: str, delta: timedelta) -> None:
    assert durationStr_to_offset(durationStr) == delta


def test_offset_durationStr_with_holida():
    date = datetime(2026, 2, 19)
    durationStr = "10 D"
    assert offset_durationStr(durationStr, date) == datetime(2026, 2, 4)


def test_offset_durationStr_no_holiday():
    date = datetime(2026, 2, 13)
    durationStr = "10 D"
    # there was a holiday in between
    assert offset_durationStr(durationStr, date) == datetime(2026, 1, 30)


@pytest.mark.parametrize(
    "datapoints,barSizeSetting,delta",
    [
        (100, "30 secs", timedelta(minutes=50)),
        (10000, "30 secs", timedelta(days=3.623188406)),
        (10_000, "1 sec", timedelta(seconds=10_000)),
        # default 23 hours per day
        (10_000, "1 min", timedelta(days=10_000 / 60 / 23)),
        # default 23 hours per day
        (100, "1 hour", timedelta(days=100 / 23)),
        (10000, "1 hour", timedelta(days=10000 / 23)),
        (100, "1 day", timedelta(days=100)),
        (1000, "1 day", timedelta(days=1000)),
        (100, "1 week", 100 * timedelta(days=5)),
        (10, "1 month", 10 * timedelta(days=22)),
        (100, "1 month", 100 * timedelta(days=22)),
    ],
)
def test_datapoints_to_timedelta(
    datapoints: int, barSizeSetting: str, delta: timedelta
) -> None:
    result = datapoints_to_timedelta(datapoints, barSizeSetting)
    # rounding to full seconds as more precision is not needed
    assert timedelta(seconds=int(result.total_seconds())) == timedelta(
        seconds=int(delta.total_seconds())
    )


@pytest.mark.parametrize(
    "datapoints,barSizeSetting,durationStr",
    [
        (100, "30 secs", "3000 S"),
        (10000, "30 secs", "4 D"),
        (10000, "5 secs", "50000 S"),
        (3600 * 23, "1 sec", "1 D"),  # exactly one session
        (3600 * 23 + 1, "1 sec", "2 D"),  # one bar more than 1 session
        (100, "5 mins", "30000 S"),
        (100, "1 hour", "5 D"),
        (1000, "1 hour", "44 D"),
        (100, "1 day", "100 D"),
        (1, "1 day", "1 D"),
        (100, "2 day", "200 D"),
        (1000, "1 day", "3 Y"),
        (100, "1 week", "2 Y"),
        (10, "1 month", "220 D"),
    ],
)
def test_datapoints_to_durationStr(
    datapoints: int, barSizeSetting: str, durationStr: str
) -> None:
    assert (
        datapoints_to_durationStr(datapoints, barSizeSetting, timedelta(hours=23))
        == durationStr
    )


@pytest.mark.parametrize("days", [1, 2, 4, 5, 6, 7, 8, 14, 20, 21])
def test_durationStr_to_datapoints_D_unit_secs(days):
    barSizeSetting = "30 secs"
    durationStr = f"{days} D"
    session_lenght = timedelta(hours=23)
    # typical session: 23 hours * 60 minues
    # * 2 (barSizeSetting: 30 secs, 2 periods in one minute)
    # * durationStr: 14 days
    assert (
        durationStr_to_datapoints(durationStr, barSizeSetting, session_lenght)
        == 23 * 60 * 2 * days
    )


@pytest.mark.parametrize("days", [1, 2, 4, 5, 6, 7, 8, 14, 20, 21])
def test_durationStr_to_datapoints_D_unit_mins(days):
    barSizeSetting = "5 mins"
    durationStr = f"{days} D"
    session_lenght = timedelta(hours=23)
    # typical session: 23 hours * 60 minutes
    # * 12 (barSizeSetting: 5 mins, 12 periods in one hour)
    # * durationStr in days
    assert (
        durationStr_to_datapoints(durationStr, barSizeSetting, session_lenght)
        == 23 * 12 * days
    )


@pytest.mark.parametrize("days", [1, 2, 4, 5, 6, 7, 8, 14, 20, 21])
def test_durationStr_to_datapoints_D_unit_hour(days):
    barSizeSetting = "1 hour"
    durationStr = f"{days} D"
    session_lenght = timedelta(hours=23)
    # typical session: 23 hours * 60 minutes
    # * durationStr in days
    assert (
        durationStr_to_datapoints(durationStr, barSizeSetting, session_lenght)
        == 23 * days
    )


@pytest.mark.parametrize("days", [1, 2, 4, 5, 6, 7, 8, 14, 20, 21])
def test_durationStr_to_datapoints_D_unit_hour_session_length_irrelevant(days):
    barSizeSetting = "1 hour"
    durationStr = f"{days} D"
    session_lenght_hours = randint(1, 23)
    session_lenght = timedelta(hours=session_lenght_hours)
    # typical session: 23 hours * 60 minutes
    # * durationStr in days
    assert (
        durationStr_to_datapoints(durationStr, barSizeSetting, session_lenght)
        == session_lenght_hours * days
    )


@pytest.mark.parametrize(
    "durationStr,expected_days",
    [("2 W", 10), ("1 W", 5), ("1 M", 22), ("2 M", 44), ("1 Y", 252), ("2 Y", 504)],
)
def test_durationStr_to_datapoints_non_D_unit_secs(durationStr, expected_days):
    barSizeSetting = "30 secs"
    session_length = timedelta(hours=23)

    # typical session: 23 hours * 60 minutes
    # * 2 (barSizeSetting: 30 secs, 2 periods in one minute)
    # * durationStr: 10 day (business days in 2 weeks)
    assert (
        durationStr_to_datapoints(durationStr, barSizeSetting, session_length)
        == 23 * 60 * 2 * expected_days
    )


@pytest.mark.parametrize(
    "durationStr,expected_days",
    [("2 W", 10), ("1 W", 5), ("1 M", 22), ("2 M", 44), ("1 Y", 252), ("2 Y", 504)],
)
def test_durationStr_to_datapoints_non_D_unit_mins(durationStr, expected_days):
    barSizeSetting = "1 mins"
    session_length = timedelta(hours=23)
    # typical session: 23 hours * 60 minutes
    # * 2 (barSizeSetting: 30 secs, 2 periods in one minute)
    # * durationStr: 10 day (business days in 2 weeks)
    assert (
        durationStr_to_datapoints(durationStr, barSizeSetting, session_length)
        == 23 * 60 * expected_days
    )


@pytest.mark.parametrize(
    "durationStr,expected_days",
    [("2 W", 10), ("1 W", 5), ("1 M", 22), ("2 M", 44), ("1 Y", 252), ("2 Y", 504)],
)
def test_durationStr_to_datapoints_non_D_unit_hours(durationStr, expected_days):
    barSizeSetting = "1 hour"
    session_length = timedelta(hours=23)
    # typical session: 23 hours * 60 minutes
    # * 2 (barSizeSetting: 30 secs, 2 periods in one minute)
    # * durationStr: 10 day (business days in 2 weeks)
    assert (
        durationStr_to_datapoints(durationStr, barSizeSetting, session_length)
        == 23 * expected_days
    )


@pytest.mark.parametrize(
    "durationStr,expected_datapoints",
    [
        ("2 S", 1),
        ("10 S", 1),
        ("60 S", 1),
        ("65 S", 2),
        ("120 S", 2),
        ("121 S", 3),
        ("3600 S", 60),
    ],
)
def test_durationStr_to_datapoints_S_unit(durationStr, expected_datapoints):
    barSizeSetting = "1 mins"
    session_length = timedelta(hours=23)
    assert (
        durationStr_to_datapoints(durationStr, barSizeSetting, session_length)
        == expected_datapoints
    )


@pytest.mark.parametrize(
    "durationStr,expected_datapoints",
    [
        ("10 S", 1),
        ("121 S", 1),
        ("3600 S", 1),
        ("7200 S", 2),
        ("10000 S", 3),
        ("36000 S", 10),
        ("36001 S", 11),
    ],
)
def test_durationStr_to_datapoints_S_unit_hours(durationStr, expected_datapoints):
    barSizeSetting = "1 hours"
    session_length = timedelta(hours=23)
    assert (
        durationStr_to_datapoints(durationStr, barSizeSetting, session_length)
        == expected_datapoints
    )


def test_date_to_delta_passed_now():
    tz = ZoneInfo("UTC")
    date = datetime(2025, 1, 11, 10, 0, tzinfo=tz)
    mock_now = datetime(2025, 1, 11, 14, 30, tzinfo=tz)  # 4.5 hours later

    with patch("haymaker.durationStr_converters.datetime") as mock_dt:
        mock_dt.now.return_value = mock_now
        end_date_or_now = datetime(2025, 1, 11, 12, 30, tzinfo=tz)
        result = date_to_delta(
            date, "1 hour", end_date_or_now=end_date_or_now, margin=0
        )
        # three bars elapsed (2.5 hours since passed end_date_or_now
        # on 1 hour bars rounded UP), if it used mock_now, it would be
        # 5 hours
        assert result == 10800


def test_date_to_delta_passed_now_no_timezone():
    date = datetime(2025, 1, 11, 10, 0, tzinfo=None)
    mock_now = datetime(2025, 1, 11, 12, 30)  # 2.5 hours later

    with patch("haymaker.durationStr_converters.datetime") as mock_dt:
        mock_dt.now.return_value = mock_now
        end_date_or_now = datetime(2025, 1, 11, 12, 30)
        result = date_to_delta(
            date, "1 hour", end_date_or_now=end_date_or_now, margin=0
        )
        assert result == 10800


def test_date_to_delta():
    tz = ZoneInfo("UTC")
    date = datetime(2025, 1, 11, 10, 0, tzinfo=tz)
    mock_now = datetime(2025, 1, 11, 12, 30, tzinfo=tz)  # 2.5 hours later

    with patch("haymaker.durationStr_converters.datetime") as mock_dt:
        mock_dt.now.return_value = mock_now

        result = date_to_delta(date, "1 hour", margin=0)

        # three bars elapsed (2.5 hours on 1 hour bars rounded UP)
        assert result == 10800


def test_date_to_accepts_date_without_timezone():
    date = datetime(2025, 1, 11, 10, 0)
    result = date_to_delta(date, "1 hour", margin=0)
    # test will throw an error if cannot handle the date doesn't make
    # sense to test actual value as it will be different depending on
    # test time
    assert result


def test_date_to_delta_sub_hour():
    tz = ZoneInfo("UTC")
    date = datetime(2026, 1, 26, 10, 0, tzinfo=tz)
    mock_now = datetime(2026, 1, 26, 10, 10, tzinfo=tz)  # 10 minutes later

    with patch("haymaker.durationStr_converters.datetime") as mock_dt:
        mock_dt.now.return_value = mock_now

        result = date_to_delta(date, "1 min", margin=0)

        # three bars elapsed (2.5 hours on 1 hour bars rounded UP)
        assert result == 600


def test_date_to_delta_margin_applied():
    tz = ZoneInfo("UTC")
    date = datetime(2025, 1, 11, 10, 0, tzinfo=tz)
    mock_now = datetime(2025, 1, 11, 12, 30, tzinfo=tz)  # 2.5 hours later

    with patch("haymaker.durationStr_converters.datetime") as mock_dt:
        mock_dt.now.return_value = mock_now

        result = date_to_delta(date, "1 hour", margin=3)

        # three bars elapsed (2.5 hours on 1 hour bars rounded UP plus
        # 3 * 1 hour bars as margin)
        assert result == 10800 + 3600 * 3


def test_date_to_delta_margin_daily_bars():
    tz = ZoneInfo("UTC")
    date = datetime(2025, 1, 11, 10, 0, tzinfo=tz)
    mock_now = datetime(2025, 1, 11, 12, 30, tzinfo=tz)  # 2.5 hours later

    with patch("haymaker.durationStr_converters.datetime") as mock_dt:
        mock_dt.now.return_value = mock_now

        result = date_to_delta(date, "1 day", margin=0)

        # one daily bar elapsed (rounded up)
        assert result == 3600 * 24


def test_date_to_delta_margin_daily_bars_multiple():
    tz = ZoneInfo("UTC")
    date = datetime(2025, 1, 11, 10, 0, tzinfo=tz)
    mock_now = datetime(2025, 1, 13, 12, 30, tzinfo=tz)  # 2 days and 2.5 hours later

    with patch("haymaker.durationStr_converters.datetime") as mock_dt:
        mock_dt.now.return_value = mock_now

        result = date_to_delta(date, "1 day", margin=0)

        # 2+ daily bar elapsed (rounded up)
        assert result == 3600 * 24 * 3


def test_date_to_delta_margin_week_bars():
    tz = ZoneInfo("UTC")
    date = datetime(2025, 1, 11, 10, 0, tzinfo=tz)
    mock_now = datetime(2025, 1, 19, 12, 30, tzinfo=tz)  # 1 week+ later

    with patch("haymaker.durationStr_converters.datetime") as mock_dt:
        mock_dt.now.return_value = mock_now

        result = date_to_delta(date, "1 week", margin=0)

        # 2 weekly bars (rounded up)
        assert result == 3600 * 24 * 7 * 2


def test_delta_to_durationStr_more_than_one_day():
    # 1,000,000 secs = 1,000,000 / 3600  = 277.77 hours / 24 = 11.57 days
    # rounded up to 12 days
    assert delta_to_durationStr(1_000_000) == "12 D"


def test_delta_to_durationStr_less_than_one_day():
    assert delta_to_durationStr(100) == "100 S"


def test_date_to_delta_wrapper():
    tz = ZoneInfo("UTC")
    date = datetime(2025, 1, 11, 10, 0, tzinfo=tz)
    mock_now = datetime(2025, 1, 11, 12, 30, tzinfo=tz)  # 2.5 hours later

    with patch("haymaker.durationStr_converters.datetime") as mock_dt:
        mock_dt.now.return_value = mock_now

        result = date_to_delta_wrapper(date, "1 hour", margin=0)

        # three bars elapsed (2.5 hours on 1 hour bars rounded UP)
        assert result == "10800 S"


def test_date_to_delta_wrapper_now_passed_as_argument():
    tz = ZoneInfo("UTC")
    date = datetime(2025, 1, 11, 10, 0, tzinfo=tz)
    mock_now = datetime(2025, 1, 11, 14, 30, tzinfo=tz)  # 4.5 hours later
    end_date_or_now = datetime(2025, 1, 11, 12, 30, tzinfo=tz)  # 2.5 hours later

    with patch("haymaker.durationStr_converters.datetime") as mock_dt:
        mock_dt.now.return_value = mock_now

        result = date_to_delta_wrapper(
            date, "1 hour", margin=0, end_date_or_now=end_date_or_now
        )

        # three bars elapsed (2.5 hours on 1 hour bars rounded UP)
        assert result == "10800 S"
