from datetime import date, datetime, timedelta

from .time_policy import normalize_point

# source:
# https://ibkrcampus.com/campus/ibkr-api-page/twsapi-doc/#historical-bars
BAR_SIZE_SECONDS = {
    "1 secs": 1,
    "5 secs": 5,
    "10 secs": 10,
    "15 secs": 15,
    "30 secs": 30,
    "1 min": 60,
    "2 mins": 120,
    "3 mins": 180,
    "5 mins": 300,
    "10 mins": 600,
    "15 mins": 900,
    "20 mins": 1200,
    "30 mins": 1800,
    "1 hour": 3600,
    "2 hours": 7200,
    "3 hours": 10800,
    "4 hours": 14400,
    "8 hours": 28800,
    "1 day": 3600 * 23,
    "1 week": 3600 * 23 * 5,
    "1 month": 3600 * 23 * 22,
}

# IBKR's current max-duration table allows only seconds requests for 1-second
# bars. Other supported dataloader bar sizes share the documented 68-year cap.
MAX_DURATION_BY_BAR_SIZE = {
    **{bar_size: "68 Y" for bar_size in BAR_SIZE_SECONDS},
    "1 secs": "2000 S",
}

VALID_BAR_SIZES = list(BAR_SIZE_SECONDS)


def _validate_bar_size(barSize: str) -> None:
    if barSize not in BAR_SIZE_SECONDS:
        raise ValueError(f"Invalid IB bar size: {barSize}")


def duration_in_secs(barSize: str) -> int:
    """Given duration string return duration in seconds int"""

    _validate_bar_size(barSize)
    return BAR_SIZE_SECONDS[barSize]


def duration_str(duration_in_secs: int) -> str:
    """
    Given duration in seconds return duration str acceptable by IB.

    """

    days = int(duration_in_secs / 60 / 60 / 23)

    if days:
        years = int(days / 252)
        if years:
            return f"{years} Y"
        months = int(days / 22)
        if months:
            return f"{months} M"
        return f"{days} D"

    return f"{duration_in_secs} S"


def timedelta_normalizer(seconds: float) -> int:
    """
    I'm assuming week is: 60secs * 60min * 23hour * 5days
    timedelta assumes: 60secs * 60min * 24hours * 7days
    So adjustments are needed...

    Holidays are ignored (irrelevant error given the objective).
    """
    # period shorter than one day
    if seconds < 23 * 60 * 60:
        return int(seconds)
    elif seconds <= 5 * 24 * 60 * 60:
        return int(seconds * 23 / 24)
    elif seconds <= 7 * 24 * 60 * 60:
        return timedelta_normalizer(5 * 24 * 60 * 60)
    else:
        weeks = int(seconds / (7 * 24 * 60 * 60))
        adjusted_week = 5 * 23 * 60 * 60
        adjusted = weeks * adjusted_week
        rest = seconds - weeks * 7 * 24 * 60 * 60
        return adjusted + timedelta_normalizer(rest)


def timedelta_to_duration_in_secs(
    duration: timedelta, bar_size_in_secs: int, max_bars: int
) -> int:
    max_bars_duration_in_secs = max_bars * bar_size_in_secs
    return min(
        timedelta_normalizer(duration.total_seconds()), max_bars_duration_in_secs
    )


def timedelta_and_barSize_to_duration_str(
    duration: timedelta, barSize: str, max_bars: int = 100_000
) -> str:
    """
    Given bar size str and duration, return optimal duration str.

    Parameters:
    -----------

    duration - timedelta of the period for which we're trying to
               generate duration str barSize - IB's barSize param

    barSize - barSize str acceptable by IB's reqHistoricalData

    max_bars - maximum number of bars to be requested at once

    """
    _validate_bar_size(barSize)
    bar_size_in_secs = duration_in_secs(barSize)
    duration_seconds = max(
        timedelta_to_duration_in_secs(duration, bar_size_in_secs, max_bars), 30
    )

    return _cap_duration_str(
        duration_str(duration_seconds), MAX_DURATION_BY_BAR_SIZE[barSize]
    )


def _cap_duration_str(durationStr: str, maxDurationStr: str) -> str:
    """Cap duration strings to IBKR's documented max request span."""

    number, unit = durationStr.split(" ")
    max_number, max_unit = maxDurationStr.split(" ")
    if max_unit == "S":
        seconds = min(
            int(duration_to_timedelta(durationStr).total_seconds()), int(max_number)
        )
        return duration_str(seconds)
    if unit == max_unit == "Y" and int(number) > int(max_number):
        return maxDurationStr
    return durationStr


def duration_to_timedelta(duration: str) -> timedelta:
    """Convert duration string of reqHistoricalData into datetime.timedelta"""
    str_number, time = duration.split(" ")
    number = int(str_number)
    if time == "S":
        return timedelta(seconds=number)
    if time == "D":
        return timedelta(days=number)
    if time == "W":
        return timedelta(weeks=number)
    if time == "M":
        return timedelta(days=31)
    if time == "Y":
        return timedelta(days=365)
    raise ValueError(f"Unknown duration string: {duration}")


def datetime_normalizer(dt: datetime, barsize: str) -> datetime | date:
    return normalize_point(dt, barsize)


def strjoin(*args: str):
    return "".join(args)
