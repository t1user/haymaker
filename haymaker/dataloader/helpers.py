from datetime import date, datetime, timedelta

# source:
# https://www.interactivebrokers.com/campus/ibkr-api-page/twsapi-doc/#hist-bar-size
valid_bar_sizes = {
    "secs": (1, 5, 10, 15, 30),
    "mins": (1, 2, 3, 5, 10, 15, 20, 30),
    "hours": (1, 2, 3, 4, 8),
    "day": (1,),
    "weeks": (1,),
    "months": (1,),
}

VALID_BAR_SIZES = [
    f"{size} {unit}" for unit, sizes in valid_bar_sizes.items() for size in sizes
]


def duration_in_secs(barSize: str) -> int:
    """Given duration string return duration in seconds int"""

    number, time = barSize.split(" ")
    time = time[:-1] if time.endswith("s") else time
    multiplier = {
        "sec": 1,
        "min": 60,
        "hour": 3600,
        "day": 3600 * 23,
        "week": 3600 * 23 * 5,
    }
    return int(number) * multiplier[time]


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
    bar_size_in_secs = duration_in_secs(barSize)

    if bar_size_in_secs == 1:
        return duration_str(2000)
    else:
        return duration_str(
            max(timedelta_to_duration_in_secs(duration, bar_size_in_secs, max_bars), 30)
        )


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


DAILY_STRINGS = ["1 day", "1 week", "1 month"]


def datetime_normalizer(dt: datetime, barsize: str) -> datetime | date:
    if barsize in DAILY_STRINGS:
        return dt.date()
    else:
        return dt
