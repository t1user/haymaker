from datetime import date, datetime, timedelta
from typing import Union

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


def duration_in_secs(barSize: str):
    """Given duration string return duration in seconds int"""
    number, time = barSize.split(" ")
    time = time[:-1] if time.endswith("s") else time
    multiplier = {
        "sec": 1,
        "min": 60,
        "hour": 3600,
        "day": 3600 * 24,
        "week": 3600 * 24 * 7,
    }
    return int(number) * multiplier[time]


def duration_str(
    duration_in_secs: float, aggression: float, from_bar: bool = True
) -> str:
    """
    Given duration in seconds return acceptable duration str.

    :from_bar:
    if True it's assumed that the duration_in_secs number comes from barSize
    and appropriate multiplier is used to get to optimal duration. Otherwise
    duration_in_secs is converted into duration_str directly without
    any multiplication.
    """
    if from_bar:
        multiplier = 2000 if duration_in_secs == 1 else 15000 * aggression
    else:
        multiplier = 1
    duration = int(duration_in_secs * multiplier)
    days = int(duration / 60 / 60 / 23)
    if days:
        years = int(days / 250)
        if years:
            return f"{years} Y"
        months = int(days / 20)
        if months:
            return f"{months} M"
        return f"{days} D"
    return f"{duration} S"


def barSize_to_duration(s: str, aggression) -> str:
    """
    Given bar size str return optimal duration str,

    :aggression: how many data points will be pulled at a time,
                 should be between 0.5 and 3,
                 larger numbers might result in more throttling,
                 requires research what's optimal number for fastest
                 downloads
    """
    return duration_str(duration_in_secs(s), aggression)


def duration_to_timedelta(duration):
    """Convert duration string of reqHistoricalData into datetime.timedelta"""
    number, time = duration.split(" ")
    number = int(number)
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


def datetime_normalizer(dt: datetime, barsize: str) -> Union[datetime, date]:
    if barsize in DAILY_STRINGS:
        return dt.date()
    else:
        return dt
