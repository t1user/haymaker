"""
durationStr allowed values: 'S', 'D', 'W', 'M', 'Y';
IB works like this: 'S' and 'D' will will return
the requested number of points ('7 D' = 7 business days,
which typically will be 1 business week plus 2 days;
'W', 'M', 'Y' will return all data available in given calendar period
('1W' = 7 calendar days, which typically will correspond to 5 business
days)

barSizeSetting allowed values:
'1 day', '1 week', '1 month';
secs: 1, 5, 10, 15, 30
mins: 1, 2, 3, 5, 10, 15, 20, 30
hours: 1, 2, 3, 4, 8
"""

from datetime import datetime, timedelta

import pandas as pd

# from pandas.tseries.offsets import BusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import BaseOffset, CustomBusinessDay

# reusable business day object with holidays
us_business_days_offset = CustomBusinessDay(calendar=USFederalHolidayCalendar())

barSizeSetting_deltas = {
    "sec": timedelta(seconds=1),
    "min": timedelta(minutes=1),
    "hour": timedelta(hours=1),
    "day": timedelta(days=1),
    # 1 calendar week
    "week": timedelta(days=7),
    # 1 calendar month
    "month": timedelta(days=30),
}


barSizeSetting_deltas_adjusted = {
    # ignoring non-business days that barSizeSetting would include
    "sec": timedelta(seconds=1),
    "min": timedelta(minutes=1),
    "hour": timedelta(hours=1),
    "day": timedelta(days=1),
    # 1 week bar will cover 5 business days corresponding to 7 calendar days
    "week": timedelta(days=5),
    # 1 month bar will cover ca. 22 business days corresponding to ca. 30 calendar days
    "month": timedelta(days=22),
}


def barSizeSetting_to_timedelta(barSizeSetting, adjusted=True) -> timedelta:
    number, time = barSizeSetting.split(" ")
    time = time[:-1] if time.endswith("s") else time
    deltas = barSizeSetting_deltas_adjusted if adjusted else barSizeSetting_deltas
    return deltas[time] * int(number)


durationStr_deltas = {
    "S": timedelta(seconds=1),
    # durationStr of "7 D" will get 7 business days, which will be
    # 1 calendar week plus 2 days (ignoring potential holidays)
    "D": timedelta(days=1),  # this needs adjustment
    "W": timedelta(weeks=1),
    "M": timedelta(days=30),
    "Y": timedelta(days=365),
}


def durationStr_to_offset(durationStr) -> timedelta | BaseOffset:
    number_str, unit = durationStr.split(" ")
    number = int(number_str)
    # 'D' durationStr needs adjustment if converting to calendar timedelta
    # as it includes only business days
    return (
        durationStr_deltas[unit] * number
        if unit != "D"
        else us_business_days_offset * number
    )


def offset_durationStr(durationStr, date: datetime) -> datetime:
    return (pd.Timestamp(date) - durationStr_to_offset(durationStr)).to_pydatetime()


def datapoints_to_timedelta(
    datapoints: int,
    barSizeSetting: str,
    session_length: timedelta = timedelta(hours=23),
) -> timedelta:
    barSize_timedelta = barSizeSetting_to_timedelta(barSizeSetting)
    bars_per_session = session_length / barSize_timedelta
    sessions_required = (
        datapoints / bars_per_session
        if barSize_timedelta < timedelta(days=1)
        else barSize_timedelta * datapoints / timedelta(days=1)
    )
    if sessions_required < 1:
        return barSize_timedelta * datapoints
    else:
        return timedelta(days=1) * sessions_required


def datapoints_to_durationStr(
    datapoints: int, barSizeSetting: str, session_length: timedelta
) -> str:
    """
    durationStr required to obtain at least the given number of
    datapoints.
    """

    barSizeSetting_timedelta = barSizeSetting_to_timedelta(
        barSizeSetting, adjusted=True
    )
    bars_per_session = session_length / barSizeSetting_timedelta

    if bars_per_session < 1:
        number_of_sessions: float = barSizeSetting_timedelta.days * datapoints
    else:
        number_of_sessions = datapoints / bars_per_session

    if number_of_sessions < 1:
        return f"{int((number_of_sessions * session_length).total_seconds())} S"
    elif number_of_sessions > 365:
        return f"{int(-(-number_of_sessions // 365))} Y"
    else:
        return f"{int(-(-number_of_sessions // 1))} D"


def durationStr_to_datapoints(
    durationStr: str, barSizeSetting: str, session_length: timedelta
) -> int:
    """
    Estimated number of datapoints in a given durationStr
    parameter passed to IB.
    """

    unit_dict = {
        "D": 1,
        "W": 5,
        "M": 22,
        "Y": 252,
    }

    durationStr_value, duration_unit = durationStr.split(" ")
    duration_value = int(durationStr_value)

    barSizeSetting_timedelta = barSizeSetting_to_timedelta(barSizeSetting)

    if duration_unit == "S":
        bars = timedelta(seconds=duration_value) / barSizeSetting_timedelta
    else:
        bars_per_session = session_length / barSizeSetting_timedelta
        number_of_sessions = unit_dict[duration_unit] * duration_value
        bars = bars_per_session * number_of_sessions

    # round UP and then convert to int
    return int(-(-bars // 1))


def date_to_delta(
    start_date: datetime,
    barSizeSetting: str,
    *,
    end_date_or_now: datetime | None = None,
    margin: int = 1,
) -> int:
    """
    Return number of seconds since date.  Makes sure that at least the
    minimum required number of bars is requested.  Used to determine
    durationStr required to pull data since the date.

    Args:
    =====

    date: since when delta is to be calculated

    barSizeSetting: bar size definition

    margin: number of extra bars

    """
    now = end_date_or_now or datetime.now(start_date.tzinfo)
    secs = (now - start_date).total_seconds()
    bar_size = barSizeSetting_to_timedelta(
        barSizeSetting, adjusted=False
    ).total_seconds()
    # at least the covered period, number of bars rounded up
    bars = int((secs + bar_size - 1) // bar_size)
    # with some margin for delays, duplicated bars will be filtered out anyway
    return int(max((bars + margin) * bar_size, bar_size))


def delta_to_durationStr(seconds: int) -> str:
    """
    Assume that the durationStr is shorter than 1 year.  Ignore that
    session if offen shorter than 24 hours.  These are irrelevant
    given the purpose of the function is to return a durationStr for
    requesting data since last restart.
    """
    print(f"{seconds=}")
    if seconds < 86400:
        return f"{seconds} S"
    else:
        # round up: ceil(a/b) = (a + b - 1) // b
        return f"{(seconds + 86400 - 1) // 86400} D"


def date_to_delta_wrapper(
    start_date: datetime,
    barSizeSetting: str,
    *,
    end_date_or_now: datetime | None = None,
    margin: int = 1,
) -> str:
    return delta_to_durationStr(
        date_to_delta(
            start_date, barSizeSetting, end_date_or_now=end_date_or_now, margin=margin
        )
    )
