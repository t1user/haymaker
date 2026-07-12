"""Convert dataloader bar sizes and time spans to IB duration strings."""

from datetime import timedelta

# IBKR recommends only a few thousand bars per request. The dataloader uses a
# deliberately moderate transitional target while preserving request throughput.
DEFAULT_TARGET_BARS_PER_REQUEST = 25_000

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

# IBKR max-duration rule: one-second bars accept at most 2000 seconds. Other
# supported dataloader bar sizes share the current table's 68-year outer cap;
# the target above normally produces a much shorter duration first.
MAX_DURATION_BY_BAR_SIZE = {
    **{bar_size: "68 Y" for bar_size in BAR_SIZE_SECONDS},
    "1 secs": "2000 S",
}

VALID_BAR_SIZES = list(BAR_SIZE_SECONDS)


def _validate_bar_size(barSize: str) -> None:
    if barSize not in BAR_SIZE_SECONDS:
        raise ValueError(f"Invalid IB bar size: {barSize}")


def duration_in_secs(barSize: str) -> int:
    """Return the nominal number of seconds represented by an IB bar size.

    Args:
        barSize: Canonical IB bar-size string supported by the dataloader.

    Returns:
        Nominal bar duration in seconds. Daily and longer bars use the
        dataloader's session-length estimates.

    Raises:
        ValueError: If ``barSize`` is unsupported.
    """

    _validate_bar_size(barSize)
    return BAR_SIZE_SECONDS[barSize]


def duration_str(duration_in_secs: int) -> str:
    """Return an IB-compatible duration string for a nominal time span.

    Args:
        duration_in_secs: Duration expressed in dataloader-normalized seconds.

    Returns:
        Duration using IB's ``S``, ``D``, ``M``, or ``Y`` unit syntax.
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
    """Convert calendar seconds to the dataloader's nominal session seconds.

    The estimate assumes a 23-hour session and five-session week. It is used to
    shape requests, not to predict the exact number of bars IB will return.

    Args:
        seconds: Calendar duration in seconds.

    Returns:
        Estimated tradable-session duration in whole seconds.
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
    duration: timedelta,
    bar_size_in_secs: int,
    target_bars_per_request: int = DEFAULT_TARGET_BARS_PER_REQUEST,
) -> int:
    """Clamp a requested span to the internal target bars per request.

    Args:
        duration: Calendar range that remains to be downloaded.
        bar_size_in_secs: Nominal duration of one bar in seconds.
        target_bars_per_request: Internal target number of bars in one request.

    Returns:
        Estimated request duration in seconds.
    """

    target_duration_in_secs = target_bars_per_request * bar_size_in_secs
    return min(timedelta_normalizer(duration.total_seconds()), target_duration_in_secs)


def timedelta_and_barSize_to_duration_str(
    duration: timedelta,
    barSize: str,
    target_bars_per_request: int = DEFAULT_TARGET_BARS_PER_REQUEST,
) -> str:
    """Return a bounded IB duration for the remaining download range.

    Args:
        duration: Calendar range that remains to be downloaded.
        barSize: Canonical IB bar-size string.
        target_bars_per_request: Internal target number of bars in one request.

    Returns:
        Duration string accepted by ``reqHistoricalData`` and capped by the
        documented maximum span for ``barSize``.

    Raises:
        ValueError: If ``barSize`` is unsupported.

    """
    _validate_bar_size(barSize)
    bar_size_in_secs = duration_in_secs(barSize)
    duration_seconds = max(
        timedelta_to_duration_in_secs(
            duration, bar_size_in_secs, target_bars_per_request
        ),
        30,
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
    """Convert an IB duration string to an approximate ``timedelta``.

    Args:
        duration: Number and unit separated by one space, for example ``2000 S``
            or ``1 Y``.

    Returns:
        Approximate calendar duration. Months and years use fixed estimates.

    Raises:
        ValueError: If the duration unit is unknown.
    """
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
