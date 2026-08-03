"""Date and time policy for IB historical dataloader requests."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd

DATE_BAR_SIZES = {"1 day", "1 week", "1 month"}


def is_date_bar(bar_size: str) -> bool:
    """Return whether IB returns date-only values for the configured bar size."""

    return bar_size in DATE_BAR_SIZES


def normalize_point(value: date | datetime, bar_size: str) -> date | datetime:
    """Normalize a scheduling point for the configured bar size.

    Args:
        value: Date or datetime from IB, datastore, or current runtime state.
        bar_size: IB bar size setting for the dataloader run.

    Returns:
        ``date`` for daily-like bars, otherwise UTC-aware ``datetime``.

    Raises:
        ValueError: If an intraday timestamp is naive.
        TypeError: If an intraday point is date-only.
    """

    if is_date_bar(bar_size):
        if isinstance(value, datetime):
            return value.date()
        return value

    if not isinstance(value, datetime):
        raise TypeError(f"Intraday bars require datetime points, got {value!r}.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"Intraday bars require timezone-aware datetimes: {value!r}.")
    return value.astimezone(timezone.utc)


def normalize_optional_point(
    value: date | datetime | None, bar_size: str
) -> date | datetime | None:
    """Normalize an optional scheduling point for the configured bar size."""

    if value is None:
        return None
    return normalize_point(value, bar_size)


def normalize_index(index: pd.Index, bar_size: str) -> pd.Index:
    """Normalize a dataframe index used for dataloader scheduling.

    Args:
        index: Dataframe index read from the historical datastore.
        bar_size: IB bar size setting for the dataloader run.

    Returns:
        Index containing only canonical date or datetime points.
    """

    return pd.Index([normalize_point(value, bar_size) for value in index])
