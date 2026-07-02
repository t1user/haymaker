"""Signal and blip transformer helpers for research workflows."""

from typing import Literal

import numpy as np
import pandas as pd

__all__ = [
    "combine_signals",
    "crosser",
    "extreme_reversal_blip",
    "inout_range",
    "range_blip",
    "signal_generator",
]


def extreme_reversal_blip(ind: pd.Series, threshold: float) -> pd.Series:
    """Return reversal blips when an oscillator leaves an extreme zone.

    Returns a one-bar ``-1`` blip when the series crosses down through
    ``threshold`` from above, a one-bar ``1`` blip when it crosses up through
    ``-threshold`` from below, and ``0`` otherwise. This is useful for
    oscillators where returning from an extreme is interpreted as a contrarian
    reversal event.
    """
    previous = ind.shift()
    return pd.Series(
        np.select(
            [
                (previous > threshold) & (ind <= threshold),
                (previous < -threshold) & (ind >= -threshold),
            ],
            [-1, 1],
            default=0,
        ),
        index=ind.index,
    )


def signal_generator(
    series: pd.Series,
    threshold: float = 0,
    handle_na: Literal["ignore", "drop", "raise"] = "ignore",
) -> pd.Series:
    """Return ``1``, ``0``, or ``-1`` based on a threshold comparison.

    Args:
        series: Input indicator.
        threshold: Comparison level.
        handle_na: Missing-value policy. ``"ignore"`` treats missing values as
            flat ``0``. ``"drop"`` drops missing input rows before generating
            signals. ``"raise"`` raises if any missing values are present.
    """
    if handle_na == "raise":
        if series.isna().any():
            raise ValueError("series contains NaN values")
    elif handle_na == "drop":
        series = series.dropna()
    elif handle_na != "ignore":
        raise ValueError("handle_na must be one of: 'ignore', 'drop', 'raise'")

    return (
        (series > threshold) * 1 - (series < threshold) * 1 + (series == threshold) * 0
    )


def combine_signals(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """Filter one signed signal by another signed signal.

    Args:
        series1: Primary signal.
        series2: Filter signal.

    Returns:
        ``series1`` where both inputs have the same sign, otherwise ``0``.
    """
    return ((np.sign(series1) == np.sign(series2)) * series1).astype(int, copy=False)


def crosser(ind: pd.Series, threshold: float = 0) -> pd.Series:
    """Return threshold-crossing blips for an indicator series.

    A ``1`` is emitted when the indicator crosses from below ``threshold`` to
    above it, and ``-1`` when it crosses from above to below. Values exactly at
    ``threshold`` are neutral: they emit no blip, but moving away from the
    threshold on the next row emits the new side. Rows with missing indicator
    values do not emit blips and do not create a crossing on the following row.
    """
    centered = ind - threshold
    side = pd.Series(np.sign(centered), index=ind.index).mask(ind.isna())
    crossed = side.notna() & side.shift().notna() & (side.shift() * side).le(0)
    return side.where(crossed, 0).astype(int)


def inout_range(
    s: pd.Series,
    threshold: float | pd.Series = 0,
    inout: Literal["inside", "outside"] = "inside",
) -> pd.Series:
    """Return whether values are inside or outside a symmetric threshold range.

    Args:
        s: Input series.
        threshold: Positive or negative threshold. The absolute value is used.
        inout: ``"inside"`` for values inside ``(-threshold, threshold)`` or
            ``"outside"`` for values outside that range.

    Returns:
        Boolean series aligned to ``s``.
    """

    if isinstance(threshold, (int, float)) and threshold == 0:
        raise ValueError("threshold cannot be zero, use: crosser()")
    threshold = abs(threshold)  # type: ignore
    excess = s.abs() - threshold
    if inout == "outside":
        result = excess > 0
    elif inout == "inside":
        result = excess < 0
    else:
        raise ValueError("'inout' parameter must be either 'inside' or 'outside'")
    result.name = inout
    return result


def range_blip(
    indicator: pd.Series,
    threshold: float | pd.Series = 0,
    inout: Literal["inside", "outside"] = "inside",
) -> pd.Series:
    """Return a signed blip when an indicator enters the selected range.

    The range is selected by :func:`inout_range`. With ``inout="inside"``, a
    blip is emitted when the indicator moves from outside to inside
    ``(-threshold, threshold)``. With ``inout="outside"``, a blip is emitted
    when the indicator leaves that range. The blip is signed the same as the
    indicator value on the entry bar.
    """

    indicator = indicator.dropna()

    selected_range = inout_range(indicator, threshold, inout)
    entry = selected_range.astype(int).diff().eq(1).astype(int)
    return (entry * np.sign(indicator)).astype(int)
