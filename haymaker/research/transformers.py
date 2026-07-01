"""Signal and blip transformer helpers for research workflows."""

from typing import Literal

import numpy as np
import pandas as pd

__all__ = [
    "breakout",
    "breakout_blip",
    "combine_signals",
    "extreme_reversal_blip",
    "inout_range",
    "min_max_blip",
    "range_blip",
    "signal_generator",
    "zero_crosser",
]


def min_max_blip(
    price: pd.Series, period: int, buff: float | pd.Series = 0
) -> pd.Series:
    """Return breakout blips from the previous rolling high or low.

    Args:
        price: Price series.
        period: Lookback window for the previous rolling high and low.
        buff: Price-unit buffer added to the upper breakout threshold and
            subtracted from the lower breakout threshold. A positive value makes
            breakouts harder to trigger. If passed as a Series, it must have the
            same index as ``price``.

    Returns:
        Series with ``1`` when price breaks above the previous rolling high plus
        ``buff``, ``-1`` when price breaks below the previous rolling low minus
        ``buff``, and ``0`` otherwise. Blips are recorded on the bar where they
        are detected.
    """
    if isinstance(buff, pd.Series) and not buff.index.equals(price.index):
        raise ValueError("buff index must match price index")

    prior_max = price.rolling(period, closed="left").max()
    prior_min = price.rolling(period, closed="left").min()
    df = pd.DataFrame(
        {
            "max": (price > prior_max + buff) * 1,
            "min": (price < prior_min - buff) * 1,
        }
    )
    df["blip"] = df["max"] - df["min"]
    return df["blip"].rename(None)


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


def breakout(
    price: pd.Series,
    lookback: int,
    stop_frac: float = 0.5,
    always_on: bool = False,
) -> pd.Series:
    """Return stateful breakout signal from rolling high/low blips.

    Args:
        price: Price series, typically close prices.
        lookback: Rolling high/low lookback.
        stop_frac: Fraction of ``lookback`` used for opposite-side close blips.
            Must be in ``(0, 1]``. If ``stop_frac == 1``, opposite breakout
            blips are interpreted directly by ``always_on``.
        always_on: Relevant when ``stop_frac == 1``. If ``True``, an opposite
            breakout blip reverses the signal immediately. If ``False``, an
            opposite breakout blip closes the existing signal to zero.

    Returns:
        Series where ``1`` means long signal, ``-1`` short signal, and ``0`` no
        desired market exposure at that bar. The signal is recorded on the bar
        where the breakout is detected; for next-bar execution, shift the output
        before deriving executable positions.
    """

    if not isinstance(price, pd.Series):
        raise TypeError("price must be a pandas series")
    if not (stop_frac > 0 and stop_frac <= 1):
        raise ValueError("stop_frac must be from (0, 1>")

    df = pd.DataFrame({"price": price})
    df["in"] = min_max_blip(df["price"], lookback)
    if stop_frac == 1:
        from .numba_tools import _blip_to_signal_converter

        df["break"] = _blip_to_signal_converter(
            df["in"].to_numpy(), always_on=always_on
        )
    else:
        from .numba_tools import _in_out_blip_unifier

        df["out"] = min_max_blip(df["price"], int(lookback * stop_frac))
        df["break"] = _in_out_blip_unifier(df[["in", "out"]].to_numpy())
    return df["break"]


def breakout_blip(
    price: pd.Series,
    lookback: int,
    stop_frac: float = 0.5,
) -> pd.Series:
    """Return breakout event blips instead of a stateful breakout signal.

    Blips are raw generated events recorded on the bar where the breakout
    information becomes known. Do not pre-shift them before passing them to
    blip-aware consumers such as ``stop_loss`` or ``no_stop``. For direct
    next-bar execution, shift the returned blips before deriving positions.

    Args:
        price: Price series, typically close prices.
        lookback: Rolling high/low lookback.
        stop_frac: Fraction of ``lookback`` used for opposite-side close blips.

    Returns:
        Series with ``1`` and ``-1`` breakout blips, and ``0`` otherwise.
    """

    if not isinstance(price, pd.Series):
        raise TypeError("price must be a pandas series")
    if not (stop_frac > 0 and stop_frac <= 1):
        raise ValueError("stop_frac must be from (0, 1>")

    df = pd.DataFrame({"price": price})
    df["in"] = min_max_blip(df["price"], lookback)
    df["out"] = min_max_blip(df["price"], int(lookback * stop_frac))
    df["break"] = (df["in"] + df["out"]).clip(-1, 1)
    return df["break"]


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


def zero_crosser(indicator: pd.Series) -> pd.Series:
    """Return signed blips when an indicator crosses zero.

    The blip sign matches the indicator sign after the crossing. If an
    indicator value is exactly zero, the next nonzero value is treated as a
    crossing from zero.
    """
    indicator = indicator.fillna(0)
    return (((indicator.shift() * indicator) <= 0) * np.sign(indicator)).astype(int)


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
        raise ValueError("theshold cannot be zero, use: <zero_crosser>")
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
