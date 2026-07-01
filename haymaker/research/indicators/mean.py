"""Moving-average and weighted-mean indicators."""

from typing import Literal

import numpy as np
import pandas as pd

__all__ = [
    "mmean",
    "rolling_weighted_mean",
    "rolling_weighted_std",
    "weighted_zscore",
]


def _validate_periods(periods: int) -> None:
    if periods < 1:
        raise ValueError("periods must be at least 1")


def _validate_weighted_inputs(
    price: pd.Series, weights: pd.Series, periods: int
) -> None:
    _validate_periods(periods)
    if not price.index.equals(weights.index):
        raise ValueError("price and weights indexes must match")
    if weights.lt(0).any():
        raise ValueError("weights must be non-negative")


def _wilder_average(series: pd.Series, lookback: int) -> pd.Series:
    """Return Wilder's average-off smoothing for a positive series."""
    _validate_periods(lookback)

    rolling = series.rolling(lookback, min_periods=lookback).mean()
    out = pd.Series(np.nan, index=series.index, dtype=float)
    first_valid = rolling.first_valid_index()
    if first_valid is None:
        return out

    first_position = series.index.get_loc(first_valid)
    if not isinstance(first_position, int):
        raise ValueError("series index must not contain duplicate labels")

    out.iloc[first_position] = rolling.iloc[first_position]
    for position in range(first_position + 1, len(series)):
        value = series.iloc[position]
        if pd.isna(value):
            out.iloc[position] = out.iloc[position - 1]
        else:
            out.iloc[position] = (
                out.iloc[position - 1] + (value - out.iloc[position - 1]) / lookback
            )
    return out


def mmean(
    series: pd.Series,
    periods: int,
    smooth_type: Literal["simple", "exponential", "wilder"] = "exponential",
    **kwargs,
) -> pd.Series:
    """Return a moving average using the selected smoothing method.

    Args:
        series: Input series.
        periods: Moving-average lookback.
        smooth_type: Smoothing method:

            ``"simple"``
                Simple moving average,
                ``mean(series[t - periods + 1], ..., series[t])``.

            ``"exponential"``
                Pandas span-based exponential moving average,
                ``series.ewm(span=periods).mean()``. This uses
                ``alpha = 2 / (periods + 1)``.

            ``"wilder"``
                Wilder's average-off smoothing. The first value is seeded with
                the simple average of the first ``periods`` values, then updated
                as ``previous + (current - previous) / periods``. This is
                equivalent to an EMA with ``alpha = 1 / periods`` after the
                initial seed.
        **kwargs: Passed to pandas for ``"simple"`` and ``"exponential"``.
    """
    _validate_periods(periods)
    if smooth_type == "wilder":
        out = _wilder_average(series, periods)
    elif smooth_type == "exponential":
        out = series.ewm(span=periods, **kwargs).mean()
    elif smooth_type == "simple":
        out = series.rolling(periods, **kwargs).mean()
    else:
        raise ValueError(
            "smooth_type must be one of: 'simple', 'exponential', 'wilder'"
        )
    return out


def rolling_weighted_mean(
    price: pd.Series, weights: pd.Series, periods: int
) -> pd.Series:
    """Return rolling weighted mean.

    Args:
        price: Input values.
        weights: Non-negative weights aligned to ``price``.
        periods: Rolling window length.

    Returns:
        Weighted mean for each full rolling window. A window whose weight sum is
        zero returns ``NaN``.
    """
    _validate_weighted_inputs(price, weights, periods)
    price_vol = price * weights
    return price_vol.rolling(periods).sum() / weights.rolling(periods).sum()


def rolling_weighted_std(
    price: pd.Series,
    weights: pd.Series,
    periods: int,
    weighted_mean: pd.Series | None = None,
) -> pd.Series:
    """Return rolling population-style weighted standard deviation.

    Args:
        price: Input values.
        weights: Non-negative weights aligned to ``price``.
        periods: Rolling window length.
        weighted_mean: Optional precomputed rolling weighted mean.

    Returns:
        Weighted standard deviation. The denominator is the rolling sum of
        weights, not a sample-variance correction.
    """
    _validate_weighted_inputs(price, weights, periods)
    if weighted_mean is not None and not weighted_mean.index.equals(price.index):
        raise ValueError("weighted_mean index must match price index")
    if weighted_mean is None:
        weighted_mean = rolling_weighted_mean(price, weights, periods)

    weighted_second_moment = rolling_weighted_mean(price**2, weights, periods)
    weighted_var = (weighted_second_moment - weighted_mean**2).clip(lower=0)
    return weighted_var.pow(0.5)


def weighted_zscore(df: pd.DataFrame, lookback: int) -> pd.Series:
    """Return volume-weighted z-score of the ``close`` column.

    Args:
        df: Dataframe with ``close`` and non-negative ``volume`` columns.
        lookback: Rolling window length.

    Returns:
        Series of z-scores where enough data and non-zero weighted volatility
        are available. Rows with unavailable z-scores are dropped.

    Raises:
        ValueError: If required columns are missing or ``lookback`` is invalid.
    """
    missing = {"close", "volume"} - set(df.columns)
    if missing:
        raise ValueError(f"weighted_zscore() missing required columns: {missing}.")
    _validate_weighted_inputs(df["close"], df["volume"], lookback)

    wmean = rolling_weighted_mean(df["close"], df["volume"], lookback)
    wstd = rolling_weighted_std(df["close"], df["volume"], lookback, wmean)
    return ((df["close"] - wmean) / wstd).dropna()
