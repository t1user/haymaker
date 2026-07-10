"""Resampling helpers for market data and lower-frequency indicators."""

from functools import partial
from typing import Any, Callable

import numpy as np
import pandas as pd

from .technical import atr

__all__ = [
    "downsampled_atr",
    "downsampled_func",
    "resample",
    "weighted_resample",
]


def resample(
    df: pd.DataFrame | pd.Series,
    freq: str,
    how: dict[str, Any] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Resample OHLC data or a single price series.

    Args:
        df: DataFrame with any supported OHLC-style columns, or a Series.
        freq: Pandas offset string for the target frequency.
        how: Optional aggregation rules for additional DataFrame columns.
        **kwargs: Passed to ``DataFrame.resample`` or ``Series.resample``.

    Returns:
        Resampled data. DataFrame columns use OHLC-style aggregation where
        available: first open, maximum high, minimum low, last close, and summed
        volume/bar count.
    """
    if how is None:
        how = {}

    if isinstance(df, pd.Series):
        return df.resample(freq, **kwargs).last().dropna()  # type: ignore

    elif isinstance(df, pd.DataFrame):
        field_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "barCount": "sum",
        }
        field_dict.update(how)

        return (
            df.resample(freq, **kwargs)  # type: ignore
            .agg(
                {
                    key: field_dict[key]  # type: ignore
                    for key in df.columns
                    if key in field_dict.keys()
                }
            )
            .dropna()
        )
    else:
        raise TypeError("df must be either series or dataframe")


def weighted_resample(
    df: pd.DataFrame, freq: str, price_column: str = "average", **kwargs: Any
) -> pd.DataFrame:
    """Resample OHLC data and calculate a volume-weighted average price.

    Args:
        df: DataFrame with OHLC data, ``volume``, and ``price_column``.
        freq: Pandas resampling frequency.
        price_column: Column used for the weighted price calculation.
        **kwargs: Passed to all resampling operations.

    Returns:
        Resampled OHLC data with an ``average`` column. Groups with zero total
        volume get ``NaN`` for ``average`` rather than a synthetic zero price.
    """

    if "volume" not in df.columns:
        raise KeyError("df must have column: volume")
    if price_column not in df.columns:
        raise KeyError(f"df must have column: {price_column}")

    volume = df["volume"].resample(freq, **kwargs).sum()
    weighted_price = (
        (df[price_column] * df["volume"]).resample(freq, **kwargs).sum(min_count=1)
    )
    out = resample(df, freq, **kwargs)
    out["average"] = weighted_price / volume.replace(0, np.nan)
    return out


def downsampled_func(
    df: pd.DataFrame | pd.Series,
    freq: str,
    func: Callable[..., pd.Series],
    *args: object,
    **kwargs: object,
) -> pd.Series:
    """Apply a lower-frequency indicator and align it to raw bars.

    The input bars are expected to be labeled at the start of each raw interval.
    For example, a 1-minute row at ``10:00`` represents the bar starting at
    ``10:00``. The resampled bars are labeled on the right edge, so the
    completed 1-hour bar for ``10:00`` through ``10:59`` is labeled ``11:00``.
    The value returned by ``func`` first appears on that right-edge timestamp
    and is then forward-filled over the next raw bars until a new
    lower-frequency value is available.

    This means lower-frequency values do not appear before their source bar is
    complete. With 1-minute input and ``freq="1h"``, the hourly value labeled
    ``11:00`` is visible on the raw rows from ``11:00`` until the next hourly
    value replaces it.

    Args:
        df: Raw input data. OHLC data should be passed as a DataFrame; a single
            price series may be passed as a Series.
        freq: Pandas resampling frequency used to build the lower-frequency
            data, for example ``"1h"`` or ``"1D"``.
        func: Function that receives the resampled data and returns a Series
            indexed by the resampled timestamps.
        *args: Positional arguments passed to ``func``.
        **kwargs: Keyword arguments passed to ``func``.

    Returns:
        Series named ``"f"`` aligned to the original index.

    Example:
        Add a 20-period moving average calculated on hourly bars to 1-minute
        data:

        .. code-block:: python

            df["hourly_ma"] = downsampled_func(
                df,
                "1h",
                lambda hourly: hourly["close"].rolling(20).mean(),
            )
    """

    if not isinstance(df, (pd.DataFrame, pd.Series)):
        raise TypeError("df must be pd.DataFrame or pd.Series")

    resampled = resample(df, freq, label="right")
    result = func(resampled, *args, **kwargs)
    if not isinstance(result, pd.Series):
        raise TypeError("func must return a pd.Series")

    return result.reindex(df.index).ffill().rename("f")


def downsampled_atr(
    df: pd.DataFrame, periods, freq: str = "B", **kwargs
) -> pd.Series:
    """Return ATR calculated on resampled data and aligned to raw bars.

    Args:
        df: Raw OHLC data.
        periods: ATR lookback on the resampled data.
        freq: Pandas resampling frequency.
        **kwargs: Passed to :func:`haymaker.research.indicators.atr`.

    Returns:
        ATR series aligned to ``df.index``. Values become available when the
        lower-frequency bar is complete and are forward-filled until the next
        value.

    Examples:
        ``downsampled_atr(df, 20, freq="B")`` calculates 20-day ATR.
        ``downsampled_atr(df, 46, freq="h", smooth_type="simple")`` calculates
        46-hour simple ATR.
    """

    return downsampled_func(df, freq, partial(atr, periods=periods), **kwargs)
