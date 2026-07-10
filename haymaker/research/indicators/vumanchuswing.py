"""VuManChu-style range-filter swing indicator helpers.

The helpers in this module calculate a smoothed range size from prices and use
it to build a stepped range filter with upper and lower bands. A rising price
must move more than the current range size above the previous filter value
before the filter steps higher; a falling price must move more than the current
range size below the previous filter value before the filter steps lower.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from numba import njit  # type: ignore

__all__ = ["range_filter", "range_size", "vu_man_chu_swing"]


@njit
def range_filter(
    price: np.ndarray, range_size: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return upper band, lower band, and stepped range filter arrays.

    Args:
        price: Price values used to update the filter.
        range_size: Per-row range size in price units. It must have the same
            length as ``price``.

    Returns:
        Tuple of ``(hi_band, lo_band, range_filter)`` arrays.
    """
    range_filt = price.copy()
    hi_band = price.copy()
    lo_band = price.copy()

    for i, p in enumerate(price):
        if i == 0:
            continue
        if p > range_filt[i - 1]:
            if p < range_filt[i - 1] + range_size[i]:
                range_filt[i] = range_filt[i - 1]
            else:
                range_filt[i] = p - range_size[i]
        else:
            if p > range_filt[i - 1] - range_size[i]:
                range_filt[i] = range_filt[i - 1]
            else:
                range_filt[i] = p + range_size[i]
        hi_band[i] = range_filt[i] + range_size[i]
        lo_band[i] = range_filt[i] - range_size[i]
    return hi_band, lo_band, range_filt


def range_size(
    price: pd.Series, range_period: int, diff_periods: int = 1
) -> pd.Series:
    """Return the smoothed absolute price-change range used by the filter.

    Args:
        price: Price series, usually close prices.
        range_period: Span used for both exponential moving averages.
        diff_periods: Number of rows used for the absolute price difference.

    Returns:
        Series on the same index as ``price``. Initial values may be ``NaN``
        while the price difference is unavailable.
    """
    return (
        price.diff(diff_periods)
        .abs()
        .ewm(span=range_period)
        .mean()
        .ewm(span=range_period)
        .mean()
    )


def vu_man_chu_swing(
    df: pd.DataFrame, range_size: pd.Series, range_multiplier: float
) -> pd.DataFrame:
    """Return VuManChu-style range-filter bands for OHLC data.

    Args:
        df: DataFrame with a ``close`` column.
        range_size: Per-row range size in price units. Use
            :func:`range_size`, or provide another same-index series.
        range_multiplier: Multiplier applied to ``range_size`` before the
            filter is calculated.

    Returns:
        DataFrame indexed like ``df`` with:

        - ``hi``: upper band, ``filter + adjusted_range_size``.
        - ``lo``: lower band, ``filter - adjusted_range_size``.
        - ``filter``: stepped range-filter value.

    Raises:
        KeyError: If ``df`` does not contain ``close``.
        ValueError: If ``range_size`` is not indexed like ``df``.
    """
    if not range_size.index.equals(df.index):
        raise ValueError("range_size must have the same index as df.")

    price = df["close"]
    rs = range_size * range_multiplier
    hi, lo, rfilter = range_filter(price.to_numpy(), rs.to_numpy())
    return pd.DataFrame({"hi": hi, "lo": lo, "filter": rfilter}, index=df.index)
