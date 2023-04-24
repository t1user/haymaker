# source: https://medium.com/@chris_42047/the-vuman-chu-swing-trading-strategy-python-tutorial-e7eba705aa48

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from numba import njit  # type: ignore


@njit
def range_filter(
    price: np.ndarray, range_size: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    price: pd.Series, range_period: int, range_multiplier: float, diff_periods: int = 1
):
    return (
        price.diff(diff_periods)
        .abs()
        .ewm(span=range_period)
        .mean()
        .ewm(span=(range_period))
        .mean()
        * range_multiplier
    )


def vu_man_chu_swing(
    df: pd.DataFrame,
    range_period: int,
    range_multiplier: int,
    diff_periods: int = 1,
    rs: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """This needs some work."""
    price = df["close"]
    if rs is None:
        rs = range_size(price, range_period, range_multiplier, diff_periods).fillna(0)
    hi, lo, rfilter = range_filter(price.to_numpy(), rs.to_numpy())
    return pd.DataFrame({"hi": hi, "lo": lo, "filter": rfilter}, index=df.index)
