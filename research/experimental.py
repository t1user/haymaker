import numpy as np
import pandas as pd  # type: ignore
from numba import jit  # type: ignore
from typing import Callable
from functools import partial

from indicators import resample


def resampled_series(row: pd.Series, *, series: pd.Series, field: str, f: Callable):
    """TOO SLOW TO BE PRACTICAL.

    Merge downsampled series by adding last value to it. To be used
    as argument to apply() on dataframe.


    Args:

    row: df.apply() will supply a series with all columns and index

    series: downsampled series on which calculation is to be done

    field: which field from row should be used func: function to be
    applied on series + last field from row.

    Usage:

    tt = partial(resampled_series, series=h_close, field='close',
    func=lambda x: x.iloc[-10:].mean())

    df.apply(tt, axis=1)

    """

    return f(
        pd.concat([series.loc[: row.name], pd.Series(row[field], index=[row.name])])
    )


def last_downsampled_func_p(df: pd.DataFrame, freq: str, func, *args) -> pd.Series:
    """
    func - accepts series returns series
    """

    series = resample(df, freq).close
    o = partial(resampled_series, series=series, field="close", f=func)
    return df.apply(o, axis=1)


@jit(nopython=True)
def down(
    downsampled_price: np.ndarray,
    price: np.ndarray,
    index: np.ndarray,
    func: Callable,
    *args
) -> np.ndarray:
    """Numba implementation, to be called only from
    last_downsampled_func, which needs to prepare data.

    Args:

    downsampled_price: downsampled price data to be used to calculate indicator

    price: price data before downsampling

    index: mapping of price to downsampled_price

    func: function to get the indicator, must be numba optimized

    """
    out = np.zeros(index.shape[0])
    for n, (v, i) in enumerate(zip(price, index)):
        out[n] = func(np.append(downsampled_price[: i + 1], v), *args)
    return out


@jit(nopython=True)
def n_mean(series: np.ndarray, periods: int) -> np.ndarray:
    """numba enhanced rolling mean for use with down"""

    return np.mean(series[-periods:])


@jit(nopython=True)
def n_ewma(data: np.ndarray, span: int):
    """numba enhanced expotential mean for use with down

    NOT TESTED. DON'T USE."""

    alpha = 2 / (span + 1.0)
    alpha_rev = 1 - alpha
    n = data.shape[0]

    pows = alpha_rev ** (np.arange(n + 1))

    scale_arr = 1 / pows[:-1]
    offset = data[0] * pows[1:]
    pw0 = alpha * alpha_rev ** (n - 1)

    mult = data * pw0 * scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums * scale_arr[::-1]
    return out[-1]


def last_downsampled_func(price: pd.Series, freq: str, func, *args) -> pd.Series:
    """
    Calculate indicator on downsampled data but include latest price.

    Resampled series will be appended with last price - so keeping
    number of rolling periods constant will get rid of the first
    element. Rolling indicator will use the latest available value, it
    will also mean that the last rolling periods will be shorter than
    all previous periods.
    """

    # the way index works, it's correct - no double use of the same value
    # and no forward snooping
    # even though resampled is not shifted back as in downsampled_func
    df = pd.DataFrame({"price": price})
    resampled = resample(price, freq, label="right")
    df["resampled"] = resampled
    df["resampled"] = df["resampled"].fillna(0)
    df["i"] = resampled.reset_index().reset_index().set_index("date")["index"]
    df["i"] = df["i"].ffill()
    df = df.dropna()
    df["i"] = df["i"].astype(int)
    df["out"] = down(
        resampled.to_numpy(), df["price"].to_numpy(), df["i"].to_numpy(), func, *args
    )
    return df["out"]
