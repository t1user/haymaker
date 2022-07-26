from functools import partial
from typing import Dict, Callable, Tuple

import pandas as pd  # type: ignore
import numpy as np

from research.numba_tools import (  # type: ignore
    _in_out_signal_unifier,
    _blip_to_signal_converter,
    swing,
)


def true_range(df: pd.DataFrame, bar: int = 1) -> pd.Series:

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pd.DataFrame")

    if not set(["open", "high", "low", "close"]).issubset(set(df.columns)):
        raise ValueError("df must have columns: 'open', 'high', 'low', 'close'")

    d = pd.DataFrame()
    d["A"] = df["high"].rolling(bar).max() - df["low"].rolling(bar).min()
    d["B"] = (df["high"].rolling(bar).max() - df["close"].shift(bar)).abs()
    d["C"] = (df["low"].rolling(bar).min() - df["close"].shift(bar)).abs()
    d["TR"] = d.max(axis=1)
    return d["TR"]


def mmean(d: pd.Series, periods: int, exp: bool = True) -> pd.Series:
    """Shotcut to select type of mean over a series (modified mean)."""
    if exp:
        out = d.ewm(span=periods).mean()
    else:
        out = d.rolling(periods).mean()
    return out


def atr(df: pd.DataFrame, periods: int, exp: bool = True) -> pd.Series:
    """
    Return Series with ATRs.

    Args:
    ---------
    data: must have columns: 'high', 'low', 'close'
    periods: lookback period
    exp: True - expotential mean, False - simple rolling mean
    """

    return mmean(true_range(df, 1), periods, exp)


# depricated function name
get_ATR = atr


def resample(
    df: pd.DataFrame, freq: str, how: Dict[str, str] = {}, **kwargs
) -> pd.DataFrame:
    """Shortcut function to resample ohlc dataframe or close price series.
    Args:

    df - DataFrame to be resampled

    freq - pandas offset key

    how - dict of functions for aggregation of non-standard fields

    **kwargs - will be passed directly to pd.DataFrame.resample
      function

    """

    if isinstance(df, pd.Series):
        return df.resample(freq, **kwargs).last().dropna()

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
            df.resample(freq, **kwargs)
            .agg(
                {key: field_dict[key] for key in df.columns if key in field_dict.keys()}
            )
            .dropna()
        )
    else:
        raise TypeError("df must be either series or dataframe")


def downsampled_func(
    df: pd.DataFrame, func: Callable, freq: str = "B", *args, **kwargs
) -> pd.Series:
    """
    Calculate <func> over lower frequency than df and fill forward values.


    Exapmple use: signals could be based on 1-minute df, but they would be filtered
    by trends based on 1-hour data.


    Usage:
    -----
    downsampled_func(df, lambda x: x.close.rolling(10).mean(), 'B')

    this will return daily rolling mean over 10 days, which will be filled forward to
    every point in time next day.

    Args:
    -----

    df - must have columns 'open', 'high', 'low', 'close'

    func - callable to be applied to lower frequency data (must take
    df and return series)

    freq -  pandas offset string representing frequency of data to which df
    will be resampled before being passed to <func>

    func_kwargs - will be passed to <func>

    Returns:
    -------
    pd.Series that can be inserted as column to the original df ensuring
    no forward data snooping

    """

    df = df.copy()

    # label="right" ensures no forward data snooping
    resampled = resample(df, freq, label="right")
    df["f"] = func(resampled, *args, **kwargs)
    df["f"] = df["f"].shift(-1)
    df["f"] = df["f"].ffill()
    return df["f"]


def downsampled_atr(df: pd.DataFrame, periods, freq: str = "B", **kwargs) -> pd.Series:

    """
    Shortcut function to simplify generating downsampled atr.

    **kwargs will be passed directly to atr function.

    Usage:
    ------

    downsampled_atr(df, 20, freq='B') - 20 day atr

    downsampled_atr(df, 46, freq='H', exp=False) - 46 hour atr, simple moving average
    """

    return downsampled_func(df, partial(atr, periods=periods), freq, **kwargs)


def cont_downsampled_func(d: pd.Series, bar: int, func, *args, **kwargs) -> pd.Series:
    """
    Apply <func> to continuous resampled series.  At every df row, last
    <bar> rows will be treated as one bar.  It's different from downsampled_func,
    where price series is first resampled to a new frequency and then any missing
    values are forward filled from last available point.

    So for hourly calculation at half hour point, downsampled_func will
    give the last value from the hour top, wherus cont_downsampled_func
    will calculate value based on last sixty minutes.

    Args:
    -----

    d - input data, must be a pd.Series

    bar - number of df rows to treat as one (e.g. if d has 30s data
    and hourly <func> is required, <bar> value should be 120)
    """

    # easier to reshape in numpy
    values = d.values
    start_index = values.shape[0] % bar
    step1 = pd.DataFrame(values[start_index:].reshape((-1, bar)))
    step2 = func(step1, *args, **kwargs)
    out = step2.values.reshape((1, -1)).T
    # out is an ndarray, has no index and needs to be put back into correct place in d
    d = d.iloc[start_index:]
    return pd.Series(out.flatten(), index=d.index)


def cont_downsampled_atr(
    df: pd.DataFrame, bar: int, periods: int, exp: bool = True
) -> pd.Series:
    """Shortcut function to get continues downsampled atr
    (cont_downsampled_func where func is atr).

    exp - True = expotential moving average, False = simple moving average
    """

    return cont_downsampled_func(
        true_range(df, bar), bar, partial(mmean), periods=periods, exp=exp
    )


def min_max_blip(price: pd.Series, period: int) -> pd.Series:
    """
    Return Series of blips (one of: -1, 0 or 1) dependig on whether
    price broke out above max (1) or below min (-1) over last <period>
    observations.

    Args:
    ---------
    price: price Series
    period: lookback period
    """
    return ((price > price.rolling(period, closed="left").max()) * 1) - (
        (price < price.rolling(period, closed="left").min()) * 1
    )


def min_max_signal(data: pd.Series, period: int) -> pd.Series:
    """
    DEPRECATED
    For backward compatibility only.  Correct nomenclature for the
    returned type of series is 'blip'.
    """
    return min_max_blip(data, period)


def min_max_buffer_signal(data: pd.Series, period: int, buff: float = 0) -> pd.Series:
    """
    Return Series of signals (one of: -1, 0 or 1) dependig on whether
    price broke out above max + <buff> (1) or below min - <buff> (-1) over last
    <period> observations.

    This is a general case of min_max_signal, not combined because this one
    will be removed if not found useful.

    Args:
    ---------
    data: price Series
    period: lookback period
    buff: buffor expressing sensitivity filter for deciding whether min or max
          has been breached, given in price units (default: 0)
    """
    df = pd.DataFrame(
        {
            "max": ((data - data.shift(1).rolling(period).max() + buff) > 0) * 1,
            "min": ((data.shift(1).rolling(period).min() + buff - data) > 0) * 1,
        }
    )
    df["signal"] = df["max"] - df["min"]
    return df["signal"]


def get_std(data, periods):
    returns = np.log(data.avg_price.pct_change() + 1)
    return returns.rolling(periods).std() * data.avg_price


def get_min_max(data: pd.Series, period: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "max": (data - data.shift(1).rolling(period).max()) > 0,
            "min": (data.shift(1).rolling(period).min() - data) > 0,
        }
    )


def majority_function(data: pd.DataFrame) -> pd.Series:
    """
    Return a reduced Series where every row = True, if majority of values in
    input row is True, else False.

    Args:
    ---------
    data: every row contains bool values, on which majority function is applied
    """
    return (0.5 + ((data.sum(axis=1) - 0.5) / data.count(axis=1))).apply(np.floor)


def get_min_max_df(
    data: pd.Series,
    periods: Tuple[int],
    func: Callable[[pd.Series, int], pd.DataFrame] = get_min_max,
) -> Dict[str, pd.DataFrame]:
    """
    Given a list of periods, return func on each of those periods.
    This is bullshit.
    """
    min_max_func = partial(func, data)
    mins = pd.DataFrame()
    maxs = pd.DataFrame()
    for period in periods:
        df = min_max_func(period)
        mins[period] = df["min"]
        maxs[period] = df["max"]
    return {"min": mins, "max": maxs}


def get_signals(data: pd.Series, periods: Tuple[int]) -> pd.Series:
    min_max = get_min_max_df(data, periods)
    return pd.DataFrame(
        {
            "signal": majority_function(min_max["max"])
            - majority_function(min_max["min"])
        }
    )


def any_signal(data: pd.Series, periods: Tuple[int]) -> pd.Series:
    min_max = get_min_max_df(data, periods)
    return min_max["max"].any(axis=1) * 1 - min_max["min"].any(axis=1) * 1


def rsi(
    price: pd.Series, lookback: int, periods: int = 1, *args, **kwargs
) -> pd.Series:
    df = pd.DataFrame({"price": price})
    df["change"] = df["price"].diff(periods).fillna(0)
    df["up"] = ((df["change"] > 0) * df["change"]).rolling(lookback).sum()
    df["down"] = ((df["change"] < 0) * df["change"].abs()).rolling(lookback).sum()
    df["rs"] = df["up"] / df["down"]
    df["rsi"] = 100 - (100 / (1 + df["rs"]))
    return df["rsi"]


def modified_rsi(rsi: pd.Series) -> pd.Series:
    """
    Rescale passed rsi to -100 to 100.
    """
    return 2 * (rsi - 50)


def carver(price: pd.Series, lookback: int) -> pd.Series:
    """
    Return modified version of price placing it on a min-max scale
    over recent lookback periods expressed on a scale of -100 to 100
    (modified stochastic oscilator, after Rob Carver:
    https://qoppac.blogspot.com/2016/05/a-simple-breakout-trading-rule.html).
    """
    df = pd.DataFrame({"price": price})
    df["max"] = df["price"].rolling(lookback).max()
    df["min"] = df["price"].rolling(lookback).min()
    df["mid"] = df[["min", "max"]].mean(axis=1)
    df["carver"] = 200 * ((df["price"] - df["mid"]) / (df["max"] - df["min"]))
    return df["carver"]


def range_crosser(ind: pd.Series, threshold: float) -> pd.Series:
    """
    For an ind like rsi, returns signal (-1, 0, 1) when ind crosses
    threshold from above or -threshold from below.
    THIS IS LIKELY CRAP. CHECK BEFORE USE!!!
    """
    df = pd.DataFrame({"ind": ind})
    df["inside"] = df["ind"].abs() < threshold
    df["ss"] = ~(df["inside"].shift().fillna(False)) & df["inside"]
    df["s"] = np.sign(df["ind"].diff())
    df["signal"] = df["ss"] * df["s"]
    return df["signal"]


def adx(data: pd.DataFrame, lookback: int) -> pd.Series:
    """
    Average directional movement index (expotentially weighted).

    data - must have columns: 'high' and 'low'

    references:
    https://en.wikipedia.org/wiki/Average_directional_movement_index
    https://www.investopedia.com/terms/d/dmi.asp
    https://www.fmlabs.com/reference/default.htm?url=ADX.htm
    """
    df = data.copy()
    df["atr"] = atr(df, lookback)
    df["deltaHigh"] = (df["high"] - df["high"].shift()).clip(lower=0)
    df["deltaLow"] = (df["low"] - df["low"].shift()).clip(lower=0)
    df["plusDM"] = (df["deltaHigh"] > df["deltaLow"]) * df["deltaHigh"]
    df["minusDM"] = (df["deltaLow"] > df["deltaHigh"]) * df["deltaLow"]
    df["pdm_s"] = df["plusDM"].ewm(span=lookback).mean()
    df["mdm_s"] = df["minusDM"].ewm(span=lookback).mean()
    df["pDI"] = (df["pdm_s"] / df["atr"]) * 100
    df["mDI"] = (df["mdm_s"] / df["atr"]) * 100
    df["dx"] = ((df["pDI"] - df["mDI"]).abs() / (df["pDI"] + df["mDI"])) * 100
    df["adx"] = df["dx"].ewm(span=lookback).mean()
    return df["adx"]


def breakout(
    price: pd.Series,
    lookback: int,
    stop_frac: float = 0.5,
) -> pd.Series:
    """
    Create a Series with signal representing buying upside breakout
    beyond lookback periods maximum and selling breakout below
    lookback periods minimum.  Once generated, signal stays constant
    until canceled or reversed.

    Args:
    -----

    data: price series on which the breakouts are to be determined,
    typically close prices

    lookback: number of periods for max/min calculation

    stop_frac: number of periods expresed as fraction of lookback
    periods; breakout above/blow max/min of this number of periods
    (rounded to nearest int) in the opposite direction to the existing
    position will close out existing position; must be between 0 and
    1; if equal to 1, strategy is always in the market (oposite signal
    reverses existing position); default: .5

    Returns:
    --------

    Series where 1 means long signal, -1 short position, 0 no position
    at a given index point.
    """

    if not isinstance(price, pd.Series):
        raise TypeError("price must be a pandas series")
    if not (stop_frac > 0 and stop_frac <= 1):
        raise ValueError("stop_frac must be from (0, 1>")

    df = pd.DataFrame({"price": price})
    df["in"] = min_max_blip(df["price"], lookback)
    if stop_frac == 1:
        df["break"] = _blip_to_signal_converter(df["in"].to_numpy())
    else:
        df["out"] = min_max_blip(df["price"], int(lookback * stop_frac))
        df["break"] = _in_out_signal_unifier(df[["in", "out"]].to_numpy())
    return df["break"]


def strength_oscillator(df: pd.DataFrame, periods) -> pd.Series:
    d = df.copy()
    d["momentum"] = d.close.diff()
    d["high_low"] = d["high"] - d["low"]
    return (d["momentum"] / d["high_low"]).rolling(periods).mean()


def join_swing(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    """
    Return original df with added columns that result from applying
    swing.  All args and kwargs must be compatible with swing function
    requirements.
    """
    return df.join(pd.DataFrame(swing(df, *args, **kwargs)._asdict(), index=df.index))
