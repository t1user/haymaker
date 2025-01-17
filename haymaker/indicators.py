from functools import partial
from typing import Callable, Literal

import numpy as np
import pandas as pd

from .research.decorators import ensure_df
from .research.numba_tools import (
    _blip_to_signal_converter,
    _in_out_signal_unifier,
    rolling_min_max_index,
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


def mmean(d: pd.Series, periods: int, exp: bool = True, **kwargs) -> pd.Series:
    """Shotcut to select type of mean over a series (modified mean)."""
    if exp:
        out = d.ewm(span=periods, **kwargs).mean()
    else:
        out = d.rolling(periods, **kwargs).mean()
    return out


def atr(df: pd.DataFrame, periods: int, exp: bool = True, **kwargs) -> pd.Series:
    """
    Return Series with ATRs.

    Args:
    ---------
    data: must have columns: 'high', 'low', 'close'
    periods: lookback period
    exp: True - expotential mean, False - simple rolling mean
    **kwargs will be passed to averaging function
    """

    return mmean(true_range(df, 1), periods, exp, **kwargs).rename("ATR")


# depricated function name
get_ATR = atr


def resample(
    df: pd.DataFrame | pd.Series, freq: str, how: dict[str, str] | None = None, **kwargs
) -> pd.DataFrame:
    """Shortcut function to resample ohlc dataframe or close price series.
    Args:

    df - DataFrame to be resampled

    freq - pandas offset key

    how - dict of functions for aggregation of non-standard fields

    **kwargs - will be passed directly to pd.DataFrame.resample
      function

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
    df: pd.DataFrame, freq: str, price_column: str = "average", **kwargs
) -> pd.DataFrame:
    """
    While performing resample calculate also volume weighted price.

    Args:
    ----------

    df, freq - same as in resample

    price_column - price column to be used for caluclation, typically
    should be volume weighted average price for the bar.

    **kwargs - will be passed to resample function

    """

    if "volume" not in df.columns:
        raise KeyError("df must have column: volume")

    df = df.copy()
    df["weights"] = df.volume.resample(freq).transform(lambda x: x / x.sum())
    df["weight_price"] = df["weights"] * df[price_column]
    return resample(df, freq, how={"weight_price": "sum"}, **kwargs).rename(
        columns={"weight_price": "average"}
    )


def downsampled_func(
    df: pd.DataFrame | pd.Series, freq: str, func: Callable, *args, **kwargs
) -> pd.Series:
    """
    Calculate <func> over lower frequency than df and fill forward values.


    Exapmple use: signals could be based on 1-minute df, but they would be filtered
    by trends based on 1-hour data.


    Usage:
    -----
    downsampled_func(df, 'B', lambda x: x.close.rolling(10).mean())

    this will return daily rolling mean over 10 days, which will be filled forward to
    every point in time next day.

    Args:
    -----

    df - must have columns 'open', 'high', 'low', 'close'

    freq -  pandas offset string representing frequency of data to which df
    will be resampled before being passed to <func>

    func - callable to be applied to lower frequency data (must take
    the same type as df and return series)

    func_kwargs - will be passed to <func>

    Returns:
    -------
    pd.Series that can be inserted as column to the original df ensuring
    no forward data snooping

    """

    # label="right" ensures no forward data snooping
    resampled = resample(df, freq, label="right")

    if isinstance(df, pd.DataFrame):
        df = df.copy()
    elif isinstance(df, pd.Series):
        df = pd.DataFrame({"data": df})
    else:
        raise TypeError("df must be pd.DataFrame or pd.Series")

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

    return downsampled_func(df, freq, partial(atr, periods=periods), **kwargs)


def cont_downsampled_func(d: pd.Series, bar: int, func, *args, **kwargs) -> pd.Series:
    """Apply <func> to continuous resampled series.  At every df row, last
    <bar> rows will be treated as one bar.  It's different from downsampled_func,
    where price series is first resampled to a new frequency and then any missing
    values are forward filled from last available point.

    So for hourly calculation at half hour point, downsampled_func will
    give the last value from the hour top, wherus cont_downsampled_func
    will calculate value based on last sixty minutes.

    If result compared to resample, resample will need closed='right',
    label='right'.

    Usage:
    -----

    For d being an OHLC one minute price dataframe, this is hourly
    standard deviation over rolling 23 hours (i.e. 1 day):

    cont_downsampled_func(d.close, 60, lambda x: x.rolling(23).std())


    Args:
    -----

    d - input data, must be a pd.Series

    bar - number of df rows to treat as one (e.g. if d has 30s data
    and hourly <func> is required, <bar> value should be 120)

    """

    # easier to reshape in numpy
    values = d.values
    assert isinstance(values, np.ndarray)
    start_index = values.shape[0] % bar
    step1 = pd.DataFrame(values[start_index:].reshape((-1, bar)))
    try:
        # for functions that calculate dataframe columnwise
        step2 = func(step1, *args, **kwargs)
    except ValueError:
        # for functions that require a series
        step2 = step1.apply(lambda x: func(x, *args, **kwargs), axis=0)
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
        true_range(df, bar), bar, mmean, periods=periods, exp=exp
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
    periods: tuple[int],
    func: Callable[[pd.Series, int], pd.DataFrame] = get_min_max,
) -> dict[str, pd.DataFrame]:
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


def get_signals(data: pd.Series, periods: tuple[int]) -> pd.DataFrame:
    min_max = get_min_max_df(data, periods)
    return pd.DataFrame(
        {
            "signal": majority_function(min_max["max"])
            - majority_function(min_max["min"])
        }
    )


def any_signal(data: pd.Series, periods: tuple[int]) -> pd.Series:
    min_max = get_min_max_df(data, periods)
    return min_max["max"].any(axis=1) * 1 - min_max["min"].any(axis=1) * 1


def modified_rsi(rsi: pd.Series) -> pd.Series:
    """
    Rescale passed rsi to -1 to 1.
    """
    return (2 * (rsi - 50)) / 100


def rsi(
    price: pd.Series,
    lookback: int,
    periods: int = 1,
    exp: bool = True,
    rescale=False,
    *args,
    **kwargs
) -> pd.Series:
    """
    Rsi indicator on a scale of 0 - 100.

    Parameteres:
    -----------

    periods:
        number of periods over which ups and downs are to be caluclated

    exp:
        wheather expotential or simple moving average should be used

    rescale:
        False - return classic RSI
        True - rescale classic RSI to (-1, 1)

    note:
    ----
    Resluts not matching talib.RSI -> for that averaging function must be:
    .ewm(span=1/lookback).mean()
    """
    df = pd.DataFrame({"price": price})
    df["change"] = df["price"].diff(periods).fillna(0)
    df["up"] = (df["change"] > 0) * df["change"]
    df["down"] = (df["change"] < 0) * df["change"].abs()
    if exp:
        df["up_roll"] = df["up"].ewm(span=lookback).mean()
        df["down_roll"] = df["down"].ewm(span=lookback).mean()
    else:
        df["up_roll"] = df["up"].rolling(lookback).mean()
        df["down_roll"] = df["down"].rolling(lookback).mean()
    df["rs"] = df["up_roll"] / df["down_roll"]
    df["rsi"] = 100 - (100 / (1 + df["rs"]))
    if rescale:
        return modified_rsi(df["rsi"])
    else:
        return df["rsi"]


def macd(
    price: pd.Series, fastperiod: int, slowperiod: int, signalperiod: int
) -> pd.DataFrame:
    """
    Results checked against talib.MACD, maching exactly.
    """
    df = pd.DataFrame({"close": price})
    df["fast_trendline"] = df["close"].ewm(span=fastperiod).mean()
    df["slow_trendline"] = df["close"].ewm(span=slowperiod).mean()
    df["macd"] = df["fast_trendline"] - df["slow_trendline"]
    df["macdsignal"] = df["macd"].ewm(span=signalperiod).mean()
    df["macdhist"] = df["macd"] - df["macdsignal"]
    return df[["macd", "macdsignal", "macdhist"]]


def tsi(price: pd.Series, lookback1, lookback2):
    """
    True Strenght Index. Kaufman 2013, p. 404.
    """
    return (
        price.diff().ewm(span=lookback1).mean().ewm(span=lookback2).mean()
        / price.diff().abs().ewm(span=lookback1).mean().ewm(span=lookback2).mean()
    )


def carver(price: pd.Series, lookback: int, ratio: int = 1) -> pd.Series:
    """
    Return modified version of price placing it on a min-max scale
    over recent lookback periods expressed on a scale of -ratio to
    +ratio (-1 to +1 by default).  It's modified stochastic oscilator,
    after Rob Carver:
    https://qoppac.blogspot.com/2016/05/a-simple-breakout-trading-rule.html
    """
    df = pd.DataFrame({"price": price})
    df["max"] = df["price"].rolling(lookback).max()
    df["min"] = df["price"].rolling(lookback).min()
    df["mid"] = df[["min", "max"]].mean(axis=1)
    df["carver"] = 2 * ratio * ((df["price"] - df["mid"]) / (df["max"] - df["min"]))
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
    Create a Series with signal representing buying breakout
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


def breakout_blip(
    price: pd.Series,
    lookback: int,
    stop_frac: float = 0.5,
) -> pd.Series:
    """
    Same as :func:`.breakout`, but generating a series with blips
    rather than signals.

    Args:
    -----

    data: price series on which the breakouts are to be determined,
    typically close prices

    lookback: number of periods for max/min calculation

    stop_frac: number of periods expresed as fraction of lookback
    periods

    Returns:
    --------

    Series with blips representing breakout strategy entry/exit points.
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


def strength_oscillator(df: pd.DataFrame, periods) -> pd.Series:
    """Kaufman 2013, p. 408"""
    d = df.copy()
    d["momentum"] = d.close.diff().fillna(0).rolling(periods).mean()
    d["high_low"] = (d["high"] - d["low"]).rolling(periods).mean()
    return (d["momentum"] / d["high_low"]).dropna()


def chande_ranking(price: pd.Series, lookback: int) -> pd.Series:
    """
    Trend strength ranking indicator. Kaufman book 2013 edition, p. 1069.
    """
    df = pd.DataFrame(index=price.index)
    df["log_return"] = np.log(price.pct_change(lookback) + 1)
    df["one_period_returns"] = np.log(price.pct_change() + 1)
    df["std"] = df["one_period_returns"].rolling(lookback).std()
    return df["log_return"] / (df["std"] * np.sqrt(lookback))


def chande_momentum_indicator(
    price: pd.Series, lookback: int, diff_period: int = 1
) -> pd.Series:
    df = pd.DataFrame({"price": price})
    df["diff"] = df["price"].diff(diff_period)
    df["ups"] = (df["diff"] * (df["diff"] > 0)).rolling(lookback).sum()
    df["downs"] = (df["diff"] * (df["diff"] < 0)).rolling(lookback).sum()
    df["numerator"] = df["ups"] - df["downs"]
    df["denominator"] = df["ups"] + df["downs"]
    return (df["numerator"] / df["denominator"]).ffill()


def join_swing(
    df: pd.DataFrame,
    f: float | np.ndarray | pd.Series,
    margin: float | pd.Series | None = None,
) -> pd.DataFrame:
    """
    Return original df with added columns that result from applying
    swing.  All args and kwargs must be compatible with swing function
    requirements.
    """
    s = swing(df, f, margin, output_as_df=True)
    assert isinstance(s, pd.DataFrame)
    return df.join(s)


def spread(df: pd.DataFrame, lookback: int) -> pd.Series:
    df = df.copy()
    if not set(["high", "low"]).issubset(set(df.columns)):
        if "close" in df.columns:
            df["high"] = df["low"] = df["close"]
        else:
            raise ValueError(
                "Required columns for spread are: 'high' and 'low' or 'close'"
            )
    df["max"] = df["high"].rolling(lookback).max()
    df["min"] = df["low"].rolling(lookback).min()
    return df["max"] - df["min"]


def momentum(price, periods):
    """Double smoothed momentum"""
    return price.diff(periods).ewm(span=periods).mean().ewm(span=periods).mean()


@ensure_df
def divergence_index(df, fast, factor=1):
    """
    Kaufman 2013 p. 384
    """
    df = df.copy()
    slow = 10 * fast
    df["fast_ema"] = df["close"].ewm(span=fast).mean()
    df["slow_ema"] = df["close"].ewm(span=slow).mean()
    df["diff"] = df["close"].diff(slow)
    numerator = df["fast_ema"] - df["slow_ema"]
    denominator = (df["diff"].rolling(slow).std()) ** 2
    df["di"] = numerator / denominator
    band = df["di"].rolling(slow).std()
    df["upper"] = factor * band
    df["lower"] = -factor * band
    return df[["di", "upper", "lower"]]


def signal_generator(series: pd.Series, threshold: float = 0) -> pd.Series:
    return (
        (series > threshold) * 1 - (series < threshold) * 1 + (series == threshold) * 0
    )


def combine_signals(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """
    Series2 is filter. If input signals disagree, no signal is
    output. If they agree, series1 signal is the output.
    """
    return ((np.sign(series1) == np.sign(series2)) * series1).astype(int, copy=False)


# ### Range blip and zero crosser ###


def zero_crosser(indicator: pd.Series) -> pd.Series:
    """
    Blip when indicator crosses zero. Blip is signed the same as sign of the indicator.
    When indicator value is exactly zero at some point, next value will be treated as
    having crossed zero.
    """
    indicator = indicator.fillna(0)
    return (((indicator.shift() * indicator) <= 0) * np.sign(indicator)).astype(int)


def inout_range(
    s: pd.Series,
    threshold: float | pd.Series = 0,
    inout: Literal["inside", "outside"] = "inside",
) -> pd.Series:
    """Given a threshold, return True/False series indicating whether s prices
    are inside/outside (-threshold, threshold) range.
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


def _range_entry(s: pd.Series) -> pd.Series:
    """
    s is the output of inout_range
    """

    return -((s.shift() - s) * s)


def _signed_range_entry(entry: pd.Series, sign: pd.Series) -> pd.Series:
    """
    entry is the output of _range_entry

    entry will be signed same as price when entering range.
    """

    return (_range_entry(entry) * np.sign(sign)).dropna().astype(int)


def range_blip(
    indicator: pd.Series,
    threshold: float | pd.Series = 0,
    inout: Literal["inside", "outside"] = "inside",
) -> pd.Series:
    """
    Blip when indicator enters or leaves range. Blip is signed the same as sign of
    the indicator.
    """

    indicator = indicator.dropna()

    r = inout_range(indicator, threshold, inout)
    return _signed_range_entry(_range_entry(r), indicator)


def min_max_index(
    price: pd.Series, lookback: int, cutoff_value: int = 1, binary: bool = True
) -> pd.Series:
    """
    Difference between min and max index.  Negative means downtrend,
    positive uptrend.

    Args:
    -----

    lookback: rolling window over which the index will be calculated

    cutoff_value: any value below this, considered to be unchanged
    from previous value, prevents zig-zaging during periods without
    any changes

    binary: if True will convert values to binary signal (1: long, -1: short)
    """

    df = pd.DataFrame(
        rolling_min_max_index(price.to_numpy(), lookback),
        columns=["min", "max"],
        index=price.index,
    )
    df["ind"] = df["min"] - df["max"]
    df.loc[df[df["ind"].abs() < cutoff_value].index, "ind"] = 0
    df = df.replace(0, np.nan).ffill()
    if binary:
        return np.sign(df["ind"])
    else:
        return df["ind"]
