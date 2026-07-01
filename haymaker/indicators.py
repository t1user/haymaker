from functools import partial, wraps
from typing import Callable, Literal

import numpy as np
import pandas as pd

from .research.numba_tools import (
    _blip_to_signal_converter,
    _in_out_blip_unifier,
    rolling_min_max_index,
    swing,
)


def ensure_df(func):
    """
    Decorator allowing signal producing functions to work with either
    dataframe or series.
    """

    @wraps(func)
    def verify(data, *args, **kwargs) -> pd.DataFrame:
        if isinstance(data, pd.Series):
            data = pd.DataFrame({"close": data})
        elif isinstance(data, pd.DataFrame):
            data = data.copy()
        else:
            raise TypeError(
                f"Data must be either Series or DataFrame with column 'close'"
                f" containing prices, not {type(data)}."
            )
        return func(data, *args, **kwargs)

    return verify


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


def min_max_buffer_signal(data: pd.Series, period: int, buff: float = 0) -> pd.Series:
    """Return breakout blips with a price-unit buffer.

    A positive ``buff`` makes breakouts harder to trigger: price must move above
    the previous rolling maximum plus ``buff`` or below the previous rolling
    minimum minus ``buff``.
    """
    prior_max = data.shift(1).rolling(period).max()
    prior_min = data.shift(1).rolling(period).min()
    df = pd.DataFrame(
        {
            "max": (data > prior_max + buff) * 1,
            "min": (data < prior_min - buff) * 1,
        }
    )
    df["signal"] = df["max"] - df["min"]
    return df["signal"]


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
    **kwargs,
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
    """Return Blau's True Strength Index.

    Kaufman 2013, pp. 404-406 describes TSI as double-smoothed momentum
    divided by double-smoothed absolute momentum. This implementation omits the
    conventional ``100`` multiplier and returns the ratio directly.
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


def adx(data: pd.DataFrame, lookback: int) -> pd.Series:
    """Return Wilder-style Average Directional Movement Index.

    ADX measures trend strength, not direction: higher values mean stronger
    directional movement whether the market is rising or falling. Kaufman cites
    common trend filters such as ADX crossing above 25 for trending markets and
    below 20 for consolidating markets.

    ``+DM`` is today's high minus yesterday's high when that upward move is
    larger than the downward move. ``-DM`` is yesterday's low minus today's low
    when that downward move is larger. The directional movement components,
    true range, and DX are exponentially smoothed with ``alpha=1/lookback``.
    This follows Kaufman's/Wilder's directional movement and DX construction,
    but it also smooths ADX itself with ``alpha=1/lookback`` rather than the
    fixed 0.133 constant Kaufman gives for 14-day ADX, and it returns ADX rather
    than ADXR.

    References:
        Perry J. Kaufman, *Trading Systems and Methods*, 5th ed., Chapter 23.
    """
    df = data.copy()
    up_move = df["high"].astype(float).diff()
    down_move = df["low"].astype(float).shift() - df["low"].astype(float)
    df["plusDM"] = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    df["minusDM"] = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    df["atr"] = true_range(df, 1).ewm(alpha=1 / lookback, adjust=False).mean()
    df["pdm_s"] = df["plusDM"].ewm(alpha=1 / lookback, adjust=False).mean()
    df["mdm_s"] = df["minusDM"].ewm(alpha=1 / lookback, adjust=False).mean()
    df["pDI"] = (df["pdm_s"] / df["atr"]) * 100
    df["mDI"] = (df["mdm_s"] / df["atr"]) * 100
    df["dx"] = ((df["pDI"] - df["mDI"]).abs() / (df["pDI"] + df["mDI"])) * 100
    df["adx"] = df["dx"].ewm(alpha=1 / lookback, adjust=False).mean()
    return df["adx"]


def breakout(
    price: pd.Series,
    lookback: int,
    stop_frac: float = 0.5,
    always_on: bool = False,
) -> pd.Series:
    """Return stateful breakout signal from rolling high/low blips.

    The returned signal is recorded on the bar where the breakout
    information becomes known. It is not an executable position. For
    stop-loss workflows prefer :func:`breakout_blip` or raw blip columns
    passed to ``stop_loss``.

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
        desired market exposure at that bar.
    """

    if not isinstance(price, pd.Series):
        raise TypeError("price must be a pandas series")
    if not (stop_frac > 0 and stop_frac <= 1):
        raise ValueError("stop_frac must be from (0, 1>")

    df = pd.DataFrame({"price": price})
    df["in"] = min_max_blip(df["price"], lookback)
    if stop_frac == 1:
        df["break"] = _blip_to_signal_converter(
            df["in"].to_numpy(), always_on=always_on
        )
    else:
        df["out"] = min_max_blip(df["price"], int(lookback * stop_frac))
        df["break"] = _in_out_blip_unifier(df[["in", "out"]].to_numpy())
    return df["break"]


def breakout_blip(
    price: pd.Series,
    lookback: int,
    stop_frac: float = 0.5,
) -> pd.Series:
    """
    Same as :func:`.breakout`, but generating a series with blips
    rather than signals.

    Blips are raw generated events recorded on the bar where the breakout
    information becomes known. Do not pre-shift them before passing them to
    blip-aware consumers such as ``stop_loss`` or ``no_stop``.

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
    """Return Kaufman's trend-strength oscillator.

    This follows the idea in Kaufman 2013, p. 408: average close-to-close
    momentum divided by average high-low range over the same window.
    """
    d = df.copy()
    d["momentum"] = d.close.diff().fillna(0).rolling(periods).mean()
    d["high_low"] = (d["high"] - d["low"]).rolling(periods).mean()
    return (d["momentum"] / d["high_low"]).dropna()


def chande_ranking(price: pd.Series, lookback: int) -> pd.Series:
    """Return a volatility-normalized trend-strength ranking.

    This follows the market-ranking idea referenced by Kaufman 2013 around the
    Directional Movement discussion: compare lookback log return with realized
    one-period volatility over the same lookback. Higher absolute values imply
    stronger trend relative to recent noise.
    """
    df = pd.DataFrame(index=price.index)
    df["log_return"] = np.log(price.pct_change(lookback) + 1)
    df["one_period_returns"] = np.log(price.pct_change() + 1)
    df["std"] = df["one_period_returns"].rolling(lookback).std()
    return df["log_return"] / (df["std"] * np.sqrt(lookback))


def chande_momentum_indicator(
    price: pd.Series, lookback: int, diff_period: int = 1
) -> pd.Series:
    """Return Chande Momentum Oscillator on a ``-100`` to ``100`` scale."""
    df = pd.DataFrame({"price": price})
    df["diff"] = df["price"].diff(diff_period)
    df["ups"] = df["diff"].clip(lower=0).rolling(lookback).sum()
    df["downs"] = (-df["diff"].clip(upper=0)).rolling(lookback).sum()
    df["numerator"] = df["ups"] - df["downs"]
    df["denominator"] = df["ups"] + df["downs"]
    df["cmo"] = 100 * df["numerator"] / df["denominator"]
    return df["cmo"].ffill()


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
    """Return Kaufman's Divergence Index and adaptive bands.

    Kaufman 2013, pp. 384-385 defines DI as the volatility-adjusted difference
    between fast and slow moving averages. This implementation uses an EMA with
    ``fast`` and ``10 * fast`` spans, then bands based on the rolling standard
    deviation of DI.
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
    return s.astype(int).diff().eq(1).astype(int)


def _signed_range_entry(entry: pd.Series, sign: pd.Series) -> pd.Series:
    """
    entry is the output of _range_entry

    entry will be signed same as price when entering range.
    """

    return (entry * np.sign(sign)).astype(int)


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
        return np.sign(df["ind"])  # type: ignore
    else:
        return df["ind"]
