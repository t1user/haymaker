from functools import partial
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

__all__ = [
    "adx",
    "atr",
    "carver",
    "chande_momentum_indicator",
    "chande_ranking",
    "divergence_index",
    "downsampled_atr",
    "downsampled_func",
    "join_swing",
    "macd",
    "mmean",
    "momentum",
    "resample",
    "rsi",
    "spread",
    "strength_oscillator",
    "true_range",
    "tsi",
    "weighted_resample",
]


def true_range(df: pd.DataFrame, bar: int = 1) -> pd.Series:
    """Return true range over ``bar`` periods.

    For ``bar=1`` this is the standard true range: the maximum of high-low,
    high-previous close, and low-previous close in absolute price units.
    Larger ``bar`` values compare the rolling high/low range against the close
    ``bar`` rows ago.
    """
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


def mmean(
    d: pd.Series,
    periods: int,
    smooth_type: Literal["simple", "exponential", "wilder"] = "exponential",
    **kwargs,
) -> pd.Series:
    """Return a moving average using the selected smoothing method.

    Args:
        d: Input series.
        periods: Moving-average lookback.
        smooth_type: Smoothing method:

            ``"simple"``
                Simple moving average,
                ``mean(d[t - periods + 1], ..., d[t])``.

            ``"exponential"``
                Pandas span-based exponential moving average,
                ``d.ewm(span=periods).mean()``. This uses
                ``alpha = 2 / (periods + 1)``.

            ``"wilder"``
                Wilder's average-off smoothing. The first value is seeded with
                the simple average of the first ``periods`` values, then updated
                as ``previous + (current - previous) / periods``. This is
                equivalent to an EMA with ``alpha = 1 / periods`` after the
                initial seed.
        **kwargs: Passed to pandas for ``"simple"`` and ``"exponential"``.
    """
    if smooth_type == "wilder":
        out = _wilder_average(d, periods)
    elif smooth_type == "exponential":
        out = d.ewm(span=periods, **kwargs).mean()
    elif smooth_type == "simple":
        out = d.rolling(periods, **kwargs).mean()
    else:
        raise ValueError(
            "smooth_type must be one of: 'simple', 'exponential', 'wilder'"
        )
    return out


def atr(
    df: pd.DataFrame,
    periods: int,
    smooth_type: Literal["simple", "exponential", "wilder"] = "exponential",
    **kwargs,
) -> pd.Series:
    """Return Average True Range in price units.

    Args:
        df: DataFrame with ``high``, ``low``, and ``close`` columns.
        periods: Moving-average lookback.
        smooth_type: Moving-average smoothing type. See :func:`mmean` for the
            formulas. The default ``"exponential"``.
        **kwargs: Passed to the selected pandas averaging method.

    Returns:
        Series named ``"ATR"``.

    Notes:
        This function uses the standard true range calculation. With
        ``smooth_type="exponential"`` it uses pandas' span-based exponential
        mean, which is not Wilder's original average-off smoothing. Use
        ``smooth_type="simple"`` for a simple moving average ATR, or
        ``smooth_type="wilder"`` for Wilder ATR.
    """

    return mmean(true_range(df, 1), periods, smooth_type, **kwargs).rename("ATR")


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


def downsampled_atr(df: pd.DataFrame, periods, freq: str = "B", **kwargs) -> pd.Series:
    """Return ATR calculated on resampled data and aligned to raw bars.

    Args:
        df: Raw OHLC data.
        periods: ATR lookback on the resampled data.
        freq: Pandas resampling frequency.
        **kwargs: Passed to :func:`atr`.

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


def _wilder_average(series: pd.Series, lookback: int) -> pd.Series:
    """Return Wilder's average-off smoothing for a positive series."""
    if lookback <= 0:
        raise ValueError("lookback must be positive")

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


def rsi(
    price: pd.Series,
    lookback: int,
    diff_period: int = 1,
    rescale: bool = False,
) -> pd.Series:
    """Return Wilder/Kaufman Relative Strength Index.

    RSI compares the average upward price changes with the average downward
    price changes and scales the result from ``0`` to ``100``. Kaufman describes
    the usual interpretation as overbought near ``70`` and oversold near ``30``.

    Args:
        price: Price series, typically close prices.
        lookback: Number of changes included in Wilder's average-off smoothing.
            Kaufman notes Wilder's common default of 14.
        diff_period: Price-difference lag. The Kaufman/Wilder definition uses
            ``1``; larger values calculate RSI from multi-period changes.
        rescale: If ``False``, return RSI on the usual ``0..100`` scale. If
            ``True``, return the same oscillator on ``-1..1``, which is easier
            to combine with symmetric threshold helpers such as
            :func:`extreme_reversal_blip`.

    Returns:
        RSI series aligned to ``price``.
    """
    df = pd.DataFrame({"price": price})
    df["change"] = df["price"].diff(diff_period)
    df["up"] = df["change"].clip(lower=0)
    df["down"] = -df["change"].clip(upper=0)
    df["up_roll"] = _wilder_average(df["up"], lookback)
    df["down_roll"] = _wilder_average(df["down"], lookback)
    df["rs"] = df["up_roll"] / df["down_roll"]
    df["rsi"] = 100 - (100 / (1 + df["rs"]))
    if rescale:
        return (2 * (df["rsi"] - 50)) / 100
    else:
        return df["rsi"]


def macd(
    price: pd.Series, fastperiod: int, slowperiod: int, signalperiod: int
) -> pd.DataFrame:
    """Return Moving Average Convergence/Divergence components.

    MACD is the difference between a fast and slow exponential moving average.
    The signal line is an exponential moving average of MACD, and the histogram
    is MACD minus the signal line.

    Args:
        price: Price series, typically close prices.
        fastperiod: Span for the fast EMA.
        slowperiod: Span for the slow EMA.
        signalperiod: Span for the MACD signal EMA.

    Returns:
        DataFrame with ``macd``, ``macdsignal``, and ``macdhist`` columns.

    Notes:
        This uses pandas ``ewm(span=...)`` defaults. Results can differ from
        libraries that use different EMA seeding or adjustment rules.
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
    divided by double-smoothed absolute momentum. The returned value is the raw
    ratio; multiply by ``100`` if you need the percentage-style scale.
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


def strength_oscillator(df: pd.DataFrame, periods) -> pd.Series:
    """Return Kaufman's trend-strength oscillator.

    This follows the idea in Kaufman 2013, p. 408: average close-to-close
    momentum divided by average high-low range over the same window.
    Positive values indicate upward movement, negative values indicate downward
    movement, and the magnitude is expressed in average bar-range units. The
    result is not bounded to a fixed scale.
    """
    d = df.copy()
    d["momentum"] = d.close.diff().fillna(0).rolling(periods).mean()
    d["high_low"] = (d["high"] - d["low"]).rolling(periods).mean()
    return d["momentum"] / d["high_low"]


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
    from ..numba_tools import swing

    s = swing(df, f, margin, output_as_df=True)
    assert isinstance(s, pd.DataFrame)
    return df.join(s)


def spread(df: pd.DataFrame, lookback: int) -> pd.Series:
    """Return the rolling high-low spread over ``lookback`` rows.

    If ``high`` and ``low`` are missing but ``close`` is present, ``close`` is
    used as both high and low. This gives a zero spread for flat close-only
    windows and keeps the function usable on single-price data.
    """
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


def momentum(
    price: pd.Series,
    periods: int,
    smooth_periods: int | tuple[int, int] | None = None,
) -> pd.Series:
    """Return Blau/Kaufman double-smoothed momentum.

    Momentum is the price change over ``periods`` rows. Kaufman describes Blau's
    double-smoothed momentum as applying exponential smoothing to that price
    change, then smoothing the result again. This keeps the indicator centered
    around zero while reducing noise.

    Args:
        price: Price series, typically close prices.
        periods: Price-difference lag.
        smooth_periods: EMA span used for smoothing. If ``None``, use
            ``periods`` for both smoothing passes. If an integer, use it for
            both passes. If a ``(first, second)`` tuple, use separate spans for
            the first and second smoothing passes.

    Returns:
        Double-smoothed momentum in price units.
    """
    if smooth_periods is None:
        first_smooth = second_smooth = periods
    elif isinstance(smooth_periods, tuple):
        first_smooth, second_smooth = smooth_periods
    else:
        first_smooth = second_smooth = smooth_periods

    return (
        price.diff(periods).ewm(span=first_smooth).mean().ewm(span=second_smooth).mean()
    )


def divergence_index(
    data: pd.Series | pd.DataFrame, fast: int, factor: float = 1
) -> pd.DataFrame:
    """Return Kaufman's Divergence Index and adaptive bands.

    Kaufman 2013, pp. 384-385 defines DI as the volatility-adjusted difference
    between fast and slow moving averages. This function uses EMA spans of
    ``fast`` and ``10 * fast``. The EMA difference is divided by the rolling
    standard deviation of one-period price changes. Upper/lower bands are based
    on the rolling standard deviation of DI.

    Args:
        data: Price series, or dataframe with a ``close`` column.
        fast: Span for the fast EMA. The slow EMA span is ``10 * fast``.
        factor: Multiplier for the upper and lower DI bands.

    Returns:
        DataFrame with ``di``, ``upper``, and ``lower`` columns.
    """
    if isinstance(data, pd.Series):
        df = pd.DataFrame({"close": data})
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError(
            "data must be either a Series or a DataFrame with a 'close' column, "
            f"not {type(data)}."
        )

    slow = 10 * fast
    df["fast_ema"] = df["close"].ewm(span=fast).mean()
    df["slow_ema"] = df["close"].ewm(span=slow).mean()
    df["diff"] = df["close"].diff()
    numerator = df["fast_ema"] - df["slow_ema"]
    denominator = df["diff"].rolling(slow).std()
    df["di"] = numerator / denominator
    band = df["di"].rolling(slow).std()
    df["upper"] = factor * band
    df["lower"] = -factor * band
    return df[["di", "upper", "lower"]]
