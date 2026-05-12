from multiprocessing import Pool, cpu_count
from typing import cast

import numpy as np
import pandas as pd

from .backtester import Results, get_min_tick


def true_sharpe(ret):
    """
    Given Series with daily returns (simple, non-log ie. P1/P0-1),
    return indicators comparing simplified Sharpe to 'true' Sharpe
    (defined by different caluclation conventions).
    """
    r = pd.Series()
    df = pd.DataFrame({"returns": ret})
    df["cummulative_return"] = (df["returns"] + 1).cumprod()
    df["log_returns"] = np.log(df["returns"] + 1)
    r["cummulative_return"] = df["cummulative_return"][-1] - 1
    r["annual_return"] = ((r["cummulative_return"] + 1) ** (252 / len(df.index))) - 1
    r["mean"] = df["returns"].mean() * 252
    r["mean_log"] = df["log_returns"].mean() * 252
    r["vol"] = df["returns"].std() * np.sqrt(252)
    r["vol_log"] = df["log_returns"].std() * np.sqrt(252)
    r["sharpe"] = r["mean"] / r["vol"]
    r["sharpe_log"] = r["mean_log"] / r["vol_log"]
    return r


def rolling_sharpe(returns: pd.Series, months: float) -> pd.DataFrame:
    ret = pd.DataFrame({"returns": returns})
    ret["mean"] = ret["returns"].rolling(int(12 * months)).mean() * 252
    ret["vol"] = ret["returns"].rolling(int(12 * months)).std() * np.sqrt(252)
    ret["sharpe"] = ret["mean"] / ret["vol"]
    ret = ret.dropna()
    return ret


def plot_rolling_sharpe(returns: pd.Series, months: float) -> None:
    rolling = rolling_sharpe(returns, months)
    rolling["mean_sharpe"] = rolling["sharpe"].mean()
    rolling[["sharpe", "mean_sharpe"]].plot(figsize=(20, 5), grid=True)


def plot_rolling_vol(returns: pd.Series, months: float) -> None:
    rolling = rolling_sharpe(returns, months)
    rolling["mean_vol"] = rolling["vol"].mean()
    rolling[["vol", "mean_vol"]].plot(figsize=(20, 5), grid=True)


def sampler(data, start=None, end=None, period_length=25, paths=100):
    if start:
        data = data.loc[start:]
    if end:
        data = data.loc[:end]

    daily = data.resample("B").first()
    lookup_table = pd.Series(daily.index.shift(period_length), index=daily.index)
    d = np.random.choice(daily.index[:-period_length], size=paths)
    output = []
    for i in d:
        p = data.loc[i : lookup_table[i]].iloc[:-1]
        # p.set_index(pd.date_range(freq='min', start=data.index[0],
        #                          periods=len(p), name='date'), inplace=True)
        output.append(p)
    return output


def m_proc(dfs, func):
    """
    Run func on every element of dfs in mulpiple processes.
    dfs is a list of DataFrames
    """
    pool = Pool(processes=cpu_count())
    results = [pool.apply_async(func, args=(df,)) for df in dfs]
    output = [p.get() for p in results]
    return output


def crosser(ind: pd.Series, threshold: float) -> pd.Series:
    """
    Generate blips only at points where indicator goes above/below threshold.
    """
    df = pd.DataFrame({"ind": ind})
    df["above_below"] = (df["ind"] >= threshold) * 1 - (df["ind"] < threshold) * 1
    df["blip"] = ((df["above_below"].shift() + df["above_below"]) == 0) * df[
        "above_below"
    ]
    df = df.dropna()
    return df["blip"]


def gap_tracer(df: pd.DataFrame, runs: int = 6, gap_freq: int = 1) -> pd.DataFrame:
    """
    Verify consistency of price data df.  Return all points where
    series ends at a non-standard time point or otherwise is suspected
    of missing data.

    Parameters:
    -----------
    runs - number of iterations, each run
    determines the most frequent closing time and treatss all gaps
    starting at those regular times as normal

    gap_freq - for each run, frequency at which gap duration must
    occur to be treated as normal

    """
    df = df.copy()
    df["timestamp"] = df.index
    df["gap"] = df.timestamp.diff()
    df["gap_bool"] = df["gap"] > df["gap"].mode()[0]
    df["from"] = df["timestamp"].shift()
    # all gaps in timeseries
    gaps = df[df["gap_bool"]]

    # non standard gaps
    out = pd.DataFrame({"from": gaps["from"], "to": gaps["timestamp"]}).reset_index(
        drop=True
    )
    out["duration"] = out["to"] - out["from"]
    out = out[1:]

    out["from_time"] = out["from"].apply(lambda x: x.time())

    # most frequent time cutoff (end of day)
    def time_cut(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
        df = df.copy()
        cutoff_time = df["from_time"].mode()[0]
        gapless = df[(df["from_time"] != cutoff_time)].reset_index(drop=True)
        return gapless, cutoff_time

    # non standard gap duration based on frequency for every time cutoff
    def duration_cut(cf: pd.Timestamp) -> pd.DataFrame:
        duration_counts = out[out["from_time"] == cf].duration.value_counts()
        duration_count_thresholds = set(
            duration_counts[duration_counts > gap_freq].index
        )
        suspicious = out[
            (out["from_time"] == cf)
            & out["duration"].apply(lambda x: x not in duration_count_thresholds)
        ].reset_index(drop=True)
        return suspicious

    cutoffs = []

    non_standard_gaps = out

    for _ in range(runs):
        try:
            non_standard_gaps, cutoff = time_cut(non_standard_gaps)
            cutoffs.append(cutoff)
        except KeyError:
            break

    suspicious = [duration_cut(cf) for cf in cutoffs]
    suspicious.append(non_standard_gaps)
    out_df = pd.concat(suspicious).sort_values("from").reset_index(drop=True)
    del out_df["from_time"]

    return out_df


def rolling_weighted_mean(
    price: pd.Series, weights: pd.Series, periods: int
) -> pd.Series:
    price_vol = price * weights
    return price_vol.rolling(periods).sum() / weights.rolling(periods).sum()


def rolling_weighted_std(
    price: pd.Series,
    weights: pd.Series,
    periods: int,
    weighted_mean: pd.Series | None = None,
) -> pd.Series:
    """weighted_mean can be passed to save one computation"""

    if weighted_mean is None:
        weighted_mean = rolling_weighted_mean(price, weights, periods)

    diff_vol = ((price - weighted_mean) ** 2) * weights
    weighted_var = diff_vol.rolling(periods).sum() / weights.rolling(periods).sum()
    return np.sqrt(weighted_var)  # type: ignore


def weighted_zscore(df: pd.DataFrame, lookback: int) -> pd.Series:
    """
    Weighted z-score. Can be used to test whether price is within/outside
    Bollinger Bands.

    """
    wmean = rolling_weighted_mean(df["close"], df["volume"], lookback)
    wstd = rolling_weighted_std(df["close"], df["volume"], lookback, wmean)
    return ((df["close"] - wmean) / wstd).dropna()


def long_short_returns(r: Results) -> pd.DataFrame:
    """Return df with log returns of long and short positions in r"""
    pos = r.positions
    pos["return"] = np.log(pos["pnl"] / pos["open"].abs())
    pos = pos.set_index("date_c")
    long = pos[pos["open"] > 0]
    short = pos[pos["open"] < 0]
    combined = pd.DataFrame({"long": long["return"], "short": short["return"]})
    combined = (combined + 1).fillna(1).cumprod()
    return combined


def paths(r: Results, cumsum: bool = True, log_return: bool = True) -> pd.DataFrame:
    """Split simulation results into long, short positions, total
    strategy return and underlying instrument return

    Args:
    --------
    cumsum - running sum of all values accross columns, useful for path chart

    log_return - if True logarithmic returns, else absolute values in
    price points

    """

    rdf = r.df.copy()
    price_column = "price" if "price" in rdf.columns else "bar_price"
    if log_return:
        field = "lreturn"
        price = np.log(rdf[price_column].pct_change() + 1)  # type: ignore
    else:
        field = "pnl"
        price = rdf[price_column].diff()  # type: ignore

    if {"transaction", "curr_price"}.issubset(rdf.columns):
        # this is to deal with always-on strategies where transaction is -2 or 2
        # this line will result with np.inf when transaction is zero
        # it's fixed subsequently
        half_return = rdf[field] / rdf["transaction"].abs()
        rdf["_return"] = half_return.mask(
            half_return.replace([-np.inf, np.inf], np.nan).isna(), rdf[field]
        )
        # for 'double' transactions: they are included both in longs and shorts
        # (the size has been halved previously)
        longs = rdf[(rdf["curr_price"] > 0) | (rdf["position"] == 1)]
        shorts = rdf[(rdf["curr_price"] < 0) | (rdf["position"] == -1)]
    else:
        rdf["_return"] = rdf[field]
        direction = rdf["position"].where(
            rdf["position"] != 0, rdf["position"].shift().fillna(0)
        )
        if "open_price" in rdf.columns:
            direction = direction.mask(
                (direction == 0) & (rdf["open_price"] != 0),
                np.sign(rdf["open_price"]),  # type: ignore
            )
        if "close_price" in rdf.columns:
            direction = direction.mask(
                (direction == 0) & (rdf["close_price"] != 0),
                -np.sign(rdf["close_price"]),  # type: ignore
            )
        if "stop_price" in rdf.columns:
            direction = direction.mask(
                (direction == 0) & (rdf["stop_price"] != 0),
                -np.sign(rdf["stop_price"]),  # type: ignore
            )
        longs = rdf[direction > 0]
        shorts = rdf[direction < 0]
    df = pd.DataFrame(
        {
            "price": price,
            "longs": longs["_return"],
            "shorts": shorts["_return"],
            "strategy": rdf[field],
        }
    ).fillna(0)

    if cumsum:
        return df.cumsum()
    else:
        return df


def always_on(series: pd.Series) -> bool:
    """
    Based on passed position series determine whether system is always
    in the market (closing position always means opening opposite position).
    """
    start = min(series.idxmax(), series.idxmin())
    start_index = cast(int, series.index.get_indexer([start])[0])
    return bool(series.iloc[start_index:].eq(0).sum() == 0)


def round_tick(series: pd.Series) -> pd.Series:
    tick = get_min_tick(series)
    floor = series // tick
    remainder = series % tick
    return floor * tick + 1 * (remainder > tick / 2)


def multiply(series: pd.Series, multiplier: float) -> pd.Series:
    """
    Floor multiplied price series to the nearest tick.
    """
    if multiplier != 1:
        series = series * multiplier
    return round_tick(series)
