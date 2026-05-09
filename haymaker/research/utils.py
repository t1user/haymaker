import warnings
from multiprocessing import Pool, cpu_count  # type: ignore
from typing import Callable, List, Literal, Optional, Sequence, Tuple, Union, cast

import numpy as np
import pandas as pd  # type: ignore

from haymaker.indicators import (  # noqa
    combine_signals,
    range_blip,
    range_crosser,
    signal_generator,
    zero_crosser,
)

from .backtester import Results, get_min_tick
from .signal_converters import sig_pos
from .stop import stop_loss


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


def bootstrap(data, start=None, end=None, period_length=3, paths=100, replace=True):
    """
    Generate hypothetical time series by randomly drawing from price data.
    """
    if start:
        data = data.loc[start:]
    if end:
        data = data.loc[:end]

    daily = data.resample("B").first()
    data_indexed = pd.DataFrame(
        {
            "open": data["open"] / data["close"],
            "high": data["high"] / data["close"],
            "low": data["low"] / data["close"],
            "close": data["close"].pct_change(),
            "volume": data["volume"],
            "barCount": data["barCount"],
        }
    )
    data_indexed = data_indexed.iloc[1:]

    days = len(daily.index)
    draws = int(days / period_length)

    d = np.random.choice(daily.index[:-period_length], size=(draws, paths))
    lookup_table = pd.Series(daily.index.shift(period_length), index=daily.index)

    output = []
    for path in d.T:
        p = pd.concat([data_indexed.loc[i : lookup_table[i]].iloc[:-1] for i in path])
        p.set_index(
            pd.date_range(freq="min", start=data.index[0], periods=len(p), name="date"),
            inplace=True,
        )

        p["close"] = (p["close"] + 1).cumprod() * data.iloc[0]["close"]
        o = pd.DataFrame(
            {
                "open": p["open"] * p["close"],
                "high": p["high"] * p["close"],
                "low": p["low"] * p["close"],
                "close": p["close"],
                "volume": p["volume"],
                "barCount": p["barCount"],
            }
        )
        output.append(o)
    return output


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
    def time_cut(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Timestamp]:
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


def upsample(
    hf_df: pd.DataFrame,
    lf_df: pd.DataFrame,
    *,
    label: Literal["left", "right"] = "left",
    sparse: Optional[Union[str, Sequence[str]]] = None,
) -> pd.DataFrame:
    """
    Upsample time series by combining higher frequency and lower
    frequency dataframes.

    hf_df: higher frequency data, all columns will be kept

    lf_df: lower frequency data. Non-overlapping columns from this dataframe
    will be added to ``hf_df``. ``position`` must not be passed in this
    dataframe because it is already executable state; upsample generated
    signals or blips first, then derive ``position`` on the upsampled frame.

    label: how lf_df is labeled. ``"left"`` means each lower-frequency row is
    labeled by the first higher-frequency bar in the group. ``"right"`` means it
    is labeled by the bar where the lower-frequency value becomes known.
    Lower-frequency values are first aligned to the higher-frequency index point
    where they become known. Ordinary lower-frequency values are then
    propagated with forward-fill from that availability point onward.

    sparse: optional column or columns that should remain sparse events instead
    of being propagated. Sparse columns are filled with ``0`` on higher-frequency
    rows where they did not occur. Use this for event-like values, not for state
    or feature values. ``blip`` and ``close_blip`` are always treated as sparse
    events because they are generated at a bar, not state values to forward-fill.
    When either canonical blip column is upsampled, its original lower-frequency
    label is preserved as ``raw_blip`` or ``raw_close_blip`` if that column does
    not already exist. Raw blip provenance columns are also always sparse.

    All other lower-frequency columns are propagated by default.

    """

    def verify_sparse(data: Union[Sequence[str], str]) -> List[str]:
        columns = [data] if isinstance(data, str) else list(data)
        unique_columns = list(dict.fromkeys(columns))
        sparse_set = set(unique_columns)
        if not sparse_set.issubset(upsampled_columns):
            raise ValueError(
                f"Cannot upsample sparse columns: "
                f"{list(sparse_set.difference(upsampled_columns))} - not in columns."
            )
        return unique_columns

    def align(
        hf_df: pd.DataFrame,
        lf_df: pd.DataFrame,
        upsampled_columns: List[str],
        label: Literal["left", "right"],
    ) -> pd.DataFrame:
        """
        Align lower-frequency rows to the higher-frequency index point where
        their values become known.
        """
        if not hf_df.index.is_monotonic_increasing:
            raise ValueError("Higher frequency dataframe index must be sorted.")
        if not lf_df.index.is_monotonic_increasing:
            raise ValueError("Lower frequency dataframe index must be sorted.")
        if label not in ("left", "right"):
            raise ValueError(f"label must be 'left' or 'right', '{label}' given")
        if not upsampled_columns:
            return lf_df[upsampled_columns].iloc[:0]

        high_index = hf_df.index
        low_index = lf_df.index

        if len(high_index) == 0 or len(low_index) == 0:
            return lf_df[upsampled_columns].iloc[:0]

        if label == "left":
            high_locs = np.empty(len(low_index), dtype=np.int64)
            if len(low_index) > 1:
                high_locs[:-1] = (
                    high_index.searchsorted(low_index[1:].to_numpy(), side="left") - 1
                )
            high_locs[-1] = len(high_index) - 1
            rows = np.arange(len(low_index), dtype=np.int64)
            valid = (
                (low_index <= high_index[-1])
                & (high_locs >= 0)
                & (high_index.take(high_locs.clip(min=0)) >= low_index)
            )
        elif label == "right":
            high_locs = high_index.searchsorted(low_index.to_numpy(), side="right") - 1
            rows = np.arange(len(low_index), dtype=np.int64)
            valid = (
                (low_index >= high_index[0])
                & (low_index <= high_index[-1])
                & (high_locs >= 0)
            )
        high_locs = high_locs[valid]
        rows = rows[valid]
        aligned = lf_df.iloc[rows][upsampled_columns].copy()
        aligned.index = high_index.take(high_locs)
        if aligned.index.has_duplicates:
            raise ValueError(
                "Cannot upsample: multiple lower-frequency rows align to the "
                "same higher-frequency index point."
            )
        return aligned

    def difference(columns: List[str], excluded: Sequence[str]) -> List[str]:
        excluded_set = set(excluded)
        return [column for column in columns if column not in excluded_set]

    def union(first: List[str], second: Sequence[str]) -> List[str]:
        seen = set(first)
        combined = list(first)
        for column in second:
            if column not in seen:
                combined.append(column)
                seen.add(column)
        return combined

    if "position" in lf_df.columns:
        raise ValueError(
            "Cannot upsample 'position'. Position is already executable state; "
            "upsample generated signals or blips first, then derive position on "
            "the upsampled dataframe."
        )

    position_like_columns = [
        column for column in lf_df.columns if "position" in column.lower()
    ]
    if position_like_columns:
        warnings.warn(
            "Columns containing 'position' are not treated as executable state by "
            f"upsample: {position_like_columns}. Ensure these are generated "
            "values, not already-shifted positions.",
            UserWarning,
            stacklevel=2,
        )

    upsampled_columns = [
        column for column in lf_df.columns if column not in hf_df.columns
    ]
    blip_columns = [
        column for column in ("blip", "close_blip") if column in upsampled_columns
    ]
    raw_source_columns = [
        column
        for column in ("raw_blip", "raw_close_blip")
        if column in upsampled_columns
    ]
    raw_blip_map = {
        column: f"raw_{column}"
        for column in blip_columns
        if f"raw_{column}" not in hf_df.columns and f"raw_{column}" not in lf_df.columns
    }
    raw_blip_columns = raw_source_columns + list(raw_blip_map.values())
    aligned_columns = difference(upsampled_columns, raw_source_columns)
    # preserve types to be able to cast back into them
    types = lf_df[upsampled_columns].dtypes.to_dict()
    types.update(
        {raw_column: lf_df[column].dtype for column, raw_column in raw_blip_map.items()}
    )

    # ffill and subsequent dropnas depend on lf_df not having n/a's
    if len(lf_df[lf_df.isna().any(axis=1)]) != 0:
        raise ValueError("Lower frequency dataframe (lf_df) must not have n/a values.")

    aligned = align(hf_df, lf_df, aligned_columns, label)
    joined_df = hf_df.join(aligned)
    raw_frames = []
    if raw_source_columns:
        raw_frames.append(lf_df[raw_source_columns])
    if raw_blip_map:
        raw_frames.append(lf_df[list(raw_blip_map)].rename(columns=raw_blip_map))
    if raw_frames:
        joined_df = joined_df.join(pd.concat(raw_frames, axis=1))

    sparse_columns = verify_sparse(sparse) if sparse is not None else []
    sparse_columns = union(sparse_columns, blip_columns + raw_blip_columns)
    propagated_columns = difference(upsampled_columns, sparse_columns)

    joined_df[sparse_columns] = joined_df[sparse_columns].fillna(0)
    joined_df[propagated_columns] = joined_df[propagated_columns].ffill()
    return joined_df.dropna().astype(types)  # type: ignore


def rolling_weighted_mean(
    price: pd.Series, weights: pd.Series, periods: int
) -> pd.Series:
    price_vol = price * weights
    return price_vol.rolling(periods).sum() / weights.rolling(periods).sum()


def rolling_weighted_std(
    price: pd.Series,
    weights: pd.Series,
    periods: int,
    weighted_mean: Optional[pd.Series] = None,
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


def stop_signal(df, signal_func: Callable, /, *func_args, **stop_kwargs) -> pd.Series:
    """Wrapper to allow applying stop loss to any signal function.


    Args are passed to function, kwargs to stop loss.

    THIS SHOULDN'T BE USED WITHOUT FURTHER REVIEW. IT'S PROBABLY WRONG.
    """
    _df = df.copy()
    _df["position"] = sig_pos(signal_func(_df["close"], *func_args))
    stopped = stop_loss(_df, **stop_kwargs)
    assert isinstance(stopped, pd.DataFrame)
    return stopped["position"]


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
                np.sign(rdf["open_price"]),
            )
        if "close_price" in rdf.columns:
            direction = direction.mask(
                (direction == 0) & (rdf["close_price"] != 0),
                -np.sign(rdf["close_price"]),
            )
        if "stop_price" in rdf.columns:
            direction = direction.mask(
                (direction == 0) & (rdf["stop_price"] != 0),
                -np.sign(rdf["stop_price"]),
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


def vector_grouper(
    df: pd.DataFrame,
    number: int,
    field: str = "volume",
    label: Literal["left", "right"] = "left",
) -> pd.DataFrame:
    """
    Alternative (volume) grouper. Vector based. Difference with
    numba_tools grouper is about the treatment of the first/last bar
    in a grouped candle.  This method has a small look-ahead bias,
    i.e. is not usable for anything serious.

    """
    df = df.copy()
    df = df.reset_index(drop=False)
    df["index_"] = (df[field].cumsum() // number).shift().fillna(0)
    return (
        df.groupby("index_")
        .agg(
            {
                "date": "first",
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "barCount": "sum",
            }
        )
        .set_index("date")
    )


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
