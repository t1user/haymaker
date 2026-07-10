"""Market-data sampling, validation, and price-normalization helpers."""

from datetime import time

import numpy as np
import pandas as pd

from .backtester import get_min_tick

__all__ = ["gap_tracer", "round_tick", "sampler"]


def sampler(
    data: pd.DataFrame,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    period_length: int = 25,
    paths: int = 100,
    seed: int | None = None,
) -> list[pd.DataFrame]:
    """Sample random contiguous business-day windows from a dataframe.

    This is a small exploratory helper. For production bootstrap workflows, use
    :mod:`haymaker.research.bootstrap`.

    Args:
        data: Dataframe indexed by timestamps.
        start: Optional inclusive lower bound for the source data.
        end: Optional inclusive upper bound for the source data.
        period_length: Number of business-day anchors in each sampled window.
        paths: Number of sampled windows to return.
        seed: Optional random seed for reproducible samples.

    Returns:
        List of dataframe slices. Each slice starts on a randomly selected
        business-day anchor and ends before the next anchor after
        ``period_length`` business-day anchors.

    Raises:
        TypeError: If ``data`` does not use a ``DatetimeIndex``.
        ValueError: If arguments are invalid or not enough data is available.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("data must use a DatetimeIndex.")
    if not data.index.is_monotonic_increasing:
        raise ValueError("data index must be sorted in increasing order.")
    if period_length < 1:
        raise ValueError("period_length must be at least 1.")
    if paths < 1:
        raise ValueError("paths must be at least 1.")

    if start:
        data = data[data.index >= pd.Timestamp(start)]
    if end:
        data = data[data.index <= pd.Timestamp(end)]
    if data.empty:
        raise ValueError("data is empty after applying start/end filters.")

    daily = data.resample("B").first().dropna(how="all")
    if len(daily.index) <= period_length:
        raise ValueError("not enough business-day anchors for period_length.")

    candidates = daily.index[:-period_length]
    lookup_table = pd.Series(daily.index[period_length:], index=candidates)
    rng = np.random.default_rng(seed)
    selected = rng.choice(candidates.to_numpy(), size=paths)
    output = []
    for i in selected:
        anchor = pd.Timestamp(i)
        p = data.loc[anchor : lookup_table[anchor]].iloc[:-1]
        output.append(p)
    return output


def gap_tracer(df: pd.DataFrame, runs: int = 6, gap_freq: int = 1) -> pd.DataFrame:
    """Return suspicious gaps in timestamp-indexed market data.

    The most common index interval is treated as the normal bar interval. Gaps
    larger than that interval are then grouped by the time of day where the gap
    begins. Repeated gap patterns, such as overnight or weekend breaks, are
    treated as normal; less frequent gaps are returned for review.

    Args:
        df: Dataframe indexed by timestamps.
        runs: Number of repeated start times to treat as normal scheduled gaps.
        gap_freq: Minimum repeated duration count for a scheduled gap pattern.

    Returns:
        DataFrame with ``from``, ``to``, and ``duration`` columns.
    """
    empty = pd.DataFrame(columns=["from", "to", "duration"])
    if len(df.index) < 2:
        return empty

    df = df.copy()
    df["timestamp"] = df.index
    df["gap"] = df.timestamp.diff()
    gap_mode = df["gap"].mode()
    if gap_mode.empty:
        return empty

    df["gap_bool"] = df["gap"] > gap_mode[0]
    df["from"] = df["timestamp"].shift()
    gaps = df[df["gap_bool"]]

    out = pd.DataFrame({"from": gaps["from"], "to": gaps["timestamp"]}).reset_index(
        drop=True
    )
    out["duration"] = out["to"] - out["from"]
    out["from_time"] = out["from"].apply(lambda x: x.time())

    def time_cut(df: pd.DataFrame) -> tuple[pd.DataFrame, time]:
        df = df.copy()
        cutoff_mode = df["from_time"].mode()
        if cutoff_mode.empty:
            raise KeyError("No gap cutoff available.")
        cutoff_time = cutoff_mode[0]
        gapless = df[(df["from_time"] != cutoff_time)].reset_index(drop=True)
        return gapless, cutoff_time

    def duration_cut(cf: time) -> pd.DataFrame:
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


def round_tick(series: pd.Series) -> pd.Series:
    """Round prices to the minimum tick inferred from the series."""
    tick = get_min_tick(series)
    if tick == 0:
        return series.copy()
    floor = series // tick
    remainder = series % tick
    return floor * tick + tick * (remainder > tick / 2)
