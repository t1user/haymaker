import numpy as np
import pandas as pd

from .backtester import Results, get_min_tick


def true_sharpe(ret: pd.Series) -> pd.Series:
    """Compare simple-return and log-return Sharpe calculations.

    Args:
        ret: Period returns as simple returns, for example ``P1 / P0 - 1``.

    Returns:
        A series with cumulative return, annualized return, annualized means,
        annualized volatility, and Sharpe ratios calculated from both simple
        and log returns.

    Raises:
        ValueError: If ``ret`` is empty.
    """
    if ret.empty:
        raise ValueError("ret must contain at least one return.")

    r = pd.Series(dtype=float)
    df = pd.DataFrame({"returns": ret.astype(float)})
    df["cumulative_return"] = (df["returns"] + 1).cumprod()
    df["log_returns"] = np.log(df["returns"] + 1)
    r["cumulative_return"] = df["cumulative_return"].iloc[-1] - 1
    r["annual_return"] = ((r["cumulative_return"] + 1) ** (252 / len(df.index))) - 1
    r["mean"] = df["returns"].mean() * 252
    r["mean_log"] = df["log_returns"].mean() * 252
    r["vol"] = df["returns"].std() * np.sqrt(252)
    r["vol_log"] = df["log_returns"].std() * np.sqrt(252)
    r["sharpe"] = r["mean"] / r["vol"]
    r["sharpe_log"] = r["mean_log"] / r["vol_log"]
    return r


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
        # p.set_index(pd.date_range(freq='min', start=data.index[0],
        #                          periods=len(p), name='date'), inplace=True)
        output.append(p)
    return output


def crosser(ind: pd.Series, threshold: float) -> pd.Series:
    """Return threshold-crossing blips for an indicator series.

    Values equal to ``threshold`` are treated as above the threshold. The first
    row is always ``0`` because there is no previous side to cross from.
    """
    side = pd.Series(np.where(ind >= threshold, 1, -1), index=ind.index)
    crossed = (side.shift() + side).eq(0)
    return side.where(crossed, 0).astype(int)


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
    # all gaps in timeseries
    gaps = df[df["gap_bool"]]

    # non standard gaps
    out = pd.DataFrame({"from": gaps["from"], "to": gaps["timestamp"]}).reset_index(
        drop=True
    )
    out["duration"] = out["to"] - out["from"]

    out["from_time"] = out["from"].apply(lambda x: x.time())

    # most frequent time cutoff (end of day)
    def time_cut(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
        df = df.copy()
        cutoff_mode = df["from_time"].mode()
        if cutoff_mode.empty:
            raise KeyError("No gap cutoff available.")
        cutoff_time = cutoff_mode[0]
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
    """Return rolling weighted mean.

    A window whose weight sum is zero returns ``NaN``.
    """
    price_vol = price * weights
    return price_vol.rolling(periods).sum() / weights.rolling(periods).sum()


def rolling_weighted_std(
    price: pd.Series,
    weights: pd.Series,
    periods: int,
    weighted_mean: pd.Series | None = None,
) -> pd.Series:
    """Return rolling population-style weighted standard deviation.

    The denominator is the rolling sum of weights, not a sample-variance
    correction. ``weighted_mean`` can be passed to save one computation.
    """

    if weighted_mean is None:
        weighted_mean = rolling_weighted_mean(price, weights, periods)

    weighted_second_moment = rolling_weighted_mean(price**2, weights, periods)
    weighted_var = (weighted_second_moment - weighted_mean**2).clip(lower=0)
    return np.sqrt(weighted_var)  # type: ignore


def weighted_zscore(df: pd.DataFrame, lookback: int) -> pd.Series:
    """Return volume-weighted z-score of the ``close`` column.

    Args:
        df: Dataframe with ``close`` and ``volume`` columns.
        lookback: Rolling window length.

    Raises:
        ValueError: If required columns are missing or ``lookback`` is invalid.
    """
    missing = {"close", "volume"} - set(df.columns)
    if missing:
        raise ValueError(f"weighted_zscore() missing required columns: {missing}.")
    if lookback < 1:
        raise ValueError("lookback must be at least 1.")

    wmean = rolling_weighted_mean(df["close"], df["volume"], lookback)
    wstd = rolling_weighted_std(df["close"], df["volume"], lookback, wmean)
    return ((df["close"] - wmean) / wstd).dropna()


def long_short_returns(r: Results) -> pd.DataFrame:
    """Return cumulative long-only and short-only trade-return paths.

    This is a hypothetical decomposition of completed trades in
    ``Results.positions``. Long trades are included only in the ``long`` path
    and short trades only in the ``short`` path. Per-trade returns are computed
    as log returns, ``log1p(pnl / abs(open))``, then accumulated and converted
    back to cumulative simple-return paths.
    """
    pos = r.positions.copy()
    pos["return"] = np.log1p(pos["pnl"] / pos["open"].abs())
    pos = pos.set_index("date_c")
    long = pos[pos["open"] > 0]
    short = pos[pos["open"] < 0]
    combined = pd.DataFrame({"long": long["return"], "short": short["return"]})
    return combined.fillna(0).cumsum().apply(np.exp)


def paths(r: Results, cumsum: bool = True, log_return: bool = False) -> pd.DataFrame:
    """Return chart-ready strategy, long, short, and underlying paths.

    This helper is meant for quick visual comparison between the strategy,
    the underlying asset movement, and the parts of strategy performance
    contributed by long and short exposure.

    Args:
        r: Backtester result returned by :func:`haymaker.research.backtester.perf`.
        cumsum: If ``True``, return running sums suitable for a path chart.
            With ``log_return=True``, these are cumulative log returns. With
            ``log_return=False``, these are cumulative price-point/PnL values.
        log_return: If ``True``, use bar log returns from the strategy and
            underlying asset. If ``False``, use absolute price-point movement
            for the underlying and absolute PnL for the strategy.

    Returns:
        DataFrame with four columns:

        - ``price``: movement of the underlying asset.
        - ``longs``: strategy bar result attributed to long exposure.
        - ``shorts``: strategy bar result attributed to short exposure.
        - ``strategy``: total strategy bar result.

        For current ``perf()`` output, exit rows are attributed to the side
        being closed. This matters on reversal bars, where ``position`` already
        contains the new side but the closing mark-to-market PnL belongs to the
        previous side.
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
        direction = pd.Series(0.0, index=rdf.index)
        direction = direction.mask(rdf["position"] != 0, rdf["position"])
        if "open_price" in rdf.columns:
            direction = direction.mask(
                rdf["open_price"] != 0,
                np.sign(rdf["open_price"]),  # type: ignore
            )
        if "close_price" in rdf.columns:
            direction = direction.mask(
                rdf["close_price"] != 0,
                -np.sign(rdf["close_price"]),  # type: ignore
            )
        if "stop_price" in rdf.columns:
            direction = direction.mask(
                rdf["stop_price"] != 0,
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
    """Return whether a position series stays in-market after first entry.

    Leading flat rows are ignored. A series with only zero positions, or an
    empty series, returns ``False``.
    """
    non_zero_positions = np.flatnonzero(series.ne(0).to_numpy())
    if len(non_zero_positions) == 0:
        return False
    return bool(series.iloc[non_zero_positions[0] :].ne(0).all())


def round_tick(series: pd.Series) -> pd.Series:
    tick = get_min_tick(series)
    if tick == 0:
        return series.copy()
    floor = series // tick
    remainder = series % tick
    return floor * tick + tick * (remainder > tick / 2)
