import matplotlib.pyplot as plt  # type: ignore
import pandas as pd  # type: ignore
import numpy as np
from multiprocessing import Pool, cpu_count  # type: ignore
from typing import Union, Optional, List, Tuple, Set, Sequence

# sys.path.append('/home/tomek/ib_tools')  # noqa


def plot(*data):
    """
    Plot every Series or column of given DataFrame on a separate vertical
    sub-plot.

    Args:
        Must be one or more Series or one DataFrame.
    """
    # split DataFrames into separate Series
    columns = []
    for d in data:
        if isinstance(d, pd.Series):
            columns.append(d)
        elif isinstance(d, pd.DataFrame):
            columns.extend([d[c] for c in d.columns])
        else:
            raise ValueError("Arguments must be Series' or a Dataframe")
    # plot the charts
    fig = plt.figure(figsize=(20, len(columns) * 5))
    num_plots = len(columns)
    for n, p in enumerate(columns):
        if n == 0:
            ax = fig.add_subplot(num_plots, 1, n + 1)
        else:
            ax = fig.add_subplot(num_plots, 1, n + 1, sharex=ax)
        ax.plot(p)
        ax.grid()
        ax.set_title(p.name)
    plt.show()


def chart_price(price_series, signal_series, threshold=0):
    """
    Plot a price chart marking where long and short positions would be,
    given values of signal.
    price_series: instrument prices
    signal_series: indicator based on which signals will be generated
    position will be:
    long for signal_series > threshold
    short for signal_series < -threshold
    """
    chart_data = pd.DataFrame()
    chart_data["out"] = price_series
    chart_data["long"] = (signal_series > threshold) * price_series
    chart_data["short"] = (signal_series < -threshold) * price_series
    chart_data.replace(0, np.nan, inplace=True)
    return chart_data.plot(figsize=(20, 10), grid=True)


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
    ret["mean"] = ret["returns"].rolling(22 * months).mean() * 252
    ret["vol"] = ret["returns"].rolling(22 * months).std() * np.sqrt(252)
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


def crosser(ind: pd.Series, threshold: float) -> pd.Series:
    df = pd.DataFrame({"ind": ind})
    df["above_below"] = (df["ind"] >= threshold) * 1 - (df["ind"] < threshold) * 1
    df["blip"] = ((df["above_below"].shift() + df["above_below"]) == 0) * df[
        "above_below"
    ]
    df = df.dropna()
    return df["blip"]


def gap_tracer(df: pd.DataFrame, runs: int = 6) -> pd.DataFrame:
    """
    Verify consistency of price data df.  Return all points where
    series ends at a non-standard time point.
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
    def cut(df):
        df = df.copy()
        cutoff_time = df["from_time"].mode()[0]
        print(cutoff_time)
        gapless = df[df["from_time"] != cutoff_time].reset_index(drop=True)
        return gapless

    non_standard_gaps = out
    for _ in range(runs):
        non_standard_gaps = cut(non_standard_gaps)

    del non_standard_gaps["from_time"]

    return non_standard_gaps


def chande_ranking(price: pd.Series, lookback: int) -> pd.Series:
    """
    Trend strength ranking indicator. Kaufman book 2013 edition, p. 1069.
    """
    df = pd.DataFrame(index=price.index)
    df["log_return"] = np.log(price.pct_change(lookback) + 1)
    df["one_period_returns"] = np.log(price.pct_change() + 1)
    df["std"] = df["one_period_returns"].rolling(lookback).std()
    return df["log_return"] / (df["std"] * np.sqrt(lookback))


def upsample(
    df: pd.DataFrame,
    dfg: pd.DataFrame,
    keep: Optional[Union[str, Sequence[str]]] = None,
    propagate: Optional[Union[str, Sequence[str]]] = None,
) -> pd.DataFrame:
    """Upsample time series by combining higher frequency and lower frequency dataframes.

    df: higher frequency data, all columns will be kept

    dfg: lower frequency data, only non-overlapping columns will be kept, unless
    columns to keep are given explicitly

    propagate: columns, that will be propagated

    keep: don't propagate these columns; forward-fill NA with with zeros.

    If none of propagate or keep given, all columns will be propagated.

    If only one given, non specified columns will be included in the other group.

    If both propagate and keep specified, but they don't include all upsampled columns,
    non specified columns will be propagated.

    Columns specified in keep and propagate must not overlap.

    """

    def warn_blip(columns: Union[Set[str], List[str]]) -> None:
        """Blips should not be propagated. If any column name contains word 'blip' warn user
        that they may be making a mistake."""
        blip_columns = [b for b in columns if "blip" in b]
        if blip_columns:
            print(f"Warning: blip is being propagated: {blip_columns}")

    def verify(data: Union[Sequence[str], str, Set]) -> List[str]:
        if isinstance(data, (str, int, float)):
            data = [
                data,
            ]
        data = set(data)  # ensure no double entries inserted by user
        if not data.issubset(upsampled_columns):
            raise ValueError(
                f"Cannot upsample: {list(data.difference(upsampled_columns))} "
                f"- not in columns."
            )
        return list(data)

    upsampled_columns = set(dfg.columns) - set(df.columns)
    joined_df = df.join(dfg[upsampled_columns])
    if not (keep or propagate):
        warn_blip(upsampled_columns)
        return joined_df.ffill().dropna()
    elif keep and propagate:
        keep = verify(keep)
        propagate = verify(propagate)
        assert not set(keep).intersection(
            propagate
        ), "Columns in keep and propagate must not overlap."
        propagate.extend(list(set(upsampled_columns) - set(keep) - set(propagate)))
    else:
        if keep:
            keep = verify(keep)
            propagate = list(set(upsampled_columns) - set(keep))
        else:
            assert propagate is not None
            propagate = verify(propagate)
            keep = list(set(upsampled_columns) - set(propagate))
    joined_df[keep] = joined_df[keep].fillna(0)
    joined_df[propagate] = joined_df[propagate].ffill()
    warn_blip(propagate)
    return joined_df.dropna()
