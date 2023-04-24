from functools import singledispatchmethod
from typing import Dict, List, Literal, NamedTuple, Optional, Tuple, Union

import numpy as np
import pandas as pd  # type: ignore
from pyfolio.timeseries import perf_stats  # type: ignore

from ib_tools.research.signal_converters import sig_pos

# from scipy import stats as scipy_stats  # type: ignore

# sys.path.append("/home/tomek/ib_tools")


def daily_returns_log_based(lreturn: pd.Series) -> pd.DataFrame:
    """
    Create daily simple (non-log) returns from log returns at every bar.
    Columns other than 'returns' is just fucking around. Simple daily
    returns are needed for pyfolio stats.
    """
    daily = pd.DataFrame()
    daily["lreturn"] = lreturn.resample("B").sum().dropna()
    daily["returns"] = np.exp(daily["lreturn"]) - 1
    daily["balance"] = (daily["returns"] + 1).cumprod()
    return daily


def daily_returns_pnl_based(price_pnl: pd.DataFrame) -> pd.DataFrame:
    """
    Daily returns from absolute pnl and price values.
    There is no mark-to-market for open positions: i.e. full pnl recognized
    at the moment position is closed.
    """
    daily = price_pnl.resample("B").agg({"price": "first", "pnl": "sum"}).dropna()
    assert isinstance(daily, pd.DataFrame)
    daily["returns"] = daily["pnl"] / daily["price"]
    daily["balance"] = (daily["returns"] + 1).cumprod()
    return daily


def extract_open_stop_positions(df: pd.DataFrame) -> pd.DataFrame:
    op_s = df[df["open_stop"] != 0]
    op_s = op_s[["open_price", "stop_price"]]
    op_s = op_s.reset_index()
    op_s["date_o"] = op_s["date"]
    op_s["date_c"] = op_s["date"]
    del op_s["date"]
    op_s = op_s.rename(columns={"open_price": "open", "stop_price": "close"})
    order = ["date_o", "open", "date_c", "close"]
    op_s = op_s.reindex(columns=order)
    return op_s


# this can be used for debugging
class Positions(NamedTuple):
    positions: pd.DataFrame
    opens: pd.DataFrame
    closes: pd.DataFrame
    open_stop_positions: Optional[pd.DataFrame] = None
    df: Optional[pd.DataFrame] = None


def pos(
    price: pd.Series,
    transaction: pd.Series,
    position: pd.Series,
    cost: float = 0,
    open_stop_positions: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Match open and close transactions to create position list.

    Returns:
        NameTuple with:
            positions: pd.DataFrame
            opens: pd.DataFrame all open transactions
            closes: pd.DataFrame all close transactions
            transactions: int number of all transactions
    """
    df = pd.DataFrame(
        {
            "price": price,
            "transaction": transaction,
            "position": position,
        }
    )

    df["c_transaction"] = ((df["transaction"] != 0) * (-df["position"].shift())).fillna(
        0
    )
    df["o_transaction"] = df["transaction"] - df["c_transaction"]

    # DataFrame calculating per transaction stats
    _opens = df[df.o_transaction != 0]
    _closes = df[df.c_transaction != 0]

    # close out final open position (if any)
    if not len(_opens) == len(_closes):
        _closes = _closes.append(df.iloc[-1])  # type: ignore
        _closes.c_transaction[-1] = _opens.o_transaction[-1] * -1

    # create positions df
    opens = (_opens["price"] * _opens["o_transaction"]).reset_index()
    closes = (_closes["price"] * _closes["c_transaction"]).reset_index()
    opens.columns = pd.Index(["date", "open"])
    closes.columns = pd.Index(["date", "close"])

    positions = opens.join(closes, how="outer", lsuffix="_o", rsuffix="_c")
    if open_stop_positions is not None:
        positions = (
            pd.concat([positions, open_stop_positions])
            .sort_values("date_o")
            .reset_index(drop=True)
        )

    positions["g_pnl"] = -(positions["open"] + positions["close"])
    positions["pnl"] = positions["g_pnl"] - cost * 2
    positions["duration"] = positions["date_c"] - positions["date_o"]

    # return Positions(
    #         positions, opens, closes, transactions, open_stop_positions, df
    #     )

    return positions


def get_min_tick(data: pd.Series) -> float:
    """
    Estimate min_tick value from price data.

    Args:
       data: price series
    """
    ps = data.sort_values().diff().abs().dropna()
    ps = ps[ps != 0]
    min_tick = ps.mode().iloc[0]
    # print(f'estimated min-tick: {min_tick}')
    return min_tick


def _skip_last_open(df: pd.DataFrame) -> pd.DataFrame:
    """
    If df ends in an open position, get rid of the last open position.
    """
    if df["position"].iloc[-1] == 0:
        return df
    position = df["position"]
    # index of last transaction
    i = position[position.shift() != position].index[-1]
    assert isinstance(i, int)
    df = df[:i]  # .iloc[:-1]
    df["position"].iloc[-1] = 0
    return df


def _perf(
    price: pd.Series,
    position: pd.Series,
    cost: float,
    ocs: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Convert price, position and transaction cost information into
    logarithmic returns.

    Args:
    -----

    price - transacions will be executed at this price on the bar when
    position value changes

    position - indicates what position is held at each bar, should be
    changed one bar after signal is generated, but this is
    resposibility of function that generates position.

    cost - transaction cost expressed in price points

    skip_last_open -

    ocs - open, close, stop prices returned from stop function
    """

    assert price.index.equals(
        position.index
    ), "Price and position must have the same index"

    df = pd.DataFrame(
        {
            "price": price,
            "position": position,
        }
    )

    df["position"] = df["position"].astype(int)
    df["transaction"] = (df["position"] - df["position"].shift().fillna(0)).astype(int)
    df["slippage"] = df["transaction"].abs() * cost

    df["curr_price"] = (df["position"] - df["transaction"]) * df["price"]
    df["base_price"] = (df["price"].shift(1) * df["position"].shift()).fillna(0)
    df["g_pnl"] = df["curr_price"] - df["base_price"]

    if ocs is not None:
        df = df.join(ocs)
        df["open_stop"] = -(
            (ocs[["open_price", "stop_price"]].sum(axis=1))
            * (ocs[["open_price", "stop_price"]].astype(bool).all(axis=1))
        ).fillna(0)
        df["stop_adj"] = (
            (df["price"] - df["stop_price"].abs())
            * np.sign(df["stop_price"])
            * (df["stop_price"] != 0)
            * (df["open_stop"] == 0)
        )
        df["t_count"] = ocs.astype(bool).sum(axis=1)
        # overriding slippage on purpose
        df["slippage"] = df["t_count"] * cost
        df["g_pnl"] += df["stop_adj"] + df["open_stop"]

    # final open positions must account for closing cost
    if df.position[-1] != 0:
        df.slippage[-1] += np.abs(df.position[-1]) * cost

    df["pnl"] = df["g_pnl"] - df["slippage"]

    df["lreturn"] = np.log(
        (df["pnl"] / df["price"].shift()) + 1  # type: ignore
    ).fillna(0)

    return df


class Results(NamedTuple):
    stats: pd.Series
    daily: pd.DataFrame
    positions: pd.DataFrame
    df: pd.DataFrame
    warnings: List[str]
    opens: Optional[pd.DataFrame] = None
    closes: Optional[pd.DataFrame] = None
    open_stop_positions: Optional[pd.DataFrame] = None
    pdf: Optional[pd.DataFrame] = None


def efficiency(price: pd.Series, strategy_return: float) -> float:
    market_return = abs(price[-1] / price[0] - 1)
    return strategy_return / market_return


def efficiency_1(price: pd.Series, strategy_return: float) -> float:
    market_return = abs(price.max() / price.min())
    return strategy_return / market_return


def efficiency_2(price: pd.Series, strategy_return: float) -> float:
    open = price[0]
    close = price[-1]
    high = price.max()
    low = price.min()
    distance = (
        max(abs((open - high)), abs((open - low)))
        + min(abs((close - high)), abs((close - low)))
        + (high - low)
        - abs(close - open)
    )
    market_return = distance / open
    return strategy_return / market_return


def duration_warning(
    positions: pd.DataFrame, full_df: pd.DataFrame
) -> Tuple[float, float]:
    """
    Calculate the percentage of positions with duration of 0 and 1
    candle, return results as a tuple.
    """

    indices = pd.DataFrame(
        {"n": np.arange(1, len(full_df.index) + 1, 1)}, index=full_df.index
    )
    locations = positions[["date_o", "date_c"]]
    locations["i_o"] = indices.loc[locations["date_o"], "n"].reset_index(drop=True)
    locations["i_c"] = indices.loc[locations["date_c"], "n"].reset_index(drop=True)
    locations["duration"] = locations["i_c"] - locations["i_o"]
    return (
        locations[locations["duration"] == 0].duration.count()
        / locations["duration"].count(),
        locations[locations["duration"] == 1].duration.count()
        / locations["duration"].count(),
    )


def last_open_position_warning(
    positions: pd.DataFrame, threshold: float
) -> Optional[str]:
    if len(positions) == 0:
        return "No positions"
    pnl_frac = np.abs(positions.iloc[-1].pnl / positions.pnl.sum())
    durations = positions["date_c"] - positions["date_o"]
    average = durations.iloc[:-1].mean()
    last = durations.iloc[-1]
    mean_time_frac = last / average
    if pnl_frac > threshold:
        message = (
            f"Warning: last open position represents {pnl_frac:.1%} of total pnl "
            f"and is {mean_time_frac:.1f}x average duration ({last} vs {average})"
        )
        return message
    return None


# def probabilistic_sharpe(
#     sharpe: float, skew: float, kurtosis: float, sr_benchmark: float = 0.0
# ) -> float:
#     """Not in use. Formula needs verification as results are rubish."""
#     # std of the sharpe ratio estimation
#     sr_std = np.sqrt(
#         (
#             1
#             + (0.5 * sharpe**2)
#             - (skew * sharpe)
#             + ((kurtosis / 4) * sharpe**2)  # removed -3 as this is Fischer's already
#         )
#         / 251  # 252 annual observations minus one (degrees of freedom)
#     )
#     return scipy_stats.norm.cdf((sharpe - sr_benchmark) / sr_std)


def generate_positions(open_close: pd.DataFrame) -> pd.Series:
    """
    Use df with open, close, stop prices that came from stop-loss
    to generate resulting positions as of end of every bar.
    """
    df = open_close[["open_price", "close_price", "stop_price"]]
    posi = np.sign(df).astype(int)
    posi_change = posi.sum(axis=1)
    posi_end = posi_change.cumsum()
    return posi_end


def pnl_from_stops(df: pd.DataFrame, price: pd.Series) -> float:
    """
    Given object generated by stop-loss function, calculate total
    gross (before fees and slippage) pnl.
    """
    from_stop = df[["open_price", "close_price", "stop_price"]]
    counts = from_stop.astype(bool).sum()
    total = from_stop.sum(axis=1).sum()

    # deal with potentilly unclosed position
    if counts["open_price"] != (counts["close_price"] + counts["stop_price"]):
        total += price.iloc[-1] * -np.sign(
            df[df["open_price"] != 0].open_price.iloc[-1]
        )
        counts["close_price"] += 1

    assert counts["open_price"] == (counts["close_price"] + counts["stop_price"]), (
        "Stop function generated a dataframe, where number of open transactions "
        "is not equal to number of closing transactions."
    )
    return -total


class _Data:
    """Helper class to validate data for perf function."""

    position: pd.DataFrame
    open_close: Optional[pd.DataFrame] = None

    @singledispatchmethod  # type: ignore
    def __init__(self, position_or_stop_price) -> None:
        raise TypeError(
            "Parameter `position_or_stop_price` must be a Series with position or "
            "Dataframe from stop-function with columns: `open_price`, `close_price`, "
            "`stop_price`."
        )

    @__init__.register
    def __df(self, position_or_stop_price: pd.DataFrame) -> None:
        if not set(["open_price", "close_price", "stop_price"]).issubset(
            set(position_or_stop_price.columns)
        ):
            raise ValueError(
                "DataFrame coming from must have columns: "
                "`open_price`, `close_price`, `stop_price`"
            )
        self.position = generate_positions(position_or_stop_price)  # type: ignore
        if "position" in position_or_stop_price:
            assert (position_or_stop_price["position"] == self.position).all(), (
                "Stop loss generated wrong data; open, close, stop prices "
                "imply different positions than positions returned by stop-loss "
                "function."
            )
            position_or_stop_price = position_or_stop_price[
                ["open_price", "close_price", "stop_price"]
            ]
            self.open_close = position_or_stop_price

    @__init__.register
    def __series(self, position_or_stop_price: pd.Series) -> None:
        self.position = position_or_stop_price  # type: ignore

    @property
    def data(self):
        assert (
            max(self.position.max(), np.abs(self.position.min())) <= 1
        ), f"Wrong position data: {self.position[self.position.abs() > 1]}."
        return self.position, self.open_close


def get_end_date(position: pd.Series):
    if position.iloc[-1] != 0:
        return position[position != position.shift()].index[-1]
    else:
        return position.index[-1]


def perf(
    price: pd.Series,
    position_or_stop_price: Union[pd.DataFrame, pd.Series],
    slippage: float = 1.5,
    skip_last_open: bool = False,
    raise_exceptions: bool = True,
    **kwargs,
) -> Results:
    """Return performance statistics and underlying data for debuging.

    Args:
    -----

    price - in priciple 'open' price should be used as position is
    assumed to be entered and closed on the next bar after signal was
    generated

    position - position held at each bar (after), possible values (-1, 0, 1)

    slippage - transaction cost expressed as multiple of min tick

    skip_last_open - final unclosed position might represent a large
    fraction of returns while not really being the result of the
    strategy - when finally closed according to strategy rules, most
    of the pnl can be gone; if False, last open transaction will be
    assumed to be closed at the last available price point

    Returns:
    --------

    Named tuple with output DataFrames with the following properties:

    stats - summary of stats for the backtest

    daily - daily returns (includes closed positions and
    mark-to-market for open positions)

    positions - closed positions with data regarding open, close,
    durations and pnl (pd.DataFrame)

    df - source data with intermediate calculation steps (for
    debugging)

    opens - open transactions (pd.Series)

    closes - close transactions (pd.Series)

    warnings - list of generated warnings

    """
    warnings = []

    if not isinstance(price, pd.Series):
        raise TypeError(f"price must be a pd.Series, not {type(price)}")
    elif len(price) == 0:
        raise ValueError("Price series is empty.")

    # open_close is price data returned from stop-loss and None if stop loss
    # was not used
    position, open_close = _Data(position_or_stop_price).data

    if len(position[position != 0]) == 0:
        warnings.append("No positions")

    # transaction cost
    min_tick = get_min_tick(price)
    cost = min_tick * slippage

    df = _perf(price, position, cost, open_close)

    if open_close is None:
        open_stop_positions = None
        price = df["price"]
    else:
        try:
            from_stops = pnl_from_stops(open_close, price)
            from_df = df["g_pnl"].sum()
            assert (
                abs(from_stops - from_df) < 1
            ), f"pnl from stops: {from_stops}, pnl: {from_df}"
        except AssertionError as e:
            if raise_exceptions:
                raise
            else:
                warnings.append(str(e))

        open_stop_positions = extract_open_stop_positions(df)

        # it doesn't seem to matter if I do it the complicated way here
        # or the simple way below
        stop = (df["stop_price"] != 0) & (df["open_stop"] == 0)
        close = (df["stop_price"] != 0) & (df["open_stop"] != 0)
        other = df["stop_price"] == 0
        price = (
            df["stop_price"].mask(~stop, 0)
            + df["price"].mask(~close, 0)
            + df["price"].mask(~other, 0)
        ).abs()

        # price = (
        #     ((df["stop_price"] != 0) * (df["open_stop"] == 0) * df["stop_price"])
        #     + ((df["stop_price"] != 0) * (df["open_stop"] != 0) * df["price"])
        #     + ((df["stop_price"] == 0) * df["price"])
        # ).abs()

    # position stats
    p = pos(price, df["transaction"], df["position"], cost, open_stop_positions)

    # positions = p.positions
    positions = p

    try:
        # exact comparision (==) causes floating arithmetics issues
        assert abs(positions.pnl.sum() - df.pnl.sum()) < 0.01, (
            f"Dubious pnl calcs... from positions: {positions.pnl.sum()} "
            f"vs. from df: {df.pnl.sum()}"
        )
    except AssertionError as e:
        if raise_exceptions:
            raise
        else:
            warnings.append(str(e))

    if skip_last_open:
        # this is not positions but position!
        end_date = get_end_date(position)
        df = df.loc[:end_date]
        positions = positions[positions["date_o"] < end_date]
    else:
        # Warn if final unclosed position doesn't seem to make sense
        warn = last_open_position_warning(positions, 0.3)
        if warn is not None:
            warnings.append(warn)

    try:
        assert abs(df.g_pnl.sum() - positions.g_pnl.sum()) < 0.01, (
            f"gross pnl from positions: {positions.g_pnl.sum()}, "
            f"gross pnl from df: {df.g_pnl.sum()}"
        )
    except AssertionError as e:
        if raise_exceptions:
            raise
        else:
            warnings.append(str(e))

    # bar by bar to daily returns
    daily = daily_returns_log_based(df["lreturn"])

    # this is left in for potential debugging
    # differences between the two methods are minuscule
    # daily = daily_returns_pnl_based(df[["pnl", "price"]])

    duration = positions["duration"].mean()

    win_pos = positions[positions["pnl"] > 0]
    # positions with zero gain are loss making
    loss_pos = positions[positions["pnl"] <= 0]
    # =========================================

    # container for all non-pyfolio stats
    stats = pd.Series(dtype="O")
    try:
        # np.float64 and np.int64 can be divided by zero
        stats["Win ratio"] = win_pos.pnl.count() / len(positions)
        stats["Average gain"] = win_pos.pnl.sum() / win_pos.pnl.count()
        stats["Average loss"] = loss_pos.pnl.sum() / loss_pos.pnl.count()
        stats["Profit factor"] = abs(stats["Average gain"] / stats["Average loss"])
        # stats["Position EV"] = (stats["Win percent"] * stats["Average gain"]) + (
        #    (1 - stats["Win percent"]) * stats["Average loss"]
        # )

        stats["Position EV"] = positions.pnl.mean()
        stats["Position EV in ticks"] = stats["Position EV"] / min_tick

        stats["Long EV"] = positions[positions["open"] > 0].pnl.mean()
        stats["Short EV"] = positions[positions["open"] < 0].pnl.mean()
        sorted_gains = positions["pnl"].sort_values().values
        if len(sorted_gains) > 0:
            stats["Best trade"] = sorted_gains[-1]
            stats["Worst trade"] = sorted_gains[0]
            stats["Best trade as % of pnl"] = sorted_gains[-1] / positions["pnl"].sum()
            stats["Worst trade as % of pnl"] = sorted_gains[0] / positions["pnl"].sum()
        days = daily.returns.count()
        num_pos = win_pos.pnl.count() + loss_pos.pnl.count()
        stats["Positions per day"] = num_pos / days
        stats["Days per position"] = days / num_pos
        # duration is a pd.Timedelta rounded to the closes minute
        stats["Actual avg. duration"] = duration.round("min")  # type: ignore

        stats["Days"] = days
        stats["Positions"] = num_pos
        # stats["Trades"] = len(positions)*2
        stats["Monthly EV"] = stats["Positions per day"] * stats["Position EV"] * 21
        stats["Annual EV"] = 12 * stats["Monthly EV"]

    except (KeyError, ValueError) as error:
        warnings.append(str(error))
        raise

    # Generate output table
    pyfolio_stats = perf_stats(daily["returns"])

    stats["Efficiency"] = efficiency(price, pyfolio_stats["Cumulative returns"])
    stats["Efficiency_1"] = efficiency(price, df["pnl"].sum() / df["price"].iloc[0])
    stats["Efficiency_2"] = efficiency_1(price, pyfolio_stats["Cumulative returns"])
    stats["Efficiency_3"] = efficiency_2(price, pyfolio_stats["Cumulative returns"])

    # Annual return from pyfolio is with annual capitalization
    # Here return is without capitalization
    year_frac = (df.index[-1] - df.index[0]) / pd.Timedelta(days=365)  # type: ignore
    stats["Simple annual return"] = (df["pnl"].sum() / df["price"].iloc[0]) / year_frac

    stats["Position Sharpe"] = stats["Position EV"] / positions.pnl.std()
    stats["Annualized position Sharpe"] = stats["Position Sharpe"] * np.sqrt(
        stats["Positions per day"] * 252
    )

    stats = pd.concat([pyfolio_stats, stats])

    # stats["Probabilistic Sharpe"] = probabilistic_sharpe(
    #    stats["Sharpe ratio"], stats["Skew"], stats["Kurtosis"]
    # )

    warning = duration_warning(positions, df)
    for i, w in enumerate(warning):
        if w > 0.05:
            warnings.append(f"Warning: {w:.1%} positions with duration of {i} candle.")

    return Results(
        stats,
        daily,
        positions,
        df,
        warnings,
        # p.open_stop_positions,
        # p.df,
        # p.opens,
        # p.closes,
    )


def v_backtester(
    indicator: pd.Series,
    threshold: float = 0,
    signal_or_position: Literal["signal", "position", "both"] = "position",
) -> Union[pd.Series, pd.DataFrame]:
    """
    Vector backtester.

    Run a quick and dirty backtest for a system that goes long when
    indicator is above threshold and short when indicator is below
    minus threshold.  Commissions and spreads are not accounted for.
    Transactions are assumed to be executed on the next price point
    after trade signal is generated.  Positions are marked to market.

    Args:
    -----

    indicator: Series with indicator values moving around zero, same
    index as price

    threshold: strategy goes long when indicator above threshold,
    short when price below minus threshold

    signal_or_position: what columns are to be returned in the
    resulting df

    Returns:
    --------

    Series with 'signal' or 'position' or DataFrame with 'indicator' ,
    'signal', and 'position' depending on the value of argument
    'signal_or_position'
    """

    assert signal_or_position in (
        "signal",
        "position",
        "both",
    ), "Acceptable values for 'signal_or_position' are 'signal', 'position',"
    " 'both'"

    df = pd.DataFrame({"indicator": indicator})
    df["signal"] = ((df["indicator"] > threshold) * 1) + (
        (df["indicator"] < -threshold) * -1
    )
    df["position"] = (df["signal"].shift(1).fillna(0)).astype("int")
    if signal_or_position == "position":
        return df["position"]
    elif signal_or_position == "signal":
        return df["signal"]
    else:
        return df


Out = NamedTuple(
    "Out",
    [
        ("stats", pd.DataFrame),
        ("dailys", pd.DataFrame),
        ("returns", pd.DataFrame),
        ("positions", Dict[float, pd.DataFrame]),
        ("dfs", Dict[float, pd.DataFrame]),
    ],
)


def summary(
    data: Union[pd.Series, pd.DataFrame],
    indicator: Optional[pd.Series] = None,
    slip: float = 0,
    threshold: Optional[Union[List, float]] = None,
    price_field_name: str = "open",
) -> Out:
    """
    Return stats summary of strategy for various thresholds run on the
    indicator.  The strategy is long when indicator > threshold and
    short when indicator < -threshold.

    Trades are executed on the given price.  Price and indicator must
    have the same index.

    In principle, signals should be generated on 'close' price and
    transactions executed on next bar's 'open'.  Passed indicator is
    assumed to have been generated on 'close' price so next bar's
    'open' is used for transactions.  If required a different price
    can be passed as a Series or a specific column can be selected
    from passed DataFrame via 'price_field_name' argument

    Args:
    -----

    data: if Series, it's price used to execute transactions, if
    DataFrame may contain also indicator

    slip: transaction cost expressed a multiples of min tick

    threshold: simulation run assuming long position held when
    indicator > threshold, short positions when indicator <
    -threshold; if given as a list of values, simulation will be run
    for every element of the list, if nothing passed, simulation run
    for a range of values

    price_field_name: if data is DataFrame, which column to use as
    price; ignored if data given as Series

    Returns:
    --------

    Named tuple with stats and underlying simulation results.
    """
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise ValueError("Data must be either Series or DataFrame")

    if isinstance(data, pd.DataFrame):
        if not (price_field_name in data.columns):
            raise ValueError(
                "Use 'price_field_name' argument to indicate which column in "
                "'data' contains price"
            )

        price = data[price_field_name]

        if indicator is None:
            try:
                indicator = data.forecast
            except KeyError:
                raise KeyError(
                    "Indicator has to be passed directly "
                    "or passed df must have column 'forecast'"
                )
    elif isinstance(data, pd.Series):
        price = data

    if threshold is None:
        threshold = [0, 1, 3, 5, 6, 10, 15, 17, 19, 20]
    elif isinstance(threshold, (int, float)):
        threshold = [threshold]

    stats = pd.DataFrame()
    dailys = pd.DataFrame()
    returns = pd.DataFrame()
    positions = {}
    dfs = {}
    for i in threshold:
        try:
            assert indicator is not None
            b = v_backtester(indicator, i)
            assert isinstance(b, pd.Series)
            r = perf(price, b, slippage=slip)
        except ZeroDivisionError:
            continue
        stats[i] = r.stats
        dailys[i] = r.daily.balance
        returns[i] = r.daily["returns"]
        positions[i] = r.positions
        dfs[i] = r.df
    return Out(stats, dailys, returns, positions, dfs)


def optimize(df, func, start_param, scope=range(20)):
    results = pd.DataFrame(index=scope, columns=["Sharpe", "Return", "Vol"])
    for i in scope:
        param = start_param * 2**i
        data = sig_pos(func(df["close"], param))
        out = perf(df["open"], data)
        results.loc[i, "Sharpe"] = out[0].loc["Sharpe ratio"]
        results.loc[i, "Return"] = out[0].loc["Annual return"]
        results.loc[i, "Vol"] = out[0].loc["Annual volatility"]
    return results


def excursions(
    high_low: pd.DataFrame, positions: pd.DataFrame, divisor: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Calculate maximum adverse and favourable excursions.

    Arguments:
    ---------

    high_low: dataframe with high and low prices

    positions: dataframe with positions returned from perf function

    divisor: number that favourable and adverse excursions will be
    divided by; typically atr or other volatility indicator, allows to
    express results in terms of vol multiples; it's callers
    resposibility to ensure that open price is alligned with divisor
    that would be available at that point in time


    Returns:
    -------

    Dataframe with columns:

    pnl_mul: [available only if divisor is not None], pnl exressed as
    multiple of divisor

    fav: maximum favourable excursion

    adv: maximum adverse excursion

    eff: efficiency; ratio of position pnl to difference between
    highest high and lowest low during position's time in the market
    """

    if not isinstance(high_low, pd.DataFrame):
        raise ValueError(f"high_low must be a pandas series not {type(high_low)}")
    if not set(["high", "low"]).issubset(set(high_low.columns)):
        raise ValueError("high_low must have columns named 'high' and 'low'")

    high_low = high_low.copy()

    if divisor is None:
        high_low["divisor"] = 1
    elif isinstance(divisor, pd.Series) and (divisor.index == high_low.index).all():
        high_low["divisor"] = divisor
    else:
        raise ValueError(
            "divisor, if given must be a pd.Series with the same index as high_low"
        )

    def extremes(
        high_low: pd.DataFrame, positions: pd.DataFrame
    ) -> List[Tuple[float, float, float]]:
        data = []
        for p in positions[["date_o", "date_c"]].itertuples():
            h_l = high_low.loc[p.date_o : p.date_c]
            # last candle shouldn't be covered fully, because might have exited
            # before the extreme value was reached
            # for single bar positions, we assume bar's extremes were
            # maximum excursions
            if len(h_l) > 1:
                h_l = h_l.iloc[:-1]
            data.append((h_l["high"].max(), h_l["low"].min(), h_l["divisor"].iloc[0]))
        return data

    out = positions.join(
        pd.DataFrame(extremes(high_low, positions), columns=["high", "low", "divisor"])
    )
    # here we have to establish wheather closing price was an extreme
    out["close_"] = out["close"].abs()
    out["high"] = (out[["close_", "high"]]).max(axis=1)
    out["low"] = (out[["close_", "low"]]).min(axis=1)

    out["_fav"] = ((out["open"] > 0) * out["high"]) + ((out["open"] < 0) * out["low"])
    out["_adv"] = ((out["open"] > 0) * out["low"]) + ((out["open"] < 0) * out["high"])

    out["fav"] = ((out["open"].abs() - out["_fav"]).abs() / out["divisor"]).round(2)
    out["adv"] = ((out["open"].abs() - out["_adv"]).abs() / out["divisor"]).round(2)
    out["eff"] = (out["g_pnl"] / (out["high"] - out["low"])).round(2)

    if divisor is None:
        return out[["fav", "adv", "eff"]]
    else:
        out["pnl_mul"] = (out["g_pnl"] / out["divisor"]).round(2)
        return out[["pnl_mul", "fav", "adv", "eff"]]


def profitable_excursions(
    df: pd.DataFrame,
    results: Results,
    divisor: Optional[pd.Series] = None,
    full: bool = False,
) -> Union[pd.Series, pd.DataFrame]:
    """
    A shortcut accepting raw dataframe with ohlc prices, result of perf simulation
    and returning data allowing for calculation of adverse excursions.
    """
    positions = results.positions
    exc = positions.join(excursions(df[["high", "low"]], positions, divisor))
    out = exc[exc["pnl"] > 0]
    if full:
        return out
    else:
        return out.adv.describe()


# I know I'm going to call it this even if its not correct
adverse_excursions = profitable_excursions


def blip_extractor(signal: pd.Series) -> pd.Series:
    return (signal.shift() != signal) * signal


def factor_extractor(
    pos: pd.DataFrame,
    df: pd.DataFrame,
    field: Union[str, List[str]],
    shift: bool = True,
) -> pd.DataFrame:
    """Join positions with a column from the original dataframe that produced
    those positions. Helps determine factors that impact the pnl.

    Parameters:
    ----------

    pos:
        dataframe with position list produced by perf function

    df: dataframe with a column with factor data, must have the same
        index as df used to produce the positions

    field:
        name of the column in df, which contains factor data

    shift: whether df should be shifted prior to calculations -
        important to use point in time at which signal was generated
    """
    pos = pos.set_index("date_o")
    if shift:
        df = df.shift()

    if isinstance(field, str):
        field = [field]

    overlap = list(set(pos.columns).intersection(set(field)))
    cols = [c for c in pos.columns if c not in overlap]
    pos = pos[cols]

    return pos.join(df.loc[pos.index, field]).reset_index()
