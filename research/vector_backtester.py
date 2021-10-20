import sys

from typing import NamedTuple, List, Union, Optional, Dict, Literal

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from pyfolio.timeseries import perf_stats  # type: ignore # noqa

from signal_converters import sig_pos


sys.path.append('/home/tomek/ib_tools')


def daily_returns_log_based(lreturn: pd.Series) -> pd.DataFrame:
    daily = pd.DataFrame()
    daily['lreturn'] = lreturn.resample('B').sum().dropna()
    daily['returns'] = np.exp(daily['lreturn']) - 1
    daily['balance'] = (daily['returns'] + 1).cumprod()
    return daily


class Positions(NamedTuple):
    positions: pd.DataFrame
    opens: pd.DataFrame
    closes: pd.DataFrame
    transactions: pd.DataFrame


def pos(price: pd.Series,
        transaction: pd.Series,
        position: pd.Series,
        cost: float = 0) -> Positions:
    """
    Match open and close transactions to create position list.

    Returns:
        NameTuple with:
            positions: pd.DataFrame
            opens: pd.DataFrame all open transactions
            closes: pd.DataFrame all close transactions
            transactions: int number of all transactions
    """
    df = pd.DataFrame({'price': price,
                       'transaction': transaction,
                       'position': position})

    df['c_transaction'] = ((df['transaction'] != 0) *
                           (-df['position'].shift(1))).fillna(0)
    df['o_transaction'] = df['transaction'] - df['c_transaction']

    # DataFrame calculating per transaction stats
    _opens = df[df.o_transaction != 0]
    _closes = df[df.c_transaction != 0]
    transactions = len(_opens) + len(_closes)

    # close out open trade (if any)
    if not len(_opens) == len(_closes):
        _closes = _closes.append(df.iloc[-1])
        _closes.c_transaction[-1] = _opens.o_transaction[-1] * -1

    # create positions df
    opens = (_opens['price'] * _opens['o_transaction']).reset_index()
    closes = (_closes['price'] * _closes['c_transaction']).reset_index()
    opens.columns = ['date', 'open']
    closes.columns = ['date', 'close']

    positions = opens.join(closes, how='outer', lsuffix='_o', rsuffix='_c')
    positions['pnl'] = -(positions['open'] + positions['close']) - cost*2
    positions['duration'] = positions['date_c'] - positions['date_o']
    return Positions(positions, opens, closes, transactions)


def get_min_tick(data: pd.Series) -> float:
    """
    Estimate min_tick value from price data.

    Args:
       data: price series
    """
    ps = data.sort_values().diff().abs().dropna()
    ps = ps[ps != 0]
    min_tick = ps.mode()[0]
    # print(f'estimated min-tick: {min_tick}')
    return min_tick


def _perf(price: pd.Series,
          position: pd.Series,
          cost: float) -> pd.DataFrame:
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
    """
    assert price.index.equals(position.index), \
        'Price and position must have the same index'

    df = pd.DataFrame({
        'price': price,
        'position': position,
    })

    df['transaction'] = (df['position'] - df['position'].shift(1)
                         .fillna(0)).astype('int')

    df['slippage'] = df['transaction'].abs() * cost
    if (df.position[-1] != 0):  # & (df.transaction[-1] == 0):
        df.slippage[-1] += np.abs(df.position[-1]) * cost

    df['curr_price'] = (df['position'] - df['transaction']) * df['price']

    df['base_price'] = (df['price'].shift(
        1) * df['position'].shift(1)).fillna(0)
    df['pnl'] = df['curr_price'] - df['base_price'] - df['slippage']
    # however convoluted, probably is correct
    slip_return = np.log((-df['slippage'] / df['price']) + 1).fillna(0)
    price_return = np.log(((df['curr_price'] - df['base_price'])
                           / abs(df['base_price'])) + 1).fillna(0)
    df['lreturn'] = slip_return + price_return

    return df


class Results(NamedTuple):
    stats: pd.Series
    daily: pd.DataFrame
    positions: pd.DataFrame
    df: pd.DataFrame
    opens: pd.Series
    closes: pd.Series


def perf(price: pd.Series,
         position: pd.Series,
         slippage: float = 1.5) -> Results:
    """
    Return performance statistics and underlying data for debuging.

    Args:
    -----

    price - in priciple 'open' price should be used as position is
    assumed to be entered and closed on the next bar after signal was
    generate

    position - what position is held at each bar, possible values (-1,
    0, 1)

    slippage - transaction cost expressed as multiple of min tick

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
    """

    # transaction cost
    cost = get_min_tick(price) * slippage

    # generate returns
    df = _perf(price, position, cost)

    # bar by bar to daily returns
    daily = daily_returns_log_based(df['lreturn'])

    # position stats
    p = pos(df['price'], df['transaction'], df['position'], cost=cost)
    positions = p.positions
    assert round(positions.pnl.sum(), 4) == round(df.pnl.sum(), 4), \
        f'Dubious pnl calcs... {positions.pnl.sum()} vs. {df.pnl.sum()}'

    duration = positions['duration'].mean()
    win_pos = positions[positions['pnl'] > 0]
    # positions with zero gain are loss making
    loss_pos = positions[positions['pnl'] <= 0]
    # =========================================

    # container for all non-pyfolio stats
    stats = pd.Series()
    try:
        stats['Win percent'] = len(win_pos) / len(positions)
        stats['Average gain'] = win_pos.pnl.sum() / len(win_pos)
        stats['Average loss'] = loss_pos.pnl.sum() / len(loss_pos)
        stats['Avg gain/loss ratio'] = abs(stats['Average gain'] /
                                           stats['Average loss'])
        stats['Position EV'] = ((stats['Win percent'] * stats['Average gain'])
                                + ((1 - stats['Win percent'])
                                   * stats['Average loss']))

        # this is sanity check for Positions EV (should be the same)
        # stats['Average pnl'] = positions.pnl.mean()

        stats['Long EV'] = positions[positions['open'] > 0].pnl.mean()
        stats['Short EV'] = positions[positions['open'] < 0].pnl.mean()

        days = daily.returns.count()
        num_pos = len(win_pos) + len(loss_pos)
        stats['Positions per day'] = num_pos/days
        stats['Days per position'] = days/num_pos
        stats['Actual avg. duration'] = duration.round('min')

        stats['Days'] = days
        stats['Positions'] = num_pos
        stats['Trades'] = p.transactions
        stats['Monthly EV'] = int(stats['Positions per day'] *
                                  stats['Position EV'] * 21)
        stats['Annual EV'] = 12 * stats['Monthly EV']
    except (ZeroDivisionError, KeyError):
        pass

    # Generate output table
    pyfolio_stats = perf_stats(daily['returns'])
    stats = pyfolio_stats.append(stats)

    return Results(stats, daily, positions, df, p[1], p[2])


def v_backtester(indicator: pd.Series,
                 threshold: float = 0,
                 signal_or_position: Literal['signal', 'position',
                                             'both'] = 'position'
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

    assert signal_or_position in ('signal', 'position', 'both'), \
        "Acceptable values for 'signal_or_position' are 'signal', 'position',"
    " 'both'"

    df = pd.DataFrame({'indicator': indicator})
    df['signal'] = ((df['indicator'] > threshold) * 1) \
        + ((df['indicator'] < -threshold) * -1)
    df['position'] = (df['signal'].shift(1).fillna(0)).astype('int')
    if signal_or_position == 'position':
        return df['position']
    elif signal_or_position == 'signal':
        return df['signal']
    else:
        return df


Out = NamedTuple('Out', [('stats', pd.DataFrame),
                         ('dailys', pd.DataFrame),
                         ('returns', pd.DataFrame),
                         ('positions', Dict[float, pd.DataFrame]),
                         ('dfs', pd.DataFrame),
                         ])


def summary(data: Union[pd.Series, pd.DataFrame],
            indicator: Optional[pd.Series] = None,
            slip: float = 0,
            threshold: Optional[Union[List, float]] = None,
            price_field_name: str = 'open') -> Out:
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
    assert isinstance(data, (pd.DataFrame, pd.Series)), \
        'Data must be either Series or DataFrame'

    if isinstance(data, pd.DataFrame):
        assert price_field_name in data.columns, \
            "Use 'price_field_name' argument to indicate which column in "
        "'data' contains price"

        price = data[price_field_name]

        if indicator is None:
            try:
                indicator = data.forecast
            except KeyError:
                raise KeyError("Indicator has to be passed directly "
                               "or passed df must have column 'forecast'")

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
            b = v_backtester(indicator, i)
            r = perf(price, b, slippage=slip)
        except ZeroDivisionError:
            continue
        stats[i] = r.stats
        dailys[i] = r.daily.balance
        returns[i] = r.daily['returns']
        positions[i] = r.positions
        dfs[i] = r.df
    return Out(stats, dailys, returns, positions, dfs)


def optimize(df, func, start_param, scope=range(20)):
    results = pd.DataFrame(index=scope, columns=['Sharpe', 'Return', 'Vol'])
    for i in scope:
        param = start_param * 2**i
        data = sig_pos(func(df['close'], param))
        out = perf(df['open'], data)
        results.loc[i, 'Sharpe'] = out[0].loc['Sharpe ratio']
        results.loc[i, 'Return'] = out[0].loc['Annual return']
        results.loc[i, 'Vol'] = out[0].loc['Annual volatility']
    return results
