from multiprocessing import Pool, cpu_count
import sys
sys.path.append('/home/tomek/ib_tools')  # noqa
from collections import namedtuple
from typing import NamedTuple, List, Union, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyfolio.timeseries import perf_stats

from research.grouper import group_by_volume


from indicators import get_ATR, get_signals  # noqa


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
            raise ValueError('Arguments must be Series or a Dataframe')
    # plot the charts
    fig = plt.figure(figsize=(20, len(columns)*5))
    num_plots = len(columns)
    for n, p in enumerate(columns):
        if n == 0:
            ax = fig.add_subplot(num_plots, 1, n+1)
        else:
            ax = fig.add_subplot(num_plots, 1, n+1, sharex=ax)
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
    chart_data['out'] = price_series
    chart_data['long'] = (signal_series > threshold) * price_series
    chart_data['short'] = (signal_series < -threshold) * price_series
    chart_data.replace(0, np.nan, inplace=True)
    return chart_data.plot(figsize=(20, 10), grid=True)


def daily_returns(pnl, start=0):
    """
    Calculate daily returns from a series of intra-day pnl data.

    Pnl can be given in points or in dollars. Start is either initial
    portfolio value or first price in the series.
    """

    # DataFrame calculating daily returns for pyfolio stats
    data = pd.DataFrame({'pnl': pnl, })
    data['cum_pnl'] = data['pnl'].cumsum()
    data['balance'] = start + data['cum_pnl']
    daily = data['balance'].resample('B').last().fillna(method='ffill')
    daily[daily.index[0]-pd.Timedelta(days=1)] = start
    daily.sort_index(inplace=True)
    daily = pd.DataFrame({'balance': daily})
    daily['returns'] = daily.balance.pct_change().fillna(0)
    daily['path'] = (daily.returns + 1).cumprod() - 1
    return daily


def daily_returns_pct_based(pnl: pd.Series, price: pd.Series) -> pd.DataFrame:
    data = pd.DataFrame({'pnl': pnl, 'price': price})
    daily = data.resample('B').agg({'pnl': 'sum', 'price': 'first'}).dropna()
    daily['returns'] = daily['pnl'] / daily['price']
    daily['balance'] = (daily['returns'] + 1).cumprod()
    return daily


def daily_returns_log_based(df: pd.DataFrame) -> pd.DataFrame:
    data = pd.DataFrame({'pnl': df.pnl, 'price': df.price,
                         'lreturn': df.lreturn})
    daily = data.resample('B').agg({'pnl': 'sum', 'price': 'first',
                                    'lreturn': 'sum'}).dropna()
    daily['returns'] = np.exp(daily['lreturn']) - 1
    daily['balance'] = (daily['returns'] + 1).cumprod()
    return daily


def pos(price: pd.Series,
        transaction: pd.Series,
        position: pd.Series,
        reason: pd.Series = None,
        cost: float = 0) -> NamedTuple:
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
    if reason is not None:
        df['reason'] = reason
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

    if reason is not None:
        closes['reason'] = (_closes.reset_index())['reason']

    positions = opens.join(closes, how='outer', lsuffix='_o', rsuffix='_c')
    positions['pnl'] = -(positions['open'] + positions['close']) - cost*2
    positions['duration'] = positions['date_c'] - positions['date_o']
    Results = namedtuple('Result', 'positions, opens, closes, transactions')
    return Results(positions, opens, closes, transactions)


def compound_pnl(df,  bankroll):
    """
    Given df with 'position' and 'pnl_dollars' determine when position size
    would be multiplied based on account balance. Position size aims to keep
    to stay in similar proportion to balance as it was initially to bankroll.
    """
    df = df.copy()
    df['comp_pnl_dollars'] = 0
    df['size'] = 0
    df['balance'] = bankroll
    df.reset_index(inplace=True)

    for i in df.index:
        if i == df.index[0]:
            continue
        df.loc[i, 'size'] = (
            df.loc[i-1, 'position'] != df.loc[i, 'position']
            & df.loc[i, 'position'] != 0) * round(
            df.loc[i-1, 'balance']/bankroll)

        if df.loc[i, 'size'] == 0:
            df.loc[i, 'size'] = df.loc[i-1, 'size'] * (
                ((df.loc[i-1, 'position'] == df.loc[i, 'position']) &
                 (df.loc[i, 'position'] != 0))
                | (df.loc[i, 'pnl_dollars'] != 0))

        df.loc[i, 'comp_pnl_dollars'] = df.loc[
            i, 'size'] * df.loc[i, 'pnl_dollars']
        df.loc[i, 'balance'] = df.loc[:i, 'comp_pnl_dollars'].sum() + bankroll
    df.set_index('date', inplace=True)
    return df


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


def perf(df: pd.DataFrame,
         multiplier: int = 0,
         bankroll: float = 15000,
         output: bool = True,
         compound: bool = False,
         price_column_name: str = 'price',
         slippage: float = 0) -> NamedTuple:
    """
    Extract performance indicators from simulation done by other functions.

    Args:
        df:         must have columns: 'price', 'position', all the information
                    about when and at what price position is entered and closed
                    is extracted from those two columns
        multiplier: futures multiplier to be used in fixed capital simulation,
                    if not given or zero simulation will be variable capital
                    without leverage
        output:     whether output is to be printed out
        compound:   for fixed capital simulation whether position size should
                    be adjusted based on current balance, ignored for variable
                    capital simulation
        price_column_name: which column in df contains price data
        slippage:   transaction cost expressed as multiple of min-tick

    Returns:
        Named tuple of resulting DataFrames for inspection:
        daily: daily returns (includes closed positions and mark-to-market
               for open positions)
        positions: closed positions
        transactions: source df for positions (for debugging)
        df: source df with signals (for debugging really)
    """
    df = df.copy()
    if price_column_name != 'price':
        df.rename(columns={price_column_name: 'price'}, inplace=True)

    if slippage:
        cost = get_min_tick(df.price) * slippage
    else:
        cost = 0

    df['transaction'] = (df['position'] - df['position'].shift(1)
                         .fillna(0)).astype('int')

    df['slippage'] = df['transaction'].abs() * cost
    if (df.position[-1] != 0):  # & (df.transaction[-1] == 0):
        df.slippage[-1] += np.abs(df.position[-1]) * cost

    df['curr_price'] = (df['position'] - df['transaction']) * df['price']

    df['base_price'] = (df['price'].shift(
        1) * df['position'].shift(1)).fillna(0)
    df['pnl'] = df['curr_price'] - df['base_price'] - df['slippage']

    slip_return = np.log((-df['slippage'] / df['price']) + 1).fillna(0)
    price_return = np.log(((df['curr_price'] - df['base_price'])
                           / abs(df['base_price'])) + 1).fillna(0)
    df['lreturn'] = slip_return + price_return

    # get daily returns
    if multiplier:
        df['pnl_dollars'] = df['pnl'] * multiplier
        if compound:
            c = compound_pnl(
                df[['pnl_dollars', 'position', 'transaction']], bankroll)
            df['size'] = c['size']  # for debugging only
            df['comp_pnl_dollars'] = c['comp_pnl_dollars']  # for debugging
            df['balance'] = c['balance']  # for debugging
            daily = daily_returns(c['comp_pnl_dollars'], bankroll)
        else:
            daily = daily_returns(df['pnl_dollars'], bankroll)
    else:
        daily = daily_returns_log_based(df)

    # get position stats
    if 'reason' in df.columns:
        p = pos(df['price'], df['transaction'],
                df['position'], df['reason'].shift(1), cost=cost)
    else:
        p = pos(df['price'], df['transaction'], df['position'], cost=cost)
    positions = p.positions
    assert round(positions.pnl.sum(), 4) == round(df.pnl.sum(), 4), \
        f'Dubious pnl calcs... {positions.pnl.sum()} vs. {df.pnl.sum()}'

    if multiplier:
        positions['pnl'] = positions['pnl'] * multiplier
    # pnl = positions['pnl'].sum()

    duration = positions['duration'].mean()
    win_pos = positions[positions['pnl'] > 0]
    # positions with zero gain are loss making
    loss_pos = positions[positions['pnl'] <= 0]
    # =========================================

    # container for all non-pyfolio stats
    stats = pd.Series()
    stats['Win percent'] = len(win_pos) / len(positions)
    stats['Average gain'] = win_pos.pnl.sum() / len(win_pos)
    stats['Average loss'] = loss_pos.pnl.sum() / len(loss_pos)
    stats['Avg gain/loss ratio'] = abs(stats['Average gain'] /
                                       stats['Average loss'])
    stats['Position EV'] = ((stats['Win percent'] * stats['Average gain'])
                            + ((1 - stats['Win percent'])
                               * stats['Average loss']))
    days = daily.returns.count()
    num_pos = len(win_pos) + len(loss_pos)
    stats['Positions per day'] = num_pos/days
    stats['Days per position'] = days/num_pos
    stats['Actual avg. duration'] = duration.round('min')

    stats['Days'] = days
    stats['Positions'] = num_pos
    stats['Trades'] = p.transactions
    stats['Monthly EV'] = (stats['Positions per day'] *
                           stats['Position EV'] * 21)
    stats['Annual EV'] = 12 * stats['Monthly EV']

    # Generate output table
    pyfolio_stats = perf_stats(daily['returns'])
    stats = pyfolio_stats.append(stats)
    if output:
        print(stats.to_string())
        daily.path.plot(figsize=(20, 10), grid=True)
        # daily.balance.plot(figsize=(20, 10), grid=True)
    Results = namedtuple(
        'Result', 'stats, daily, positions, df, opens, closes')
    return Results(stats, daily, positions, df, p[1], p[2])


def perf_var(df: pd.DataFrame,
             output: bool = True,
             price_column_name: str = 'price',
             slippage: int = 0):
    """
    Shortcut interface function for perf to generate variable capital
    simulation.

    Args:
        df:         must have columns: 'price', 'position', all the information
                    about when and at what price position if entered and closed
                    is extracted from those two columns
        output:     whether output is to be printed out
        price_column_name: which column in df contains price data

    """

    return perf(df, output=output, price_column_name=price_column_name,
                slippage=slippage)


def v_backtester(price: pd.Series,
                 indicator: pd.Series,
                 threshold: float,) -> pd.DataFrame:
    """
    Vector backtester.

    Run a quick and dirty backtest for a system that goes long
    when indicator is above threshold and short when indicator
    is below minus threshold. Commissions and spreads are not accounted for.
    Transactions are assumed to be executed on the next price point after
    trade signal is generated. Positions are marked to market.

    Args:
        price:       Series with instrument prices,
                     must be indexed with Timestamps
        indicator:   Series with indicator values moving around zero,
                     same index as price
        threshold:   strategy goes long when indicator above threshold,
                     short when price below minus threshold

    Returns:
       DataFame with columns: 'signal', 'position'
    """

    df = pd.DataFrame({'price': price,
                       'indicator': indicator})
    df['signal'] = ((df['indicator'] > threshold) * 1) \
        + ((df['indicator'] < -threshold) * -1)
    df['position'] = (df['signal'].shift(1).fillna(0)).astype('int')
    return df


def c_backtester(data: pd.DataFrame,
                 sl_atr: float = 50,
                 trailing_sl: bool = True,
                 active_close: bool = False,
                 block_stop: bool = True,
                 take_profit: int = 0,
                 ) -> pd.DataFrame:
    """
    Consecutive (event driven) backtester.

    Given df with 'signal' (-1 short, 0 out, 1 long) return df with 'position',
    taking into account:
       - position can be taken on the next row after signal is generated
       - stop-losse (either relative to entry or high water mark)
       - filtered_signal if given allows for additional condition that must
         be met to initiate position. Positions are closed regardless of
         filter.

    Args:
        data:        must have columns: 'price', 'close', 'signal', 'atr';
                     'filtered_signal' is optional, if not given,
                     filtred_signal = signal
                     'price' used for transactions
                     'close' to decide whether stop-loss has been triggered
        sl_atr:      stop-loss distance in multiples of ATRs
                     (if no stop loss required use very high number,
                      default 50)
        trailing_sl: if True, stop-loss calculated off high watermark,
                     if False, entry price
        active_close: if True close signal is the signal opposite to the
                      direction of the position, if False close signal is
                      lack of signal in
                      the direction of the position
        block_stop:  if True, after stop loss no position will be entered in
                     the same same direction as stoped out position until
                     opposite signal is generated
        take_profit: take profit distance expressed as multiple of stop-loss
                     distance, 0 means no take profit

    Returns:
        DataFrame with column 'position' to be processed by another function.
    """

    for c in ['price', 'close', 'signal', 'atr']:
        assert c in data.columns, f"'{c}' is a required column"

    data = data.copy()

    # while in position maintain open price and transaction direction
    data['position'] = 0
    # flag to execute transaction at next data point
    data['mark'] = False
    # note the reason for transaction at next data point
    data['reason'] = ''
    # record transaction price
    data['t_price'] = 0
    # entry price for stop loss calculation
    data['entry'] = 0
    # for stop-loss calculation
    data['high_water'] = 0
    # whether stop loss is trailing or fixed
    trailing_sl = trailing_sl
    # restrict re-entering positions after stop loss
    # (1=long positions blocked, -1=short positions blocked)
    block = 0

    if trailing_sl:
        sl_field = 'high_water'
    else:
        sl_field = 'entry'

    if 'date' not in data.columns:
        data.reset_index(inplace=True)

    if 'filtered_signal' not in data.columns:
        data['filtered_signal'] = data['signal']

    for item in data.itertuples():
        # first row doesn't have to check for positions or execute transactions
        if not item.Index == 0:
            # starting position is the same as previous day's position
            data.loc[item.Index, 'position'] = data.loc[(
                item.Index - 1), 'position']
            data.loc[item.Index, 'entry'] = data.loc[(item.Index - 1), 'entry']
            # execute transactions
            if data.loc[(item.Index - 1), 'mark']:
                # close position
                if data.loc[item.Index, 'position']:
                    data.loc[item.Index, 'position'] = 0
                    data.loc[item.Index, 'entry'] = 0
                    # record transaction price
                    data.loc[item.Index, 't_price'] = item.price * \
                        np.sign(data.loc[(item.Index - 1), 'entry']) * -1
                # open position
                else:
                    data.loc[item.Index, 'position'] = data.loc[(
                        item.Index - 1), 'signal']
                    data.loc[item.Index, 'entry'] = item.price * \
                        data.loc[(item.Index - 1), 'signal']
                    # record transaction price and high water mark
                    data.loc[item.Index, 't_price'] = item.price * \
                        data.loc[(item.Index - 1), 'signal']
                    data.loc[item.Index,
                             'high_water'] = data.loc[item.Index, 't_price']

        # update high water mark
        if not item.Index == 0:  # skip first row
            if data.loc[item.Index-1, 'position'] != 0:
                data.loc[item.Index, 'high_water'] = max(
                    data.loc[item.Index - 1, 'high_water'],
                    item.close*data.loc[item.Index, 'position'])

        # check for close signal
        if active_close:
            if data.loc[item.Index, 'position'] != 0 and np.sign(item.signal) != 0:
                if np.sign(data.loc[item.Index, 'position']) != np.sign(item.signal):
                    data.loc[item.Index, 'mark'] = True
                    data.loc[item.Index, 'reason'] = 'close'

        # check for stop-loss signal
        # long positions
        if data.loc[item.Index, 'position'] > 0:
            if item.close <= (
                    data.loc[item.Index, sl_field] - (item.atr * sl_atr)):
                data.loc[item.Index, 'mark'] = True
                data.loc[item.Index, 'reason'] = 'stop-out'
                if block_stop:
                    block = 1
        # short positions
        if data.loc[item.Index, 'position'] < 0:
            if item.close >= abs(
                    (data.loc[item.Index, sl_field] - (item.atr * sl_atr))):
                data.loc[item.Index, 'mark'] = True
                data.loc[item.Index, 'reason'] = 'stop-out'
                if block_stop:
                    block = -1

        # check for take profit
        if take_profit:
            # long positions
            if data.loc[item.Index, 'position'] > 0:
                if item.close >= (
                        data.loc[item.Index, 'entry'] +
                        (item.atr * sl_atr * take_profit)):
                    data.loc[item.Index, 'mark'] = True
                    data.loc[item.Index, 'reason'] = 'take-profit'
                    block = 1
            # short positions
            if data.loc[item.Index, 'position'] < 0:
                if item.close <= abs(
                        (data.loc[item.Index, 'entry'] +
                         (item.atr * sl_atr * take_profit))):
                    data.loc[item.Index, 'mark'] = True
                    data.loc[item.Index, 'reason'] = 'take-profit'
                    block = -1

        # check for entry signal
        if data.loc[item.Index, 'position'] == 0:
            if item.filtered_signal != 0 and item.filtered_signal != block:
                data.loc[item.Index, 'mark'] = True
                data.loc[item.Index, 'reason'] = 'entry'
                block = 0

    data.set_index('date', inplace=True, drop=True)
    return data


def true_sharpe(ret):
    """
    Given Series with daily returns (simple, non-log ie. P1/P0-1),
    return indicators comparing simplified Sharpe to 'true' Sharpe
    (defined by different caluclation conventions).
    """
    r = pd.Series()
    df = pd.DataFrame({'returns': ret})
    df['cummulative_return'] = (df['returns'] + 1).cumprod()
    df['log_returns'] = np.log(df['returns']+1)
    r['cummulative_return'] = df['cummulative_return'][-1] - 1
    r['annual_return'] = ((r['cummulative_return'] + 1)
                          ** (252 / len(df.index))) - 1
    r['mean'] = df['returns'].mean() * 252
    r['mean_log'] = df['log_returns'].mean() * 252
    r['vol'] = df['returns'].std() * np.sqrt(252)
    r['vol_log'] = df['log_returns'].std() * np.sqrt(252)
    r['sharpe'] = r['mean'] / r['vol']
    r['sharpe_log'] = r['mean_log'] / r['vol_log']
    return r


def rolling_sharpe(returns: pd.Series, months: float) -> pd.DataFrame:
    ret = pd.DataFrame({'returns': returns})
    ret['mean'] = ret['returns'].rolling(22*months).mean() * 252
    ret['vol'] = ret['returns'].rolling(22*months).std() * np.sqrt(252)
    ret['sharpe'] = (ret['mean'] / ret['vol'])
    ret = ret.dropna()
    return ret


def plot_rolling_sharpe(returns: pd.Series, months: float) -> None:
    rolling = rolling_sharpe(returns, months)
    rolling['mean_sharpe'] = rolling['sharpe'].mean()
    rolling[['sharpe', 'mean_sharpe']].plot(figsize=(20, 5), grid=True)


def plot_rolling_vol(returns: pd.Series, months: float) -> None:
    rolling = rolling_sharpe(returns, months)
    rolling['mean_vol'] = rolling['vol'].mean()
    rolling[['vol', 'mean_vol']].plot(figsize=(20, 5), grid=True)


def breakout_strategy(contract: pd.DataFrame,
                      time_int: float,
                      periods: list,
                      ema_fast: int,
                      ema_slow: int,
                      atr_periods: int,
                      sl_atr: float,
                      start: str = None,
                      end: str = None,
                      trailing_sl: bool = True,
                      active_close: bool = True,
                      block_stop: bool = False,
                      stop_loss: float = 0,
                      take_profit: float = 0) -> pd.DataFrame:
    """
    Implementation of breakout strategy.

    Returns:
    Dataframe breaking out position at given timepoints.
    """
    if start:
        contract = contract.loc[start:]
    if end:
        contract = contract.loc[:end]
    avg_vol = contract.volume.rolling(time_int).sum().mean()
    vol_candles = group_by_volume(contract, avg_vol)
    vol_candles['atr'] = get_ATR(vol_candles, atr_periods)
    data = vol_candles.copy()
    data['ema_fast'] = data.close.ewm(span=ema_fast).mean()
    data['ema_slow'] = data.close.ewm(span=ema_slow).mean()
    rows_to_drop = max(*periods, ema_slow)
    data = data.iloc[rows_to_drop:]
    data.reset_index(drop=True, inplace=True)
    data['signal'] = get_signals(data.close, periods)
    data['filter'] = np.sign(data['ema_fast'] - data['ema_slow'])
    data['filtered_signal'] = data['signal'] * \
        ((data['signal'] * data['filter']) == 1)
    data.rename(columns={'open': 'price'}, inplace=True)
    return c_backtester(data,
                        sl_atr=sl_atr,
                        trailing_sl=trailing_sl,
                        active_close=active_close,
                        block_stop=block_stop,
                        take_profit=take_profit)


def bootstrap(data, start=None, end=None, period_length=3, paths=100,
              replace=True):
    """
    Generate hypothetical time series by randomly drawing from price data.
    """
    if start:
        data = data.loc[start:]
    if end:
        data = data.loc[:end]

    daily = data.resample('B').first()
    data_indexed = pd.DataFrame({
        'open': data['open'] / data['close'],
        'high': data['high'] / data['close'],
        'low': data['low'] / data['close'],
        'close': data['close'].pct_change(),
        'volume': data['volume'],
        'barCount': data['barCount']})
    data_indexed = data_indexed.iloc[1:]

    days = len(daily.index)
    draws = int(days / period_length)

    d = np.random.choice(daily.index[:-period_length], size=(draws, paths))
    lookup_table = pd.Series(
        daily.index.shift(period_length), index=daily.index)

    output = []
    for path in d.T:
        p = pd.concat([data_indexed.loc[i:lookup_table[i]].iloc[:-1]
                       for i in path])
        p.set_index(pd.date_range(freq='min', start=data.index[0],
                                  periods=len(p), name='date'), inplace=True)

        p['close'] = (p['close'] + 1).cumprod() * data.iloc[0]['close']
        o = pd.DataFrame({
            'open': p['open'] * p['close'],
            'high': p['high'] * p['close'],
            'low': p['low'] * p['close'],
            'close': p['close'],
            'volume': p['volume'],
            'barCount': p['barCount']})
        output.append(o)
    return output


def sampler(data, start=None, end=None, period_length=25, paths=100):
    if start:
        data = data.loc[start:]
    if end:
        data = data.loc[:end]

    daily = data.resample('B').first()
    lookup_table = pd.Series(
        daily.index.shift(period_length), index=daily.index)
    d = np.random.choice(daily.index[:-period_length], size=paths)
    output = []
    for i in d:
        p = data.loc[i:lookup_table[i]].iloc[:-1]
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


out = NamedTuple('Out', [('stats', pd.DataFrame),
                         ('dailys', pd.DataFrame),
                         ('returns', pd.DataFrame),
                         ('positions', Dict[float, pd.DataFrame]),
                         ('dfs', pd.DataFrame),
                         ])


def summary(price: Union[pd.Series, pd.DataFrame],
            indicator: Optional[pd.Series] = None,
            slip: float = 0,
            threshold: Optional[Union[List, float]] = None) -> out:
    """
    Return stats summary of strategy for various thresholds
    run on the indicator. The strategy is long when indicator > threshold
    and short when indicator < -threshold.

    Trades are executed on the given price. Price and indicator
    must have the same index.
    """

    if isinstance(price, pd.DataFrame) and indicator is None:
        indicator = price.forecast
        price = price.open

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
            b = v_backtester(price, indicator, i)
            r = perf_var(b, False, slippage=slip)
        except ZeroDivisionError:
            continue
        stats[i] = r.stats
        dailys[i] = r.daily.balance
        returns[i] = r.daily['returns']
        positions[i] = r.positions
        dfs[i] = r.df
    return out(stats, dailys, returns, positions, dfs)
