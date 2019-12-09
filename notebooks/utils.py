import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyfolio.timeseries import perf_stats
from collections import namedtuple
from typing import Tuple, NamedTuple


def plot(*data):
    """
    Plot every Series or column of given DataFrame on a separate vertical sub-plot.
    """
    # split DataFrames into separate Series
    columns = []
    for d in data:
        if isinstance(d, pd.Series):
            columns.append(d)
        elif isinstance(d, pd.DataFrame):
            columns.extend([d[c] for c in d.columns])
        else:
            raise ValueError('Arguments must be Series or Dataframes')
    # plot the charts
    fig = plt.figure(figsize=(20, 16))
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
    chart_data.plot(figsize=(20, 10), grid=True)


def backtest(price: pd.Series, indicator: pd.Series,
             threshold: float, multiplier: int,
             bankroll: float) -> Tuple[pd.DataFrame]:
    """
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
        multiplier:  multiplier for the futures contract that corresponds
                     to price
        bankroll:    bankroll to be used for daily returns assuming
                     trading with one contract

    Returns:
        Named tuple of resulting DataFrames for inspection:
        daily: daily returns (includes closed positions and mark-to-market
               for open positions)
        positions: closed positions
        transactions: source df for positions (for debugging)
        df: source df with signals (for debugging really)
    """
    # Main simulation DataFrame
    df = pd.DataFrame({'price': price,
                       'indicator': indicator})
    df['long'] = (df['indicator'] > threshold) * 1
    df['short'] = (df['indicator'] < -threshold) * -1
    df['signal'] = (df.long + df.short).shift(1)
    df['signal'] = df['signal'].fillna(0)
    df['position'] = df['signal'] * df['price']
    df['base'] = (df['signal'] - df['signal'].shift(1)) * df.price
    df['base'] = df['base'] + df['price'].shift(1) * df['signal'].shift(1)
    df['base'] = df['base'].fillna(0)
    df['pnl'] = df['position'] - df['base']

    df['pnl_dollars'] = df['pnl'] * multiplier
    df['cum_pnl_dollars'] = df['pnl_dollars'].cumsum()
    df['balance'] = bankroll + df['cum_pnl_dollars']

    # DataFrame calculating daily returns for pyfolio stats
    daily = df.balance.resample('B').last().fillna(method='ffill')
    daily[daily.index[0]-pd.Timedelta(days=1)] = bankroll
    daily.sort_index(inplace=True)
    daily = pd.DataFrame({'balance': daily})
    daily['returns'] = daily.balance.pct_change()
    daily['returns'].iloc[0] = 0
    #daily = daily.iloc[1:]
    pyfolio_stats = perf_stats(daily['returns'])

    # DataFrame calculating per transaction stats
    transactions = df[df.signal !=
                      df.signal.shift(-1)][['signal', 'cum_pnl_dollars']]
    transactions.merge(
        df[df.signal != df.signal.shift(1)][['signal', 'cum_pnl_dollars']], how='outer')
    transactions.reset_index(inplace=True)
    positions = transactions[transactions.index % 2 == 0]
    last_row = transactions[transactions.index % 2 != 0].iloc[-1]
    if last_row.date != positions.date.iloc[-1]:
        positions = positions.append(last_row)
    positions['transaction_pnl'] = positions['cum_pnl_dollars'].diff()
    positions = positions.iloc[1:]
    positions.set_index('date', inplace=True)
    win_trans = positions[positions.transaction_pnl > 0]
    loss_trans = positions[positions.transaction_pnl < 0]

    # container for all non-pyfolio stats
    stats = pd.Series()
    num_pos = positions.signal.count()  # number of positions
    stats['Win percent'] = len(win_trans) / num_pos
    stats['Average gain'] = win_trans.transaction_pnl.sum() / len(win_trans)
    # positions with zero gain are loss making
    stats['Average loss'] = (loss_trans.transaction_pnl.sum()
                             / (num_pos - len(win_trans)))
    stats['Avg gain/loss ratio'] = abs(stats['Average gain'] /
                                       stats['Average loss'])
    stats['Position EV'] = ((stats['Win percent'] * stats['Average gain'])
                            + ((1 - stats['Win percent']) * stats['Average loss']))
    days = daily.returns.count()
    stats['Positions per day'] = num_pos/days
    stats['Days per position'] = days/num_pos
    stats['Days'] = days
    stats['Positions'] = num_pos
    stats['Trades'] = transactions.signal.count()
    print(pyfolio_stats.to_string())
    print()
    print(stats.to_string())
    daily.balance.plot(figsize=(20, 10), grid=True)
    Results = namedtuple('Result', 'daily, positions, transactions, df')
    return Results(daily, positions, transactions, df)


def v_backtester(price: pd.Series,
                 indicator: pd.Series,
                 threshold: float,
                 multiplier: int,
                 bankroll: float,
                 output: bool = True) -> NamedTuple:
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
        multiplier:  multiplier for the futures contract that corresponds
                     to price
        bankroll:    bankroll to be used for daily returns assuming
                     trading with one contract
        output:      whether results are to be printed out

    Returns:
        Named tuple of resulting DataFrames for inspection:
        daily: daily returns (includes closed positions and mark-to-market
               for open positions)
        positions: closed positions
        transactions: source df for positions (for debugging)
        df: source df with signals (for debugging really)
    """
    # Main simulation DataFrame
    df = pd.DataFrame({'price': price,
                       'indicator': indicator})
    df['signal'] = ((df['indicator'] > threshold) * 1) \
        + ((df['indicator'] < -threshold) * -1)
    df['position'] = (df['signal'].shift(1).fillna(0)).astype('int')


def stats_calculator(price, position):
    """
    price and position are pd.Series with the same index
    """
    df = pd.DataFrame({'price': price,
                       'position': position})
    df['transaction'] = ((df['position'] - df['position'].shift(1)
                          ).fillna(0)).astype('int')
    df['curr_price'] = (df['position'] - df['transaction']) * df['price']
    df['base_price'] = (df['price'].shift(
        1) * df['position'].shift(1)).fillna(0)
    df['pnl'] = df['curr_price'] - df['base_price']
    df['c_transaction'] = ((df['transaction'] != 0) *
                           (-df['position'].shift(1))).fillna(0)
    df['o_transaction'] = df['transaction'] - df['c_transaction']

    df['pnl_dollars'] = df['pnl'] * multiplier
    df['cum_pnl_dollars'] = df['pnl_dollars'].cumsum()
    df['balance'] = bankroll + df['cum_pnl_dollars']

    # DataFrame calculating daily returns for pyfolio stats
    daily = df.balance.resample('B').last().fillna(method='ffill')
    daily[daily.index[0]-pd.Timedelta(days=1)] = bankroll
    daily.sort_index(inplace=True)
    daily = pd.DataFrame({'balance': daily})
    daily['returns'] = daily.balance.pct_change()
    daily['returns'].iloc[0] = 0
    #daily = daily.iloc[1:]
    pyfolio_stats = perf_stats(daily['returns'])

    # ==========================================
    # DataFrame calculating per transaction stats
    opens = df[df.o_transaction != 0]
    closes = df[df.c_transaction != 0]

    transactions = len(opens) + len(closes)  # for reporting

    # close out open trade (if any)
    if not len(opens) == len(closes):
        closes = closes.append(df.iloc[-1])
        closes.c_transaction[-1] = opens.o_transaction[-1] * -1

    # create positions df
    opens = (opens['price'] * opens['o_transaction']).reset_index()
    opens.columns = ['date', 'open']
    closes = (closes['price'] * closes['c_transaction']).reset_index()
    closes.columns = ['date', 'close']
    positions = opens.join(closes, how='outer', lsuffix='_o', rsuffix='_c')
    # get position stats
    positions['pnl'] = -(positions['open'] + positions['close'])
    positions['pnl_dollars'] = positions['pnl'] * multiplier
    pnl = positions['pnl_dollars'].sum()
    positions['duration'] = positions['date_c'] - positions['date_o']
    duration = positions['duration'].mean()
    win_pos = positions[positions['pnl'] > 0]
    # positions with zero gain are loss making
    loss_pos = positions[positions['pnl'] <= 0]
    # =========================================

    # container for all non-pyfolio stats
    stats = pd.Series()
    stats['Win percent'] = len(win_pos) / len(positions)
    stats['Average gain'] = win_pos.pnl_dollars.sum() / len(win_pos)
    stats['Average loss'] = loss_pos.pnl_dollars.sum() / len(loss_pos)
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
    stats['Trades'] = transactions
    stats = pyfolio_stats.append(stats)
    if output:
        print(stats.to_string())
        daily.balance.plot(figsize=(20, 10), grid=True)
    Results = namedtuple(
        'Result', 'stats, daily, positions, df, opens, closes')
    return Results(stats, daily, positions, df, opens, closes)


def c_backtester(data, sl_atr=20, commission=0, trailing_sl=True):
    """
    Consecutive (event driven) backtester.
    data: DataFrame with columns: open, close, signal, filtered_signal, atr
    """
    for c in ['open', 'close', 'signal']:
        assert c in data.columns, f"'{c}' is a required column"

    data = data.copy()

    # while in position maintain open price and transaction direction
    data['position'] = 0
    # flag to execute transaction at next data point
    data['mark'] = False
    # note the reason for transaction at next data point
    data['reason'] = ''
    # record commission paid
    data['commission'] = 0
    # record transaction price
    data['price'] = 0
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

    if 'date' in data.columns:
        data.reset_index(inplace=True)

    if 'filtered_signal' not in data.columns:
        data['filtered_signal'] = data['signal']

    for item in data.itertuples():
        # first row doesn't have to check if we have positions or execute transactions
        if not item.Index == 0:
            # starting position is the same as previous day position
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
                    data.loc[item.Index, 'price'] = item.open * \
                        np.sign(data.loc[(item.Index - 1), 'entry']) * -1
                # open position
                else:
                    data.loc[item.Index, 'position'] = data.loc[(
                        item.Index - 1), 'signal']
                    data.loc[item.Index, 'entry'] = item.open * \
                        data.loc[(item.Index - 1), 'signal']
                    # record transaction price and high water mark
                    data.loc[item.Index, 'price'] = item.open * \
                        data.loc[(item.Index - 1), 'signal']
                    data.loc[item.Index,
                             'high_water'] = data.loc[item.Index, 'price']
                # record commission paid
                data.loc[item.Index, 'commission'] = commission

        # update high water mark
        if not item.Index == 0:  # skip first row
            if data.loc[item.Index-1, 'position'] != 0:
                data.loc[item.Index, 'high_water'] = max(
                    data.loc[item.Index - 1, 'high_water'], item.close*data.loc[item.Index, 'position'])

        # check for close signal
        if data.loc[item.Index, 'position'] != 0 \
           and item.signal != data.loc[item.Index, 'position']:
            data.loc[item.Index, 'mark'] = True
            data.loc[item.Index, 'reason'] = 'close'
        # check for stop-loss signal
        # long positions
        if data.loc[item.Index, 'position'] > 0:
            if item.close <= (
                    data.loc[item.Index, sl_field] - (item.atr * sl_atr)):
                data.loc[item.Index, 'mark'] = True
                data.loc[item.Index, 'reason'] = 'stop-out'
                block = 1
        # short positions
        if data.loc[item.Index, 'position'] < 0:
            if item.close >= abs(
                    (data.loc[item.Index, sl_field] - (item.atr * sl_atr))):
                data.loc[item.Index, 'mark'] = True
                data.loc[item.Index, 'reason'] = 'stop-out'
                block = -1

        # check for entry signal
        if data.loc[item.Index, 'position'] == 0:
            if item.filtered_signal != 0 and item.filtered_signal != block:
                data.loc[item.Index, 'mark'] = True
                data.loc[item.Index, 'reason'] = 'entry'
                block = 0

    # close any open positions
    if data[data.price != 0].price.count() % 2 != 0:
        data.loc[data.index[-1], 'price'] = data.open.iloc[-1] * \
            np.sign(data.entry.iloc[-1]) * -1
        data.loc[data.index[-1], 'entry'] = 0
    data.set_index('date', inplace=True, drop=True)
    return data
