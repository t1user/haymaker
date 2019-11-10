import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyfolio.timeseries import perf_stats


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


def chart_price(price_series, signal_series, long_threshold=0, short_threshold=0):
    """
    Plot a price chart marking where long and short positions would be,
    given values of signal.
    price_series: instrument prices
    signal_series: indicator based on which signals will be generated
    position will be:
    long for signal_series > long_threshold
    short for signal_series < short_threshold
    """
    chart_data = pd.DataFrame()
    chart_data['out'] = price_series
    chart_data['long'] = (signal_series > long_threshold) * price_series
    chart_data['short'] = (signal_series < short_threshold) * price_series
    chart_data.replace(0, np.nan, inplace=True)
    chart_data.plot(figsize=(20, 10), grid=True)


def backtest(price: pd.Series, indicator: pd.Series,
             threshold: float, multiplier: int, bankroll: float) -> None:
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
                     short when price below minus threshold short
        multiplier:  multiplier for the futures contract that corresponds
                     to price
        bankroll:    bankroll to be used for daily returns assuming
                     trading with one contract
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
    daily = daily.iloc[1:]
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
    positions = positions.signal.count()
    stats['Win percent'] = len(win_trans) / positions
    stats['Average gain'] = win_trans.transaction_pnl.sum() / len(win_trans)
    # positions with zero gain are loss making
    stats['Average loss'] = (loss_trans.transaction_pnl.sum()
                             / (positions - len(win_trans)))
    stats['Avg gain/loss ratio'] = abs(stats['Average gain'] /
                                       stats['Average loss'])
    stats['Position EV'] = ((stats['Win percent'] * stats['Average gain'])
                            + ((1 - stats['Win percent']) * stats['Average loss']))
    days = daily.returns.count()
    stats['Positions per day'] = positions/days
    stats['Days per position'] = days/positions
    stats['Days'] = days
    stats['Positions'] = positions
    stats['Trades'] = transactions.signal.count() - 1
    print(pyfolio_stats.to_string())
    print()
    print(stats.to_string())
    daily.balance.plot(figsize=(20, 10), grid=True)
