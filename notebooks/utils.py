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
    Plot a price chart marking where long and short positions would be
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
    Transactions are assumed to be executed on the next price point after signal.

    Args:
    price: Series with instrument prices, must be indexed with Timestamps
    indicator: Series with indicator values moving around zero, same index as price
    threshold: above threshold strategy goes long below minus threshold short
    multiplier: multiplier of the futures contract that corresponds to price
    bankroll: bankroll to be used for daily returns assuming trading with one 
              contract
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
    transactions = df[df.signal != df.signal.shift()].iloc[1:][[
        'signal', 'cum_pnl_dollars']]
    transactions['transaction_pnl'] = transactions['cum_pnl_dollars'].diff()
    win_trans = transactions[transactions.transaction_pnl > 0]
    los_trans = transactions[transactions.transaction_pnl < 0]

    # container for all non-pyfolio stats
    stats = pd.Series()
    stats['Win percent'] = (len(win_trans)
                            / (len(win_trans) + len(los_trans)))
    stats['Average gain'] = (win_trans.transaction_pnl.sum()
                             / len(win_trans))
    stats['Average loss'] = (los_trans.transaction_pnl.sum()
                             / len(los_trans))
    stats['Avg Win/avg loss ratio'] = abs(
        stats['Average gain'] / stats['Average loss'])
    stats['Transaction EV'] = ((stats['Win percent'] * stats['Average gain'])
                               + ((1 - stats['Win percent']) * stats['Average loss']))
    stats['Number of transactions'] = len(win_trans) + len(los_trans)

    trades = df[df.signal != df.signal.shift()].signal.count()
    days = daily.returns.count()
    stats['Per day trades'] = trades/days
    stats['Per trade days'] = days/trades
    print(pyfolio_stats)
    print()
    print(stats)
    daily.balance.plot(figsize=(20, 10), grid=True)
