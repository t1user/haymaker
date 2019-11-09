import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
    chart_data['long'] = (signal_series > long_threshold) * price_series
    chart_data['short'] = (signal_series < short_threshold) * price_series
    if long_threshold != short_threshold:
        chart_data['out'] = ~((signal_series >= short_threshold) & (
            signal_series <= long_threshold)) * price_series
    chart_data.replace(0, np.nan, inplace=True)
    chart_data.plot(figsize=(20, 10), grid=True)
