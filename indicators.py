import pandas as pd
import numpy as np
from functools import partial


def get_ATR(data, periods):
    TR = pd.DataFrame({'A': (data['high'] - data['low']),
                       'B': (data['high'] - data['close']).abs(),
                       'C': (data['low'] - data['close']).abs()
                       })
    TR['TR'] = TR.max(axis=1)
    TR['ATR'] = TR['TR'].ewm(span=periods).mean()
    #TR['ATR'] = TR['TR'].rolling(periods).mean()
    return TR.ATR


def get_std(data, periods):
    returns = np.log(data.avg_price.pct_change()+1)
    return returns.rolling(periods).std() * data.avg_price


def get_min_max(data, period):
    return pd.DataFrame({
        'max': (data - data.shift(1).rolling(period).max()) > 0,
        'min': (data.shift(1).rolling(period).min() - data) > 0
    })


def majority_function(data):
    return (
        0.5 + ((data.sum(axis=1) - 0.5) / data.count(axis=1))).apply(np.floor)


def get_min_max_df(data, periods, func=get_min_max):
    min_max_func = partial(func, data)
    mins = pd.DataFrame()
    maxs = pd.DataFrame()
    for period in periods:
        df = min_max_func(period)
        mins[period] = df['min']
        maxs[period] = df['max']
    return {'min': mins,
            'max': maxs}


def get_signals(data, periods, func=get_min_max_df):
    min_max = func(data, periods)
    # return min_max['min']

    return pd.DataFrame({
        'signal': majority_function(min_max['max']) - majority_function(min_max['min'])
    })
