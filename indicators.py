from functools import partial
from typing import List, Dict, Callable, Tuple

import pandas as pd
import numpy as np
import sys


def atr(data: pd.DataFrame, periods: int, exp: bool = True) -> pd.Series:
    """
    Return Series with ATRs.

    Args:
    ---------
    data: must have columns: 'high', 'low', 'close'
    periods: lookback period
    exp: True - expotential mean, False - simple rolling mean 
    """
    TR = pd.DataFrame({'A': (data['high'] - data['low']),
                       'B': (data['high'] - data['close'].shift()).abs(),
                       'C': (data['low'] - data['close'].shift()).abs()
                       })
    TR['TR'] = TR.max(axis=1)
    if exp:
        TR['ATR'] = TR['TR'].ewm(span=periods).mean()
    else:
        TR['ATR'] = TR['TR'].rolling(periods).mean()
    return TR.ATR


# depricated function name
get_ATR = atr


def min_max_signal(data: pd.Series, period: int) -> pd.Series:
    """
    Return Series of signals (one of: -1, 0 or 1) dependig on whether
    price broke out above max (1) or below min (-1) over last <period>
    observations.

    Args:
    ---------
    data: price Series
    period: lookback period
    """
    df = pd.DataFrame({
        'max': ((data - data.shift(1).rolling(period).max()) > 0) * 1,
        'min': ((data.shift(1).rolling(period).min() - data) > 0) * 1,
    })
    df['signal'] = df['max'] - df['min']
    return df['signal']


def min_max_buffer_signal(data: pd.Series, period: int,
                          buff: float = 0) -> pd.Series:
    """
    Return Series of signals (one of: -1, 0 or 1) dependig on whether
    price broke out above max + <buff> (1) or below min - <buff> (-1) over last
    <period> observations.

    This is a general case of min_max_signal, not combined because this one
    will be removed if not found useful.

    Args:
    ---------
    data: price Series
    period: lookback period
    buff: buffor expressing sensitivity filter for deciding whether min or max
          has been breached, given in price units (default: 0)
    """
    df = pd.DataFrame({
        'max': ((data - data.shift(1).rolling(period).max() + buff) > 0) * 1,
        'min': ((data.shift(1).rolling(period).min() + buff - data) > 0) * 1,
    })
    df['signal'] = df['max'] - df['min']
    return df['signal']


def get_std(data, periods):
    returns = np.log(data.avg_price.pct_change()+1)
    return returns.rolling(periods).std() * data.avg_price


def get_min_max(data, period):
    return pd.DataFrame({
        'max': (data - data.shift(1).rolling(period).max()) > 0,
        'min': (data.shift(1).rolling(period).min() - data) > 0
    })


def majority_function(data: pd.DataFrame) -> pd.Series:
    """
    Return a reduced Series where every row = True, if majority of values in
    input row is True, else False.

    Args:
    ---------
    data: every row contains bool values, on which majority function is applied
    """
    return (
        0.5 + ((data.sum(axis=1) - 0.5) / data.count(axis=1))).apply(np.floor)


def get_min_max_df(data: pd.DataFrame, periods: Tuple[int],
                   func: Callable[[pd.DataFrame, int],
                                  pd.DataFrame] = get_min_max
                   ) -> Dict[str, pd.DataFrame]:
    """
    Given a list of periods, return func on each of those periods.
    """
    min_max_func = partial(func, data)
    mins = pd.DataFrame()
    maxs = pd.DataFrame()
    for period in periods:
        df = min_max_func(period)
        mins[period] = df['min']
        maxs[period] = df['max']
    return {'min': mins,
            'max': maxs}


def get_signals(data: pd.Series, periods: Tuple[int]) -> pd.Series:
    min_max = get_min_max_df(data, periods)
    return pd.DataFrame({
        'signal': majority_function(
            min_max['max']) - majority_function(min_max['min'])
    })


def any_signal(data: pd.Series, periods: Tuple[int]) -> pd.Series:
    min_max = get_min_max_df(data, periods)
    return min_max['max'].any(axis=1) * 1 - min_max['min'].any(axis=1) * 1


def rsi(price: pd.Series, lookback: int) -> pd.Series:
    df = pd.DataFrame({'price': price})
    df['change'] = df['price'].diff().fillna(0)
    df['up'] = ((df['change'] > 0) * df['change']).rolling(lookback).sum()
    df['down'] = ((df['change'] < 0) * df['change'].abs()
                  ).rolling(lookback).sum()
    df['rs'] = df['up'] / df['down']
    df['rsi'] = (100 - (100/(1+df['rs'])))
    return df['rsi']


def modified_rsi(rsi: pd.Series) -> pd.Series:
    """
    Rescale passed rsi to -100 to 100.
    """
    return 2*(rsi - 50)


def carver(price: pd.Series, lookback: int) -> pd.Series:
    """
    Return modified version of price placing it on a min-max scale
    over recent lookback periods expressed on a scale of -100 to 100
    (modified stochastic oscilator, after Rob Carver:
    https://qoppac.blogspot.com/2016/05/a-simple-breakout-trading-rule.html).
    """
    df = pd.DataFrame({'price': price})
    df['max'] = df['price'].rolling(lookback).max()
    df['min'] = df['price'].rolling(lookback).min()
    df['mid'] = df[['min', 'max']].mean(axis=1)
    df['carver'] = 200*((df['price'] - df['mid']) / (df['max'] - df['min']))
    return df['carver']


def range_crosser(ind: pd.Series, threshold: float) -> pd.Series:
    """
    For an ind like rsi, returns signal (-1, 0, 1) when ind crosses
    threshold from above or -threshold from below.
    """
    df = pd.DataFrame({'ind': ind})
    df['inside'] = (df['ind'].abs() < threshold)
    df['ss'] = ~(df['inside'].shift().fillna(False)) & df['inside']
    df['s'] = np.sign(df['ind'].diff())
    df['signal'] = df['ss'] * df['s']
    return df['signal']
