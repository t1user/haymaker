from datetime import timedelta
from functools import partial
from typing import Callable, List, Optional, Tuple
import pandas as pd

from grouper import group_by_volume


TIME_INT = 120
VOL_LOOKBACK = 200
PERIODS = [5, 10, 20, 40, 80, 160]
SMOOTH = partial(lambda x, y: x.ewm(span=max((int(y), 1))).mean())
START_DATE = '20180901'
END_DATE = '20191231'
CALIBRATION_MONTHS = 3


def get_data(contract, start_date=START_DATE, end_date=END_DATE):
    return pd.read_pickle(
        f'data/minute_{contract}_cont_non_active_included.pickle'
    ).loc[start_date:end_date]


def get_avg_vol(data, time_int=TIME_INT):
    return data.volume.rolling(time_int).sum().mean()


def get_candles(data, avg_vol):
    return group_by_volume(data, avg_vol)


def get_vol(data, vol_lookback):
    data = data.copy()
    data['returns'] = data['close'].pct_change()
    data['vol_returns'] = data['returns'].ewm(
        span=vol_lookback, min_periods=int(vol_lookback*.9)).std()
    data['vol_price'] = data['vol_returns'] * data['close']
    return data['vol_price']


def calibrate(data: pd.DataFrame, ind: Callable,  vols: pd.Series,
              periods: List = PERIODS, adjustment: Optional[pd.Series] = None,
              multiplier: Optional[float] = None, smooth: int = SMOOTH
              ) -> Tuple[pd.Series, pd.Series, float, pd.DataFrame]:

    inds = pd.DataFrame([ind(data, p, smooth, vols)
                         for p in periods]).T.dropna()

    if adjustment is not None:
        adjustments = pd.Series([adjustment]*len(periods), index=inds.columns)
    else:
        adjustments = 10/inds.abs().mean()

    scaled_inds = (inds * adjustments).clip(lower=-20, upper=20)
    target_vol = scaled_inds.abs().std().mean()
    corr = scaled_inds.corr()
    weights = (1/corr.mean()) / (1/corr.mean()).sum()
    scaled_inds_combined = (scaled_inds * weights).sum(axis=1)

    if multiplier is None:
        multiplier = target_vol / scaled_inds_combined.abs().std()

    return weights, adjustments, multiplier, corr


def simulate(data: pd.DataFrame, ind: Callable,
             vols: pd.Series, weights: pd.Series,
             adjustments: pd.Series,
             multiplier: float, periods: List[int] = PERIODS,
             smooth: int = SMOOTH) -> pd.DataFrame:

    inds = pd.DataFrame([ind(data, p, smooth, vols)
                         for p in periods]).T
    scaled_inds = (inds * adjustments).clip(lower=-20, upper=20)
    scaled_inds_combined = (scaled_inds * weights).sum(axis=1)
    forecasts = (
        multiplier*scaled_inds_combined).clip(lower=-20, upper=20)
    return forecasts


def run(contract: str,
        ind: Callable[[pd.DataFrame, List[int], int, pd.Series], pd.Series],
        periods: List[int] = PERIODS,
        adjustment: Optional[float] = None, multiplier: Optional[float] = None,
        vol_lookback: int = VOL_LOOKBACK,
        start_date: str = START_DATE, end_date: str = END_DATE,
        calibration_months: int = CALIBRATION_MONTHS,
        time_int: int = TIME_INT,
        smooth: int = SMOOTH,
        output: bool = False) -> pd.DataFrame:

    data = get_data(contract, start_date, end_date)
    cal_data_end = data.resample('M').last(
    ).index[calibration_months] + timedelta(days=1)
    cal_data = data.loc[:cal_data_end]
    candles = get_candles(data, get_avg_vol(
        cal_data, time_int)).set_index('date')
    cal_candles = candles.loc[:cal_data_end]
    weights, adjustments, multiplier, corr = calibrate(
        cal_candles, ind, get_vol(cal_candles, vol_lookback), periods,
        adjustment, multiplier, smooth)
    forecasts = simulate(candles, ind, get_vol(candles, vol_lookback), weights,
                         adjustments, multiplier, periods, smooth)
    if output:
        print(
            f'weights: \n{weights.to_string()}\n\nadjustments:'
            f'\n{adjustments.to_string()}\n\n'
            f'multiplier:\n{multiplier}\n\ncorrelations:\n{corr}')
        print(f'\nsimulation start date: {cal_data_end}')
    return pd.DataFrame({'open': candles.open,
                         'close': candles.close,
                         'forecast': forecasts}).loc[cal_data_end:]
