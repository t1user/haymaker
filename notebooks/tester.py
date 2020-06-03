from datetime import timedelta
from functools import partial
from typing import Callable, List, Optional, Tuple
import pandas as pd

from grouper import group_by_volume


TIME_INT = 60
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
        span=vol_lookback, min_periods=int(vol_lookback*.7)).std()
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

    # negative correlations capped at zero
    corr_non_negative = corr.copy()
    corr_non_negative[corr_non_negative < 0] = 0

    reverse_sum_corr = 1 / corr_non_negative.mean()
    weights = reverse_sum_corr / reverse_sum_corr.sum()
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


def calibrate_multiple(
        contract: str,
        ind: Callable[[pd.DataFrame, List[int], int, pd.Series], pd.Series],
        periods: List[int] = PERIODS,
        adjustment: Optional[float] = None, multiplier: Optional[float] = None,
        vol_lookback: int = VOL_LOOKBACK,
        start_date: str = START_DATE, end_date: str = END_DATE,
        calibration_months: int = CALIBRATION_MONTHS,  # only to keep api consistent
        time_int: int = TIME_INT,
        smooth: int = SMOOTH,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:

    data = get_data(contract, start_date, end_date)
    months = [g for n, g in data.groupby(pd.Grouper(freq='M'))]
    candles = get_candles(data, get_avg_vol(
        data, time_int)).set_index('date')
    weights = {}
    adjustments = {}
    multipliers = {}

    for month in months:
        start = month.index[0]
        end = month.index[-1]

        # skip initial months where not enough data available
        vols = get_vol(candles, vol_lookback)
        vols_period = vols.loc[start:end]
        if len(vols_period) - len(vols_period.dropna()) > 3:
            continue

        # skip partial months
        if (end-start).days < 19:
            continue

        period = candles.loc[start:end]
        _weights, _adjustments, _multiplier, _corr = calibrate(
            period, ind, vols, periods, adjustment,
            multiplier, smooth)
        weights[start] = _weights
        adjustments[start] = _adjustments
        multipliers[start] = _multiplier

    # return (pd.DataFrame(weights), pd.DataFrame(adjustments),
    #        pd.Series(multipliers))

    return (pd.DataFrame(weights).T.mean(), pd.DataFrame(adjustments).T.mean(),
            pd.Series(multipliers).mean())


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
    cal_data_start = data.index[0]
    cal_data_end = data.resample('M').last(
    ).index[calibration_months] + timedelta(days=1)
    # using all data to get candle size -> avg out over the period
    candles = get_candles(data, get_avg_vol(data, time_int)).set_index('date')
    vols = get_vol(candles, vol_lookback).dropna()
    vols_start = vols.index[0]
    offset = vols_start - cal_data_start
    cal_data_end += offset
    cal_candles = candles.loc[:cal_data_end]
    weights, adjustments, multiplier, corr = calibrate(
        cal_candles, ind, vols, periods, adjustment, multiplier, smooth)
    forecasts = simulate(
        candles, ind, vols, weights, adjustments, multiplier, periods, smooth)
    if output:
        print(
            f'weights: \n{weights.to_string()}\n\nadjustments:'
            f'\n{adjustments.to_string()}\n\n'
            f'multiplier:\n{multiplier}\n\ncorrelations:\n{corr}')
        print(f'\nsimulation start date: {cal_data_end}')
    return pd.DataFrame({'open': candles.open,
                         'close': candles.close,
                         'forecast': forecasts}).loc[cal_data_end:]
