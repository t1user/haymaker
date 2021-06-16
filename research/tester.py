from datetime import timedelta
from functools import partial
from typing import Callable, List, Optional, Tuple, Union
import pandas as pd

from grouper import group_by_volume, VolumeGrouper
from datastore import Store


TIME_INT = 30
VOL_LOOKBACK = 200
PERIODS = [5, 10, 20, 40, 80, 160]
SMOOTH = partial(lambda x, y: x.ewm(span=max((int(y), 1))).mean())
START_DATE = '20180901'
END_DATE = '20191231'
CALIBRATION_MONTHS = 3

symbol_dict = {
    'CL': '/cont/min/CL_20200320_NYMEX_USD',
    'ES': '/cont/min/ES_20200320_GLOBEX_USD',
    'GC': '/cont/min/GC_20200428_NYMEX_USD',
    'NQ': '/cont/min/NQ_20200320_GLOBEX_USD',
}

store = Store()

vol_dict = {'NQ': 10000, 'ES': 33000, 'GC': 5500, 'CL': 11500}

min_dict = {'NQ': 30, 'ES': 120, 'GC': 30, 'CL': 30}


def get_data(contract, start_date=START_DATE, end_date=END_DATE):
    return store.read(symbol_dict[contract]
                      ).sort_index().loc[start_date: end_date]

    # return pd.read_pickle(
    #    f'data/minute_{contract}_cont_non_active_included.pickle'
    # ).loc[start_date:end_date]


def get_fixed_vol(symbol: str) -> float:
    return vol_dict[symbol]


def get_avg_vol(data, time_int=TIME_INT):
    return data.volume.rolling(time_int).sum().mean()


def get_candles(data, volume):
    return group_by_volume(data, volume)


def get_vol(data, vol_lookback):
    data = data.copy()
    data['returns'] = data['close'].pct_change()
    data['vol_returns'] = data['returns'].ewm(
        span=vol_lookback, min_periods=int(vol_lookback*.7)).std()
    data['vol_price'] = data['vol_returns'] * data['close']
    return data['vol_price']


def calibrate(inds: pd.DataFrame, adjustment: Optional[float] = None,
              multiplier: Optional[float] = None
              ) -> Tuple[pd.Series, pd.Series, float, pd.DataFrame]:

    if adjustment is not None:
        adjustments = pd.Series([adjustment]*len(inds.columns),
                                index=inds.columns)
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


def _simulate(inds: pd.DataFrame, weights: pd.Series, adjustments: pd.Series,
              multiplier: float) -> pd.DataFrame:

    scaled_inds = (inds * adjustments).clip(lower=-20, upper=20)
    scaled_inds_combined = (scaled_inds * weights).sum(axis=1)
    scaled_inds['forecast'] = (
        multiplier*scaled_inds_combined).clip(lower=-20, upper=20)
    return scaled_inds


def _data(contract: str,
          ind: Callable[[pd.DataFrame, List[int], int, pd.Series], pd.Series],
          periods: List[int] = PERIODS,
          vol_lookback: int = VOL_LOOKBACK,
          start_date: str = START_DATE,
          end_date: str = END_DATE,
          time_int: int = TIME_INT,
          smooth: Callable = SMOOTH,
          candle_volume: str = 'fixed'  # fixed, rolling, average
          ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    data = get_data(contract, start_date, end_date)
    # using all data to get candle size -> avg out over the period
    if candle_volume == 'fixed':
        candles = get_candles(data, get_fixed_vol(contract))
    elif candle_volume == 'average':
        candles = get_candles(data, get_avg_vol(data, time_int))
    else:
        candles = VolumeGrouper(
            data, dynamic=True, multiple=min_dict[contract]).df
    vols = get_vol(candles, vol_lookback).dropna()
    inds = pd.DataFrame([ind(candles, p, smooth, vols)
                         for p in periods]).T.dropna()
    start_date = max(vols.index[0], inds.index[0], candles.index[0])
    data = data.loc[start_date:]
    candles = candles.loc[start_date:]
    vols = inds.loc[start_date:]
    inds = inds.loc[start_date:]
    return data, candles, vols, inds


def calibrate_multiple(
        contract: str,
        ind: Callable[[pd.DataFrame, List[int], int, pd.Series], pd.Series],
        periods: List[int] = PERIODS,
        adjustment: Optional[float] = None, multiplier: Optional[float] = None,
        vol_lookback: int = VOL_LOOKBACK,
        start_date: str = START_DATE, end_date: str = END_DATE,
        calibration_months: int = CALIBRATION_MONTHS,  # only to keep api consistent
        time_int: int = TIME_INT,
        smooth: Callable = SMOOTH,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:

    data, candles, vols, inds = _data(contract, ind, periods, vol_lookback,
                                      start_date, end_date, time_int, smooth)

    weights = {}
    adjustments = {}
    multipliers = {}

    months = [g for n, g in inds.groupby(pd.Grouper(freq='M'))]

    for month in months:
        start = month.index[0]
        end = month.index[-1]

        # skip partial months
        if (end - start).days < 19:
            continue

        period = inds.loc[start:end]
        _weights, _adjustments, _multiplier, _corr = calibrate(
            period, adjustment, multiplier)
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
        output: bool = False,
        save_params: bool = False
        ) -> Union[pd.DataFrame,
                   Tuple[pd.DataFrame, Tuple[pd.Series, pd.Series, float]]]:

    data, candles, vols, inds = _data(contract, ind, periods, vol_lookback,
                                      start_date, end_date, time_int, smooth)

    # data for calibration
    cal_data_end = inds.resample('M').last(
    ).index[calibration_months] + timedelta(days=1)
    cal_inds = inds.loc[:cal_data_end]

    # calibrate
    weights, adjustments, multiplier, corr = calibrate(
        cal_inds, adjustment, multiplier)

    # simulate (exclude calibration period)
    sim_inds = inds.loc[cal_data_end:]
    forecasts = _simulate(sim_inds, weights, adjustments, multiplier)
    if output:
        print(
            f'weights: \n{weights.to_string()}\n\nadjustments:'
            f'\n{adjustments.to_string()}\n\n'
            f'multiplier:\n{multiplier}\n\ncorrelations:\n{corr}')
        print(f'\nsimulation start date: {cal_data_end}')

    prices = pd.DataFrame({'open': candles.open,
                           'close': candles.close,
                           })
    sim = pd.concat([prices, forecasts], axis=1).loc[cal_data_end:]

    if save_params:
        return sim, (weights, adjustments, multiplier, corr)
    return sim


def simulate(
    params: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    contract: str,
    ind: Callable[[pd.DataFrame, List[int], int, pd.Series], pd.Series],
    periods: List[int] = PERIODS,
    adjustment: Optional[float] = None, multiplier: Optional[float] = None,
    vol_lookback: int = VOL_LOOKBACK,
    start_date: str = START_DATE, end_date: str = END_DATE,
    calibration_months: int = CALIBRATION_MONTHS,
    time_int: int = TIME_INT,
    smooth: int = SMOOTH,
) -> pd.DataFrame:

    data, candles, vols, inds = _data(contract, ind, periods, vol_lookback,
                                      start_date, end_date, time_int, smooth)

    weights, adjustments, multiplier = params

    forecasts = _simulate(inds, weights, adjustments, multiplier)
    prices = pd.DataFrame({'open': candles.open,
                           'close': candles.close,
                           })

    return pd.concat([prices, forecasts], axis=1)
