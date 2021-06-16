import pandas as pd
import numpy as np


class VolumeGrouper:
    """
    Resample given df based on volume.

    After initializing the object, resampled df is available as df property.
    """

    volume = 0
    label = 0

    def __init__(self, in_df: pd.DataFrame, avg_vol: int = 1000,
                 dynamic: bool = False, multiple: int = 30,
                 days: int = 10) -> None:
        """
        Args:
        in_df: DataFrame to be resampled, must be indexed by date and have
               column 'volume'
        avg_vol: fixed volume for every new bar
        dynamic: if True, volume for bar size will be recalculated daily as
                 a rolling mean over days argument times multiple
        multiple: how many rows of in_df should be used to calculate volume
        days: how many days should the volume be based on.

        Usage:

        VolumeGrouper(df, 5000).df
        Resample df into bars with volume of ca. 5000 each.

        VolumeGrouper(df, dynamic=True, multiple=60, days=5).df
        Assuming df contains 1 minute bars, VolumeGrouper will return
        a df resampled to bars each equal to mean hourly volume rolling over
        last 5 days.
        """

        self.avg_vol = avg_vol
        self.dynamic = dynamic
        self.in_df = in_df
        if dynamic:
            self.bar_volume = self.get_mean_volume(days) * multiple

    def group_dynamic(self, data: pd.Series) -> int:
        """
        Adjust bar volume based on averages for the past days.

        To be used in:
        df[['date', 'volume']].apply(group)

        This is extremly slow... better solution pending...

        data has fields: date, volume.
        """
        self.avg_vol = self.bar_volume.get(data['date'].floor('d'))
        if self.avg_vol is None:
            return np.nan
        return self.group(data['volume'])

    def group(self, vol: int) -> int:
        """
        Method doing actual resampling.

        Consecutive numbers are used to label every row on the input df
        with index on the output df.
        """
        self.volume += vol
        if self.volume >= self.avg_vol:
            self.volume = 0
            _label = self.label
            self.label += 1
            return _label
        return self.label

    def get_mean_volume(self, days: int = 10) -> pd.Series:
        """
        Used only if dynamic=True.
        Return rolling mean volume over given days per bar on the input df.
        """
        daily_volume = self.in_df.resample('B').agg(
            {'close': 'count', 'volume': 'mean'})
        daily_volume.rename(
            columns={'close': 'count', 'volume': 'mean_volume'}, inplace=True)
        daily_volume['rolling_volume'] = daily_volume['mean_volume'].rolling(
            days).mean().round()
        daily_volume = daily_volume.dropna()
        return daily_volume['rolling_volume']

    @property
    def df(self) -> pd.DataFrame:
        """Format output. Returned df indexed by date."""
        vol_candles = self.in_df.copy().reset_index()

        if self.dynamic:
            vol_candles['label'] = vol_candles[[
                'date', 'volume']].apply(self.group_dynamic, axis=1)
            vol_candles = vol_candles.dropna()
        else:
            vol_candles['label'] = vol_candles.volume.apply(self.group)

        # for debuging and inspection
        self.labeled = vol_candles.set_index('date')

        vol_candles = vol_candles.groupby('label').agg(
            {'date': 'last',
             'open': 'first',
             'high': 'max',
             'low': 'min',
             'close': 'last',
             'barCount': 'sum',
             'volume': 'sum', }
        ).set_index('date')
        return vol_candles


def group_by_volume(df, volume=1000, dynamic=False, multiple=30, days=10):
    """
    Interface method to provide compatibility access to grouper for older
    notebooks.
    """
    return VolumeGrouper(df, volume, dynamic, multiple, days).df


def group_by_time(df, time_int):
    vol_candles = pd.DataFrame()
    vol_candles['open'] = df.open.resample(time_int).first()
    vol_candles['high'] = df.high.resample(time_int).max()
    vol_candles['low'] = df.low.resample(time_int).min()
    vol_candles['close'] = df.close.resample(time_int).last()
    vol_candles = vol_candles.reset_index()
    return vol_candles
