import pandas as pd


class VolumeGrouper:
    def __init__(self, avg_vol):
        self.volume = 0
        self.counter = 0
        self.avg_vol = avg_vol

    def group(self, vol):
        self.volume += vol
        if self.volume >= self.avg_vol:
            self.volume -= self.avg_vol
            self.counter += 1
        return self.counter


def group_by_volume(df, volume):
    vol_candles = df.copy().reset_index()
    grouper = VolumeGrouper(volume)
    vol_candles['label'] = vol_candles.volume.apply(grouper.group)
    vol_candles = vol_candles.groupby('label').agg({'date': 'last',
                                                    'open': 'first',
                                                    'high': 'max',
                                                    'low': 'min',
                                                    'close': 'last',
                                                    'barCount': 'sum',
                                                    'volume': 'sum'})
    return vol_candles


def group_by_time(df, time_int):
    vol_candles = pd.DataFrame()
    vol_candles['open'] = df.open.resample(time_int).first()
    vol_candles['high'] = df.high.resample(time_int).max()
    vol_candles['low'] = df.low.resample(time_int).min()
    vol_candles['close'] = df.close.resample(time_int).last()
    vol_candles = vol_candles.reset_index()
    return vol_candles
