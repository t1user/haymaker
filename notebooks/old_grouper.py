import pandas as pd


class VolumeGrouper:
    def __init__(self, avg_vol=1000, dynamic=False):
        self.volume = 0
        self.counter = 0
        self.avg_vol = avg_vol
        self.accumulator = []
        self.dynamic = dynamic

    def group(self, vol):
        self.volume += vol
        self.accumulator.append(vol)
        if self.volume >= self.avg_vol:
            #self.volume -= self.avg_vol
            self.volume = 0
            self.counter += 1
            if self.dynamic:
                self.adjust_avg()
        return self.counter

    def adjust_avg(self):
        ewma = pd.Series(self.accumulator).rolling(
            30).sum().ewm(span=4140).mean().iloc[-1]
        # print(ewma)
        if abs(self.avg_vol - ewma)/self.avg_vol > .4:
            self.avg_vol = ewma


def group_by_volume(df, volume=1000, dynamic=False):
    vol_candles = df.copy().reset_index()
    grouper = VolumeGrouper(volume, dynamic)
    vol_candles['label'] = vol_candles.volume.apply(grouper.group)
    vol_candles = vol_candles.groupby('label').agg({'date': 'last',
                                                    'open': 'first',
                                                    'high': 'max',
                                                    'low': 'min',
                                                    'close': 'last',
                                                    'barCount': 'sum',
                                                    'volume': 'sum', })
    return vol_candles
