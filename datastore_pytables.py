import pandas as pd
from ib_insync.contract import Future, ContFuture, Contract
from config import default_path


class Store:

    def __init__(self, path=default_path, what='TRADES'):
        self.store = pd.HDFStore(f'{default_path}/{what}.h5')

    def write(self, symbol, data, freq='min'):
        # print(symbol)
        self.store.append(self._symbol(symbol, freq), data)

    def read(self, symbol, freq='min', *args, **kwargs):
        symbol = self._symbol(symbol, freq)
        return self.store.select(symbol, *args, **kwargs)

    def check_earliest(self, symbol, freq='min'):
        try:
            return self.read(symbol, freq=freq).index.min()
        except KeyError:
            return None

    def check_latest(self, symbol, freq='min'):
        try:
            return self.read(symbol, freq=freq).index.max()
        except KeyError:
            return None

    def _symbol(self, s, freq):
        if isinstance(s, ContFuture):
            string = (f'cont/{freq}/{s.symbol}_'
                      f'{s.lastTradeDateOrContractMonth}_{s.exchange}'
                      f'_{s.currency}')
            return string
        elif isinstance(s, (Future, Contract)):
            string = (f'{freq}/{s.symbol}_{s.lastTradeDateOrContractMonth}'
                      f'_{s.exchange}_{s.currency}')
            return string

        else:
            return s

    def clean(self):
        for key in self.store.keys():
            df = self.read(key).sort_index(ascending=False)
            df.drop(index=df[df.index.duplicated()].index, inplace=True)
            self.store.remove(key)
            self.write(key, df)

    def keys(self):
        return self.store.keys()
