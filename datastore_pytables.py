import pandas as pd
from ib_insync.contract import Future, ContFuture, Contract
from config import default_path


class Store:

    def __init__(self, path=default_path, what='TRADES'):
        self.store = pd.HDFStore(f'{default_path}/{what}.h5')

    def write(self, symbol, data, freq='min'):
        # print(symbol)
        self.store.append(self._symbol(symbol, freq), data)

    def read(self, symbol, freq='min', **kwargs):
        symbol = self._symbol(symbol, freq)
        return self.store.select(symbol, **kwargs)

    def check_earliest(self, symbol, freq='min'):
        try:
            return self.read(symbol, freq=freq).index[0]
        except KeyError:
            return None

    def check_latest(self, symbol, freq='min'):
        try:
            return self.read(symbol, freq=freq).index[-1]
        except KeyError:
            return None

    def _symbol(self, s, freq):
        if isinstance(s, (Future, Contract)):
            return f'{freq}/{s.symbol}_{s.lastTradeDateOrContractMonth}_{s.exchange}_{s.currency}'
        elif isinstance(s, ContFuture):
            return f'cont/{freq}/{s.symbol}'
        else:
            return s
