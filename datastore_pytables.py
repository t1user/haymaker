import pandas as pd
from ib_insync.contract import Future, ContFuture
from config import default_path


class Store:

    def __init__(self, path=default_path, what='TRADES'):
        self.store = pd.HDFStore(f'{default_path}/{what}.h5')

    def write(self, symbol, data, freq='min'):
        self.store.append(self._s(symbol, freq), data)

    def read(self, symbol, freq='min', **kwargs):
        return self.store.select(self._symbol(symbol, freq), **kwargs)

    def check_earliest(self, symbol, freq='min', what='TRADES'):
        return self.read(symbol, freq, what).iloc[0].index

    def check_latest(self, symbol, freq='min', what='TREADES'):
        return self.read(symbol, freq, what).iloc[-1].index

    def _symbol(s, freq):
        if isinstance(s, [Future, ContFuture]):
            symbol = f'{freq}/{s.symbol}_{s.lastTradeDateOrContractMonth}_{s.exchange}_{s.currency}'
            if isinstance(s, ContFuture):
                symbol += '_cont'
        else:
            return f'{freq}/{s}'
