import pandas as pd
from ib_insync.contract import Future, ContFuture
from config import default_path
from functools import partial


class Store:

    def __init__(self, path=default_path, what='cont_fut_only'):
        path = f'{default_path}/{what}.h5'
        self.store = partial(pd.HDFStore, path)

    def write(self, symbol, data, freq='min'):
        with self.store() as store:
            store.append(self._symbol(symbol, freq), data)

    def date_string(self, start_date=None, end_date=None):
        dates = []
        if start_date:
            if isinstance(start_date, pd.Timestamp):
                start_date = start_date.strftime('%Y%m%d')
            dates.append(f'index >= {start_date}')
        if end_date:
            if isinstance(end_date, pd.Timestamp):
                end_date = end_date.strftime('%Y%m%d')
            dates.append(f'index <= {end_date}')
        if len(dates) == 2:
            return ' & '.join(dates)
        else:
            return dates[0]

    def read(self, symbol, freq='min', start_date=None, end_date=None):
        date_string = None
        if start_date or end_date:
            date_string = self.date_string(start_date, end_date)
        symbol = self._symbol(symbol, freq)
        with self.store() as store:
            if date_string:
                data = store.select(symbol, date_string)
            else:
                data = store.select(symbol)
        return data

    def remove(self, symbol, freq='min', *args, **kwargs):
        symbol = self._symbol(symbol, freq)
        with self.store() as store:
            store.remove(symbol)

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
        elif isinstance(s, Future):
            string = (f'{freq}/{s.symbol}_{s.lastTradeDateOrContractMonth}'
                      f'_{s.exchange}_{s.currency}')
            return string

        else:
            return s

    def clean(self):
        with self.store() as store:
            for key in store.keys():
                df = store.select(key).sort_index(ascending=False)
                df.drop(index=df[df.index.duplicated()].index, inplace=True)
                store.remove(key)
                store.append(key, df)

    def keys(self):
        with self.store() as store:
            keys = store.keys()
        return keys

        """
        TODO:
        implement pickle store
        df = pd.read_pickle('notebooks/data/minute_NQ_cont_non_active_included.pickle'
                            ).loc['20180201':].sort_index(ascending=False)
        """
