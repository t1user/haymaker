from arctic import Arctic, CHUNK_STORE
import pandas as pd

from config import default_library


class Store:
    """
    freq in ['min', 'hour', 'day']
    """
    chunk_size = {
        'min': 'D',
        'hour': 'D',
        'day': 'M'
    }

    def __init__(self, library=default_library, db='localhost', **kwargs):
        self.conn = Arctic(db)

    def write(self, symbol, data, freq='min', what='TRADES'):
        lib_name = f'{what}_{freq}'
        if lib_name in self.conn.list_libraries():
            lib = self.conn[lib_name]
            lib.append(symbol, data)
        else:
            self.conn.initialize_library(lib_name)
            lib = self.conn[lib_name]
            lib.write(symbol, data, lib_type=CHUNK_STORE,
                      chunk_size=self.chunk_size['freq'])

    def read(self, symbol, freq='min', what='TRADES'):
        lib_name = f'{what}_{freq}'
        lib = self.conn[lib_name]
        return lib.read(symbol, **kwargs)

    def check_data_availability(self, symbol, ):
        pass
