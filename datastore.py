from functools import partial
from typing import List, Dict, Union, Optional, Tuple, Any, DefaultDict
from abc import ABC, abstractmethod
import pickle
from collections import defaultdict

import pandas as pd
from ib_insync.contract import Future, ContFuture, Contract
from arctic import Arctic
from arctic.exceptions import NoDataFoundException
from arctic.store.versioned_item import VersionedItem
from arctic.date import DateRange
from logbook import Logger

from config import default_path

log = Logger(__name__)


class AbstractBaseStore(ABC):
    """
    Interface for accessing datastores. To be inherited by particular store
    type implementation.
    """

    _latest_contfutures = None

    @abstractmethod
    def write(self, symbol: Union[str, Contract]):
        """
        Write data to datastore. Implementation has to recognize whether
        string or Contract was passed, extract metadata and save it in
        implementation specific format. Implementation is responsible
        for veryfying and cleaning data. In principle, if symbol exists
        in store data is to be overriden (other behaviour possible in
        implementations).
        """
        pass

    @abstractmethod
    def read(self, symbol: Union[str, Contract]):
        """
        Read data from store for a given symbol. Implementation has to
        recognize whether str or Contract was passed and read metadata
        in implementation specific manner.
        """
        pass

    @abstractmethod
    def delete(self, symbol: Union[str, Contract]):
        """
        Implementation responsible for distinguishing between str and Contract.
        """
        pass

    @abstractmethod
    def keys(self) -> List[str]:
        """Return a list of symbols available in store."""
        pass

    def _symbol(self, sym: Union[Contract, str]) -> str:
        """
        If Contract passed extract string that is used as key.
        Otherwise return the string passed.
        """
        if isinstance(sym, Contract):
            return f'{"_".join(sym.localSymbol.split())}_{sym.secType}'
        else:
            return sym

    def _metadata(self, obj: Union[Contract, str]) -> Dict[str, Any]:
        """
        If Contract passed extract metadata into a dict.
        Otherwise return empty dict.
        """
        if isinstance(obj, Contract):
            return {**obj.nonDefaults(),
                    **{'repr': repr(obj),
                       'secType': obj.secType,
                       'object': obj}}
        else:
            return {}

    @abstractmethod
    def read_metadata(self, symbol: Union[Contract, str]) -> Dict:
        """
        Public method for reading metadata for given symbol.
        Implementation must distinguish between str and Contract.
        """
        pass

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure no duplicates and ascending sorting of diven df."""
        df = df.sort_index(ascending=True)
        df.drop(index=df[df.index.duplicated()].index, inplace=True)
        return df

    def check_earliest(self, symbol: str) -> pd.datetime:
        """Return earliest date of available data for a given symbol."""
        try:
            return self.read(symbol).index.min()
        except (KeyError, AttributeError):
            return None

    def check_latest(self, symbol: str) -> pd.datetime:
        """Return earliest date of available data for a given symbol."""
        try:
            return self.read(symbol).index.max()
        except (KeyError, AttributeError):
            return None

    def date_range(self) -> pd.DataFrame:
        """
        For every key in library return start date and end date of
        available data.
        """
        range = {key: (self.check_earliest(key), self.check_latest(key))
                 for key in self.keys()}
        return pd.DataFrame(range).T.rename(columns={0: 'from', 1: 'to'})

    def _contfutures(self) -> List[str]:
        """Return keys that correspond to contfutures"""
        return [c for c in self.keys() if c.endswith('CONTFUT')]

    def _contfutures_dict(self) -> DefaultDict[str, Dict[pd.datetime, str]]:
        """
        Return a dictionary, where:
              keys: trading class (exchange symbol) for every contfuture
              values: dict of expiry date: symbol
        """
        contfutures = defaultdict(dict)
        for cf in self._contfutures():
            meta = self.read_metadata(cf)
            date = pd.to_datetime(meta['lastTradeDateOrContractMonth'])
            contfutures[meta['tradingClass']].update({date: cf})
        return contfutures

    @property
    def latest_contfutures(self) -> Dict[str, str]:
        """
        Return a dictionary of most recent contfutures {tradingClass: symbol}.
        Pull data from database only once.
        """
        if self._latest_contfutures is None:
            self._latest_contfutures = {
                c: d[max(d.keys())] for c, d in self._contfutures_dict().items()
            }
        return self._latest_contfutures

    def contfuture(self, symbol: str) -> pd.DataFrame:
        """
        Return data for latest contfuture for given exchange symbol
        (tradingClass).
        """
        return self.read(self.latest_contfutures[symbol])


class ArcticStore(AbstractBaseStore):

    def __init__(self, lib: str, host: str = 'localhost') -> None:
        """
        Library name is whatToShow + barSize, eg.
        TRADES_1_min
        BID_ASK_1_hour
        MIDPOINT_30_secs
        """
        lib = lib.replace(' ', '_')
        self.db = Arctic(host)
        self.db.initialize_library(lib)
        self.store = self.db[lib]

    def write(self, symbol: Union[str, Contract],
              data: pd.DataFrame, meta: dict = {}) -> VersionedItem:
        metadata = self._metadata(symbol)
        metadata.update(meta)
        version = self.store.write(
            self._symbol(symbol),
            self._clean(data),
            metadata=metadata)
        if version:
            return f'symbol: {version.symbol} version: {version.version}'
        return version

    def read(self, symbol: Union[str, Contract]) -> Optional[pd.DataFrame]:
        try:
            return self.read_object(symbol).data
        except AttributeError:
            return None

    def read_object(self, symbol: Union[str, Contract]
                    ) -> Optional[VersionedItem]:
        """
        Return Arctic object, which contains data and full metadata.

        This object has properties: symbol, library, data, version,
        metadata, host.

        Metadata is a dict with its properties mostly copied from ib.
        It's keys are: secType, conId, symbol,
        lastTradeDateOrContractMonth, multiplier, exchange, currency,
        localSymbol, tradingClass, repr, object.

        Repr is __repr__() of ib contract object.
        Object is the actual ib contract object.
        """
        try:
            return self.store.read(self._symbol(symbol))
        except NoDataFoundException:
            return None

    def delete(self, symbol: Union[str, Contract]) -> None:
        self.store.delete(self._symbol(symbol))

    def keys(self) -> List[str]:
        return self.store.list_symbols()

    def read_metadata(self, symbol: Union[str, Contract]
                      ) -> Optional[Dict[str, Dict[str, str]]]:
        try:
            return self.store.read_metadata(self._symbol(symbol)).metadata
        except AttributeError:
            return

    def _metadata(self, obj: Union[Contract, str]) -> Dict[str, Dict[str, str]]:
        if isinstance(obj, Contract):
            meta = super()._metadata(obj)
            meta.update({'object': pickle.dumps(obj)})
            # meta.update({'object': None})
        else:
            meta = {}
        return meta


class PyTablesStore(AbstractBaseStore):
    """Pandas HDFStore fixed format."""

    def __init__(self, lib: str, path: str = default_path) -> None:
        lib = lib.replace(' ', '_')
        path = f'{path}/{lib}.h5'
        self.store = partial(pd.HDFStore, path)
        self.metastore = f'{path}/meta.pickle'

    def write(self, symbol: Union[str, Contract],
              data: pd.DataFrame, meta: dict = {}) -> str:
        _symbol = self._symbol(symbol)
        with self.store() as store:
            store.put(_symbol,
                      self._clean(data))
        self._write_meta(_symbol, self._metadata(symbol))
        return f'{_symbol}'

    def read(self, symbol: Union[str, Contract]) -> Optional[pd.DataFrame]:
        with self.store() as store:
            data = store.get(self._symbol(symbol))
        return data

    def delete(self, symbol: Union[str, Contract]) -> None:
        self.store.remove(self._symbol(symbol))

    def keys(self) -> List[str]:
        with self.store() as store:
            keys = store.keys()
        return keys

    def read_metadata(self, symbol: Union[str, Contract]) -> dict:
        """Return metadata for given symbol"""
        return self._read_meta()[self._symbol(symbol)]

    def _read_meta(self) -> dict:
        """Return full metadata dictionary (for all symbols)."""
        with open(self.metastore, 'rb') as metastore:
            meta = pickle.load(metastore)
        return meta

    def _write_meta(self, symbol, data):
        meta = self._read_meta()
        meta[symbol] = data
        with open(self.metastore, 'wb') as metastore:
            pickle.dump(meta, metastore)


class PickleStore(AbstractBaseStore):

    def __init__(self, lib: str, path: str = default_path) -> None:
        lib = lib.replace(' ', '_')
        pass


# ==================================================================
# DEPRECATED
# ==================================================================


class Store:
    """Pandas HDFStore table format"""

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

    def clean_store(self):
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
