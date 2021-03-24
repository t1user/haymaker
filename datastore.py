from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional, Any, DefaultDict
from functools import partial
from datetime import datetime
from collections import defaultdict
import pickle

import pandas as pd
import numpy as np
from logbook import Logger
from arctic.date import DateRange
from arctic.store.versioned_item import VersionedItem
from arctic.exceptions import NoDataFoundException
from arctic import Arctic
from ib_insync.contract import Future, ContFuture, Contract

from config import default_path


log = Logger(__name__)


def symbol_extractor(func):
    """
    Not in use. TODO.

    Decorator that handles checking if method received Contract object or
    string. If Contract received convert it to approriate string.
    """
    def wrapper(symbol, *args, **kwargs):
        if isinstance(symbol, Contract):
            symbol = f'{"_".join(symbol.localSymbol.split())}_{symbol.secType}'
        return func(symbol, *args, **kwargs)
    return wrapper


class AbstractBaseStore(ABC):
    """
    Interface for accessing datastores. To be inherited by particular store
    type implementation.
    """

    @abstractmethod
    def write(self, symbol: Union[str, Contract]):
        """
        Write data to datastore. Implementation has to recognize whether
        string or Contract was passed, extract metadata and save it in
        implementation specific format. Implementation is responsible
        for veryfying and cleaning data. In principle, if symbol exists
        in store data is to be overriden (different behaviour possible in
        implementations).
        """
        pass

    @abstractmethod
    def read(self, symbol: Union[str, Contract],
             start_date: Optional[str] = None, end_date: Optional[str] = None
             ) -> Optional[pd.DataFrame]:
        """
        Read data from store for a given symbol. Implementation has to
        recognize whether str or Contract was passed and read metadata
        in implementation specific manner.

        Return df with data or None if the symbol is not in datastore.
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

    @abstractmethod
    def read_metadata(self, symbol: Union[Contract, str]) -> Dict:
        """
        Public method for reading metadata for given symbol.
        Implementation must distinguish between str and Contract.
        """
        pass

    @abstractmethod
    def write_metadata(self, symbol: Union[Contract, str], meta: Dict) -> Any:
        """
        Public method for writing metadata for given symbol.
        Implementation must distinguish between str and Contract.
        Metadata should be updated rather than overriden.
        """
        pass

    @abstractmethod
    def override_metadata(self, symbol: str, meta: Dict[str, Any]) -> Any:
        """
        Delete any existing metadata for symbol and replace it with meta.
        """
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

    def _update_metadata(self, symbol: Union[Contract, str],
                         meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        To be used in implementations that override metadata. Read existing
        metadata, update by meta and return updated dictionary. Relies on
        other methods to actually write the updated metadata.
        """
        _meta = self.read_metadata(symbol)
        if _meta:
            _meta.update(meta)
        else:
            _meta = meta
        return _meta

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

    def delete_metadata_item(self, symbol: str, key: str) -> Any:
        """
        Delete an entry from metadata for a given symbol.
        Return None if symbol or key not present in datastore.
        """
        meta = self.read_metadata(symbol)
        if meta:
            try:
                del meta[key]
            except KeyError:
                return None
            return self.override_metadata(symbol, meta)

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure no duplicates and ascending sorting of diven df."""
        df = df.sort_index(ascending=True)
        df.drop(index=df[df.index.duplicated()].index, inplace=True)
        return df

    def check_earliest(self, symbol: str) -> datetime:
        """Return earliest date of available data for a given symbol."""
        try:
            return self.read(symbol).index.min()
        except (KeyError, AttributeError):
            return None

    def check_latest(self, symbol: str) -> datetime:
        """Return earliest date of available data for a given symbol."""
        try:
            return self.read(symbol).index.max()
        except (KeyError, AttributeError):
            return None

    def date_range(self) -> pd.DataFrame:
        """
        For every key in datastore return start date and end date of
        available data.
        """
        range = {}
        for key in self.keys():
            df = self.read(key)
            try:
                range[key] = (df.index[0], df.index[-1])
            except IndexError:
                range[key] = (None, None)
        return pd.DataFrame(range).T.rename(columns={0: 'from', 1: 'to'})

    def review(self, *field: str) -> pd.DataFrame:
        """
        Return df with date_range together with some contract details.
        """
        fields = ['symbol', 'tradingClass',
                  'currency', 'min_tick', 'lastTradeDateOrContractMonth',
                  'name']
        if field:
            fields.extend(field)
        df = self.date_range()
        details = defaultdict(list)
        for key in df.to_dict('index').keys():
            meta = self.read_metadata(key)
            for f in fields:
                details[f].append('' if meta is None else meta.get(f))
        for k, v in details.items():
            df[k] = v
        return df

    def _contfutures(self) -> List[str]:
        """Return keys that correspond to contfutures"""
        return [c for c in self.keys() if c.endswith('CONTFUT')]

    def _contfutures_dict(self, field: str = 'tradingClass'
                          ) -> DefaultDict[str, Dict[datetime, str]]:
        """
        Args:
        ----------
        field: which metadata field is to be used as a key in returned dict

        Returns:
        ----------
        dictionary, where:
              keys: field (default: 'tradingClass') for every contfuture,
                    if to be used to lookup future in ib, 'symbol' should be
                    used
              values: dict of expiry date: symbol sorted ascending by expiry
                      date
        """
        contfutures = defaultdict(dict)
        for cf in self._contfutures():
            meta = self.read_metadata(cf)
            date = pd.to_datetime(meta['lastTradeDateOrContractMonth'])
            contfutures[meta[field]].update({date: cf})

        # sorting
        ordered_contfutures = defaultdict(dict)
        for k, v in contfutures.items():
            for i in sorted(v):
                ordered_contfutures[k].update({i: v[i]})

        # for k, v in contfutures.items():
        #    contfutures[k] = sorted(v, key=lambda x: x[0])
        return ordered_contfutures

    def latest_contfutures(self, index: int = -1,
                           field: str = 'tradingClass') -> Dict[str, str]:
        """
        Return a dictionary of contfutures for every tradingClass
        {tradingClass: symbol}. Relies on self._contfutures_dict.values()
        sorted ascending.

        Args:
        ----------
        index: standard python zero based indexing. (-1 means most recent
        contract, -2 second most recent, 0 first, 1 second, etc.)
                Oldest available contract if index is lower than the number of
                available contracts. Latest available contract if index is
                greater than number of contracts minus one.
        field: which field of metadata dict is to be searched to determine
               which contract is a contfutures.

        Returns:
        ----------
        Dictionary of {symbol: datastore key}

        """
        _latest_contfutures = {}
        for c, d in self._contfutures_dict(field).items():
            keys_list = list(d.keys())
            _latest_contfutures[c] = d[keys_list[np.clip(index,
                                                         -len(keys_list),
                                                         (len(keys_list)-1))]]
        return _latest_contfutures

    def contfuture(self, symbol: str, index: int = -1,
                   field: str = 'tradingClass',
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Return data for latest contfuture for symbol

        Args:
        ----------
        symbol: symbol to look-up
        field (default: 'tradingClass'): which field of metadata dict is to be
              looked-up
        index: -1 for latest contract, -2 for second latest, etc.
        start_date:
        end_date:

        Returns:
        ----------
        DataFrame with price/volume data for given contract.
        """
        return self.read(self.latest_contfutures(index, field)[symbol],
                         start_date, end_date)

    def contfuture_contract_object(self, symbol: str, index: int = -1,
                                   field: str = 'tradingClass'
                                   ) -> Optional[Contract]:
        """
        Return ib_insync object for latest contfuture for given exchange symbol
        (tradingClass).

        Usage:
        contfuture_contract_object('NQ')
        Returns:
        ContFuture(conId=371749745, symbol='NQ',
        lastTradeDateOrContractMonth='20200918', multiplier='20',
        exchange='GLOBEX', currency='USD', localSymbol='NQU0',
        tradingClass='NQ')
        """
        meta = self.read_metadata(
            self.latest_contfutures(index, field).get(symbol))
        if meta:
            try:
                return pickle.loads(meta['object'])
            except TypeError:
                return


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
            metadata=self._update_metadata(symbol, metadata))
        if version:
            return f'symbol: {version.symbol} version: {version.version}'

    def read(self, symbol: Union[str, Contract],
             start_date: Optional[str] = None,
             end_date: Optional[str] = None
             ) -> Optional[pd.DataFrame]:
        try:
            return self.read_object(symbol, start_date, end_date).data
        except (AttributeError, NoDataFoundException):
            return None

    def read_object(self, symbol: Union[str, Contract],
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None
                    ) -> Optional[VersionedItem]:
        """
        Return Arctic object, which contains data and full metadata.

        This object has properties: symbol, library, data, version,
        metadata, host.

        Metadata is a dict with its properties mostly copied from ib.
        It's keys are: secType, conId, symbol,
        lastTradeDateOrContractMonth, multiplier, exchange, currency,
        localSymbol, tradingClass, repr, object.
        It can also be updated by utils function to contain additional
        details.

        Repr is __repr__() of ib contract object.
        Object is the actual ib contract object.
        """
        date_range = DateRange(start_date, end_date)
        try:
            return self.store.read(self._symbol(symbol), date_range=date_range)
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
        except (AttributeError, NoDataFoundException):
            return

    def write_metadata(self, symbol: Union[Contract, str], meta: Dict[str, Any]
                       ) -> Optional[VersionedItem]:
        return self.store.write_metadata(self._symbol(symbol),
                                         self._update_metadata(symbol, meta))

    def override_metadata(self, symbol: str, meta: Dict[str, Any]
                          ) -> Optional[VersionedItem]:
        return self.store.write_metadata(symbol, meta)

    def _metadata(self, obj: Union[Contract, str]) -> Dict[str, Dict[str, str]]:
        if isinstance(obj, Contract):
            meta = super()._metadata(obj)
            meta.update({'object': pickle.dumps(obj)})
            # Unconmment following line if pickled object is not to be saved
            # Will make data unusable for backtester
            # meta.update({'object': None})
        else:
            meta = {}
        return meta


class PyTablesStore(AbstractBaseStore):
    """
    Pandas HDFStore fixed format.
    THIS HAS NOT BEEN TESTED AND LIKELY DOESN'T WORK PROPERLY. TODO.
    """

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
            keys=store.keys()
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
