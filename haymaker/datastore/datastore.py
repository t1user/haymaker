from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import ib_insync as ibi
import pandas as pd
from arctic import Arctic  # type: ignore
from arctic.date import DateRange  # type: ignore
from arctic.exceptions import NoDataFoundException  # type: ignore
from arctic.store.versioned_item import VersionedItem  # type: ignore

if TYPE_CHECKING:
    from pymongo import MongoClient  # type: ignore


log = logging.getLogger(__name__)


class AbstractBaseStore(ABC):
    """
    Interface for accessing datastores. To be inherited by particular store
    type implementation.

    Datastore is for saving and reading pandas dataframes and a dict of metadata.
    """

    @abstractmethod
    def write(
        self, symbol: str | ibi.Contract, data: pd.DataFrame, meta: dict = {}
    ) -> Any:
        """
        Write data to datastore. Implementation has to recognize whether
        string or Contract was passed, extract metadata and save it in
        implementation specific format. Implementation is responsible
        for veryfying and cleaning data. In principle, if symbol exists
        in store data is to be overriden (different behaviour possible in
        implementations).
        """
        ...

    @abstractmethod
    def read(
        self,
        symbol: str | ibi.Contract,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame | None:
        """
        Read data from store for a given symbol. Implementation has to
        recognize whether str or Contract was passed and read metadata
        in implementation specific manner.

        Return df with data or None if the symbol is not in datastore.
        """
        ...

    @abstractmethod
    def delete(self, symbol: str | ibi.Contract):
        """
        Delete given symbol, if `ibi.Contract` is passed,
        implementation must convert it to actual symbol first.
        """
        symbol = self._symbol(symbol)
        # delete symbol here
        ...

    @abstractmethod
    def keys(self) -> list[str]:
        """Return a list of symbols available in store."""
        ...

    @abstractmethod
    def read_metadata(self, symbol: ibi.Contract | str) -> dict:
        """
        Public method for reading metadata for given symbol.
        Implementation must distinguish between str and Contract.
        """
        ...

    @abstractmethod
    def write_metadata(self, symbol: ibi.Contract | str, meta: dict) -> Any:
        """
        Public method for writing metadata for given symbol.
        Implementation must distinguish between str and Contract.
        Metadata should be updated rather than overriden.
        """
        ...

    @abstractmethod
    def override_metadata(self, symbol: str, meta: dict[str, Any]) -> Any:
        """
        Delete any existing metadata for symbol and replace it with meta.
        """
        ...

    def _symbol(self, sym: ibi.Contract | str) -> str:
        """
        If Contract passed extract string that is used as key.
        Otherwise return the string passed.
        """
        if isinstance(sym, ibi.Contract):
            return f'{"_".join(sym.localSymbol.split())}_{sym.secType}'
        else:
            return sym

    def _update_metadata(
        self, symbol: ibi.Contract | str, meta: dict[str, Any]
    ) -> dict[str, Any]:
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

    def _metadata(self, obj: ibi.Contract | str) -> dict[str, Any]:
        """
        If Contract passed extract metadata into a dict.
        Otherwise return empty dict.
        """
        if isinstance(obj, ibi.Contract):
            return {**ibi.util.dataclassNonDefaults(obj), "repr": repr(obj)}
        else:
            return {}

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure no duplicates and ascending sorting of given df."""
        df = df.sort_index(ascending=True)
        df = df[~df.index.duplicated()]
        # df.drop(index=df[df.index.duplicated()].index, inplace=True)
        return df

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"({', '.join([f'{k}={v}' for k, v in self.__dict__.items()])})"
        )


class ArcticStore(AbstractBaseStore):
    def __init__(self, lib: str, host: str | MongoClient = "localhost") -> None:
        """
        Library name is whatToShow + barSize, eg.
        TRADES_1_min
        BID_ASK_1_hour
        MIDPOINT_30_secs
        """
        lib = lib.replace(" ", "_")
        self.lib = lib
        self.host = host
        self.db = Arctic(host)
        self.db.initialize_library(lib)
        self.store = self.db[lib]

    def write(
        self,
        symbol: str | ibi.Contract,
        data: pd.DataFrame,
        meta: dict | None = None,
    ) -> str:
        metadata = self._metadata(symbol)
        if meta is not None:
            metadata.update(meta)
        version = self.store.write(
            self._symbol(symbol),
            self._clean(data),
            metadata=self._update_metadata(symbol, metadata),
        )
        return f"symbol: {version.symbol} version: {version.version}"

    def read(
        self,
        symbol: str | ibi.Contract,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame | None:
        data = self.read_object(symbol, start_date, end_date)
        if data:
            return data.data
        else:
            return None

    def read_object(
        self,
        symbol: str | ibi.Contract,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> VersionedItem | None:
        """
        Return Arctic object, which contains data and full metadata.

        This object has properties: symbol, library, data, version,
        metadata, host.

        Metadata is a dict with its properties mostly copied from ib.
        It's keys are: secType, conId, symbol,
        lastTradeDateOrContractMonth, multiplier, exchange, currency,
        localSymbol, tradingClass.
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

    def delete(self, symbol: str | ibi.Contract) -> None:
        self.store.delete(self._symbol(symbol))

    def keys(self) -> list[str]:
        return self.store.list_symbols()

    def read_metadata(self, symbol: str | ibi.Contract) -> dict:
        try:
            return self.store.read_metadata(self._symbol(symbol)).metadata
        except (AttributeError, NoDataFoundException):
            return {}

    def write_metadata(
        self, symbol: ibi.Contract | str, meta: dict[str, Any]
    ) -> VersionedItem | None:
        return self.store.write_metadata(
            self._symbol(symbol), self._update_metadata(symbol, meta)
        )

    def override_metadata(
        self, symbol: str, meta: dict[str, Any]
    ) -> VersionedItem | None:
        return self.store.write_metadata(symbol, meta)

    def _metadata(self, obj: ibi.Contract | str) -> dict[str, dict[str, str]]:
        if isinstance(obj, ibi.Contract):
            meta = super()._metadata(obj)
        else:
            meta = {}
        return meta

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(lib={self.lib}, host={self.host})"
