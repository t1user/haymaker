from collections import defaultdict
from dataclasses import fields
from datetime import datetime
from typing import Any

import ib_insync as ibi
import numpy as np
import pandas as pd

from haymaker.datastore import AbstractBaseStore


class DataStoreWrapper:
    def __init__(self, store: AbstractBaseStore):
        self.store = store

    def delete_metadata_item(self, symbol: str, key: str) -> Any:
        """
        Delete an entry from metadata for a given symbol.
        Return None if symbol or key not present in datastore.
        """
        meta = self.store.read_metadata(symbol)
        if meta:
            try:
                del meta[key]
            except KeyError:
                return None
            return self.store.override_metadata(symbol, meta)

    def check_earliest(self, symbol: str) -> datetime | None:
        """Return earliest date of available data for a given symbol."""

        try:
            data = self.store.read(symbol)
        except KeyError:
            return None

        if data is None:
            return None

        return data.index.min()

    def check_latest(self, symbol: str) -> datetime | None:
        """Return earliest date of available data for a given symbol."""

        try:
            data = self.store.read(symbol)
        except KeyError:
            return None

        if data is None:
            return None

        return data.index.max()

    def date_range(self, symbol: str | None = None) -> pd.DataFrame:
        """
        For every key in datastore return start date and end date
        of available data. If symbol is given, only keys related to
        the symbol will be included.
        """

        range: dict = {}
        keys = self.store.keys()
        if symbol:
            keys = [k for k in keys if k.startswith(symbol)]

        for key in keys:
            df = self.store.read(key)
            if df is None:
                range[key] = (None, None)
            else:
                try:
                    range[key] = (df.index[0], df.index[-1])
                except IndexError:
                    range[key] = (None, None)
        return pd.DataFrame(range).T.rename(columns={0: "from", 1: "to"})

    def review(self, *field: str, symbol: str | None = None) -> pd.DataFrame:
        """
        Return df with date_range together with some contract
        details. If symbol given, include only contracts related to
        the symbol.

        Very slow and ineffective.

        Parameters:
        ----------
        *field - positional arguments will be treated as required additional columns

        symbol - must be given as keyword only argument

        """
        fields = [
            "symbol",
            "tradingClass",
            "currency",
            "min_tick",
            "lastTradeDateOrContractMonth",
            "name",
        ]
        if field:
            fields.extend(field)
        df = self.date_range(symbol=symbol)
        details = defaultdict(list)
        for row in df.itertuples():
            meta = self.read_metadata(row.Index)  # type: ignore
            for f in fields:
                details[f].append("" if meta is None else meta.get(f))
        for k, v in details.items():
            df[k] = v
        df["lastTradeDateOrContractMonth"] = pd.to_datetime(
            df["lastTradeDateOrContractMonth"]
        )
        try:
            df["multiplier"] = df["multiplier"].astype(float)
        except KeyError:
            pass
        return df

    def _contfutures(self) -> list[str]:
        """Return keys that correspond to contfutures"""
        return [c for c in self.store.keys() if c.endswith("CONTFUT")]

    def _contfutures_dict(
        self, field: str = "tradingClass"
    ) -> defaultdict[str, dict[datetime, str]]:
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
        contfutures: defaultdict[str, dict[datetime, str]] = defaultdict(dict)
        for cf in self._contfutures():
            meta = self.store.read_metadata(cf)
            date = pd.to_datetime(meta["lastTradeDateOrContractMonth"])
            contfutures[meta[field]].update({date: cf})

        # sorting
        ordered_contfutures: defaultdict[str, dict[datetime, str]] = defaultdict(dict)
        for k, v in contfutures.items():
            for i in sorted(v):
                ordered_contfutures[k].update({i: v[i]})

        # for k, v in contfutures.items():
        #    contfutures[k] = sorted(v, key=lambda x: x[0])
        return ordered_contfutures

    def latest_contfutures(
        self, index: int = -1, field: str = "tradingClass"
    ) -> dict[str, str]:
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
            _latest_contfutures[c] = d[
                keys_list[np.clip(index, -len(keys_list), (len(keys_list) - 1))]
            ]
        return _latest_contfutures

    def contfuture(
        self,
        symbol: str,
        index: int = -1,
        field: str = "tradingClass",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
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
        data = self.store.read(
            self.latest_contfutures(index, field)[symbol], start_date, end_date
        )
        assert data is not None, "contfuture data cannot be None"
        return data

    def contfuture_contract_object(
        self, symbol: str, index: int = -1, field: str = "tradingClass"
    ) -> ibi.Contract | None:
        """
        Return ib_insync object for latest contfuture for given symbol.
        """
        meta = self.store.read_metadata(self.latest_contfutures(index, field)[symbol])
        if meta:
            try:
                return ibi.ContFuture(
                    **{k: v for k, v in meta.items() if k in fields(ibi.ContFuture)}
                )
            except Exception:
                return None
        else:
            return None
