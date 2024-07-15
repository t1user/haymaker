import functools
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Callable, Optional, Union

import ib_insync as ibi
import pandas as pd

from haymaker.datastore import AbstractBaseStore


@dataclass
class StoreWrapper:
    contract: ibi.Contract
    store: AbstractBaseStore
    now: Union[date, datetime]

    @property
    def data(self) -> Optional[pd.DataFrame]:
        """Available data in datastore for contract or None"""
        return self.store.read(self.contract)

    @functools.cached_property
    def from_date(self) -> Optional[datetime]:
        """Earliest point in datastore"""
        # second point in the df to avoid 1 point gap
        return self.data.index[1] if self.data is not None else None  # type: ignore

    @functools.cached_property
    def to_date(self) -> Optional[datetime]:
        """Latest point in datastore"""
        date = self.data.index.max() if self.data is not None else None
        return date

    @staticmethod
    def cast_expiry(func: Callable, *args, **kwargs) -> Callable:
        def wrapper(self):
            d = func(self, *args, **kwargs)
            if isinstance(self.now, datetime):
                d = datetime(d.year, d.month, d.day).replace(tzinfo=timezone.utc)
            return d

        return wrapper

    @functools.cached_property
    @cast_expiry
    def expiry(self) -> Optional[datetime]:  # this maybe an error
        """Expiry date for expirable contracts or ''"""
        e = self.contract.lastTradeDateOrContractMonth
        return (
            None
            if not e
            else datetime.strptime(e, "%Y%m%d").replace(tzinfo=timezone.utc)
        )

    def expiry_or_now(self):
        """
        It's mean to set the latest point to which it's possible to
        download data, which is either present moment or contract
        expiry whichever is earlier.  Contract expiry exists only for
        some types of contracts, if it doesn't exist, it should be
        disregarded.
        """
        return min(self.expiry, self.now) if self.expiry else self.now

    def __getattr__(self, attr):
        return getattr(self.store, attr)
