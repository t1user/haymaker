import asyncio
import functools
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from functools import wraps
from typing import Callable, Union

import ib_insync as ibi
import pandas as pd

from haymaker.datastore import AbstractBaseStore
from haymaker.misc import async_cached_property


@dataclass
class StoreWrapper:
    contract: ibi.Contract
    store: AbstractBaseStore
    now: Union[date, datetime]
    _loop: asyncio.AbstractEventLoop = field(
        default_factory=asyncio.get_running_loop, repr=False
    )

    async def data_async(self) -> pd.DataFrame | None:
        """Available data in datastore for contract or None"""
        return await self._loop.run_in_executor(None, self.store.read, self.contract)

    async def write_async(self, contract: ibi.Contract, data: pd.DataFrame) -> str:
        # save asynchronously to a synchronous store using executor
        return await self._loop.run_in_executor(None, self.write, contract, data)

    @async_cached_property
    async def from_date_async(self) -> datetime | None:
        # not in use
        data = await self.data_async()
        if data is not None:
            return data.index[1]
        else:
            return None

    @async_cached_property
    async def to_date_async(self) -> datetime | None:
        # not in use
        data = await self.data_async()
        if data is not None:
            return data.index[1]
        else:
            return None

    @property
    def data(self) -> pd.DataFrame | None:
        """Available data in datastore for contract or None"""
        return self.store.read(self.contract)

    @functools.cached_property
    def from_date(self) -> datetime | None:
        """Earliest point in datastore"""
        # second point in the df to avoid 1 point gap
        return self.data.index[1] if self.data is not None else None  # type: ignore

    @functools.cached_property
    def to_date(self) -> datetime | None:
        """Latest point in datastore"""
        date = self.data.index.max() if self.data is not None else None
        return date

    @staticmethod
    def cast_expiry(func: Callable, *args, **kwargs) -> Callable:
        @wraps(func)
        def wrapper(self):
            d = func(self, *args, **kwargs)
            if isinstance(self.now, datetime):
                d = datetime(d.year, d.month, d.day).replace(tzinfo=timezone.utc)
            return d

        return wrapper

    @functools.cached_property
    @cast_expiry
    def expiry(self) -> datetime | None:  # this maybe an error
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
        # everything not defined delegate to the wrapped object
        return getattr(self.store, attr)
