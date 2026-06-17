import functools
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from functools import wraps
from typing import Any, Callable, Union

import ib_insync as ibi
import pandas as pd

from haymaker.datastore import AsyncAbstractBaseStore


@dataclass
class StoreWrapper:
    """Async datastore view for one contract's currently persisted data."""

    contract: ibi.Contract
    store: AsyncAbstractBaseStore
    now: Union[date, datetime]
    data: pd.DataFrame | None = field(default=None, init=False, repr=False)

    @classmethod
    async def create(
        cls,
        contract: ibi.Contract,
        store: AsyncAbstractBaseStore,
        now: Union[date, datetime],
    ) -> "StoreWrapper":
        """Create a wrapper and preload the existing dataframe once."""

        wrapper = cls(contract, store, now)
        await wrapper.read()
        return wrapper

    async def read(self) -> pd.DataFrame | None:
        """Read and cache available datastore data for this contract."""

        self.data = await self.store.read(self.contract)
        return self.data

    async def write_async(self, contract: ibi.Contract, data: pd.DataFrame) -> Any:
        """Persist data through the async datastore and refresh local cache."""

        version = await self.store.async_write(contract, data)
        self.data = data
        return version

    @functools.cached_property
    def from_date(self) -> datetime | None:
        """Earliest point in datastore"""
        if self.data is None or self.data.empty:
            return None
        if len(self.data.index) == 1:
            return self.data.index[0]
        # second point in the df to avoid 1 point gap
        return self.data.index[1]

    @functools.cached_property
    def to_date(self) -> datetime | None:
        """Latest point in datastore"""
        if self.data is None or self.data.empty:
            return None
        date = self.data.index.max() if self.data is not None else None
        return date

    @staticmethod
    def cast_expiry(func: Callable, *args, **kwargs) -> Callable:
        @wraps(func)
        def wrapper(self):
            d = func(self, *args, **kwargs)
            if d is not None and isinstance(self.now, datetime):
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
