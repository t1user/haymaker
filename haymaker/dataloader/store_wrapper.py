from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from functools import wraps
from typing import Any, Callable, Union

import ib_insync as ibi
import pandas as pd

from haymaker.datastore import AsyncAbstractBaseStore

from .time_policy import normalize_index, normalize_optional_point, normalize_point


@dataclass
class AsyncStoreView:
    """Read-only async datastore view for one contract's persisted data."""

    contract: ibi.Contract
    store: AsyncAbstractBaseStore
    now: Union[date, datetime]
    bar_size: str
    data: pd.DataFrame | None = field(default=None, init=False, repr=False)

    @classmethod
    async def create(
        cls,
        contract: ibi.Contract,
        store: AsyncAbstractBaseStore,
        now: Union[date, datetime],
        bar_size: str,
    ) -> "AsyncStoreView":
        """Create a wrapper and preload the existing dataframe once."""

        wrapper = cls(contract, store, normalize_point(now, bar_size), bar_size)
        await wrapper.read()
        return wrapper

    async def read(self) -> pd.DataFrame | None:
        """Read and cache available datastore data for this contract."""

        data = await self.store.read(self.contract)
        if data is not None and not data.empty:
            data = data.copy()
            data.index = normalize_index(data.index, self.bar_size)
        self.data = data
        return data

    @property
    def from_date(self) -> date | datetime | None:
        """Earliest point in datastore"""
        if self.data is None or self.data.empty:
            return None
        if len(self.data.index) == 1:
            return self.data.index[0]
        # second point in the df to avoid 1 point gap
        return self.data.index[1]

    @property
    def to_date(self) -> date | datetime | None:
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
            return normalize_optional_point(d, self.bar_size)

        return wrapper

    @property
    @cast_expiry
    def expiry(self) -> date | datetime | None:
        """Return precise contract expiry when available."""

        e = self.contract.lastTradeDateOrContractMonth
        if not e or len(e) == 6:
            return None
        return datetime.strptime(e, "%Y%m%d").replace(tzinfo=timezone.utc)

    def expiry_or_now(self):
        """
        It's mean to set the latest point to which it's possible to
        download data, which is either present moment or contract
        expiry whichever is earlier.  Contract expiry exists only for
        some types of contracts, if it doesn't exist, it should be
        disregarded.
        """
        return min(self.expiry, self.now) if self.expiry else self.now


@dataclass
class HistorySink:
    """Persist raw downloaded historical chunks for one contract."""

    contract: ibi.Contract
    store: AsyncAbstractBaseStore

    async def write(self, new_data: pd.DataFrame) -> Any:
        """Merge downloaded data with stored history and rewrite the collection.

        The sink preserves the dataframe exactly as supplied by IB-side callers.
        Index normalization for scheduling lives in :class:`AsyncStoreView`;
        final sorting, duplicate removal, and metadata belong to the datastore.

        Args:
            new_data: Newly downloaded bars indexed by bar timestamp.

        Returns:
            Datastore-specific write result.
        """

        existing = await self.store.read(self.contract)
        if existing is None:
            existing = pd.DataFrame()
        data = pd.concat([existing, new_data])
        return await self.store.async_write(self.contract, data)
