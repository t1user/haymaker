from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Union

import ib_insync as ibi
import pandas as pd

from haymaker.datastore import AsyncAbstractBaseStore

from .time_policy import normalize_index, normalize_optional_point, normalize_point


@dataclass
class AsyncStoreView:
    """Preloaded read-only scheduling view of one persisted contract series.

    Use :meth:`create` so data and optional metadata are loaded before reading
    boundary properties. Intraday indices are normalized to UTC-aware
    datetimes; daily and longer indices are normalized to dates.

    Args:
        contract: Contract whose existing series is being planned.
        store: Async datastore used for reads.
        now: Run-scoped latest point considered by scheduling.
        bar_size: Canonical IB bar size controlling date normalization.
    """

    contract: ibi.Contract
    store: AsyncAbstractBaseStore
    now: Union[date, datetime]
    bar_size: str
    data: pd.DataFrame | None = field(default=None, init=False, repr=False)
    metadata: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    async def create(
        cls,
        contract: ibi.Contract,
        store: AsyncAbstractBaseStore,
        now: Union[date, datetime],
        bar_size: str,
    ) -> "AsyncStoreView":
        """Create a view and preload its dataframe and optional metadata."""

        wrapper = cls(contract, store, normalize_point(now, bar_size), bar_size)
        await wrapper.read()
        await wrapper.read_metadata()
        return wrapper

    async def read(self) -> pd.DataFrame | None:
        """Read and cache available datastore data for this contract."""

        data = await self.store.read(self.contract)
        if data is not None and not data.empty:
            data = data.copy()
            data.index = normalize_index(data.index, self.bar_size)
        self.data = data
        return data

    async def read_metadata(self) -> dict[str, Any]:
        """Read and cache optional datastore metadata for this contract."""

        self.metadata = await self.store.read_metadata(self.contract) or {}
        return self.metadata

    @property
    def backfill_boundary(self) -> date | datetime | None:
        """Earliest stored point safe to use as a backfill request boundary."""

        if self.data is None or self.data.empty:
            return None
        if len(self.data.index) == 1:
            return self.data.index[0]
        # second point in the df to avoid 1 point gap
        return self.data.index[1]

    @property
    def to_date(self) -> date | datetime | None:
        """Return the latest stored scheduling point, if any."""
        if self.data is None or self.data.empty:
            return None
        return self.data.index.max()

    @property
    def backfill_exhausted(self) -> bool:
        """Return whether older backfill was marked unavailable."""

        return self.metadata.get("backfill_exhausted") is True

    @property
    def expiry(self) -> date | datetime | None:
        """Return precise contract expiry when available."""

        e = self.contract.lastTradeDateOrContractMonth
        if not e or len(e) == 6:
            return None
        expiry = datetime.strptime(e, "%Y%m%d").replace(tzinfo=timezone.utc)
        return normalize_optional_point(expiry, self.bar_size)

    def expiry_or_now(self) -> date | datetime:
        """Return the latest point eligible for this contract's download."""
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

    async def mark_backfill_exhausted(self) -> Any:
        """Mark this series as unavailable for older backfills."""

        existing = await self.store.read(self.contract)
        if existing is None or existing.empty:
            return None
        return self.store.write_metadata(self.contract, {"backfill_exhausted": True})
