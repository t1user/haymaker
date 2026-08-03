from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

import ib_insync as ibi
import pandas as pd

from haymaker.async_wrappers import (
    QueueRunner,
    QueueShutdownPolicy,
    SyncQueueRunner,
    finish_on_cancel,
    make_async,
)

from .datastore import ArcticStore
from .symbol_namer import SymbolNamer

if TYPE_CHECKING:
    from pymongo import MongoClient  # type: ignore


class AsyncDataStore(Protocol):
    """Awaited dataframe datastore operations.

    A mutation returning normally means the backend operation has completed.
    Implementations must surface backend failures at the await site.
    """

    async def write(
        self, symbol: str | ibi.Contract, data: pd.DataFrame, meta: dict | None = None
    ) -> Any:
        """Write a dataframe and wait for backend completion."""

        ...

    async def append(
        self,
        symbol: str | ibi.Contract,
        data: pd.DataFrame,
        meta: dict | None = None,
        upsert: bool = True,
    ) -> Any:
        """Append a dataframe and wait for backend completion."""

        ...

    async def read(
        self,
        symbol: str | ibi.Contract,
        start_date: str | datetime | None = None,
        end_date: str | datetime | None = None,
    ) -> pd.DataFrame | None:
        """Read a dataframe from the backend."""

        ...

    async def keys(self) -> list[str]:
        """Return persisted symbol names."""

        ...

    async def read_metadata(self, symbol: ibi.Contract | str) -> dict[str, Any]:
        """Read metadata for one symbol."""

        ...

    async def write_metadata(
        self, symbol: ibi.Contract | str, meta: dict[str, Any]
    ) -> Any:
        """Update metadata and wait for backend completion."""

        ...


class QueuedDataSink(Protocol):
    """Queue dataframe mutations without waiting for backend completion.

    Enqueue methods return after work has been accepted. Final durability and
    failure handling depend on :attr:`shutdown_policy`; ``DRAIN`` reports
    processing failures or drain timeouts during queue shutdown, while
    ``DISCARD`` is best effort.
    """

    @property
    def shutdown_policy(self) -> QueueShutdownPolicy:
        """Return the policy applied when the underlying queue is closed."""

        ...

    def enqueue_write(
        self, symbol: str | ibi.Contract, data: pd.DataFrame, meta: dict | None = None
    ) -> None:
        """Queue a dataframe replacement."""

        ...

    def enqueue_append(
        self,
        symbol: str | ibi.Contract,
        data: pd.DataFrame,
        meta: dict | None = None,
        upsert: bool = True,
    ) -> None:
        """Queue a dataframe append."""

        ...

    def enqueue_write_metadata(
        self, symbol: ibi.Contract | str, meta: dict[str, Any]
    ) -> None:
        """Queue a metadata update."""

        ...


class AsyncArcticStore:

    _queue = SyncQueueRunner(
        "AsyncArcticStore_queue", shutdown_policy=QueueShutdownPolicy.DISCARD
    )
    _sync_class = ArcticStore

    def __init__(
        self,
        lib: str,
        host: str | MongoClient = "localhost",
        symbol_namer: SymbolNamer | None = None,
        shutdown_policy: QueueShutdownPolicy = QueueShutdownPolicy.DISCARD,
    ) -> None:
        self.store = self._sync_class(lib, host, symbol_namer)
        if shutdown_policy is not QueueShutdownPolicy.DISCARD:
            self._queue = SyncQueueRunner(
                f"AsyncArcticStore_queue_{lib}",
                shutdown_policy=shutdown_policy,
            )

    @property
    def symbol_namer(self) -> SymbolNamer:
        """Return the wrapped store's construction-time naming policy."""

        return self.store.symbol_namer

    @property
    def shutdown_policy(self) -> QueueShutdownPolicy:
        """Return the policy applied when this store's queue is closed."""

        return self._queue.shutdown_policy

    def _enqueue(self, fn: Callable[..., Any], *args: Any) -> None:
        self._queue.enqueue(fn, *args)

    async def close(self) -> None:
        """Apply this store's shutdown policy and stop its write queue."""

        await self._queue.close()

    def enqueue_write(
        self, symbol: str | ibi.Contract, data: pd.DataFrame, meta: dict | None = None
    ) -> None:
        """Queue a dataframe replacement without waiting for the backend."""

        self._enqueue(self.store.write, symbol, data, meta)

    def enqueue_append(
        self,
        symbol: str | ibi.Contract,
        data: pd.DataFrame,
        meta: dict | None = None,
        upsert: bool = True,
    ) -> None:
        """Queue a best-effort dataframe append."""

        # guaranteed to save in the same order as received
        self._enqueue(self.store.append, symbol, data, meta, upsert)

    async def write(
        self, symbol: str | ibi.Contract, data: pd.DataFrame, meta: dict | None = None
    ) -> Any:
        """Write a dataframe and wait for backend completion."""

        return await finish_on_cancel(make_async(self.store.write, symbol, data, meta))

    async def append(
        self,
        symbol: str | ibi.Contract,
        data: pd.DataFrame,
        meta: dict | None = None,
        upsert: bool = True,
    ) -> Any:
        """Append a dataframe and wait for backend completion."""

        return await finish_on_cancel(
            make_async(self.store.append, symbol, data, meta, upsert)
        )

    async def write_metadata(
        self, symbol: str | ibi.Contract, meta: dict[str, Any]
    ) -> Any:
        """Write metadata and wait until the database operation has finished."""

        return await finish_on_cancel(
            make_async(self.store.write_metadata, symbol, meta)
        )

    async def read(
        self,
        symbol: str | ibi.Contract,
        start_date: str | datetime | None = None,
        end_date: str | datetime | None = None,
    ) -> pd.DataFrame | None:
        return await make_async(self.store.read, symbol, start_date, end_date)

    def enqueue_delete(self, symbol: str | ibi.Contract) -> None:
        """Queue deletion of one symbol."""

        self._enqueue(self.store.delete, symbol)

    async def keys(self) -> list[str]:
        return await make_async(self.store.keys)

    async def read_metadata(self, symbol: str | ibi.Contract) -> dict[str, Any]:
        return await make_async(self.store.read_metadata, symbol)

    def enqueue_write_metadata(
        self, symbol: str | ibi.Contract, meta: dict[str, Any]
    ) -> None:
        """Queue a metadata update without waiting for the backend."""

        self._enqueue(self.store.write_metadata, symbol, meta)

    def enqueue_override_metadata(self, symbol: str, meta: dict[str, Any]) -> None:
        """Queue replacement of all metadata for one symbol."""

        self._enqueue(self.store.override_metadata, symbol, meta)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(lib={self.store.lib}, "
            f"host={self.store.host})"
        )
