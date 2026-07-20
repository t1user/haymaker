from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

import ib_insync as ibi
import pandas as pd

from haymaker.async_wrappers import (
    QueueRunner,
    QueueShutdownPolicy,
    SyncQueueRunner,
    finish_on_cancel,
    make_async,
)

from .collection_namer import SymbolNamer
from .datastore import ArcticStore

if TYPE_CHECKING:
    from pymongo import MongoClient  # type: ignore


class AsyncAbstractBaseStore(ABC):
    @abstractmethod
    def write(
        self, symbol: str | ibi.Contract, data: pd.DataFrame, meta: dict | None = None
    ) -> None: ...

    @abstractmethod
    def append(
        self,
        symbol: str | ibi.Contract,
        data: pd.DataFrame,
        meta: dict | None = None,
        upsert: bool = True,
    ) -> Any: ...

    @abstractmethod
    async def read(
        self,
        symbol: str | ibi.Contract,
        start_date: str | datetime | None = None,
        end_date: str | datetime | None = None,
    ) -> pd.DataFrame | None: ...

    @abstractmethod
    async def keys(self) -> list[str]: ...

    @abstractmethod
    async def read_metadata(self, symbol: ibi.Contract | str) -> dict: ...

    @abstractmethod
    def write_metadata(self, symbol: ibi.Contract | str, meta: dict) -> Any: ...

    @abstractmethod
    def override_metadata(self, symbol: str, meta: dict[str, Any]) -> Any: ...

    @abstractmethod
    def delete(self, symbol: str | ibi.Contract) -> None: ...

    @abstractmethod
    async def async_append(
        self,
        symbol: str | ibi.Contract,
        data: pd.DataFrame,
        meta: dict | None = None,
        upsert: bool = True,
    ) -> Any: ...

    @abstractmethod
    async def async_write(
        self, symbol: str | ibi.Contract, data: pd.DataFrame, meta: dict | None = None
    ) -> Any: ...

    @abstractmethod
    async def async_write_metadata(
        self, symbol: str | ibi.Contract, meta: dict[str, Any]
    ) -> Any:
        """Write metadata and wait until the database operation has finished."""

        ...


class AsyncArcticStore(AsyncAbstractBaseStore):

    _queue = SyncQueueRunner(
        "AsyncArcticStore_queue", shutdown_policy=QueueShutdownPolicy.DISCARD
    )
    _sync_class = ArcticStore

    def __init__(
        self,
        lib: str,
        host: str | MongoClient = "localhost",
        collection_namer: SymbolNamer | None = None,
        shutdown_policy: QueueShutdownPolicy = QueueShutdownPolicy.DISCARD,
    ) -> None:
        self.store = self._sync_class(lib, host, collection_namer)
        if shutdown_policy is not QueueShutdownPolicy.DISCARD:
            self._queue = SyncQueueRunner(
                f"AsyncArcticStore_queue_{lib}",
                shutdown_policy=shutdown_policy,
            )

    @property
    def symbol_namer(self) -> SymbolNamer:
        """Return the wrapped store's construction-time naming policy."""

        return self.store.symbol_namer

    def enqueue(self, fn: Callable[..., Any], *args) -> None:
        self._queue.enqueue(fn, *args)

    async def close(self) -> None:
        """Apply this store's shutdown policy and stop its write queue."""

        await self._queue.close()

    def write(
        self, symbol: str | ibi.Contract, data: pd.DataFrame, meta: dict | None = None
    ) -> None:
        return self.enqueue(self.store.write, symbol, data, meta)

    def append(
        self,
        symbol: str | ibi.Contract,
        data: pd.DataFrame,
        meta: dict | None = None,
        upsert: bool = True,
    ) -> None:
        """
        Warning: this is best efforts, may lead to race conditions for
        very frequent writes.
        """
        # guaranteed to save in the same order as received
        self.enqueue(self.store.append, symbol, data, meta, upsert)

    async def async_write(
        self, symbol: str | ibi.Contract, data: pd.DataFrame, meta: dict | None = None
    ) -> Any:
        return await finish_on_cancel(make_async(self.store.write, symbol, data, meta))

    async def async_append(
        self,
        symbol: str | ibi.Contract,
        data: pd.DataFrame,
        meta: dict | None = None,
        upsert: bool = True,
    ) -> Any:
        return await finish_on_cancel(
            make_async(self.store.append, symbol, data, meta, upsert)
        )

    async def async_write_metadata(
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

    def delete(self, symbol: str | ibi.Contract) -> None:
        self.enqueue(self.store.delete, symbol)

    async def keys(self) -> list[str]:
        return await make_async(self.store.keys)

    async def read_metadata(self, symbol: str | ibi.Contract):
        return await make_async(self.store.read_metadata, symbol)

    def write_metadata(self, symbol: str | ibi.Contract, meta: dict[str, Any]) -> None:
        self.enqueue(self.store.write_metadata, symbol, meta)

    def override_metadata(self, symbol: str, meta: dict[str, Any]) -> None:
        self.enqueue(self.store.override_metadata, symbol, meta)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(lib={self.store.lib}, "
            f"host={self.store.host})"
        )
