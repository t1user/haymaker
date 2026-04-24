from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, Self

import ib_insync as ibi
import pandas as pd

from haymaker.async_wrappers import QueueRunner, make_async

from .datastore import AbstractBaseStore, ArcticStore

if TYPE_CHECKING:
    from pymongo import MongoClient  # type: ignore


class AsyncAbstractBaseStore(ABC):

    collection_namer: Callable[[ibi.Contract], str] = AbstractBaseStore.collection_namer

    @abstractmethod
    def override_collection_namer(
        self, namer: Callable[[ibi.Contract], str]
    ) -> Self: ...

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
    ) -> None: ...

    @abstractmethod
    async def async_write(
        self, symbol: str | ibi.Contract, data: pd.DataFrame, meta: dict | None = None
    ) -> None: ...


class AsyncStoreTask(NamedTuple):
    task: Callable
    args: tuple


class AsyncArcticStore(AsyncAbstractBaseStore):

    _library_queues: ClassVar[dict[str, QueueRunner]] = {}
    _sync_class = ArcticStore

    from_params = _sync_class.from_params

    def __init__(
        self,
        lib: str,
        host: str | MongoClient = "localhost",
        collection_namer: Callable[[ibi.Contract], str] | None = None,
    ) -> None:
        self.store = self._sync_class(lib, host, collection_namer)

    def override_collection_namer(self, collection_namer: Callable[..., str]) -> Self:
        self.store.collection_namer = collection_namer
        return self

    @property
    def _queue(self) -> QueueRunner:
        lib = self.store.lib
        if lib not in self._library_queues:
            AsyncArcticStore._library_queues[lib] = QueueRunner(
                processing_func=self._execute_store_task, owner=str(self)
            )
        return AsyncArcticStore._library_queues[lib]

    def enqueue(self, fn: Callable[..., Any], *args) -> None:
        self._queue.push(AsyncStoreTask(fn, args))

    async def _execute_store_task(self, data: AsyncStoreTask) -> None:
        func, args = data
        await asyncio.to_thread(func, *args)

    def write(
        self, symbol: str | ibi.Contract, data: pd.DataFrame, meta: dict | None = None
    ) -> None:
        """
        Warning: this is best efforts, may lead to race conditions for
        very frequent writes.
        """
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
    ) -> None:
        await make_async(self.store.write, symbol, data, meta)

    async def async_append(
        self,
        symbol: str | ibi.Contract,
        data: pd.DataFrame,
        meta: dict | None = None,
        upsert: bool = True,
    ) -> None:
        await make_async(self.store.append, symbol, data, meta, upsert)

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
