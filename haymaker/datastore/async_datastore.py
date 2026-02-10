from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any

import ib_insync as ibi
import pandas as pd

from haymaker.async_wrappers import fire_and_forget, make_async

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


class AsyncArcticStore(AsyncAbstractBaseStore):

    _sync_class = ArcticStore

    from_params = _sync_class.from_params

    def __init__(self, lib: str, host: str | MongoClient = "localhost") -> None:
        self.store = self._sync_class(lib, host)

    def write(
        self, symbol: str | ibi.Contract, data: pd.DataFrame, meta: dict | None = None
    ) -> None:
        return fire_and_forget(self.store.write, symbol, data, meta)

    def append(
        self,
        symbol: str | ibi.Contract,
        data: pd.DataFrame,
        meta: dict | None = None,
        upsert: bool = True,
    ) -> None:
        fire_and_forget(self.store.append, symbol, data, meta, upsert)

    async def read(
        self,
        symbol: str | ibi.Contract,
        start_date: str | datetime | None = None,
        end_date: str | datetime | None = None,
    ) -> pd.DataFrame | None:
        return await make_async(self.store.read, symbol, start_date, end_date)

    def delete(self, symbol: str | ibi.Contract) -> None:
        fire_and_forget(self.store.delete, symbol)

    async def keys(self) -> list[str]:
        return await make_async(self.store.keys)

    async def read_metadata(self, symbol: str | ibi.Contract):
        return await make_async(self.store.read_metadata, symbol)

    def write_metadata(self, symbol: str | ibi.Contract, meta: dict[str, Any]) -> None:
        fire_and_forget(self.store.write_metadata, symbol, meta)

    def override_metadata(self, symbol: str, meta: dict[str, Any]) -> None:
        fire_and_forget(self.store.override_metadata, symbol, meta)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(lib={self.store.lib}, "
            f"host={self.store.host})"
        )
