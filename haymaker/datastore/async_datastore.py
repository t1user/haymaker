from __future__ import annotations

from abc import ABC, abstractmethod
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
        self, symbol: str | ibi.Contract, data: pd.DataFrame, meta: dict = {}
    ) -> None: ...

    @abstractmethod
    async def read(
        self,
        symbol: str | ibi.Contract,
        start_date: str | None = None,
        end_date: str | None = None,
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
        self.arctic_store = self._sync_class(lib, host)

    def write(
        self, symbol: str | ibi.Contract, data: pd.DataFrame, meta: dict | None = None
    ) -> None:
        return fire_and_forget(self.arctic_store.write, symbol, data, meta)

    async def read(
        self,
        symbol: str | ibi.Contract,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame | None:
        return await make_async(self.arctic_store.read, symbol, start_date, end_date)

    def delete(self, symbol: str | ibi.Contract) -> None:
        fire_and_forget(self.arctic_store.delete, symbol)

    async def keys(self) -> list[str]:
        return await make_async(self.arctic_store.keys)

    async def read_metadata(self, symbol: str | ibi.Contract):
        return await make_async(self.arctic_store.read_metadata, symbol)

    def write_metadata(self, symbol: str | ibi.Contract, meta: dict[str, Any]) -> None:
        fire_and_forget(self.arctic_store.write_metadata, symbol, meta)

    def override_metadata(self, symbol: str, meta: dict[str, Any]) -> None:
        fire_and_forget(self.arctic_store.override_metadata, symbol, meta)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(lib={self.arctic_store.lib}, "
            f"host={self.arctic_store.host})"
        )
