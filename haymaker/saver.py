from __future__ import annotations

import asyncio
import csv
import logging
import pickle
from abc import ABC, abstractmethod
from collections.abc import Collection
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from .async_wrappers import fire_and_forget, make_async
from .config import CONFIG as config
from .databases import get_mongo_client
from .misc import default_path, name_str

log = logging.getLogger(__name__)


CONFIG = config.get("saver") or {}


class AbstractBaseSaver(ABC):
    """
    Api for saving data during trading/simulation.
    """

    @abstractmethod
    def save(self, data: Any, /, *args: Any) -> None: ...

    @abstractmethod
    def read(self, key=None, /, *args: Any): ...

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"({', '.join([f'{k}={v}' for k, v in self.__dict__.items()])})"
        )


class AbstractBaseFileSaver(AbstractBaseSaver):
    _suffix = ""

    def __init__(self, name: str, folder: str = "", use_timestamp: bool = True) -> None:
        self.path = default_path(folder)
        self.name = name
        self.timestamp = datetime.now(timezone.utc) if use_timestamp else None

    def _file(self, *args) -> str:
        return (
            f"{self.path}/"
            f"{name_str(self.name, *args, timestamp=self.timestamp)}"
            f".{self._suffix}"
        )

    def __str__(self) -> str:
        return f"{self.__class__.__qualname__}({self.path}, {self.name})"


class PickleSaver(AbstractBaseFileSaver):
    _suffix = "pickle"

    def save(self, data: Any, /, *args: str) -> None:
        if isinstance(data, pd.DataFrame):
            data.to_pickle(self._file(*args))
        else:
            with open(self._file(*args), "wb") as f:
                pickle.dump(data, f)

    def read(self, name: str | None = None, /, *args: Any) -> Any:
        if name is None:
            raise ValueError("name must be a string, not None")
        with open(self._file(name), "rb") as f:
            return pickle.load(f)


class CsvSaver(AbstractBaseFileSaver):
    _suffix = "csv"

    def __init__(self, name: str, folder: str = "", use_timestamp: bool = True) -> None:
        super().__init__(name, folder, use_timestamp)
        self._fieldnames: list[str] | None = None

    def _path_exists(self, *args: str) -> bool:
        return Path(self._file(*args)).exists()

    def _create_header(self, keys: Collection, *args: str) -> None:
        with open(self._file(*args), "w") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()

    def save(self, data: dict[str, Any], /, *args: str) -> None:
        if not self._path_exists(*args):
            self._create_header(data.keys(), *args)

        with open(self._file(*args), "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            writer.writerow(data)

    def save_many(
        self, data: list[dict[str, Any]], /, *args: str, override=True
    ) -> None:
        if not data:
            raise ValueError("Cannot save empty data list")
        if override or not self._path_exists(*args):
            self._create_header(data[0].keys())
        with open(self._file(*args), "a") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            for item in data:
                writer.writerow(item)

    def read(self, name: str | None = None, /, *args) -> list[dict[str, Any]]:
        if name is None:
            raise ValueError("name must be a string, not None")
        with open(self._file(*args), newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)


class MongoSaver(AbstractBaseSaver):
    def __init__(self, collection: str, query_key: str | None = None) -> None:
        """
        `query_key` if for type of records that need to be recalled
        and updated, this key will be used to identify the record (of
        course it's possible to find any record with standard pymongo
        methods, this is just a helper for a typical task done by Haymaker).

        """
        self.client = get_mongo_client()
        db = CONFIG["MongoSaver"]["db"]
        self.db = self.client[db]
        self.collection = self.db[collection]
        self.query_key = query_key

    def save(self, data: dict[str, Any], /, *args) -> None:
        try:
            if self.query_key and (key := data.get(self.query_key)):
                result = self.collection.update_one(
                    {self.query_key: key}, {"$set": data}, upsert=True
                )
            elif not all(data.keys()):
                log.error(f"Attempt to save data with empty keys: {list(data.keys())}")
            else:
                result = self.collection.insert_one(data)  # noqa
        except Exception:
            log.exception("Error saving data to MongoDB")
            log.debug(f"Data that caused error: {data}")
            raise
        # log.debug(f"{self}: transaction result: {result}")

    def read(self, key: dict | None = None, /, *args) -> Any:
        if key is None:
            key = {}
        return list(self.collection.find(key))

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(db={self.db}, collection={self.collection})"


class MongoLatestSaver(MongoSaver):
    """Read latest document from mongodb collection."""

    def read(self, *args) -> dict:
        log.debug(f"{self} will read latest.")
        try:
            data = self.collection.find_one(
                {"$query": {}, "$orderby": {"$natural": -1}}
            )
        except Exception:
            log.exception(Exception)
            raise
        return data or {}


class SavingObject(Protocol):
    def read(self, *args) -> Any: ...

    def save(self, *args) -> Any: ...


class AsyncSaveManager:
    """
    Abstract away the process of perfoming asynchronous save and read
    operations. Works as a wrapper for a saver object.
    """

    _tasks: set[asyncio.Task] = set()

    def __init__(self, saver: SavingObject, name: str = ""):
        self.saver = saver
        self.name = f"saver_{name}" if name else "saver"

    def save(self, *args: Any) -> None:
        # save is fire and forget
        fire_and_forget(self.saver.save, *args)

    async def read(self, *args: Any) -> Any:
        # you don't want to proceed until you get the result
        return await make_async(self.saver.read, *args)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}({self.saver!r})"
