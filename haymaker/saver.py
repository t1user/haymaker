from __future__ import annotations

import asyncio
import csv
import logging
import pickle
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Collection

import pandas as pd
from arctic import Arctic  # type: ignore

from .config import CONFIG as config
from .databases import get_mongo_client
from .misc import default_path, name_str

log = logging.getLogger(__name__)


CONFIG = config.get("saver") or {}

ARCTIC_SAVER_LIBRARY = CONFIG["ArcticSaver"]["library"]


class AsyncSaveManager:
    """
    Abstract away the process of perfoming asynchronous save and read
    operations. Works as a wrapper for a saver object.
    """

    _tasks: set[asyncio.Task] = set()

    def __init__(self, saver: AbstractBaseSaver, name: str = ""):
        self.saver = saver
        if name:
            self.name = f"saver_{name}"
        else:
            self.name = "saver"

    @staticmethod
    async def async_runner(func: Callable, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, func, *args)

    def save(self, data: Any, *args: str) -> None:
        # save is fire and forget
        task = asyncio.create_task(
            self.async_runner(self.saver.save, data, *args), name=self.name
        )

        # Below is preventing tasks from being garbage collected
        # Add task to the set. This creates a strong reference.
        self._tasks.add(task)

        # To prevent keeping references to finished tasks forever,
        # make each task remove its own reference from the set after
        # completion:
        task.add_done_callback(self._tasks.discard)

    async def read(self, key: Any = None) -> Any:
        # you don't want to proceed until you get the result
        return await self.async_runner(self.saver.read, key)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}({self.saver})"


class AbstractBaseSaver(ABC):
    """
    Api for saving data during trading/simulation.
    """

    @abstractmethod
    def save(self, data: Any, *args: str) -> None: ...

    @abstractmethod
    def read(self, key=None): ...

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

    def save(self, data: Any, *args: str) -> None:
        if isinstance(data, pd.DataFrame):
            data.to_pickle(self._file(*args))
        else:
            with open(self._file(*args), "wb") as f:
                pickle.dump(data, f)

    def read(self, name: str | None = None) -> Any:
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

    def save(self, data: dict[str, Any], *args: str) -> None:
        if not self._path_exists(*args):
            self._create_header(data.keys(), *args)

        with open(self._file(*args), "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            writer.writerow(data)

    def save_many(self, data: list[dict[str, Any]], *args: str, override=True) -> None:
        if not data:
            raise ValueError("Cannot save empty data list")
        if override or not self._path_exists(*args):
            self._create_header(data[0].keys())
        with open(self._file(*args), "a") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            for item in data:
                writer.writerow(item)

    def read(self, name: str | None = None, *args) -> list[dict[str, Any]]:
        if name is None:
            raise ValueError("name must be a string, not None")
        with open(self._file(*args), newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)


class ArcticSaver(AbstractBaseSaver):
    """
    Saver for Arctic VersionStore.

    WORKS ONLY ON DATAFRAMES
    """

    def __init__(
        self,
        name: str = "",
        library: str = ARCTIC_SAVER_LIBRARY,
        use_timestamp=False,
    ) -> None:
        """
        Library given at init, collection determined by misc.name_str.
        """
        self.host = get_mongo_client()
        self.library = library
        self.db = Arctic(self.host)
        self.db.initialize_library(self.library)
        self.store = self.db[self.library]
        self.name = name
        self.timestamp = datetime.now(timezone.utc) if use_timestamp else None

    def _make_key(self, *args: str) -> str:
        return name_str(self.name, *args, timestamp=self.timestamp)

    def save(self, data: pd.DataFrame, *args: str):
        self.store.write(self._make_key(*args), data)

    def keys(self) -> list[str]:
        return self.store.list_symbols()

    def read(self, key: str | None = None) -> pd.DataFrame:
        if key is None:
            raise ValueError("key must be string not None")
        return self.store.read(key)

    def __str__(self) -> str:
        return (
            f"<ArcticSaver(host={self.host}, library={self.library}, "
            f"name={self.name})>"
        )


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

    def save(self, data: dict[str, Any], *args) -> None:
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

    def read(self, key: dict | None = None) -> Any:
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
