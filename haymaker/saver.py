from __future__ import annotations

import asyncio
import csv
import logging
import pickle
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Collection

import eventkit as ev  # type: ignore
import pandas as pd
from arctic import Arctic  # type: ignore

from .config import CONFIG as config
from .databases import get_mongo_client
from .misc import default_path

log = logging.getLogger(__name__)


CONFIG = config.get("saver") or {}

ARCTIC_SAVER_LIBRARY = CONFIG["ArcticSaver"]["library"]


class AbstractBaseSaveManager(ABC):
    """
    Run savers, be an interface between :class:`Atom` and :class:`AbstractBaseSaver`.
    In particular, the purpose of this class is to allow for running savers
    asynchronously as a background task in a separate thread or process.

    Classes implementing this abstract class can be used both: as
    a descriptor and a regular instance attribute, i.e.

    * as a descriptor:
        ```
        class Example:
            save = SaveManager(saver_instance)
        ```
    * as an instance attribute:
        ```
        class Example:
            def __init__(self, saver_instance):
                self.save = SaveManager(saver_instance)
        ```
    """

    def __init__(self, saver: AbstractBaseSaver, name="", use_timestamp: bool = True):
        self.saver = saver

    def __get__(self, obj, objtype=None) -> Callable:
        return self.save

    @abstractmethod
    def save(self, data: Any, *args: str): ...

    def __call__(self, data: Any, *args: str):
        return self.save(data, *args)

    def __repr__(self):
        return f"{self.__class__.__qualname__}({self.saver})"


async def async_runner(func: Callable, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, func, *args)


class AsyncSaveManager(AbstractBaseSaveManager):
    """
    Abstract away the process of perfoming save operations.  Use
    :class:`eventkit.Event` to put :func:`.saving_function` into
    asyncio loop.
    """

    @staticmethod
    async def saving_function(data: Any, saver: AbstractBaseSaver, *args: str):
        """
        Function that actually peforms all saving.
        :class:`AbstractBaseSaver` objects wishing to save should connect
        events to it or await it directly.
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, saver.save, data, *args)

    @staticmethod
    def error_reporting_function(event, exception: Exception) -> None:
        log.error(f"Event error: {event.name()}: {exception}", exc_info=True)

    saveEvent = ev.Event("saveEvent")
    saveEvent.connect(saving_function, error=error_reporting_function)

    def save(self, data: Any, *args: str):
        self.saveEvent.emit(data, self.saver, *args)


class SyncSaveManager(AbstractBaseSaveManager):
    """
    Saves immediately in the current thread without intermediation of
    asyncio loop.
    """

    def save(self, data: Any, *args: str):
        self.saver.save(data, *args)


class AbstractBaseSaver(ABC):
    """
    Api for saving data during trading/simulation.
    """

    def __init__(
        self, name: str = "", use_timestamp: bool = True, *args, **kwargs
    ) -> None:
        if use_timestamp:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H_%M")
            self.name = f"{name}_{timestamp}"
        else:
            self.name = name

    @abstractmethod
    def save(self, data: Any, *args: str) -> None:
        """
        Save data to store.

        Args:
        -----

        data: data to be saved

        *args: any additional identifiers to be included in collection
        name
        """
        ...

    def save_many(self, data: list[dict[str, Any]]):
        raise NotImplementedError

    def read(self, key):
        raise NotImplementedError

    def read_latest(self):
        raise NotImplementedError

    def delete(self, query: dict) -> None:
        raise NotImplementedError

    @staticmethod
    def name_str(
        name: str, *args: str, timestamp: datetime | None = None, join_str: str = "_"
    ) -> str:
        """
        Helper method to generate file/collection names if necessary.

        Args:
        -----

        name - file/collection name

        *args - additional strings to be concatinated with `name`

        use_timestamp - if True timestamp (of when the object was instantiated rather
        than the time of every save) will be added after the name

        join_str - symbol used to join strings

        This `name_str` can be used by :meth:`.save`to build filename,
        database collection name, key-value store key, etc.
        """
        if timestamp:
            args_str = join_str.join((name, *args, timestamp.strftime("%Y%m%d_%H_%M")))
        else:
            args_str = join_str.join((name, *args))
        return args_str


class AbstractBaseFileSaver(AbstractBaseSaver):
    _suffix = ""

    def __init__(self, name: str, folder: str = "", use_timestamp: bool = True) -> None:
        self.path = default_path(folder)
        self.name = name
        self.timestamp = datetime.now(timezone.utc) if use_timestamp else None

    def _file(self, *args) -> str:
        return (
            f"{self.path}/"
            f"{self.name_str(self.name, *args, timestamp=self.timestamp)}"
            f".{self._suffix}"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}({self.path}, {self.name})"


class PickleSaver(AbstractBaseFileSaver):
    _suffix = "pickle"

    def save(self, data: pd.DataFrame, *args: str) -> None:
        if isinstance(data, pd.DataFrame):
            data.to_pickle(self._file(*args))
        else:
            with open(self._file(*args), "wb") as f:
                f.write(pickle.dumps(data))


class CsvSaver(AbstractBaseFileSaver):
    _suffix = "csv"
    _fieldnames: list[str] | None

    def __init__(self, folder: str, name: str = "", use_timestamp: bool = True) -> None:
        self.path = default_path(folder)
        self.name = name
        self.use_timestamp = use_timestamp

    def _create_header(self, keys: Collection, *args: str) -> None:
        with open(self._file(*args), "a") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()

    def save(self, data: dict[str, Any], *args: str) -> None:
        if not Path(self._file()).exists():
            self._create_header(data.keys(), *args)
        with open(self._file(*args), "a") as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            writer.writerow(data)

    def save_many(self, data: list[dict[str, Any]], *args: str) -> None:
        self._create_header(data[0].keys())
        with open(self._file(*args), "a") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            for item in data:
                writer.writerow(item)


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
        Library given at init, collection determined by self.name_str.
        """
        self.host = get_mongo_client()
        self.library = library
        self.db = Arctic(self.host)
        self.db.initialize_library(self.library)
        self.store = self.db[self.library]
        self.name = name
        self.timestamp = datetime.now(timezone.utc) if use_timestamp else None

    def _make_key(self, *args: str) -> str:
        return self.name_str(self.name, *args, timestamp=self.timestamp)

    def save(self, data: pd.DataFrame, *args: str):
        self.store.write(self._make_key(*args), data)

    def keys(self) -> list[str]:
        return self.store.list_symbols()

    def read(self, key: str) -> pd.DataFrame:
        return self.store.read(key)

    def __str__(self):
        return (
            f"ArcticSaver(host={self.host}, library={self.library}, "
            f"name={self.name})"
        )


class MongoSaver(AbstractBaseSaver):
    def __init__(
        self,
        collection: str,
        query_key: str | None = None,
        use_timestamp: bool = False,
    ) -> None:
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
                log.error(f"Attempt to save with wrong keys: {list(data.keys())}")
            else:
                result = self.collection.insert_one(data)  # noqa
        except Exception:
            log.exception(Exception)
            log.debug(f"Data that caused error: {data}")
            raise
        # log.debug(f"{self}: transaction result: {result}")

    def save_many(self, data: list[dict[str, Any]]) -> None:
        self.collection.insert_many(data)

    def read(self, key: dict | None = None) -> list:
        if key is None:
            key = {}
        return [i for i in self.collection.find(key)]

    def read_latest(self) -> dict:
        log.debug(f"{self} will read latest.")
        try:
            data = self.collection.find_one(
                {"$query": {}, "$orderby": {"$natural": -1}}
            )
        except Exception:
            log.exception(Exception)
            raise
        return data or {}

    def delete(self, query: dict) -> None:
        log.debug(f"Will mock delete data: {query}. DELETE METHOD NOT IMPLEMENTED")

    def __repr__(self) -> str:
        return f"MongoSaver(db={self.db}, collection={self.collection})"
