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
from pymongo import MongoClient  # type: ignore

from .config import CONFIG as config
from .misc import default_path

log = logging.getLogger(__name__)


CONFIG = config.get("saver") or {}


class AbstractBaseSaveManager(ABC):
    def __init__(self, saver: AbstractBaseSaver, name="", timestamp: bool = True):
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


async def saving_function(data: Any, saver: AbstractBaseSaver, *args: str):
    """
    Funcion that actually peforms all saving.
    :class:`AbstractBaseSaver` objects wishing to save should connect
    events to it or await it directly.
    """
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, saver.save, data, *args)


def error_reporting_function(event, exception: Exception) -> None:
    log.error(f"Event error: {event.name()}: {exception}", exc_info=True)


class AsyncSaveManager(AbstractBaseSaveManager):
    """
    Abstract away the process of perfoming save operations.  Use
    :class:`eventkit.Event` to put :func:`.saving_function` into
    asyncio loop.

    This class can be used both: as a descriptor and a regular class:

    ```
    class Example:

        save = SaveManager(saver_instance)

        ...
    ```

    or:

    ```
    class Example:

        def __init__(self):
            self.save = SaveManager(saver_instance)

        ...
    ```
    """

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

    def __init__(self, name: str = "", timestamp: bool = True) -> None:
        if timestamp:
            timestamp_ = datetime.now(timezone.utc).strftime("%Y%m%d_%H_%M")
            self.name = f"{name}_{timestamp_}"
        else:
            self.name = name

    def name_str(self, *args: str) -> str:
        """
        Return string under which the data is to be saved.  Timestamp
        and/or 'name' may be included in the name depending on how the
        object was initialized.

        This name can be used by :meth:`.save`to build filename,
        database collection name, key-value store key, etc.
        """
        args_str = "_".join(args)
        return f"{self.name}_{args_str}"

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


class PickleSaver(AbstractBaseSaver):
    def __init__(self, folder: str, name: str = "", timestamp: bool = True) -> None:
        self.path = default_path(folder)
        super().__init__(name, timestamp)

    def _file(self, *args):
        return f"{self.path}/{self.name_str(*args)}.pickle"

    def save(self, data: pd.DataFrame, *args: str) -> None:
        if isinstance(data, pd.DataFrame):
            data.to_pickle(self._file(*args))
        else:
            with open(self._file(*args), "wb") as f:
                f.write(pickle.dumps(data))

    def __repr__(self):
        return f"PickleSaver({self.path}, {self.name})"


class CsvSaver(AbstractBaseSaver):
    _fieldnames: list[str] | None

    def __init__(self, folder: str, name: str = "", timestamp: bool = True) -> None:
        self.path = default_path(folder)
        super().__init__(name, timestamp)

    @property
    def _file(self):
        return f"{self.path}/{self.name_str()}.csv"

    def _create_header(self, keys: Collection) -> None:
        with open(self._file, "a") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()

    def save(self, data: dict[str, Any], *args: str) -> None:
        if not Path(self._file).exists():
            self._create_header(data.keys())
        with open(self._file, "a") as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            writer.writerow(data)

    def save_many(self, data: list[dict[str, Any]]) -> None:
        self._create_header(data[0].keys())
        with open(self._file, "a") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            for item in data:
                writer.writerow(item)

    def __repr__(self):
        return f"CsvSaver({self.path}, {self.name})"


class ArcticSaver(AbstractBaseSaver):
    """
    Saver for Arctic VersionStore.

    WORKS ONLY ON DATAFRAMES (or does it?)
    """

    def __init__(
        self,
        library: str = "",
        name: str = "",
        timestamp=False,
    ) -> None:
        """
        Library given at init, collection determined by self.name_str.
        """
        self.host = CONFIG["ArcticSaver"]["host"]
        self.library = library or CONFIG["ArcticSaver"]["library"]
        self.db = Arctic(self.host)
        self.db.initialize_library(self.library)
        self.store = self.db[self.library]
        super().__init__(name, timestamp)

    def save(self, data: pd.DataFrame, *args: str):
        self.store.write(self.name_str(*args), data)

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
        self, collection: str, query_key: str | None = None, timestamp: bool = False
    ) -> None:
        """
        `query_key` if for type of records that need to be recalled
        and updated, this key will be used to identify the record (of
        course it's possible to find any record with standard pymongo
        methods, this is just a helper for a typical task done by Haymaker).

        `timestamp` not used in MongoSaver
        """
        host = CONFIG["MongoSaver"]["host"]
        port = CONFIG["MongoSaver"]["port"]
        db = CONFIG["MongoSaver"]["db"]
        self.client = MongoClient(host, port)
        self.db = self.client[db]
        self.collection = self.db[collection]
        self.query_key = query_key
        super().__init__("", timestamp)

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
