from __future__ import annotations

import asyncio
import csv
import logging
import pickle
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import eventkit as ev  # type: ignore
import pandas as pd
from arctic import Arctic
from pymongo import MongoClient  # type: ignore

from ib_tools.utilities import default_path

log = logging.getLogger(__name__)


async def saving_function(data: Any, saver: AbstractBaseSaver, *args: str):
    """
    Funcion that actually peforms all saving.  All objects wishing to
    save should connect saving events to it or await it directly.
    """
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, saver.save, data, *args)


class SaveManager:
    """
    This class can be used both: as a descriptor and a regular class.
    """

    saveEvent = ev.Event("saveEvent")
    saveEvent += saving_function

    def __init__(self, saver: AbstractBaseSaver, note="", timestamp: bool = True):
        self.saver = saver

    def __get__(self, obj, objtype=None) -> Callable:
        return self.save

    def save(self, data: Any, *args: str):
        self.saveEvent.emit(data, self.saver, *args)

    __call__ = save


class AbstractBaseSaver(ABC):
    """
    Api for saving data during trading/simulation.
    """

    def __init__(self, note: str = "", timestamp: bool = True):
        if timestamp:
            timestamp_ = datetime.now(timezone.utc).strftime("%Y%m%d_%H_%M")
            self.note = f"{note}_{timestamp_}"
        else:
            self.note = note

    def name_str(self, *args: str) -> str:
        """
        Return string under which the data is to be saved.  Timestamp
        and/or note may be included in the name depending on how the
        object was initialized.

        This name can be used by :meth:`.save`to build filename,
        database collection name, key-value store key, etc.
        """
        args_str = "_".join(args)
        return f"{self.note}_{args_str}"

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


class PickleSaver(AbstractBaseSaver):
    def __init__(self, folder: str, note: str = "", timestamp: bool = True) -> None:
        self.path = default_path(folder)
        super().__init__(note, timestamp)

    def _file(self, *args):
        return f"{self.path}/{self.name_str(*args)}.pickle"

    def save(self, data: pd.DataFrame, *args: str) -> None:
        if isinstance(data, pd.DataFrame):
            data.to_pickle(self._file(*args))
        else:
            with open(self._file(*args), "wb") as f:
                f.write(pickle.dumps(data))

    def __repr__(self):
        return f"PickleSaver({self.path}, {self.note})"


class CsvSaver(AbstractBaseSaver):
    _fieldnames: Optional[list[str]]

    def __init__(self, folder: str, note: str = "", timestamp: bool = True):
        self.path = default_path(folder)
        self._fieldnames = None
        super().__init__(note, timestamp)

    @property
    def _file(self):
        return f"{self.path}/{self.name_str()}.csv"

    def _create_header(self) -> None:
        with open(self._file, "w") as f:
            assert self._fieldnames
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writeheader()

    def save(self, data: dict[str, Any], *args: str) -> None:
        if not self._fieldnames:
            self._fieldnames = list(data.keys())
            self._create_header()
        with open(self._file, "a") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writerow(data)

    def save_many(self, data: list[dict[str, Any]]) -> None:
        self._fieldnames = list(data[0].keys())
        self._create_header()
        with open(self._file, "a") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            for item in data:
                writer.writerow(item)

    def __repr__(self):
        return f"CsvSaver({self.path}, {self.note})"


class ArcticSaver(AbstractBaseSaver):
    """
    Saver for Arctic VersionStore.

    WORKS ONLY ON DATAFRAMES (or does it?)
    """

    def __init__(
        self,
        host: str = "localhost",
        library: str = "test_log",
        note: str = "",
        timestamp=False,
    ) -> None:
        """
        Library given at init, collection determined by self.name_str.
        """
        self.host = host
        self.library = library
        self.db = Arctic(host)
        self.db.initialize_library(library)
        self.store = self.db[library]
        super().__init__(note, timestamp)

    def save(self, data: pd.DataFrame, *args: str):
        self.store.write(self.name_str(*args), data)

    def keys(self) -> list[str]:
        return self.store.list_symbols()

    def read(self, key: str) -> pd.DataFrame:
        return self.store.read(key)

    def __str__(self):
        return (
            f"ArcticSaver(host={self.host}, library={self.library}, "
            f"note={self.note})"
        )


class MongoSaver(AbstractBaseSaver):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 27017,
        db: str = "blotter",
        collection: "str" = "test_blotter",
    ) -> None:
        self.client = MongoClient(host, port)
        self.db = self.client[db]
        self.collection = self.db[collection]

    def save(self, data: dict[str, Any], *args) -> None:
        self.collection.insert_one(data)

    def save_many(self, data: list[dict[str, Any]]) -> None:
        self.collection.insert_many(data)

    def read(self) -> list:
        return [i for i in self.collection.find()]

    def __repr__(self) -> str:
        return f"MongoBlotter(db={self.db}, collection={self.collection})"
