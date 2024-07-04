import logging
from abc import ABC, abstractproperty
from dataclasses import dataclass
from datetime import date, datetime
from functools import cached_property
from typing import NamedTuple, Optional, Self, Union

import pandas as pd

from .store_wrapper import StoreWrapper

log = logging.getLogger(__name__)


class Dates(NamedTuple):
    from_date: Union[date, datetime]
    to_date: Union[date, datetime]


@dataclass
class TaskFactory(ABC):
    store: StoreWrapper
    head: Union[date, datetime]  # HeadTimeStamp earliest point for which IB has data

    @abstractproperty
    def from_date(self) -> Union[date, datetime, None]: ...

    @abstractproperty
    def to_date(self) -> Union[date, datetime, None]: ...

    def dates(self) -> Optional[Dates]:
        if self.from_date and self.to_date and (self.to_date > self.from_date):
            return Dates(self.from_date, self.to_date)
        else:
            return None

    def __str__(self) -> str:
        return f"<{self.__class__.__name__} from: {self.from_date} to: {self.to_date}"


@dataclass
class BackfillFactory(TaskFactory):

    @property
    def from_date(self) -> Union[date, datetime, None]:
        return self.head

    @cached_property
    def to_date(self) -> Union[date, datetime, None]:
        # data present in datastore
        if self.store.from_date:
            return self.store.from_date if self.store.from_date > self.head else None
        # data not in datastore yet
        else:
            return self.store.expiry_or_now()


@dataclass
class UpdateFactory(TaskFactory):
    """
    If no data in datastore for this symbol, there is no update.  All
    download handles by backfill.
    """

    @property
    def from_date(self) -> Union[date, datetime, None]:
        # from last point available in datastore...
        return self.store.to_date

    @cached_property
    def to_date(self) -> Union[date, datetime, None]:
        if self.store.to_date and (
            (eon := self.store.expiry_or_now()) > self.store.to_date
        ):
            return eon
        else:
            return None


@dataclass
class GapFillFactory:
    from_date: Union[date, datetime]
    to_date: Union[date, datetime]

    dates = TaskFactory.dates
    __str__ = TaskFactory.__str__

    @classmethod
    def gap_factory(cls, store: StoreWrapper) -> list[Self]:
        if (data := store.data) is None:
            return []
        data = data.copy()
        data["timestamp"] = data.index
        data["gap"] = data["timestamp"].diff()
        inferred_frequency = data["gap"].mode()[0]
        log.debug(f"inferred frequency: {inferred_frequency}")
        data["gap_bool"] = data["gap"] > inferred_frequency
        data["start"] = data.timestamp.shift()
        data["stop"] = data.timestamp.shift(-1)
        gaps = data[data["gap_bool"]]
        out = pd.DataFrame({"start": gaps["start"], "stop": gaps["stop"]}).reset_index(
            drop=True
        )
        out = out[1:]
        if len(out) == 0:
            return []
        out["start_time"] = out["start"].apply(lambda x: x.time())
        cutoff_time = out["start_time"].mode()[0]
        log.debug(f"inferred cutoff time: {cutoff_time}")
        non_standard_gaps = out[out["start_time"] != cutoff_time].reset_index(drop=True)
        return cls.from_list(
            list(non_standard_gaps[["start", "stop"]].itertuples(index=False))
        )

    @classmethod
    def from_list(cls, items: list[tuple[datetime, datetime]]) -> list[Self]:
        return [cls(*i) for i in items]


def task_factory(store: StoreWrapper, head: Union[date, datetime]) -> list[Dates]:
    return [
        t
        for t in [
            task(store, head).dates() for task in (BackfillFactory, UpdateFactory)
        ]
        if t
    ]


def task_factory_with_gaps(store: StoreWrapper, head: Union[date, datetime]):
    return [*task_factory(store, head), *GapFillFactory.gap_factory(store)]
