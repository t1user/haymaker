import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Literal, NamedTuple, Optional, Self, Union

import pandas as pd

from .store_wrapper import AsyncStoreView
from .time_policy import is_date_bar, normalize_point

log = logging.getLogger(__name__)


class Dates(NamedTuple):
    from_date: Union[date, datetime]
    to_date: Union[date, datetime]


RangeKind = Literal["gap", "backfill", "update"]


class PlannedRange(NamedTuple):
    from_date: Union[date, datetime]
    to_date: Union[date, datetime]
    kind: RangeKind


@dataclass
class TaskFactory(ABC):
    store: AsyncStoreView
    head: Union[date, datetime]  # HeadTimeStamp earliest point for which IB has data

    @property
    @abstractmethod
    def from_date(self) -> Union[date, datetime, None]: ...

    @property
    @abstractmethod
    def to_date(self) -> Union[date, datetime, None]: ...

    def __post_init__(self) -> None:
        self.head = normalize_point(self.head, self.store.bar_size)

    def dates(self) -> Optional[Dates]:
        from_date = self.from_date
        to_date = self.to_date
        if from_date and to_date and (to_date > from_date):
            return Dates(from_date, to_date)
        else:
            return None

    def __str__(self) -> str:
        return f"<{self.__class__.__name__} from: {self.from_date} to: {self.to_date}"


@dataclass
class BackfillFactory(TaskFactory):

    @property
    def from_date(self) -> Union[date, datetime, None]:
        return self.head

    @property
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

    @property
    def to_date(self) -> Union[date, datetime, None]:
        eon: date | datetime | None = None
        try:
            eon = self.store.expiry_or_now()
            if self.store.to_date and (eon > self.store.to_date):
                return eon
            else:
                return None
        except TypeError:
            log.exception(f"{self.store.contract=} | {eon=} {self.store.to_date=}")
            raise


@dataclass
class GapFillFactory:
    from_date: Union[date, datetime]
    to_date: Union[date, datetime]

    def dates(self) -> Optional[Dates]:
        if self.from_date and self.to_date and (self.to_date > self.from_date):
            return Dates(self.from_date, self.to_date)
        else:
            return None

    def __str__(self) -> str:
        return f"<{self.__class__.__name__} from: {self.from_date} to: {self.to_date}"

    @classmethod
    def gap_factory(cls, store: AsyncStoreView) -> list[Self]:
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
        if is_date_bar(store.bar_size):
            return cls.from_list(list(out[["start", "stop"]].itertuples(index=False)))
        out["start_time"] = out["start"].apply(lambda x: x.time())
        cutoff_time = out["start_time"].mode()[0]
        log.debug(f"inferred cutoff time: {cutoff_time}")
        non_standard_gaps = out[out["start_time"] != cutoff_time].reset_index(drop=True)
        return cls.from_list(
            list(non_standard_gaps[["start", "stop"]].itertuples(index=False))
        )

    @classmethod
    def from_list(
        cls, items: list[tuple[date | datetime, date | datetime]]
    ) -> list[Self]:
        return [cls(*i) for i in items]


@dataclass
class TaskPlanner:
    """Plan historical download ranges for one contract store view.

    Args:
        store: Preloaded datastore view for the contract being planned.
        head: Earliest available IB timestamp to consider.
        max_period_days: Maximum lookback window for this planning run.
        fill_gaps: Whether to include datastore gap-fill ranges before normal
            backfill/update ranges.
    """

    store: AsyncStoreView
    head: Union[date, datetime]
    max_period_days: int
    fill_gaps: bool = True

    def __post_init__(self) -> None:
        self.head = normalize_point(self.head, self.store.bar_size)

    @property
    def start(self) -> Union[date, datetime]:
        """Return the clamped earliest point for this planning run."""

        latest_available = self.store.expiry_or_now()
        return max(self.head, latest_available - timedelta(days=self.max_period_days))

    def ranges(self) -> list[Dates]:
        """Return planned download ranges in worker execution order."""

        return [Dates(r.from_date, r.to_date) for r in self.planned_ranges()]

    def planned_ranges(self) -> list[PlannedRange]:
        """Return tagged planned download ranges in worker execution order."""

        if self.fill_gaps:
            return planned_task_factory_with_gaps(self.store, self.start)
        return planned_task_factory(self.store, self.start)


def _gap_ranges(store: AsyncStoreView) -> list[Dates]:
    """Return datastore gap-fill ranges inferred from the store view."""

    return [
        dates
        for gap in GapFillFactory.gap_factory(store)
        if (dates := gap.dates()) is not None
    ]


def _range_from_factory(task: TaskFactory, kind: RangeKind) -> PlannedRange | None:
    """Return a tagged range from a task factory when it has work."""

    dates = task.dates()
    if dates is None:
        return None
    return PlannedRange(dates.from_date, dates.to_date, kind)


def planned_task_factory(
    store: AsyncStoreView, head: Union[date, datetime]
) -> list[PlannedRange]:
    """Return tagged backfill and update ranges."""

    ranges = []
    if not store.backfill_exhausted:
        ranges.append(_range_from_factory(BackfillFactory(store, head), "backfill"))
    ranges.append(_range_from_factory(UpdateFactory(store, head), "update"))
    return [r for r in ranges if r is not None]


def planned_task_factory_with_gaps(
    store: AsyncStoreView, head: Union[date, datetime]
) -> list[PlannedRange]:
    """Return tagged gap-fill, backfill, and update ranges."""

    return [
        *[PlannedRange(d.from_date, d.to_date, "gap") for d in _gap_ranges(store)],
        *planned_task_factory(store, head),
    ]


def task_factory(store: AsyncStoreView, head: Union[date, datetime]) -> list[Dates]:
    """Return backfill and update ranges for compatibility callers."""

    return [Dates(r.from_date, r.to_date) for r in planned_task_factory(store, head)]


def task_factory_with_gaps(
    store: AsyncStoreView, head: Union[date, datetime]
) -> list[Dates]:
    """Return gap-fill, backfill, and update ranges for compatibility callers."""

    return [
        Dates(r.from_date, r.to_date)
        for r in planned_task_factory_with_gaps(store, head)
    ]
