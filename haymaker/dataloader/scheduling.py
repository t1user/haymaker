import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Literal, NamedTuple, Optional, Union
from zoneinfo import ZoneInfo

import pandas as pd

from .helpers import duration_in_secs
from .store_wrapper import AsyncStoreView
from .time_policy import normalize_point

log = logging.getLogger(__name__)

SMALL_BAR_MAX_AGE = timedelta(days=180)
EXPIRED_FUTURE_MAX_AGE = timedelta(days=365 * 2)
UNAVAILABLE_EXPIRED_SECTYPES = {"OPT", "FOP", "WAR"}


class Dates(NamedTuple):
    from_date: Union[date, datetime]
    to_date: Union[date, datetime]


RangeKind = Literal["gap", "backfill", "update"]
GapFillMode = Literal["off", "heuristic", "schedule", "auto"]
HEURISTIC_GAP_PASSES = 2
HEURISTIC_MAX_TYPICAL_GAP = timedelta(hours=6)


class PlannedRange(NamedTuple):
    from_date: Union[date, datetime]
    to_date: Union[date, datetime]
    kind: RangeKind
    gap_pattern: Optional["GapPattern"] = None


class GapPattern(NamedTuple):
    """Repeatable short-gap shape used by heuristic and run-local learning."""

    from_time: time
    duration: timedelta
    timezone: str


@dataclass(frozen=True)
class SessionRange:
    """Scheduled open trading session normalized to UTC datetimes."""

    start: datetime
    end: datetime


@dataclass(frozen=True)
class GapCandidate:
    """Detected gap with both request and classification intervals.

    ``from_date`` and ``to_date`` preserve the existing request behavior:
    request through one stored bar after the gap when that point exists. The
    missing interval is narrower and is used for calendar/schedule checks.
    """

    from_date: date | datetime
    to_date: date | datetime
    missing_start: date | datetime
    missing_end: date | datetime
    duration: timedelta

    def dates(self) -> Dates | None:
        """Return request dates when the candidate has a valid range."""

        if self.to_date > self.from_date:
            return Dates(self.from_date, self.to_date)
        return None

    def pattern(self, timezone_name: str | None) -> GapPattern | None:
        """Return a short intraday pattern for heuristic/learned suppression."""

        if not isinstance(self.from_date, datetime):
            return None
        if self.duration > HEURISTIC_MAX_TYPICAL_GAP:
            return None
        tz_name = timezone_name or "UTC"
        local_start = self.from_date.astimezone(ZoneInfo(tz_name))
        return GapPattern(local_start.time(), self.duration, tz_name)


@dataclass
class BaseTask(ABC):
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
class BackfillTask(BaseTask):

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
class UpdateTask(BaseTask):
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
class TaskPlanner:
    """Plan historical download ranges for one contract store view.

    Args:
        store: Preloaded datastore view for the contract being planned.
        head: Earliest available IB timestamp to consider.
        max_period_days: Maximum lookback window for this planning run.
    """

    store: AsyncStoreView
    head: Union[date, datetime]
    max_period_days: int

    def __post_init__(self) -> None:
        self.head = normalize_point(self.head, self.store.bar_size)

    @property
    def start(self) -> Union[date, datetime]:
        """Return the clamped earliest point for this planning run."""

        latest_available = self.store.expiry_or_now()
        candidates = [
            self.head,
            latest_available - timedelta(days=self.max_period_days),
        ]
        if availability_start := historical_availability_start(self.store):
            candidates.append(availability_start)
        return max(candidates)

    def ranges(self) -> list[Dates]:
        """Return planned download ranges in worker execution order."""

        return [Dates(r.from_date, r.to_date) for r in self.planned_ranges()]

    def planned_ranges(self) -> list[PlannedRange]:
        """Return tagged backfill and update ranges in worker execution order."""

        if historical_data_unavailable(self.store):
            return []
        return planned_task_factory(self.store, self.start)


def _gap_ranges(
    store: AsyncStoreView,
    *,
    start: date | datetime | None = None,
    timezone_name: str | None = None,
) -> list[PlannedRange]:
    """Return heuristic datastore gap-fill ranges inferred from the store view."""

    return [
        PlannedRange(
            dates.from_date, dates.to_date, "gap", candidate.pattern(timezone_name)
        )
        for candidate in heuristic_gap_candidates(
            raw_gap_candidates(store, start=start), timezone_name=timezone_name
        )
        if (dates := candidate.dates()) is not None
    ]


def _sec_type(store: AsyncStoreView) -> str:
    """Return the upper-case IB security type for a store view."""

    return str(getattr(store.contract, "secType", "")).upper()


def _is_expired(store: AsyncStoreView) -> bool:
    """Return whether the contract has an exact expiry in the past."""

    expiry = store.expiry
    return expiry is not None and expiry <= store.now


def historical_data_unavailable(store: AsyncStoreView) -> bool:
    """Return whether IB documents this expired instrument as unavailable."""

    if _sec_type(store) in UNAVAILABLE_EXPIRED_SECTYPES and _is_expired(store):
        return True
    if _sec_type(store) == "FUT" and _is_expired(store):
        expiry = store.expiry
        assert expiry is not None
        return store.now > expiry + EXPIRED_FUTURE_MAX_AGE
    return False


def historical_availability_start(
    store: AsyncStoreView,
) -> Union[date, datetime, None]:
    """Return the earliest point IB availability rules allow requesting."""

    candidates: list[date | datetime] = []
    if duration_in_secs(store.bar_size) <= 30 and isinstance(store.now, datetime):
        candidates.append(store.now - SMALL_BAR_MAX_AGE)
    if _sec_type(store) == "FUT" and _is_expired(store):
        expiry = store.expiry
        assert expiry is not None
        candidates.append(expiry - EXPIRED_FUTURE_MAX_AGE)
    if candidates:
        return max(candidates)
    return None


def _planned_range_from_task(task: BaseTask, kind: RangeKind) -> PlannedRange | None:
    """Return a tagged range from a task when it has work."""

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
        ranges.append(_planned_range_from_task(BackfillTask(store, head), "backfill"))
    ranges.append(_planned_range_from_task(UpdateTask(store, head), "update"))
    return [r for r in ranges if r is not None]


def planned_task_factory_with_gaps(
    store: AsyncStoreView,
    head: Union[date, datetime],
    *,
    timezone_name: str | None = None,
) -> list[PlannedRange]:
    """Return tagged gap-fill, backfill, and update ranges."""

    return [
        *_gap_ranges(store, start=head, timezone_name=timezone_name),
        *planned_task_factory(store, head),
    ]


def raw_gap_candidates(
    store: AsyncStoreView,
    *,
    start: date | datetime | None = None,
) -> list[GapCandidate]:
    """Return raw stored-data gaps before schedule/heuristic filtering."""

    data = store.data
    if data is None or len(data.index) < 2:
        return []
    index = pd.Index(data.index).sort_values()
    gaps = pd.Series(index).diff()
    inferred_frequency = gaps.mode()
    if inferred_frequency.empty:
        return []
    expected = inferred_frequency.iloc[0]
    log.debug("inferred frequency: %s", expected)
    out: list[GapCandidate] = []
    for position in range(1, len(index)):
        duration = index[position] - index[position - 1]
        if duration <= expected:
            continue
        request_to = (
            index[position + 1] if position + 1 < len(index) else index[position]
        )
        if start is not None and request_to <= start:
            continue
        request_from = (
            max(index[position - 1], start)
            if start is not None
            else index[position - 1]
        )
        out.append(
            GapCandidate(
                request_from,
                request_to,
                index[position - 1] + expected,
                index[position] - expected,
                duration,
            )
        )
    return out


def heuristic_gap_candidates(
    candidates: list[GapCandidate],
    *,
    timezone_name: str | None,
) -> list[GapCandidate]:
    """Filter raw gap candidates using calendar and repeated short patterns."""

    if not candidates:
        return []
    if not isinstance(candidates[0].from_date, datetime):
        return [
            candidate for candidate in candidates if not _date_gap_is_weekend(candidate)
        ]

    remaining = [
        candidate
        for candidate in candidates
        if not _datetime_gap_is_weekend(candidate, timezone_name)
    ]
    for _ in range(HEURISTIC_GAP_PASSES):
        counts = pd.Series(
            [
                pattern
                for candidate in remaining
                if (pattern := candidate.pattern(timezone_name)) is not None
            ]
        ).value_counts()
        if counts.empty or counts.iloc[0] <= 1:
            break
        typical = counts.index[0]
        remaining = [
            candidate
            for candidate in remaining
            if candidate.pattern(timezone_name) != typical
        ]
    return remaining


def scheduled_gap_candidates(
    candidates: list[GapCandidate],
    sessions: list[SessionRange],
    *,
    timezone_name: str | None,
    typical_patterns: set[GapPattern] | None = None,
) -> list[GapCandidate]:
    """Return candidates whose missing interval overlaps scheduled sessions."""

    typical_patterns = typical_patterns or set()
    fillable = []
    for candidate in candidates:
        if _datetime_gap_is_weekend(candidate, timezone_name):
            continue
        if candidate.pattern(timezone_name) in typical_patterns:
            continue
        if _candidate_overlaps_session(candidate, sessions):
            fillable.append(candidate)
    return fillable


def sessions_from_historical_schedule(schedule: object) -> list[SessionRange]:
    """Parse an IB historical schedule object into UTC session ranges."""

    timezone_name = getattr(schedule, "timeZone", None)
    if not timezone_name:
        return []
    sessions = []
    for session in getattr(schedule, "sessions", []) or []:
        start = _parse_schedule_time(session.startDateTime, timezone_name)
        end = _parse_schedule_time(session.endDateTime, timezone_name)
        sessions.append(SessionRange(start, end))
    return sessions


def schedule_timezone(schedule: object) -> str | None:
    """Return the timezone name advertised by an IB schedule response."""

    timezone_name = getattr(schedule, "timeZone", None)
    return str(timezone_name) if timezone_name else None


def _parse_schedule_time(value: str, timezone_name: str) -> datetime:
    """Parse IB schedule time strings as exchange-local datetimes."""

    return (
        datetime.strptime(value, "%Y%m%d-%H:%M:%S")
        .replace(tzinfo=ZoneInfo(timezone_name))
        .astimezone(timezone.utc)
    )


def _candidate_overlaps_session(
    candidate: GapCandidate, sessions: list[SessionRange]
) -> bool:
    """Return whether a candidate's missing interval intersects a session."""

    if not isinstance(candidate.missing_start, datetime) or not isinstance(
        candidate.missing_end, datetime
    ):
        return True
    missing_start = candidate.missing_start.astimezone(timezone.utc)
    missing_end = candidate.missing_end.astimezone(timezone.utc)
    if missing_end < missing_start:
        return False
    return any(
        missing_start < session.end and missing_end > session.start
        for session in sessions
    )


def _date_gap_is_weekend(candidate: GapCandidate) -> bool:
    """Return whether a date-like missing interval contains only weekend dates."""

    if not isinstance(candidate.missing_start, date) or isinstance(
        candidate.missing_start, datetime
    ):
        return False
    days = pd.date_range(candidate.missing_start, candidate.missing_end, freq="D")
    return len(days) > 0 and all(day.weekday() >= 5 for day in days)


def _datetime_gap_is_weekend(
    candidate: GapCandidate, timezone_name: str | None
) -> bool:
    """Return whether an intraday missing interval is fully Saturday/Sunday."""

    if not isinstance(candidate.missing_start, datetime) or not isinstance(
        candidate.missing_end, datetime
    ):
        return False
    tz = ZoneInfo(timezone_name or "UTC")
    start = candidate.missing_start.astimezone(tz)
    end = candidate.missing_end.astimezone(tz)
    if end < start:
        return False
    days = pd.date_range(start.date(), end.date(), freq="D")
    return len(days) > 0 and all(day.weekday() >= 5 for day in days)
