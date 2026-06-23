"""Asynchronous Interactive Brokers historical data downloader.

The module is built around a producer/worker queue. A manager discovers the
contracts and datastore ranges that need data, producers enqueue download jobs,
and workers download historical chunks through a session-scoped request pacer.
Connection recovery is owned by the dataloader supervisor integration; request
pacing, failure recording, and resume-in-memory state live in the dataloader
session.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import AsyncGenerator, Optional, TypedDict, cast

import ib_insync as ibi
import pandas as pd
from tqdm import tqdm

from haymaker.config import CONFIG
from haymaker.databases import get_mongo_client
from haymaker.datastore import AsyncAbstractBaseStore, AsyncArcticStore
from haymaker.logging import setup_logging
from haymaker.validators import bar_size_validator, wts_validator

from . import helpers
from .connect import connection
from .contract_selectors import ContractSelector
from .pacer import RequestPacing
from .scheduling import (
    GapFillMode,
    GapPattern,
    PlannedRange,
    RangeKind,
    TaskPlanner,
    heuristic_gap_candidates,
    historical_data_unavailable,
    raw_gap_candidates,
    scheduled_gap_candidates,
    schedule_timezone,
    sessions_from_historical_schedule,
)
from .store_wrapper import AsyncStoreView, HistorySink
from .task_logger import create_task
from .time_policy import normalize_point

setup_logging(CONFIG.get("logging_config"))

log = logging.getLogger(__name__)

BARSIZE: str = bar_size_validator(CONFIG["barSize"])
WTS: str = wts_validator(CONFIG["wts"])
MAX_BARS: int = CONFIG.get("max_bars", 50000)
GAP_FILL_MODE: GapFillMode = CONFIG.get("gap_fill_mode", "off")
USE_RTH: bool = CONFIG.get("useRTH", False)
AUTO_SAVE_INTERVAL: int = CONFIG.get("auto_save_interval", 0)
NUMBER_OF_WORKERS: int = CONFIG.get("number_of_workers", 20)
SOURCE: str = CONFIG["source"]
PACER_NO_RESTRICTION: bool = CONFIG.get("pacer_no_restriction", False)
PACER_ALLOWANCE_FRACTION: float = CONFIG.get("pacer_allowance_fraction", 1.0)
MAX_PERIOD: int = CONFIG.get("max_period", 30)

if PACER_ALLOWANCE_FRACTION <= 0:
    raise ValueError("pacer_allowance_fraction must be greater than 0")

log.debug(
    f"settings: {BARSIZE=}, {WTS=}, {MAX_BARS=}, {GAP_FILL_MODE=}, {USE_RTH=}, "
    f"{AUTO_SAVE_INTERVAL=}, {NUMBER_OF_WORKERS=}, {MAX_PERIOD=}, "
    f"{PACER_ALLOWANCE_FRACTION=}"
)


def create_dataloader_store(
    wts: str = WTS,
    bar_size: str = BARSIZE,
) -> AsyncAbstractBaseStore:
    """Create the Arctic-backed async datastore used by dataloader jobs.

    Args:
        wts: IB ``whatToShow`` value used to derive the Arctic library name.
        bar_size: IB ``barSizeSetting`` value used to derive the library name.

    Returns:
        Async datastore instance for the configured historical data library.
    """

    lib = f"{wts_validator(wts)}_{bar_size_validator(bar_size)}".replace(" ", "_")
    return AsyncArcticStore(lib=lib, host=get_mongo_client())


def request_pacing_factory(
    ib: ibi.IB,
    bar_size: str = BARSIZE,
    wts: str = WTS,
) -> RequestPacing:
    """Create request pacing from dataloader request policy."""

    return RequestPacing(
        ib,
        bar_size,
        wts,
        no_restriction=PACER_NO_RESTRICTION,
        allowance_fraction=PACER_ALLOWANCE_FRACTION,
    )


def current_session_now() -> date | datetime:
    """Return current time normalized for the configured bar size."""

    return normalize_point(datetime.now(timezone.utc), BARSIZE)


class Params(TypedDict):
    contract: ibi.Contract
    endDateTime: datetime | date | str
    durationStr: str


@dataclass
class DownloadFailure:
    """Record one failed dataloader download job."""

    job: DownloadJob
    error: BaseException
    when: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class DownloadFailureRegistry:
    """Collect failed download jobs so the session can report them at completion."""

    failures: list[DownloadFailure] = field(default_factory=list)

    def record(self, job: DownloadJob, error: BaseException) -> None:
        """Record one failed download job."""

        self.failures.append(DownloadFailure(job, error))

    def log_summary(self) -> None:
        """Log a summary of failed download jobs."""

        if not self.failures:
            log.info("Dataloader completed with no failed download jobs.")
            return

        log.warning(
            f"Dataloader completed with {len(self.failures)} failed download jobs."
        )
        for failure in self.failures:
            log.warning(
                "%s failed at %s with %s: %s",
                failure.job,
                failure.when.isoformat(),
                type(failure.error).__name__,
                failure.error,
            )


CONNECTION_FAILURE_ERRORS = (ConnectionError, TimeoutError)
SCHEDULE_CHUNK_DAYS = 120
LEARNED_GAP_FAILURES = 2


class DownloadRequestError(Exception):
    """Raised when a broker historical-data request fails for one job."""


@dataclass
class RunGapLearner:
    """Track short no-data gap patterns for the current dataloader run only."""

    failures: Counter[GapPattern] = field(default_factory=Counter)

    def record_empty_gap(self, pattern: GapPattern | None) -> None:
        """Record a no-data gap-fill request when it has a learnable pattern."""

        if pattern is not None:
            self.failures[pattern] += 1

    @property
    def typical_patterns(self) -> set[GapPattern]:
        """Return patterns that failed often enough in this run to suppress."""

        return {
            pattern
            for pattern, count in self.failures.items()
            if count >= LEARNED_GAP_FAILURES
        }


def pulse_factory(interval: int) -> ibi.Event | None:
    if interval:
        return ibi.Event().timerange(interval, None, interval)
    return None


@dataclass
class DownloadJob:
    """
    Keep track of planned downloads for one contract.

    The job provides exact :meth:`ib.reqHistoricalData` params for each planned
    range, buffers downloaded bars until they are handed to persistence, and
    delegates writes to :class:`HistorySink`.

    There are potentially 3 streams of data that a job might schedule:

    * backfill - data older than the oldest data point available in
    the store

    * update - data newer than the newest data point available

    * gap-fill - any data missing inside the range available in the
    store

    public methods/attributes:

    * `contract`

    * `next_date`

    * `params`

    * `save_chunk`

    Args:
        contract: IB contract this job downloads.
        sink: Persistence boundary for downloaded bars.
        queue: Planned download ranges for this contract.
        bar_size: IB bar size used to calculate request durations.
        max_bars: Maximum bars allowed per request.
        auto_save_interval: Optional interval for flushing buffered bars.
    """

    contract: ibi.Contract
    sink: HistorySink = field(repr=False)
    queue: list[DownloadContainer] = field(default_factory=list, repr=False)
    bar_size: str = BARSIZE
    max_bars: int = MAX_BARS
    auto_save_interval: int = AUTO_SAVE_INTERVAL
    gap_learner: RunGapLearner | None = field(default=None, repr=False)
    _pulse: ibi.Event | None = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        self._pulse = pulse_factory(self.auto_save_interval)
        if self._pulse:
            self._pulse += self._onPulse
        log.debug(f"{self!r} initialized.")

    async def _onPulse(self, time: datetime):
        if self.is_done():
            self._pulse -= self._onPulse  # type: ignore
        else:
            await self.write_to_store()

    def is_done(self) -> bool:
        while True:
            if len(self.queue) != 0:
                if self._container.next_date:
                    return False
                else:
                    self.queue.pop()
            else:
                return True

    @property
    def _container(self) -> DownloadContainer:
        # there must be at least one object in the queue, otherwise it's an error
        # don't call the method without verifying the queue is not empty
        # with self.is_done()
        return self.queue[-1]

    @property
    def next_date(self) -> date | datetime | None:
        if not self.is_done():
            return self._container.next_date
        else:
            return None

    async def save_chunk(self, data: ibi.BarDataList | None) -> None:
        if data:
            log.info(f"{self!s} received bars from: {data[0].date} to {data[-1].date}")
            self._container.save_chunk(data)
            await self.write_to_store()
        else:
            if self._container.kind == "gap" and self.gap_learner is not None:
                self.gap_learner.record_empty_gap(self._container.gap_pattern)
            if self._container.next_date:
                log.warning(
                    f"{self!s} cannot download data past {self._container.next_date}"
                )
            if self._container.kind == "backfill":
                await self.sink.mark_backfill_exhausted()
            self.queue.pop()

    async def write_to_store(self) -> None:
        """
        This is the only method that actually saves data.  All other
        'save' methods don't.
        """

        _data = self._container.flush_data()

        if _data is not None:
            version = await self.sink.write(_data)
            log.debug(
                f"{self!s} written to store {_data.index[0]} - {_data.index[-1]} "
                f"{version}"
            )
            self._container.clear()

    @property
    def params(self) -> Params:
        # should've been checked before scheduling
        assert self.next_date is not None
        return {
            "contract": self.contract,
            "endDateTime": self.next_date,
            "durationStr": self.duration,
        }

    @property
    def duration(self) -> str:
        next_date = self.next_date
        assert next_date is not None
        if isinstance(next_date, datetime):
            delta = next_date - cast(datetime, self._container.from_date)
        else:
            delta = next_date - cast(date, self._container.from_date)
        return helpers.timedelta_and_barSize_to_duration_str(
            delta, self.bar_size, self.max_bars
        )

    def __str__(self) -> str:
        return f"<DownloadJob: {self.contract.localSymbol}>"

    def __repr__(self) -> str:
        return (
            f"<DownloadJob: {self.contract.localSymbol} "
            f"{[(str(i.from_date), str(i.to_date)) for i in self.queue]}>"
        )


@dataclass
class DownloadContainer:
    """Hold downloaded bars before they are saved to the datastore.

    Args:
        from_date: Earliest point this range should cover.
        to_date: Latest point this range should cover.
        kind: Download range kind used to distinguish backfill exhaustion from
            update and gap-fill misses.
        bars: Downloaded chunks buffered before persistence.

    Public methods/attributes:

    * next_date

    * save_chunk

    * clear

    * flush_data

    """

    from_date: datetime | date
    to_date: datetime | date
    bar_size: str = BARSIZE
    kind: RangeKind = "backfill"
    gap_pattern: GapPattern | None = None
    _next_date: datetime | date | None = field(init=False, repr=False)
    bars: list[ibi.BarDataList] = field(default_factory=list, repr=False)

    def __post_init__(self):
        self.from_date = normalize_point(self.from_date, self.bar_size)
        self.to_date = normalize_point(self.to_date, self.bar_size)
        assert (
            self.to_date > self.from_date
        ), f"{self.from_date=} is later than {self.to_date=}"
        self.next_date = self.to_date

    @property
    def next_date(self):
        return self._next_date

    @next_date.setter
    def next_date(self, point: date | datetime | None) -> None:
        if point is None:
            # failed to get data
            self._next_date = None
            return

        normalized = normalize_point(point, self.bar_size)
        if normalized <= self.from_date:
            # got data, that's all we need
            self._next_date = None
        else:
            # got data, there's more to get
            self._next_date = normalized

    def save_chunk(self, bars: Optional[ibi.BarDataList] | None) -> None:
        """
        Store downloaded data and if more data and update date point
        for next download.
        """

        if bars:
            self.next_date = bars[0].date
            self.bars.append(bars)
        else:
            self.next_date = None

    def flush_data(self) -> Optional[pd.DataFrame]:
        """
        Return df ready to be written to datastore or date of end
        point for additional downloads.
        """
        if self.bars:
            df = ibi.util.df(
                [b for bars in reversed(self.bars) for b in bars]
            ).set_index("date")
            return df
        else:
            return None

    def clear(self):
        """
        Delete data stored in this object.  Should be called after
        data is persisted to db.
        """
        self.bars.clear()


@dataclass
class Manager:
    ib: ibi.IB
    pacing: RequestPacing | None = None
    store: AsyncAbstractBaseStore | None = None
    source: str = SOURCE
    gap_fill_mode: GapFillMode = GAP_FILL_MODE
    use_rth: bool = USE_RTH
    max_period: int = MAX_PERIOD
    max_bars: int = MAX_BARS
    auto_save_interval: int = AUTO_SAVE_INTERVAL
    wts: str = WTS
    bar_size: str = BARSIZE
    now: date | datetime = field(default_factory=current_session_now)
    active_jobs: list[DownloadJob] = field(default_factory=list, repr=False)
    _initiated_contracts: list[ibi.Contract] = field(default_factory=list, repr=False)
    gap_learner: RunGapLearner = field(default_factory=RunGapLearner, repr=False)
    _timezone_cache: dict[tuple[str, str, str], str] = field(
        default_factory=dict, repr=False
    )
    new_job_generator: AsyncGenerator[DownloadJob, None] = field(init=False)

    def __post_init__(self) -> None:
        self.bar_size = bar_size_validator(self.bar_size)
        self.wts = wts_validator(self.wts)
        self.now = normalize_point(self.now, self.bar_size)
        if self.pacing is None:
            self.pacing = request_pacing_factory(self.ib, self.bar_size, self.wts)
        else:
            self.pacing.bar_size = self.bar_size
            self.pacing.wts = self.wts
        self.new_job_generator = self._job_generator()

    def get_store(self) -> AsyncAbstractBaseStore:
        """Return the session datastore, creating it lazily when needed."""

        if self.store is None:
            self.store = create_dataloader_store(self.wts, self.bar_size)
        return self.store

    @functools.cached_property
    def sources(self) -> list[dict]:
        return pd.read_csv(self.source, keep_default_na=False).to_dict("records")

    async def contracts(self) -> AsyncGenerator[ibi.Contract, None]:
        with tqdm(self.sources, desc="Sources") as source_pbar:
            for s in source_pbar:
                assert self.pacing is not None
                contract_selector = ContractSelector.from_kwargs(
                    pacing=self.pacing, **s
                )
                with tqdm(
                    desc=f"Contracts_{s.get('symbol')}", leave=False, total=None
                ) as contract_pbar:
                    async for contract in contract_selector.objects():
                        yield contract
                        contract_pbar.update(1)

    async def headstamps(
        self,
    ) -> AsyncGenerator[tuple[ibi.Contract, date | datetime], None]:
        async for contract in self.contracts():
            headstamp = await self.headstamp(contract)
            yield contract, headstamp

    async def tasks(
        self, store: AsyncStoreView, headstamp: datetime | date
    ) -> list[DownloadContainer]:
        """Return planned download containers for one contract store view."""

        planner = TaskPlanner(
            store,
            headstamp,
            max_period_days=self.max_period,
            gap_fill_mode=self.gap_fill_mode,
        )
        planned_ranges = await self.planned_ranges(store, planner)
        return [
            DownloadContainer(
                t.from_date,
                t.to_date,
                bar_size=self.bar_size,
                kind=t.kind,
                gap_pattern=t.gap_pattern,
            )
            for t in planned_ranges
        ]

    async def planned_ranges(
        self, store: AsyncStoreView, planner: TaskPlanner
    ) -> list[PlannedRange]:
        """Return planned ranges, resolving async gap-fill inputs as needed."""

        if historical_data_unavailable(store):
            return []
        base_planner = TaskPlanner(
            store,
            planner.head,
            max_period_days=planner.max_period_days,
            gap_fill_mode="off",
        )
        base_ranges = base_planner.planned_ranges()
        if self.gap_fill_mode == "off":
            return base_ranges
        if self.gap_fill_mode == "heuristic":
            return [*await self._heuristic_gap_ranges(store, planner), *base_ranges]
        return await self._scheduled_planned_ranges(store, planner, base_ranges)

    async def _scheduled_planned_ranges(
        self,
        store: AsyncStoreView,
        planner: TaskPlanner,
        base_ranges: list[PlannedRange],
    ) -> list[PlannedRange]:
        """Plan gap-fill ranges with IB historical schedules when available."""

        candidates = raw_gap_candidates(store, start=planner.start)
        if not candidates:
            return base_ranges
        try:
            schedules = await self._historical_schedules(store.contract, candidates)
        except Exception:
            if self.gap_fill_mode == "schedule":
                raise
            log.exception(
                "Schedule gap-fill unavailable for %s; using heuristic.",
                store.contract,
            )
            return [*await self._heuristic_gap_ranges(store, planner), *base_ranges]

        sessions = [
            session
            for schedule in schedules
            for session in sessions_from_historical_schedule(schedule)
        ]
        timezone_name = next(
            (
                tz_name
                for schedule in schedules
                if (tz_name := schedule_timezone(schedule)) is not None
            ),
            None,
        )
        if timezone_name is None:
            timezone_name = await self.contract_timezone(store.contract)
        if not sessions:
            if self.gap_fill_mode == "schedule":
                raise RuntimeError(
                    f"No historical schedule returned for {store.contract}"
                )
            return [*await self._heuristic_gap_ranges(store, planner), *base_ranges]

        gap_ranges = [
            PlannedRange(
                dates.from_date,
                dates.to_date,
                "gap",
                candidate.pattern(timezone_name),
            )
            for candidate in scheduled_gap_candidates(
                candidates,
                sessions,
                timezone_name=timezone_name,
                typical_patterns=self.gap_learner.typical_patterns,
            )
            if (dates := candidate.dates()) is not None
        ]
        return [*gap_ranges, *base_ranges]

    async def _heuristic_gap_ranges(
        self, store: AsyncStoreView, planner: TaskPlanner
    ) -> list[PlannedRange]:
        """Return heuristic gap ranges filtered by run-local learned patterns."""

        timezone_name = await self.contract_timezone(store.contract)
        ranges: list[PlannedRange] = []
        for candidate in heuristic_gap_candidates(
            raw_gap_candidates(store, start=planner.start),
            timezone_name=timezone_name,
        ):
            pattern = candidate.pattern(timezone_name)
            if pattern in self.gap_learner.typical_patterns:
                continue
            if dates := candidate.dates():
                ranges.append(
                    PlannedRange(dates.from_date, dates.to_date, "gap", pattern)
                )
        return ranges

    async def _historical_schedules(
        self, contract: ibi.Contract, candidates: list
    ) -> list[object]:
        """Request historical schedules covering candidate gaps."""

        schedules = []
        sorted_candidates = sorted(
            candidates, key=lambda candidate: candidate.missing_start
        )
        chunk: list = []
        chunk_start: date | datetime | None = None
        for candidate in sorted_candidates:
            if chunk_start is None:
                chunk_start = candidate.missing_start
            if (
                chunk
                and _days_between(chunk_start, candidate.missing_end)
                > SCHEDULE_CHUNK_DAYS
            ):
                schedules.append(await self._historical_schedule_chunk(contract, chunk))
                chunk = []
                chunk_start = candidate.missing_start
            chunk.append(candidate)
        if chunk:
            schedules.append(await self._historical_schedule_chunk(contract, chunk))
        return schedules

    async def _historical_schedule_chunk(
        self, contract: ibi.Contract, candidates: list
    ) -> object:
        """Request one historical schedule chunk for candidate gaps."""

        assert self.pacing is not None
        start = min(candidate.missing_start for candidate in candidates)
        end = max(candidate.to_date for candidate in candidates)
        num_days = max(1, min(SCHEDULE_CHUNK_DAYS, _days_between(start, end) + 2))
        return await self.pacing.historical_schedule(
            contract,
            numDays=num_days,
            endDateTime=end,
            useRTH=self.use_rth,
        )

    async def contract_timezone(self, contract: ibi.Contract) -> str | None:
        """Return best-effort exchange timezone for gap heuristics."""

        cache_key = _contract_timezone_key(contract)
        if cache_key in self._timezone_cache:
            return self._timezone_cache[cache_key]
        timezone_name = await self._contract_details_timezone(contract)
        if timezone_name is None and getattr(contract, "secType", "").upper() == "FUT":
            timezone_name = await self._futures_chain_timezone(contract)
        if timezone_name is not None:
            self._timezone_cache[cache_key] = timezone_name
        return timezone_name

    async def _contract_details_timezone(self, contract: ibi.Contract) -> str | None:
        """Return timezone from exact contract details when available."""

        assert self.pacing is not None
        details = await self.pacing.contract_details(contract)
        return _timezone_from_details(contract, details)

    async def _futures_chain_timezone(self, contract: ibi.Contract) -> str | None:
        """Return timezone from a sibling futures contract when exact details lack it."""

        assert self.pacing is not None
        chain_contract = ibi.Future(
            symbol=getattr(contract, "symbol", ""),
            exchange=getattr(contract, "exchange", ""),
            currency=getattr(contract, "currency", ""),
            includeExpired=True,
        )
        details = await self.pacing.contract_details(chain_contract)
        return _timezone_from_details(contract, details)

    async def jobs(self) -> AsyncGenerator[DownloadJob, None]:
        async for contract, headstamp in self.headstamps():
            if contract in self._initiated_contracts:
                continue
            self._initiated_contracts.append(contract)
            datastore = self.get_store()
            store = await AsyncStoreView.create(
                contract, datastore, self.now, self.bar_size
            )
            sink = HistorySink(contract, datastore)
            tasks = await self.tasks(store, headstamp)
            new_job = DownloadJob(
                contract,
                sink,
                tasks,
                bar_size=self.bar_size,
                max_bars=self.max_bars,
                auto_save_interval=self.auto_save_interval,
                gap_learner=self.gap_learner,
            )
            if new_job.is_done():
                log.debug(f"Skipping {new_job!s} - no need to download data.")
                continue
            yield new_job

    async def _job_generator(self) -> AsyncGenerator[DownloadJob, None]:
        async for new_job in self.jobs():
            self.active_jobs.append(new_job)
            yield new_job

    async def headstamp(self, contract: ibi.Contract) -> date | datetime:
        headTimeStamp = None
        while not headTimeStamp:
            assert self.pacing is not None
            headTimeStamp = await self.pacing.head_timestamp(
                contract,
                whatToShow=self.wts,
                useRTH=self.use_rth,
                formatDate=2,
            )

            # empty response after pacing retries means no headstamp is available
            if not headTimeStamp:
                five_years_ago = datetime.now(tz=timezone.utc) - timedelta(days=5 * 250)
                log.warning(
                    (
                        f"Unavailable headTimeStamp ({headTimeStamp}) for "
                        f"{contract}. Will use {five_years_ago}"
                    )
                )
                headTimeStamp = five_years_ago

        hs = normalize_point(headTimeStamp, self.bar_size)
        log.debug(f"Headstamp for: {contract.localSymbol}: {hs}")
        return hs


def validate_age(job: DownloadJob, now: date | datetime) -> bool:
    """Return whether a job is still inside IB's small-bar age limit.

    IB does not allow requests for bars 30 seconds or smaller older than six
    months.

    Args:
        job: Download job being validated.
        now: Run-scoped current point normalized for the job's bar size.

    Returns:
        ``True`` when the job can keep requesting data, otherwise ``False``.
    """
    if helpers.duration_in_secs(job.bar_size) <= 30 and job.next_date:
        assert isinstance(job.next_date, datetime)
        if isinstance(now, datetime) and (now - job.next_date).days > 180:
            return False
    return True


@dataclass
class DataloaderSession:
    """Own one dataloader runtime session and its restartable execution state."""

    ib: ibi.IB
    manager: Manager | None = None
    number_of_workers: int = NUMBER_OF_WORKERS
    failures: DownloadFailureRegistry = field(default_factory=DownloadFailureRegistry)
    queue: asyncio.Queue[DownloadJob] | None = field(default=None, init=False)
    workers: list[asyncio.Task] = field(default_factory=list, init=False)
    producer_task: asyncio.Task | None = field(default=None, init=False)
    fatal_error: BaseException | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.manager is None:
            self.manager = Manager(self.ib)
        else:
            self.manager.ib = self.ib
            if self.manager.pacing is not None:
                self.manager.pacing.ib = self.ib

    @property
    def pacing(self) -> RequestPacing:
        """Return the pacing state used by this session."""

        assert self.manager is not None
        assert self.manager.pacing is not None
        return self.manager.pacing

    async def run(self) -> None:
        """Run producer and workers until all queued download work is finished."""

        log.debug("Initializing dataloader session.")
        self.queue = asyncio.LifoQueue(maxsize=int(self.number_of_workers / 4))
        self.workers = [
            create_task(
                self.worker(f"worker_{i}", self.queue),
                logger=log,
                message="asyncio error",
                message_args=(f"worker {i}",),
            )
            for i in range(self.number_of_workers)
        ]
        self.producer_task = create_task(
            self.producer(self.queue),
            logger=log,
            message="asyncio error",
            message_args=("producer",),
        )

        try:
            try:
                await self.producer_task
            except asyncio.CancelledError:
                if self.fatal_error is not None:
                    raise self.fatal_error
                raise
            await self.queue.join()
            if self.fatal_error is not None:
                raise self.fatal_error
        finally:
            await self.cancel_execution()
            self.failures.log_summary()

    async def producer(self, queue: asyncio.Queue) -> None:
        """Queue unfinished active jobs and then discover new jobs."""

        assert self.manager is not None
        log.debug("Initializing PRODUCER")
        if self.manager.active_jobs:
            log.debug("Will start queuing jobs, which are already initialized.")
        for job in self.manager.active_jobs:
            if not job.is_done():
                await queue.put(job)
                log.debug(f"Active job {job} added to queue.")

        log.debug("Will start queuing new jobs.")
        while True:
            try:
                new_job = await anext(self.manager.new_job_generator)
                await queue.put(new_job)
                log.debug(f"New job {new_job} added to queue.")
            except StopAsyncIteration:
                log.debug("No more jobs!")
                break

        log.debug("Producer done!")

    async def worker(self, name: str, queue: asyncio.Queue) -> None:
        """Download historical data chunks for queued jobs."""

        log.debug(f"Initializing {name.upper()}")
        while True:
            job = await queue.get()

            try:
                await self.download_job(name, job)
            except asyncio.CancelledError:
                raise
            except CONNECTION_FAILURE_ERRORS as exc:
                self.abort_execution(exc, queue)
                log.exception("%s hit connection failure while loading %s.", name, job)
                raise
            except DownloadRequestError as exc:
                error = exc.__cause__ or exc
                log.exception("%s request failed while loading %s.", name, job)
                self.failures.record(job, error)
            except Exception as exc:
                self.abort_execution(exc, queue)
                log.exception("%s hit fatal failure while loading %s.", name, job)
                raise
            finally:
                queue.task_done()

    async def download_job(self, name: str, job: DownloadJob) -> None:
        """Download all currently queued chunks for one job."""

        assert self.manager is not None
        while True:
            if not validate_age(job, self.manager.now):
                await job.save_chunk(None)
                log.debug(f"{job!s} dropped before request on age validation.")
                break
            params = job.params
            log.debug(
                f"{name} loading {job!s} ending: {job.next_date} "
                f"duration: {params['durationStr']}"
            )
            try:
                chunk = await self.pacing.historical_data(
                    **params,
                    barSizeSetting=job.bar_size,
                    whatToShow=self.manager.wts,
                    useRTH=self.manager.use_rth,
                    formatDate=2,
                )
            except CONNECTION_FAILURE_ERRORS:
                raise
            except Exception as exc:
                raise DownloadRequestError(str(exc)) from exc
            await job.save_chunk(chunk)

            if job.next_date:
                if not validate_age(job, self.manager.now):
                    await job.save_chunk(None)
                    log.debug(f"{job!s} dropped on age validation.")
                    break
            else:
                log.info(f"{job!s} done!")
                break

    def cancel_tasks(self) -> None:
        """Request cancellation of active producer/worker tasks."""

        log.debug("Will cancel dataloader session tasks.")
        if self.producer_task and not self.producer_task.done():
            self.producer_task.cancel()
        if self.queue:
            shutdown_queue(self.queue)
        log.debug("Dataloader session tasks cancelled.")

    def abort_execution(
        self, error: BaseException, queue: asyncio.Queue[DownloadJob]
    ) -> None:
        """Abort the current session after a fatal worker failure."""

        if self.fatal_error is None:
            self.fatal_error = error
        if self.producer_task and not self.producer_task.done():
            self.producer_task.cancel()
        shutdown_queue(queue)

    async def cancel_execution(self) -> None:
        """Cancel active dataloader execution tasks and drain queued work."""

        await cancel_execution(self.producer_task, self.workers, self.queue)


async def main(manager: Manager, ib: ibi.IB) -> None:
    """Run a dataloader session for compatibility with older call sites."""

    await DataloaderSession(ib, manager=manager).run()


async def cancel_execution(
    producer_task: asyncio.Task | None,
    workers: list[asyncio.Task],
    queue: asyncio.Queue | None,
) -> None:
    """Cancel active dataloader execution tasks and drain queued work."""

    tasks = [task for task in [producer_task, *workers] if task is not None]
    for task in tasks:
        if not task.done():
            task.cancel()

    if queue is not None:
        shutdown_queue(queue)

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


def shutdown_queue(queue: asyncio.Queue) -> None:
    while True:
        try:
            queue.get_nowait()
            queue.task_done()
        except (asyncio.QueueEmpty, ValueError):
            break
    log.debug("Queue has been shutdown.")


def _days_between(start: date | datetime, end: date | datetime) -> int:
    """Return whole calendar days between two date-like points."""

    start_date = start.date() if isinstance(start, datetime) else start
    end_date = end.date() if isinstance(end, datetime) else end
    return (end_date - start_date).days


def _contract_timezone_key(contract: ibi.Contract) -> tuple[str, str, str]:
    """Return a stable cache key for contract timezone lookups."""

    return (
        str(getattr(contract, "symbol", "")),
        str(getattr(contract, "exchange", "")),
        str(getattr(contract, "tradingClass", "")),
    )


def _timezone_from_details(
    contract: ibi.Contract, details: list[ibi.ContractDetails]
) -> str | None:
    """Return exact or sibling timezone from contract details."""

    for detail in details:
        if (
            detail.contract is not None
            and _contract_matches(contract, detail.contract)
            and detail.timeZoneId
        ):
            return detail.timeZoneId
    for detail in details:
        if (
            detail.contract is not None
            and _contract_sibling_matches(contract, detail.contract)
            and detail.timeZoneId
        ):
            return detail.timeZoneId
    return None


def _contract_matches(contract: ibi.Contract, candidate: ibi.Contract) -> bool:
    """Return whether two contract descriptions identify the same contract."""

    con_id = getattr(contract, "conId", 0)
    local_symbol = getattr(contract, "localSymbol", "")
    return bool(con_id and con_id == getattr(candidate, "conId", 0)) or bool(
        local_symbol and local_symbol == getattr(candidate, "localSymbol", "")
    )


def _contract_sibling_matches(contract: ibi.Contract, candidate: ibi.Contract) -> bool:
    """Return whether a contract detail belongs to the same symbol family."""

    return (
        getattr(contract, "symbol", "") == getattr(candidate, "symbol", "")
        and getattr(contract, "exchange", "") == getattr(candidate, "exchange", "")
        and (
            not getattr(contract, "tradingClass", "")
            or getattr(contract, "tradingClass", "")
            == getattr(candidate, "tradingClass", "")
        )
    )


def start():
    ibi.util.patchAsyncio()
    ib = ibi.IB()
    session = DataloaderSession(ib)
    ib.errorEvent += session.pacing.onErrEvent
    asyncio.get_event_loop().set_debug(True)
    log.debug("Will start...")

    connection(ib, session.run, session.cancel_tasks)

    log.info("script finished, about to disconnect")
    ib.disconnect()
    log.debug("disconnected")
