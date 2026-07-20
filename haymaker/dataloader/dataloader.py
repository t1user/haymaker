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
from typing import AsyncGenerator, Literal, Optional, TypedDict, cast

import ib_insync as ibi
import pandas as pd
from tqdm import tqdm

from haymaker.async_wrappers import QueueShutdownPolicy
from haymaker.config.settings import StorageSettings
from haymaker.databases import StoreFactory
from haymaker.datastore import AsyncAbstractBaseStore
from haymaker.validators import bar_size_validator, wts_validator

from . import helpers
from .contract_selectors import (
    ContractQualificationError,
    ContractSelector,
    FuturesSelectionPolicy,
)
from .pacer import InFlightRequest, RequestPacing
from .scheduling import (
    GapFillMode,
    GapPattern,
    GapCandidate,
    PlannedRange,
    RangeKind,
    SMALL_BAR_MAX_AGE,
    SessionRange,
    historical_data_unavailable,
    historical_unavailability_reason,
    schedule_timezone,
    sessions_from_historical_schedule,
    TaskPlanner,
)
from .store_wrapper import AsyncStoreView, HistorySink
from .time_policy import normalize_point

log = logging.getLogger(__name__)


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

    def log_summary(
        self, *, outcome: Literal["completed", "interrupted", "failed"] = "completed"
    ) -> None:
        """Log the run outcome and any failed download jobs."""

        if not self.failures:
            log.info("Dataloader %s with no failed download jobs.", outcome)
            return

        relation = "with" if outcome == "completed" else "after"
        log.warning(
            "Dataloader %s %s %d failed download jobs.",
            outcome,
            relation,
            len(self.failures),
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
HEADSTAMP_FALLBACK_DAYS = 5 * 365
STATUS_LOG_INTERVAL = 300.0


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


@dataclass
class DownloadJob:
    """Track request progression and buffered results for one contract.

    The job provides exact :meth:`ib.reqHistoricalData` params for each planned
    range, buffers downloaded bars until they are handed to persistence, and
    delegates writes to :class:`HistorySink`.

    A job may contain update, backfill, and gap-fill ranges. Ranges are consumed
    in queue order. Successful chunks are persisted at ``save_every_chunks``;
    range completion and explicit cleanup flush any smaller remaining batch.

    Args:
        contract: IB contract this job downloads.
        sink: Persistence boundary for downloaded bars.
        queue: Planned download ranges for this contract.
        bar_size: IB bar size used to calculate request durations.
        target_bars_per_request: Target chunk size used before applying IB's
            hard duration limits.
        save_every_chunks: Number of downloaded chunks buffered before writing.

    Raises:
        ValueError: If ``save_every_chunks`` is not positive.
    """

    contract: ibi.Contract
    sink: HistorySink = field(repr=False)
    queue: list[DownloadContainer] = field(default_factory=list, repr=False)
    bar_size: str = "30 secs"
    target_bars_per_request: int = helpers.DEFAULT_TARGET_BARS_PER_REQUEST
    save_every_chunks: int = 10
    gap_learner: RunGapLearner | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.save_every_chunks, int) or isinstance(
            self.save_every_chunks, bool
        ):
            raise TypeError("save_every_chunks must be an integer")
        if self.save_every_chunks <= 0:
            raise ValueError("save_every_chunks must be a positive integer")
        log.debug(f"{self!r} initialized.")

    def is_done(self) -> bool:
        while True:
            if len(self.queue) != 0:
                if self._container.next_date:
                    return False
                else:
                    self.queue.pop(0)
            else:
                return True

    @property
    def _container(self) -> DownloadContainer:
        # there must be at least one object in the queue, otherwise it's an error
        # don't call the method without verifying the queue is not empty
        # with self.is_done()
        return self.queue[0]

    @property
    def next_date(self) -> date | datetime | None:
        if not self.is_done():
            return self._container.next_date
        else:
            return None

    async def save_chunk(
        self,
        data: ibi.BarDataList | None,
        *,
        empty_reason: str = "IB returned no bars",
    ) -> None:
        """Buffer one response or stop the current range with an explicit reason."""

        container = self._container
        if data:
            log.info(
                "IB returned %d bars for %s from %s to %s.",
                len(data),
                self,
                data[0].date,
                data[-1].date,
            )
            container.save_chunk(data)
            range_complete = container.next_date is None or self.is_continuous_future
            if len(container.bars) >= self.save_every_chunks or range_complete:
                await self.write_to_store(container)
            if range_complete and container.exhausts_update:
                await self._mark_update_exhausted()
            if self.is_continuous_future:
                self.queue.pop(0)
        else:
            learned_gap_pattern = self._learn_empty_gap_pattern()
            if container.next_date:
                log.warning(
                    "%s for %s; stopping %s range at %s.",
                    empty_reason,
                    self,
                    container.kind,
                    container.next_date,
                )
            await self.write_to_store(container)
            if container.kind == "backfill":
                await self.sink.mark_backfill_exhausted()
            elif container.exhausts_update:
                await self._mark_update_exhausted()
            self.queue.pop(0)
            self._drop_typical_gap_containers(learned_gap_pattern)

    async def _mark_update_exhausted(self) -> None:
        """Persist completion after a terminal update range finishes."""

        await self.sink.mark_update_exhausted()
        log.debug(
            "%s marked update_exhausted after completing its past-expiry "
            "update range through IB.",
            self,
        )

    def _learn_empty_gap_pattern(self) -> GapPattern | None:
        """Record an empty gap-fill response and return its learned pattern."""

        if self._container.kind != "gap" or self.gap_learner is None:
            return None
        pattern = self._container.gap_pattern
        self.gap_learner.record_empty_gap(pattern)
        return pattern

    def _drop_typical_gap_containers(self, pattern: GapPattern | None) -> None:
        """Remove remaining gap ranges with a pattern now known to be empty."""

        if (
            pattern is None
            or self.gap_learner is None
            or pattern not in self.gap_learner.typical_patterns
        ):
            return
        self.queue = [
            container
            for container in self.queue
            if container.kind != "gap" or container.gap_pattern != pattern
        ]

    async def write_to_store(self, container: DownloadContainer) -> None:
        """Persist buffered data from one download range."""

        _data = container.flush_data()

        if _data is not None:
            version = await self.sink.write(_data)
            log.debug(
                f"{self!s} written to store {_data.index[0]} - {_data.index[-1]} "
                f"{version}"
            )
            container.clear()

    async def flush_pending(self) -> None:
        """Persist all chunks buffered by this job."""

        for container in self.queue:
            await self.write_to_store(container)

    @property
    def params(self) -> Params:
        # should've been checked before scheduling
        assert self.next_date is not None
        return {
            "contract": self.contract,
            "endDateTime": "" if self.is_continuous_future else self.next_date,
            "durationStr": self.duration,
        }

    @property
    def is_continuous_future(self) -> bool:
        """Return whether this job uses IB's continuous-future request policy."""

        return getattr(self.contract, "secType", "") == "CONTFUT"

    @property
    def duration(self) -> str:
        next_date = self.next_date
        assert next_date is not None
        if isinstance(next_date, datetime):
            delta = next_date - cast(datetime, self._container.from_date)
        else:
            delta = next_date - cast(date, self._container.from_date)
        return helpers.timedelta_and_barSize_to_duration_str(
            delta, self.bar_size, self.target_bars_per_request
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
    """Hold request state and buffered bars for one planned date range.

    Args:
        from_date: Earliest point this range should cover.
        to_date: Latest point this range should cover.
        bar_size: Canonical IB bar size controlling date normalization.
        kind: Download range kind used to distinguish backfill exhaustion from
            update and gap-fill misses.
        gap_pattern: Optional run-local pattern used by heuristic gap filling.
        exhausts_update: Whether completing this range proves that an expired
            series has no later data.
        bars: Downloaded chunks waiting for persistence.
    """

    from_date: datetime | date
    to_date: datetime | date
    bar_size: str = "30 secs"
    kind: RangeKind = "backfill"
    gap_pattern: GapPattern | None = None
    exhausts_update: bool = False
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
        """Buffer returned bars and advance the next request boundary."""

        if bars:
            self.next_date = bars[0].date
            self.bars.append(bars)
        else:
            self.next_date = None

    def flush_data(self) -> Optional[pd.DataFrame]:
        """Return buffered chunks as one chronological dataframe, if present."""
        if self.bars:
            df = ibi.util.df(
                [b for bars in reversed(self.bars) for b in bars]
            ).set_index(  # type: ignore
                "date"
            )
            return df
        else:
            return None

    def clear(self) -> None:
        """Discard buffered chunks after successful persistence."""
        self.bars.clear()


@dataclass
class DownloadPlan:
    """Carry broker request containers and local completion work."""

    containers: list[DownloadContainer]
    completes_update_without_request: bool = False


@dataclass
class Manager:
    """Discover contracts and create download jobs for one dataloader run.

    Args:
        ib: Connected IB client used through ``RequestPacing``.
        pacing: Optional preconfigured request pacer. A session pacer is created
            when omitted.
        store: Optional async store, primarily for focused callers and tests.
            Arctic is created lazily when omitted.
        source: CSV path containing IB contract fields.
        gap_fill_mode: One of ``off``, ``heuristic``, ``schedule``, or ``auto``.
        use_rth: Restrict historical bars and schedules to regular hours.
        max_lookback_days: Optional positive run lookback cap. ``None`` requests
            all history permitted by IB and datastore state.
        save_every_chunks: Positive number of chunks buffered per datastore
            version.
        wts: IB ``whatToShow`` value.
        bar_size: Canonical IB historical bar size.
        now: Run-scoped current point; normalized according to ``bar_size``.

    Raises:
        ValueError: If request policy values are invalid.
    """

    ib: ibi.IB
    pacing: RequestPacing | None = None
    store: AsyncAbstractBaseStore | None = None
    store_factory: StoreFactory = field(
        default_factory=lambda: StoreFactory(StorageSettings()), repr=False
    )
    futures: FuturesSelectionPolicy = field(
        default_factory=FuturesSelectionPolicy, repr=False
    )
    source: str = "contracts.csv"
    gap_fill_mode: GapFillMode = "off"
    use_rth: bool = False
    max_lookback_days: int | None = None
    save_every_chunks: int = 10
    wts: str = "TRADES"
    bar_size: str = "30 secs"
    pacer_no_restriction: bool = False
    pacer_allowance_fraction: float = 1.0
    now: date | datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    active_jobs: list[DownloadJob] = field(default_factory=list, repr=False)
    _initiated_contracts: set[ibi.Contract] = field(default_factory=set, repr=False)
    gap_learner: RunGapLearner = field(default_factory=RunGapLearner, repr=False)
    new_job_generator: AsyncGenerator[DownloadJob, None] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source, str):
            raise TypeError("source must be a string")
        if not isinstance(self.use_rth, bool):
            raise TypeError("use_rth must be a boolean")
        if self.max_lookback_days is not None and (
            not isinstance(self.max_lookback_days, int)
            or isinstance(self.max_lookback_days, bool)
        ):
            raise TypeError("max_lookback_days must be an integer or None")
        if not isinstance(self.save_every_chunks, int) or isinstance(
            self.save_every_chunks, bool
        ):
            raise TypeError("save_every_chunks must be an integer")
        self.bar_size = bar_size_validator(self.bar_size)
        self.wts = wts_validator(self.wts)
        if self.gap_fill_mode not in {"off", "heuristic", "schedule", "auto"}:
            raise ValueError(f"Unknown gap-fill mode: {self.gap_fill_mode!r}")
        if self.max_lookback_days is not None and self.max_lookback_days <= 0:
            raise ValueError("max_lookback_days must be a positive integer or None")
        if self.save_every_chunks <= 0:
            raise ValueError("save_every_chunks must be a positive integer")
        self.now = normalize_point(self.now, self.bar_size)
        if self.pacing is None:
            self.pacing = RequestPacing(
                self.ib,
                no_restriction=self.pacer_no_restriction,
                allowance_fraction=self.pacer_allowance_fraction,
            )
        self.new_job_generator = self._job_generator()

    @property
    def datastore(self) -> AsyncAbstractBaseStore:
        """Return the session datastore, creating it lazily when needed."""

        if self.store is None:
            lib = f"{self.wts}_{self.bar_size}".replace(" ", "_")
            self.store = self.store_factory.arctic_store(
                lib, shutdown_policy=QueueShutdownPolicy.DRAIN
            )
        return self.store

    @functools.cached_property
    def sources(self) -> list[dict]:
        source = pd.read_csv(self.source, keep_default_na=False)
        if unknown := set(source.columns) - ContractSelector.contract_fields:
            names = ", ".join(sorted(unknown))
            raise ContractQualificationError(
                f"Unknown contract field(s) in {self.source}: {names}"
            )
        return source.to_dict("records")

    async def contracts(self) -> AsyncGenerator[ibi.Contract, None]:
        with tqdm(self.sources, desc="Sources") as source_pbar:
            for s in source_pbar:
                assert self.pacing is not None
                contract_selector = ContractSelector.from_kwargs(
                    pacing=self.pacing, futures=self.futures, **s
                )
                with tqdm(
                    desc=f"Contracts_{s.get('symbol')}", leave=False, total=None
                ) as contract_pbar:
                    async for contract in contract_selector.objects():
                        yield contract
                        contract_pbar.update(1)

    async def download_plan(
        self, store: AsyncStoreView, headstamp: datetime | date | None
    ) -> DownloadPlan:
        """Return broker work and local completion for one contract store view."""

        planner = self.task_planner(store, headstamp)
        planned_ranges = await self.planned_ranges(planner)
        return DownloadPlan(
            [
                DownloadContainer(
                    t.from_date,
                    t.to_date,
                    bar_size=self.bar_size,
                    kind=t.kind,
                    gap_pattern=t.gap_pattern,
                    exhausts_update=t.exhausts_update,
                )
                for t in planned_ranges
            ],
            completes_update_without_request=(planner.completes_update_without_request),
        )

    def task_planner(
        self, store: AsyncStoreView, headstamp: datetime | date | None
    ) -> TaskPlanner:
        """Create a planner from preloaded store data and an optional IB head."""

        assert self.pacing is not None
        return TaskPlanner(
            store,
            headstamp,
            max_lookback_days=self.max_lookback_days,
            gap_fill_mode=self._sync_gap_fill_mode,
            timezone_name=self.pacing.contract_timezone(store.contract),
            typical_patterns=self.gap_learner.typical_patterns,
        )

    async def planned_ranges(self, planner: TaskPlanner) -> list[PlannedRange]:
        """Return planned ranges, resolving async gap-fill inputs as needed."""

        if historical_data_unavailable(planner.store):
            return []
        if self.gap_fill_mode in {"off", "heuristic"}:
            return planner.planned_ranges()
        return await self._scheduled_planned_ranges(planner)

    @property
    def _sync_gap_fill_mode(self) -> GapFillMode:
        """Return the non-async fallback mode used before schedule resolution."""

        return "heuristic" if self.gap_fill_mode == "auto" else self.gap_fill_mode

    async def _scheduled_planned_ranges(
        self, planner: TaskPlanner
    ) -> list[PlannedRange]:
        """Plan gap-fill ranges with IB historical schedules when available."""

        candidates = planner.gap_candidates()
        if not candidates:
            return planner.base_ranges()
        try:
            schedules = await self._historical_schedules(
                planner.store.contract, candidates
            )
        except Exception:
            if self.gap_fill_mode == "schedule":
                raise
            log.exception(
                "Schedule gap-fill unavailable for %s; using heuristic.",
                planner.store.contract,
            )
            return self._planner_with_mode(planner, "heuristic").planned_ranges()

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
        if not sessions:
            if self.gap_fill_mode == "schedule":
                raise RuntimeError(
                    f"No historical schedule returned for {planner.store.contract}"
                )
            return self._planner_with_mode(planner, "heuristic").planned_ranges()

        return self._planner_with_mode(
            planner,
            "schedule",
            sessions,
            timezone_name or planner.timezone_name,
        ).planned_ranges()

    def _planner_with_mode(
        self,
        planner: TaskPlanner,
        mode: GapFillMode,
        sessions: list[SessionRange] | None = None,
        timezone_name: str | None = None,
    ) -> TaskPlanner:
        """Return a planner with resolved gap-fill inputs."""

        return TaskPlanner(
            planner.store,
            planner.head,
            planner.max_lookback_days,
            gap_fill_mode=mode,
            timezone_name=timezone_name or planner.timezone_name,
            sessions=sessions,
            typical_patterns=planner.typical_patterns,
        )

    async def _historical_schedules(
        self, contract: ibi.Contract, candidates: list[GapCandidate]
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
        self, contract: ibi.Contract, candidates: list[GapCandidate]
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

    async def jobs(self) -> AsyncGenerator[DownloadJob, None]:
        async for contract in self.contracts():
            if contract in self._initiated_contracts:
                continue
            self._initiated_contracts.add(contract)
            store = await AsyncStoreView.create(
                contract, self.datastore, self.now, self.bar_size
            )
            planner = self.task_planner(store, None)
            if reason := historical_unavailability_reason(store):
                log.debug(
                    "Skipping %s without an IB historical request: local policy "
                    "determined the series is unavailable; %s.",
                    contract.localSymbol or contract.symbol,
                    reason,
                )
                continue
            headstamp = (
                await self.headstamp(contract) if planner.needs_headstamp else None
            )
            sink = HistorySink(contract, self.datastore)
            plan = await self.download_plan(store, headstamp)
            if plan.completes_update_without_request:
                await sink.mark_update_exhausted()
                log.debug(
                    "Skipping final update request for %s: local policy determined "
                    "that exact past expiry is no more than one intraday bar after "
                    "the stored endpoint; marked update_exhausted.",
                    contract.localSymbol or contract.symbol,
                )
            new_job = DownloadJob(
                contract,
                sink,
                plan.containers,
                bar_size=self.bar_size,
                save_every_chunks=self.save_every_chunks,
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
        assert self.pacing is not None
        broker_headstamp = await self.pacing.head_timestamp(
            contract,
            whatToShow=self.wts,
            useRTH=self.use_rth,
            formatDate=2,
        )

        # empty response after pacing retries means no headstamp is available
        if broker_headstamp is None:
            headstamp = self.headstamp_fallback(contract)
            log.warning(
                (
                    f"Unavailable headTimeStamp for {contract}. "
                    f"Will probe historical data from fallback {headstamp}"
                )
            )
        else:
            headstamp = broker_headstamp

        hs = normalize_point(headstamp, self.bar_size)
        log.debug(f"Headstamp for: {contract.localSymbol}: {hs}")
        return hs

    def headstamp_fallback(self, contract: ibi.Contract) -> date | datetime:
        """Return a bounded historical start when IB provides no head timestamp."""

        expiry = _contract_expiry(contract, self.bar_size)
        latest = min(expiry, self.now) if expiry is not None else self.now
        candidates = [latest - timedelta(days=HEADSTAMP_FALLBACK_DAYS)]
        if self.max_lookback_days is not None:
            candidates.append(latest - timedelta(days=self.max_lookback_days))
        if helpers.duration_in_secs(self.bar_size) <= 30:
            candidates.append(self.now - SMALL_BAR_MAX_AGE)
        if (
            getattr(contract, "secType", "").upper() == "FUT"
            and expiry is not None
            and expiry <= self.now
        ):
            candidates.append(expiry - timedelta(days=2 * 365))
        return max(candidates)


def request_age_available(job: DownloadJob, now: date | datetime) -> bool:
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
        if isinstance(now, datetime) and now - job.next_date > SMALL_BAR_MAX_AGE:
            return False
    return True


@dataclass
class DataloaderSession:
    """Run discovered jobs through a producer and paced worker pool.

    Connection failures propagate for supervisor recovery. Individual broker
    historical-request failures are recorded while other jobs continue. Local
    processing and datastore failures terminate the session. Before returning
    or propagating an error, the session stops workers and flushes buffered
    chunks from active jobs.

    Args:
        ib: IB client used by this dataloader process.
        manager: Optional preconfigured job manager.
        number_of_workers: Number of concurrent worker tasks. Request pacing
            still limits outbound broker calls.
        failures: Registry receiving non-terminal historical-request failures.
    """

    ib: ibi.IB
    manager: Manager | None = None
    number_of_workers: int = 10
    failures: DownloadFailureRegistry = field(default_factory=DownloadFailureRegistry)
    queue: asyncio.Queue[DownloadJob] | None = field(default=None, init=False)
    workers: list[asyncio.Task] = field(default_factory=list, init=False)
    producer_task: asyncio.Task | None = field(default=None, init=False)
    status_task: asyncio.Task | None = field(default=None, init=False)
    producer_state: str = field(default="not_started", init=False)
    producer_pending_job: str | None = field(default=None, init=False)
    queued_jobs: dict[int, str] = field(default_factory=dict, init=False)
    worker_jobs: dict[str, str] = field(default_factory=dict, init=False)
    fatal_error: BaseException | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.number_of_workers, int) or isinstance(
            self.number_of_workers, bool
        ):
            raise TypeError("number_of_workers must be an integer")
        if self.number_of_workers <= 0:
            raise ValueError("number_of_workers must be positive")
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
        # FIFO preserves source order so current contracts are not overtaken by
        # later-discovered expired contracts while the queue is under pressure.
        self.queue = asyncio.Queue(maxsize=max(1, self.number_of_workers // 4))
        self.workers = [
            asyncio.create_task(
                self.worker(f"worker_{i}", self.queue),
                name=f"worker {i}",
            )
            for i in range(self.number_of_workers)
        ]
        self.producer_task = asyncio.create_task(
            self.producer(self.queue),
            name="producer",
        )
        self.status_task = asyncio.create_task(
            self.log_status(), name="dataloader-status"
        )

        outcome: Literal["completed", "interrupted", "failed"] = "failed"
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
            outcome = "completed"
        except asyncio.CancelledError:
            outcome = "interrupted"
            raise
        finally:
            await self.cancel_execution()
            await self.flush_pending()
            self.failures.log_summary(outcome=outcome)

    async def producer(self, queue: asyncio.Queue) -> None:
        """Queue unfinished active jobs and then discover new jobs."""

        assert self.manager is not None
        log.debug("Initializing PRODUCER")
        if self.manager.active_jobs:
            self.producer_state = "resuming"
            log.debug("Will start queuing jobs, which are already initialized.")
        for job in self.manager.active_jobs:
            if not job.is_done():
                await self._enqueue_job(queue, job, active=True)

        self.producer_state = "planning"
        log.debug("Will start queuing new jobs.")
        while True:
            try:
                new_job = await anext(self.manager.new_job_generator)
                await self._enqueue_job(queue, new_job, active=False)
            except StopAsyncIteration:
                log.debug("No more jobs!")
                break

        self.producer_state = "finished"
        log.debug("Producer done!")

    async def _enqueue_job(
        self, queue: asyncio.Queue, job: DownloadJob, *, active: bool
    ) -> None:
        """Queue one job while exposing producer backpressure state."""

        label = _job_label(job)
        self.producer_pending_job = label
        self.producer_state = "blocked_on_full_queue" if queue.full() else "enqueueing"
        try:
            await queue.put(job)
        finally:
            self.producer_pending_job = None
        self.queued_jobs[id(job)] = label
        self.producer_state = "resuming" if active else "planning"
        source = "Active" if active else "New"
        log.debug("%s job %s added to queue.", source, job)

    async def worker(self, name: str, queue: asyncio.Queue) -> None:
        """Download historical data chunks for queued jobs."""

        log.debug(f"Initializing {name.upper()}")
        while True:
            job = await queue.get()
            self.queued_jobs.pop(id(job), None)
            self.worker_jobs[name] = _job_label(job)

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
                self.worker_jobs.pop(name, None)
                queue.task_done()

    async def download_job(self, name: str, job: DownloadJob) -> None:
        """Download all currently queued chunks for one job."""

        assert self.manager is not None
        while True:
            if not request_age_available(job, self.manager.now):
                await job.save_chunk(
                    None,
                    empty_reason="Local IB age policy rejected the request before submission",
                )
                log.debug(f"{job!s} dropped before request on age validation.")
                break
            params = job.params
            log.debug(
                "%s prepared %s historical request ending %s with duration %s; "
                "handing it to the local pacer.",
                name,
                job,
                job.next_date,
                params["durationStr"],
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
                if not request_age_available(job, self.manager.now):
                    await job.save_chunk(
                        None,
                        empty_reason=(
                            "Local IB age policy stopped the range after the last "
                            "response"
                        ),
                    )
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

        tasks = [
            task
            for task in [self.producer_task, self.status_task, *self.workers]
            if task is not None
        ]
        for task in tasks:
            if not task.done():
                task.cancel()

        if self.queue is not None:
            shutdown_queue(self.queue)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def log_status(self) -> None:
        """Periodically explain whether work is locally queued or waiting on IB."""

        while True:
            await asyncio.sleep(STATUS_LOG_INTERVAL)
            assert self.queue is not None
            small = self.pacing.historical.status()
            large = self.pacing.large_bar_historical.status()
            discovery = self.pacing.metadata.status()
            historical_requests = tuple(
                sorted(
                    (*small.requests, *large.requests),
                    key=lambda request: request.elapsed_seconds,
                    reverse=True,
                )
            )
            workers_waiting_for_ib = sum(
                request.owner.startswith("worker ") for request in historical_requests
            )
            producer = self.producer_state
            if self.producer_pending_job:
                producer = f"{producer} (next={self.producer_pending_job})"
            log.debug(
                "Dataloader status: workers_engaged=%d/%d, "
                "workers_waiting_for_IB=%d, historical_requests_waiting_for_IB=%d, "
                "waiting_for_local_pacer=%d, waiting_for_local_concurrency=%d, "
                "contracts_waiting_for_worker=%d, producer=%s.",
                len(self.worker_jobs),
                self.number_of_workers,
                workers_waiting_for_ib,
                len(historical_requests),
                small.pacing_waiters + large.pacing_waiters,
                small.concurrency_waiters + large.concurrency_waiters,
                self.queue.qsize(),
                producer,
            )
            if historical_requests:
                log.debug(
                    "Waiting for IB (%d): %s.",
                    len(historical_requests),
                    _format_in_flight_requests(historical_requests),
                )
            if self.queued_jobs:
                log.debug(
                    "Waiting for worker (%d): %s.",
                    len(self.queued_jobs),
                    ", ".join(self.queued_jobs.values()),
                )
            if discovery.requests:
                log.debug(
                    "Discovery requests waiting for IB (%d): %s.",
                    len(discovery.requests),
                    _format_in_flight_requests(discovery.requests),
                )

    async def flush_pending(self) -> None:
        """Persist chunks buffered by active jobs after execution has stopped."""

        assert self.manager is not None
        for job in self.manager.active_jobs:
            await job.flush_pending()


def shutdown_queue(queue: asyncio.Queue) -> None:
    while True:
        try:
            queue.get_nowait()
            queue.task_done()
        except (asyncio.QueueEmpty, ValueError):
            break
    log.debug("Queue has been shutdown.")


def _job_label(job: DownloadJob) -> str:
    """Return a compact instrument label for queue and worker diagnostics."""

    contract = getattr(job, "contract", None)
    if contract is None:
        return str(getattr(job, "name", job))
    return str(contract.localSymbol or contract.symbol or contract)


def _format_in_flight_requests(requests: tuple[InFlightRequest, ...]) -> str:
    """Format named IB requests with oldest waits first."""

    return ", ".join(
        f"{request.label}={_format_elapsed(request.elapsed_seconds)}"
        for request in requests
    )


def _format_elapsed(seconds: float) -> str:
    """Return a compact elapsed duration for periodic status logs."""

    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _days_between(start: date | datetime, end: date | datetime) -> int:
    """Return whole calendar days between two date-like points."""

    start_date = start.date() if isinstance(start, datetime) else start
    end_date = end.date() if isinstance(end, datetime) else end
    return (end_date - start_date).days


def _contract_expiry(contract: ibi.Contract, bar_size: str) -> date | datetime | None:
    """Return an exact normalized contract expiry when one is available."""

    raw_expiry = str(getattr(contract, "lastTradeDateOrContractMonth", "") or "")
    if len(raw_expiry) < 8:
        return None
    expiry = datetime.strptime(raw_expiry[:8], "%Y%m%d").replace(tzinfo=timezone.utc)
    return normalize_point(expiry, bar_size)
