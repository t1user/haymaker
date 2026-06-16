"""Asynchronous Interactive Brokers historical data downloader.

The module is built around a producer/worker queue. A manager discovers the
contracts and datastore ranges that need data, producers enqueue writer jobs,
and workers download historical chunks through a session-scoped request pacer.
Connection recovery is owned by the dataloader supervisor integration; request
pacing, failure recording, and resume-in-memory state live in the dataloader
session.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from functools import partial
from typing import AsyncGenerator, Optional, TypedDict

import ib_insync as ibi
import pandas as pd
from tqdm import tqdm

from haymaker.config import CONFIG
from haymaker.datastore import AbstractBaseStore
from haymaker.logging import setup_logging
from haymaker.validators import bar_size_validator, wts_validator

from . import helpers
from .connect import connection
from .contract_selectors import ContractSelector
from .pacer import RequestPacing
from .scheduling import task_factory, task_factory_with_gaps
from .store_wrapper import StoreWrapper
from .task_logger import create_task

setup_logging(CONFIG.get("logging_config"))

log = logging.getLogger(__name__)

BARSIZE: str = bar_size_validator(CONFIG["barSize"])
WTS: str = wts_validator(CONFIG["wts"])
MAX_BARS: int = CONFIG.get("max_bars", 50000)
FILL_GAPS: bool = CONFIG.get("fill_gaps", True)
AUTO_SAVE_INTERVAL: int = CONFIG.get("auto_save_interval", 0)
NUMBER_OF_WORKERS: int = CONFIG.get("number_of_workers", 20)
STORE: AbstractBaseStore = CONFIG["datastore"]
SOURCE: str = CONFIG["source"]
PACER_NO_RESTRICTION: bool = CONFIG.get("pacer_no_restriction", False)
PACER_ALLOWANCE_FRACTION: float = CONFIG.get("pacer_allowance_fraction", 1.0)
MAX_PERIOD: int = CONFIG.get("max_period", 30)

if PACER_ALLOWANCE_FRACTION <= 0:
    raise ValueError("pacer_allowance_fraction must be greater than 0")

norm = partial(helpers.datetime_normalizer, barsize=BARSIZE)

log.debug(
    f"settings: {BARSIZE=}, {WTS=}, {MAX_BARS=}, {FILL_GAPS=}, "
    f"{AUTO_SAVE_INTERVAL=}, {NUMBER_OF_WORKERS=}, {STORE=}, "
    f"{MAX_PERIOD=}, {PACER_ALLOWANCE_FRACTION=}"
)


def request_pacing_factory(ib: ibi.IB) -> RequestPacing:
    """Create request pacing from dataloader config constants."""

    return RequestPacing(
        ib,
        BARSIZE,
        WTS,
        no_restriction=PACER_NO_RESTRICTION,
        allowance_fraction=PACER_ALLOWANCE_FRACTION,
    )


def current_session_now() -> date | datetime:
    """Return current time normalized for the configured bar size."""

    return norm(datetime.now(timezone.utc))


class Params(TypedDict):
    contract: ibi.Contract
    endDateTime: datetime | date | str
    durationStr: str


@dataclass
class DownloadFailure:
    """Record one failed dataloader writer job."""

    writer: DataWriter
    error: BaseException
    when: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class DownloadFailureRegistry:
    """Collect failed writer jobs so the session can report them at completion."""

    failures: list[DownloadFailure] = field(default_factory=list)

    def record(self, writer: DataWriter, error: BaseException) -> None:
        """Record one failed writer job."""

        self.failures.append(DownloadFailure(writer, error))

    def log_summary(self) -> None:
        """Log a summary of failed writer jobs."""

        if not self.failures:
            log.info("Dataloader completed with no failed writer jobs.")
            return

        log.warning(
            f"Dataloader completed with {len(self.failures)} failed writer jobs."
        )
        for failure in self.failures:
            log.warning(
                "%s failed at %s with %s: %s",
                failure.writer,
                failure.when.isoformat(),
                type(failure.error).__name__,
                failure.error,
            )


def pulse_factory(interval: float) -> ibi.Event | None:
    if interval:
        return ibi.Event().timerange(interval, None, interval)
    return None


@dataclass
class DataWriter:
    """
    Keep track of all download tasks for one contract, provide exact
    :meth:`ib.reqHistoricalData` params for every download task and
    save data to store.

    There are potentially 3 streams of data that writer might
    schedule:

    * backfill - data older than the oldest data point available in
    the store

    * update - data newer than the newest data point available

    * fill_gaps - any data missing inside the range available in the
    store

    public methods/attributes:

    * `contract`

    * `next_date`

    * `params`

    * `save_chunk`
    """

    contract: ibi.Contract
    store: StoreWrapper = field(repr=False)
    queue: list[DownloadContainer] = field(default_factory=list, repr=False)
    bar_size: str = BARSIZE
    max_bars: int = MAX_BARS
    auto_save_interval: int = AUTO_SAVE_INTERVAL
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

    async def save_chunk(self, data: ibi.BarDataList) -> None:
        if data:
            log.info(f"{self!s} received bars from: {data[0].date} to {data[-1].date}")
            self._container.save_chunk(data)
            await self.write_to_store()
        else:
            if self._container.next_date:
                log.warning(
                    f"{self!s} cannot download data past {self._container.next_date}"
                )
            self.queue.pop()

    async def write_to_store(self) -> None:
        """
        This is the only method that actually saves data.  All other
        'save' methods don't.
        """

        _data = self._container.flush_data()

        if _data is not None:
            data = await self.store.data_async()
            if data is None:
                data = pd.DataFrame()
            data = pd.concat([data, _data])
            version = await self.store.write_async(self.contract, data)
            log.debug(
                f"{self!s} written to store {_data.index[0]} - {_data.index[-1]} "
                f"{version}"
            )
            if version:
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
        assert self.next_date is not None
        delta = self.next_date - self._container.from_date
        return helpers.timedelta_and_barSize_to_duration_str(
            delta, self.bar_size, self.max_bars
        )

    def __str__(self) -> str:
        return f"<Writer: {self.contract.localSymbol}>"

    def __repr__(self) -> str:
        return (
            f"<Writer: {self.contract.localSymbol} "
            f"{[(str(i.from_date), str(i.to_date)) for i in self.queue]}>"
        )


@dataclass
class DownloadContainer:
    """
    Hold downloaded data before it is saved to datastore.

    Object is initiated with desided from and to dates, for which data
    is to to be loaded, subsequently it is used to store any data
    before it's permanently saved.  It also keeps track of which data
    is still missing.


    Methods:

    * next_date

    * save_chunk

    * clear

    * flush_data

    """

    from_date: datetime | date
    to_date: datetime | date
    _next_date: datetime | date | None = field(init=False, repr=False)
    bars: list[ibi.BarDataList] = field(default_factory=list, repr=False)

    def __post_init__(self):
        assert (
            self.to_date > self.from_date
        ), f"{self.from_date=} is later than {self.to_date=}"
        self.next_date = self.to_date

    @property
    def next_date(self):
        return self._next_date

    @next_date.setter
    def next_date(self, date: date | datetime | None) -> None:
        if date is None:
            # failed to get data
            self._next_date = None
        elif date <= self.from_date:
            # got data, that's all we need
            self._next_date = None
        else:
            # got data, there's more to get
            self._next_date = date

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
    store: AbstractBaseStore = STORE
    source: str = SOURCE
    fill_gaps: bool = FILL_GAPS
    max_period: int = MAX_PERIOD
    wts: str = WTS
    now: date | datetime = field(default_factory=current_session_now)
    active_writers: list[DataWriter] = field(default_factory=list, repr=False)
    _initiated_contracts: list[ibi.Contract] = field(default_factory=list, repr=False)
    new_writer_generator: AsyncGenerator[DataWriter, None] = field(init=False)

    def __post_init__(self) -> None:
        if self.pacing is None:
            self.pacing = request_pacing_factory(self.ib)
        self.new_writer_generator = self._writer_generator()

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

    def tasks(
        self, store: StoreWrapper, start: datetime | date
    ) -> list[DownloadContainer]:
        if self.fill_gaps:
            tasklist = task_factory_with_gaps(store, start)
        else:
            tasklist = task_factory(store, start)
        return [DownloadContainer(*t) for t in tasklist]

    async def writers(self) -> AsyncGenerator[DataWriter, None]:
        async for contract, headstamp in self.headstamps():
            if contract in self._initiated_contracts:
                continue
            self._initiated_contracts.append(contract)
            store = StoreWrapper(contract, self.store, self.now)
            last_bar_date = min(
                datetime.fromisoformat(contract.lastTradeDateOrContractMonth).replace(
                    tzinfo=timezone.utc
                ),
                self.now,
            )
            start = max(headstamp, last_bar_date - timedelta(days=self.max_period))
            tasks = self.tasks(store, start)
            new_writer = DataWriter(contract, store, tasks)
            if new_writer.is_done():
                log.debug(f"Skipping {new_writer!s} - no need to download data.")
                continue
            yield new_writer

    async def _writer_generator(self) -> AsyncGenerator[DataWriter, None]:
        async for new_writer in self.writers():
            self.active_writers.append(new_writer)
            yield new_writer

    async def headstamp(self, contract: ibi.Contract) -> date | datetime:
        headTimeStamp = None
        while not headTimeStamp:
            assert self.pacing is not None
            headTimeStamp = await self.pacing.head_timestamp(
                contract,
                whatToShow=self.wts,
                useRTH=False,
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

        hs = norm(headTimeStamp)
        log.debug(f"Headstamp for: {contract.localSymbol}: {hs}")
        return hs


def validate_age(writer: DataWriter) -> bool:
    """
    IB doesn't permit to request data for bars < 30secs older than 6
    months.  Trying to push it here with 30secs.

    THIS IS NOT NECCESSARY???, MERGE IT WITH EARLIES POINT DETERMINATION.
    """
    if helpers.duration_in_secs(writer.bar_size) < 30 and writer.next_date:
        assert isinstance(writer.next_date, datetime)
        # TODO: not sure if correct or necessary
        now = current_session_now()
        if isinstance(now, datetime) and (now - writer.next_date).days > 180:
            return False
    return True


@dataclass
class DataloaderSession:
    """Own one dataloader runtime session and its restartable execution state."""

    ib: ibi.IB
    manager: Manager | None = None
    number_of_workers: int = NUMBER_OF_WORKERS
    bar_size: str = BARSIZE
    wts: str = WTS
    failures: DownloadFailureRegistry = field(default_factory=DownloadFailureRegistry)
    queue: asyncio.Queue[DataWriter] | None = field(default=None, init=False)
    workers: list[asyncio.Task] = field(default_factory=list, init=False)
    producer_task: asyncio.Task | None = field(default=None, init=False)

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
            await self.producer_task
            await self.queue.join()
        finally:
            await self.cancel_execution()
            self.failures.log_summary()

    async def producer(self, queue: asyncio.Queue) -> None:
        """Queue unfinished active writers and then discover new writers."""

        assert self.manager is not None
        log.debug("Initializing PRODUCER")
        if self.manager.active_writers:
            log.debug("Will start queuing writers, which are already initialized.")
        for writer in self.manager.active_writers:
            if not writer.is_done():
                await queue.put(writer)
                log.debug(f"Active writer {writer} added to queue.")

        log.debug("Will start queing new writers.")
        while True:
            try:
                new_writer = await anext(self.manager.new_writer_generator)
                await queue.put(new_writer)
                log.debug(f"New writer {new_writer} added to queue.")
            except StopAsyncIteration:
                log.debug("No more writers!")
                break

        log.debug("Producer done!")

    async def worker(self, name: str, queue: asyncio.Queue) -> None:
        """Download historical data chunks for queued writers."""

        log.debug(f"Initializing {name.upper()}")
        while True:
            writer = await queue.get()

            try:
                await self.download_writer(name, writer)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.exception("%s failed while loading %s.", name, writer)
                self.failures.record(writer, exc)
            finally:
                queue.task_done()

    async def download_writer(self, name: str, writer: DataWriter) -> None:
        """Download all currently queued chunks for one writer."""

        while True:
            params = writer.params
            log.debug(
                f"{name} loading {writer!s} ending: {writer.next_date} "
                f"duration: {params['durationStr']}"
            )
            chunk = await self.pacing.historical_data(
                **params,
                barSizeSetting=self.bar_size,
                whatToShow=self.wts,
                useRTH=False,
                formatDate=2,
            )
            await writer.save_chunk(chunk)

            if writer.next_date:
                if not validate_age(writer):
                    await writer.save_chunk(None)
                    log.debug(f"{writer!s} dropped on age validation.")
                    break
            else:
                log.info(f"{writer!s} done!")
                break

    def cancel_tasks(self) -> None:
        """Request cancellation of active producer/worker tasks."""

        log.debug("Will cancel dataloader session tasks.")
        if self.producer_task and not self.producer_task.done():
            self.producer_task.cancel()
        if self.queue:
            shutdown_queue(self.queue)
        log.debug("Dataloader session tasks cancelled.")

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
