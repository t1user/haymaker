from __future__ import annotations

import asyncio
import functools
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import AsyncGenerator, Optional, TypedDict

import ib_insync as ibi
import pandas as pd
from tqdm import tqdm

from haymaker.config import CONFIG
from haymaker.logging import setup_logging

from . import helpers
from .connect import connection
from .contract_selectors import ContractSelector
from .pacer import PacingViolationError, RequestPacing
from .scheduling import task_factory, task_factory_with_gaps
from .settings import DataloaderSettings
from .store_wrapper import StoreWrapper
from .task_logger import create_task

"""
Async queue implementation modelled (loosely) on example here:
https://docs.python.org/3/library/asyncio-queue.html#examples
and here:
https://realpython.com/async-io-python/#using-a-queue
"""
setup_logging(CONFIG.get("logging_config"))

log = logging.getLogger(__name__)

SETTINGS = DataloaderSettings.from_config(CONFIG)
NOW: date | datetime = SETTINGS.normalize_datetime(datetime.now(timezone.utc))

log.debug(
    f"settings: {SETTINGS.bar_size=}, {SETTINGS.wts=}, {SETTINGS.max_bars=}, "
    f"{SETTINGS.fill_gaps=}, {SETTINGS.auto_save_interval=}, "
    f"{SETTINGS.number_of_workers=}, {SETTINGS.store=}, {NOW=}, "
    f"{SETTINGS.max_period=}, "
    f"{SETTINGS.pacer_allowance_fraction=}"
)


def request_pacing_factory() -> RequestPacing:
    """Create request pacing configured from dataloader settings."""

    return SETTINGS.create_pacing()


class Params(TypedDict):
    contract: ibi.Contract
    endDateTime: datetime | date | str
    durationStr: str


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
    settings: DataloaderSettings = field(repr=False)
    queue: list[DownloadContainer] = field(default_factory=list, repr=False)
    _pulse: ibi.Event | None = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        self._pulse = pulse_factory(self.settings.auto_save_interval)
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
            delta, self.settings.bar_size, self.settings.max_bars
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
    settings: DataloaderSettings = field(default_factory=lambda: SETTINGS)
    pacing: RequestPacing | None = None
    active_writers: list[DataWriter] = field(default_factory=list, repr=False)
    _initiated_contracts: list[ibi.Contract] = field(default_factory=list, repr=False)
    new_writer_generator: AsyncGenerator[DataWriter, None] = field(init=False)

    def __post_init__(self) -> None:
        if self.pacing is None:
            self.pacing = self.settings.create_pacing()
        self.new_writer_generator = self._writer_generator()

    @functools.cached_property
    def sources(self) -> list[dict]:
        return pd.read_csv(self.settings.source, keep_default_na=False).to_dict(
            "records"
        )

    async def contracts(self) -> AsyncGenerator[ibi.Contract, None]:
        with tqdm(self.sources, desc="Sources") as source_pbar:
            for s in source_pbar:
                contract_selector = ContractSelector.from_kwargs(ib=self.ib, **s)
                with tqdm(
                    desc=f"Contracts_{s.get('symbol')}", leave=False, total=None
                ) as contract_pbar:
                    async for contract in contract_selector.objects():
                        # ContractSelector has been calling the api,
                        # so to prevent pacing violation
                        await asyncio.sleep(0.5)
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
        if self.settings.fill_gaps:
            tasklist = task_factory_with_gaps(store, start)
        else:
            tasklist = task_factory(store, start)
        return [DownloadContainer(*t) for t in tasklist]

    async def writers(self) -> AsyncGenerator[DataWriter, None]:
        async for contract, headstamp in self.headstamps():
            if contract in self._initiated_contracts:
                continue
            self._initiated_contracts.append(contract)
            store = StoreWrapper(contract, self.settings.store, NOW)
            last_bar_date = min(
                datetime.fromisoformat(contract.lastTradeDateOrContractMonth).replace(
                    tzinfo=timezone.utc
                ),
                NOW,
            )
            start = max(
                headstamp, last_bar_date - timedelta(days=self.settings.max_period)
            )
            tasks = self.tasks(store, start)
            new_writer = DataWriter(contract, store, self.settings, tasks)
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
            try:
                assert self.pacing is not None
                async with self.pacing:
                    headTimeStamp = await self.ib.reqHeadTimeStampAsync(
                        contract,
                        whatToShow=self.settings.wts,
                        useRTH=False,
                        formatDate=2,
                    )
                # it was a pacing violation
                if (not headTimeStamp) and self.pacing.verify(contract):
                    raise PacingViolationError(
                        "Headstamp pacing violation ignored, job will be rescheduled."
                    )
                # no pacing violation but still no headstamp
                if not headTimeStamp:
                    five_years_ago = datetime.now(tz=timezone.utc) - timedelta(
                        days=5 * 250
                    )
                    log.warning(
                        (
                            f"Unavailable headTimeStamp ({headTimeStamp}) for "
                            f"{contract}. Will use {five_years_ago}"
                        )
                    )
                    headTimeStamp = five_years_ago
            except Exception:
                continue

        hs = self.settings.normalize_datetime(headTimeStamp)
        log.debug(f"Headstamp for: {contract.localSymbol}: {hs}")
        return hs


def validate_age(writer: DataWriter, settings: DataloaderSettings) -> bool:
    """
    IB doesn't permit to request data for bars < 30secs older than 6
    months.  Trying to push it here with 30secs.

    THIS IS NOT NECCESSARY???, MERGE IT WITH EARLIES POINT DETERMINATION.
    """
    if helpers.duration_in_secs(settings.bar_size) < 30 and writer.next_date:
        assert isinstance(writer.next_date, datetime)
        # TODO: not sure if correct or necessary
        if (NOW - writer.next_date).days > 180:
            return False
    return True


@dataclass
class DataloaderSession:
    """Own one dataloader runtime session and its restartable execution state."""

    ib: ibi.IB
    settings: DataloaderSettings = field(default_factory=lambda: SETTINGS)
    manager: Manager | None = None
    queue: asyncio.Queue[DataWriter] | None = field(default=None, init=False)
    workers: list[asyncio.Task] = field(default_factory=list, init=False)
    producer_task: asyncio.Task | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.manager is None:
            self.manager = Manager(self.ib, settings=self.settings)
        else:
            self.manager.ib = self.ib
            if not hasattr(self.manager, "settings"):
                self.manager.settings = self.settings

    @property
    def pacing(self) -> RequestPacing:
        """Return the pacing state used by this session."""

        assert self.manager is not None
        assert self.manager.pacing is not None
        return self.manager.pacing

    async def run(self) -> None:
        """Run producer and workers until all queued download work is finished."""

        log.debug("Initializing dataloader session.")
        self.queue = asyncio.LifoQueue(maxsize=int(self.settings.number_of_workers / 4))
        self.workers = [
            create_task(
                self.worker(f"worker_{i}", self.queue),
                logger=log,
                message="asyncio error",
                message_args=(f"worker {i}",),
            )
            for i in range(self.settings.number_of_workers)
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

        log.debug("Dataloader session run done!")

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

            while True:
                try:
                    async with self.pacing:
                        log.debug(
                            f"{name} loading {writer!s} ending: {writer.next_date} "
                            f"duration: {writer.params['durationStr']}"
                        )
                        chunk = await self.ib.reqHistoricalDataAsync(
                            **writer.params,
                            barSizeSetting=self.settings.bar_size,
                            whatToShow=self.settings.wts,
                            useRTH=False,
                            formatDate=2,
                            timeout=0,
                        )

                    if (not chunk) and self.pacing.verify(writer.contract):
                        # if pacing violation just happened, empty (or None) chunk
                        # doesn't mean there is no data; reschedule same params
                        raise PacingViolationError(
                            "This error is being ignored, job will be rescheduled."
                        )
                    # below will not run if error above, so
                    # `writer.next_date` will still hold the same date,
                    # i.e. the same chunk will be rescheduled
                    await writer.save_chunk(chunk)
                except ConnectionError:
                    while not self.ib.isConnected():
                        await asyncio.sleep(5)
                except Exception as e:
                    log.exception(e)
                    # prevent same request sooner than after 60 secs
                    await asyncio.sleep(60)
                    log.debug(f"{writer!s} rescheduled.")

                if writer.next_date:
                    if not validate_age(writer, self.settings):
                        await writer.save_chunk(None)
                        log.debug(f"{writer!s} dropped on age validation.")
                        break
                else:
                    log.info(f"{writer!s} done!")
                    break

            queue.task_done()

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
