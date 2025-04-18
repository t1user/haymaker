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

from haymaker.config import CONFIG
from haymaker.datastore import AbstractBaseStore
from haymaker.logging import setup_logging
from haymaker.validators import bar_size_validator, wts_validator

from . import helpers
from .connect import Connection, Mode
from .contract_selectors import ContractSelector
from .pacer import PacingViolationError, PacingViolationRegistry, pacer
from .scheduling import task_factory, task_factory_with_gaps
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


BARSIZE: str = bar_size_validator(CONFIG["barSize"])
WTS: str = wts_validator(CONFIG["wts"])
MAX_BARS: int = CONFIG.get("max_bars", 50000)
FILL_GAPS: bool = CONFIG.get("fill_gaps", True)
AUTO_SAVE_INTERVAL: int = CONFIG.get("auto_save_interval", 0)
NUMBER_OF_WORKERS: int = CONFIG.get("number_of_workers", 20)
STORE: AbstractBaseStore = CONFIG["datastore"]
norm = partial(helpers.datetime_normalizer, barsize=BARSIZE)

NOW: date | datetime = norm(datetime.now(timezone.utc))
RUN_MODE: Mode = CONFIG.get("run_mode", "reconnect")
SOURCE: str = CONFIG["source"]
PACER_NO_RESTRICTION: bool = CONFIG.get("pacer_no_restriction", False)
PACER_RESTRICTIONS: bool = CONFIG["pacer_restrictions"]
MAX_PERIOD = CONFIG.get("max_period", 30)
WORKER_TIMEOUT = CONFIG.get("worker_timeout", 60)
WRITERS_DONE: dict[str, bool] = {}

log.debug(
    f"settings: {BARSIZE=}, {WTS=}, {MAX_BARS=}, {FILL_GAPS=}, "
    f"{AUTO_SAVE_INTERVAL=}, {NUMBER_OF_WORKERS=}, {STORE=}, {NOW=}, "
    f"{MAX_PERIOD=}"
)


class Params(TypedDict):
    contract: ibi.Contract
    endDateTime: datetime | date | str
    durationStr: str


def pulse_factory(interval: float = AUTO_SAVE_INTERVAL) -> ibi.Event | None:
    if asi := AUTO_SAVE_INTERVAL:
        return ibi.Event().timerange(asi, None, asi)
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
    _pulse: ibi.Event | None = field(
        default_factory=pulse_factory, repr=False, init=False
    )

    def __post_init__(self) -> None:
        if self._pulse:
            self._pulse += self._onPulse
        log.info(f"{self!r} initialized.")

    async def _onPulse(self, time: datetime):
        if self.is_done():
            self._pulse -= self.onPulse  # type: ignore
        else:
            await self.write_to_store()

    def is_done(self) -> bool:
        return len(self.queue) == 0

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
        if self.is_done():
            log.warning(f"{self!s} is done, data: {data}")
            return
        if data:
            log.info(f"{self!s} received bars from: {data[0].date} to {data[-1].date}")
        elif self._container.next_date:
            log.warning(
                f"{self!s} cannot download data past {self._container.next_date}"
            )
        self._container.save_chunk(data)
        await self.write_to_store()
        if self._container.next_date is None:
            self.queue.pop()

    async def write_to_store(self) -> None:
        """
        This is the only method that actually saves data.  All other
        'save' methods don't.
        """

        if self.is_done():
            log.warning(f"{self}: abandoned attempt to write to store on done writer.")
            return

        _data = self._container.flush_data()

        if _data is not None:
            data = await self.store.data_async()
            if data is None:
                data = pd.DataFrame()
            data = pd.concat([data, _data])
            version = await self.store.write_async(self.contract, data)
            log.info(
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
        return helpers.timedelta_and_barSize_to_duration_str(delta, BARSIZE, MAX_BARS)

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
    store: AbstractBaseStore = STORE
    active_writers: list[DataWriter] = field(default_factory=list, repr=False)
    _initiated_contracts: list[ibi.Contract] = field(default_factory=list, repr=False)

    @functools.cached_property
    def sources(self) -> list[dict]:
        return pd.read_csv(SOURCE, keep_default_na=False).to_dict("records")

    async def contracts(self) -> AsyncGenerator[ibi.Contract, None]:
        ContractSelector.set_ib(self.ib)
        for s in self.sources:
            async for contract in ContractSelector.from_kwargs(**s).objects():
                # we've been calling api so to prevent pacing violation
                await asyncio.sleep(1)
                yield contract

    async def headstamps(
        self,
    ) -> AsyncGenerator[tuple[ibi.Contract, date | datetime], None]:
        async for contract in self.contracts():
            headstamp = await self.headstamp(contract)
            yield contract, headstamp

    def tasks(
        self, store: StoreWrapper, start: datetime | date
    ) -> list[DownloadContainer]:
        if FILL_GAPS:
            tasklist = task_factory_with_gaps(store, start)
        else:
            tasklist = task_factory(store, start)
        return [DownloadContainer(*t) for t in tasklist]

    async def writers(self) -> AsyncGenerator[DataWriter, None]:
        async for contract, headstamp in self.headstamps():
            if contract in self._initiated_contracts:
                continue
            self._initiated_contracts.append(contract)
            store = StoreWrapper(contract, STORE, NOW)
            last_bar_date = min(
                datetime.fromisoformat(contract.lastTradeDateOrContractMonth).replace(
                    tzinfo=timezone.utc
                ),
                NOW,
            )
            start = max(headstamp, last_bar_date - timedelta(days=MAX_PERIOD))
            tasks = self.tasks(store, start)
            new_writer = DataWriter(contract, store, tasks)
            self.active_writers.append(new_writer)
            yield new_writer

    async def writer_generator(self) -> AsyncGenerator[DataWriter, None]:
        for writer in self.active_writers:
            if not writer.is_done():
                yield writer
        async for new_writer in self.writers():
            yield new_writer

    async def headstamp(self, contract: ibi.Contract) -> date | datetime:
        headTimeStamp = None
        while not headTimeStamp:
            try:
                async with PACER:
                    headTimeStamp = await self.ib.reqHeadTimeStampAsync(
                        contract, whatToShow=WTS, useRTH=False, formatDate=2
                    )
                # it was a pacing violation
                if (not headTimeStamp) and PCR.verify(contract):
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
                            f"Unavailable headTimeStamp ({headTimeStamp}) for {contract}. "
                            f"Will use {five_years_ago}"
                        )
                    )
                    headTimeStamp = five_years_ago
            except Exception:
                continue

        hs = norm(headTimeStamp)
        log.debug(f"Headstamp for: {contract.localSymbol}: {hs}")
        return hs


def validate_age(writer: DataWriter) -> bool:
    """
    IB doesn't permit to request data for bars < 30secs older than 6
    months.  Trying to push it here with 30secs.

    THIS IS NOT NECCESSARY???, MERGE IT WITH EARLIES POINT DETERMINATION.
    """
    if helpers.duration_in_secs(BARSIZE) < 30 and writer.next_date:
        assert isinstance(writer.next_date, datetime)
        # TODO: not sure if correct or necessary
        if (NOW - writer.next_date).days > 180:
            return False
    return True


async def producer(manager: Manager, queue: asyncio.Queue) -> None:
    async for writer in manager.writer_generator():
        await queue.put(writer)
    WRITERS_DONE["done"] = True


async def worker(name: str, queue: asyncio.Queue, ib: ibi.IB, timeout: int) -> None:
    while True:
        log.debug(f"{name} will get a new contract.")

        # try:
        #     # producer needs to initialize after all workers are up and running
        #     writer = await asyncio.wait_for(queue.get(), timeout=timeout)
        # except TimeoutError:
        #     log.debug(f"{name} done, no more tasks in the queue.")
        #     break

        if not WRITERS_DONE:
            writer = await queue.get()
        else:
            break

        while True:
            try:
                async with PACER:
                    log.debug(
                        f"{name} loading {writer!s} ending: {writer.next_date} "
                        f"duration: {writer.params['durationStr']}"
                    )
                    chunk = await ib.reqHistoricalDataAsync(
                        **writer.params,
                        barSizeSetting=BARSIZE,
                        whatToShow=WTS,
                        useRTH=False,
                        formatDate=2,
                        timeout=0,
                    )

                if (not chunk) and PCR.verify(writer.contract):
                    # if pacing violation just happened, empty (or None) chunk doesn't
                    # mean there is no data; need to reschedule with same parameters
                    raise PacingViolationError(
                        "This error is being ignored, job will be rescheduled."
                    )
                # below will not run if error above, so
                # `writer.next_date` will still hold the same date,
                # i.e. the same chunk will be rescheduled
                await writer.save_chunk(chunk)
            except Exception as e:
                log.exception(e)
                # prevent same request sooner than after 15 secs
                await asyncio.sleep(15)
                log.debug(f"{writer!s} rescheduled.")

            if writer.next_date:
                if not validate_age(writer):
                    await writer.save_chunk(None)
                    log.debug(f"{writer!s} dropped on age validation.")
                    break
            else:
                log.info(f"{writer!s} done!")
                break

        queue.task_done()


WORKERS: list[asyncio.Task] = []
PRODUCER: list[asyncio.Task] = []


async def main(manager: Manager, ib: ibi.IB) -> None:

    queue: asyncio.Queue[DataWriter] = asyncio.LifoQueue(
        maxsize=int(NUMBER_OF_WORKERS / 4)
    )

    # just a precaution
    WORKERS.clear()
    WORKERS.extend(
        [
            create_task(
                worker(f"worker {i}", queue, ib, WORKER_TIMEOUT),
                logger=log,
                message="asyncio error",
                message_args=(f"worker {i}",),
            )
            for i in range(NUMBER_OF_WORKERS)
        ]
    )

    producer_task: asyncio.Task[None] = create_task(
        producer(manager, queue),
        logger=log,
        message="asyncio error",
        message_args=("producer",),
    )
    # just a precaution
    PRODUCER.clear()
    PRODUCER.append(producer_task)

    # wait for producer to finish
    await producer_task

    # wait for queue to empty
    await queue.join()

    # wait for workers to finish
    await asyncio.gather(*WORKERS)

    log.debug("Main done!")


def cancel_tasks():
    log.debug("Will cancel tasks.")
    assert (i := len(PRODUCER) == 1), f"Number of producers: {i}"
    PRODUCER[0].cancel()
    PRODUCER.clear()
    for task in WORKERS:
        task.cancel()
    WORKERS.clear()


def start():
    ibi.util.patchAsyncio()
    ib = ibi.IB()
    ib.errorEvent += PCR.onError
    manager = Manager(ib)
    asyncio.get_event_loop().set_debug(True)
    log.debug("Will start...")

    Connection(ib, partial(main, manager, ib), cancel_tasks, RUN_MODE)

    log.info("script finished, about to disconnect")
    ib.disconnect()
    log.debug("disconnected")


PACER = pacer(
    BARSIZE,
    WTS,
    restrictions=[] if PACER_NO_RESTRICTION else PACER_RESTRICTIONS,  # type: ignore
)
log.debug(f"Pacer initialized: {PACER}")
PCR = PacingViolationRegistry()
