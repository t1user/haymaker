from __future__ import annotations

import asyncio
import functools
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from functools import partial
from typing import ClassVar, NamedTuple, Optional, TypedDict, Union

import ib_insync as ibi
import pandas as pd

from ib_tools.config import CONFIG
from ib_tools.datastore import AbstractBaseStore
from ib_tools.logging import setup_logging
from ib_tools.validators import Validator, bar_size_validator, wts_validator

from . import helpers
from .connect import Connection
from .contract_selectors import ContractSelector
from .pacer import Pacer, pacer
from .task_logger import create_task

"""
Async queue implementation modelled (loosely) on example here:
https://docs.python.org/3/library/asyncio-queue.html#examples
and here:
https://realpython.com/async-io-python/#using-a-queue
"""
setup_logging()

log = logging.getLogger(__name__)

MAX_NUMBER_OF_WORKERS = CONFIG.get("max_number_of_workers", 40)


class Params(TypedDict):
    contract: ibi.Contract
    endDateTime: Union[datetime, date, str, None]
    durationStr: str
    barSizeSetting: str
    whatToShow: str
    useRTH: bool


@dataclass
class DataWriter:
    """
    Interface between dataloader and datastore.

    It is created for every contract for which data is to be
    downloaded.  It's responsibilities are (SRP?????):

    * determine dates for which download is necessary (based on data
    already available in datastore)

    * save data to the store

    * provide exact params for :meth:`ib.reqHistoricalData`

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

    store: AbstractBaseStore
    contract: ibi.Contract
    head: datetime
    barSize: str
    wts: str
    aggression: float = 2
    now: datetime = field(
        default_factory=partial(datetime.now, timezone.utc), repr=False
    )
    fill_gaps: bool = True
    _queue: list[DownloadContainer] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.c = self.contract.localSymbol
        if asi := CONFIG.get("auto_save_interval", 0):
            pulse = ibi.Event().timerange(asi, None, asi)
            pulse += self.onPulse
        self.schedule_tasks()
        log.info(f"Object initialized: {self}")
        # self.schedule_next()

    def onPulse(self, time: datetime):
        self.write_to_store()

    def schedule_tasks(self):

        if backfill := self.backfill():
            log.debug(f"{self.c} queued for backfill")
            self._queue.append(DownloadContainer(from_date=self.head, to_date=backfill))

        if update := self.update():
            log.debug(f"{self.c} queued for update")
            self._queue.append(
                DownloadContainer(from_date=self.to_date, to_date=update, update=True)
            )

        if self.fill_gaps and (fill_gaps := self.gap_filler()):
            for gap in fill_gaps:
                log.debug(f"{self.c} queued gap from {gap.start} to {gap.stop}")
                self._queue.append(
                    DownloadContainer(from_date=gap.start, to_date=gap.stop)
                )

    def is_done(self) -> bool:
        return len(self._queue) == 0

    @property
    def _current_container(self) -> DownloadContainer:
        # there must be at least one object in the queue, otherwise it's an error
        # don't call the method without verifying the queue is not empty
        # with self.is_done()
        return self._queue[-1]

    @property
    def next_date(self) -> Union[date, datetime, None]:
        if not self.is_done():
            return self._current_container.next_date
        else:
            return None

    # def schedule_next(self):
    #     if self._current_object:
    #         self.write_to_store()
    #     try:
    #         self._current_object = self._queue.pop()
    #     except IndexError:
    #         self.write_to_store()
    #         self.next_date = ""  # this should be None; TODO
    #         log.debug(f"{self.c} done!")
    #         return
    #     self.next_date = self._current_object.to_date
    #     log.debug(f"scheduling {self.c}: {self._current_object}")

    def save_chunk(self, data: ibi.BarDataList):
        assert not self.is_done()
        container = self._current_container
        if data:
            log.debug(f"{self} received bars from: {data[0].date} to {data[-1].date}")
        elif container.next_date:
            log.warning(f"Cannot download data past {container.next_date}")
        container.save_chunk(data)
        next_date = container.next_date
        log.debug(f"{self}: chunk saved, next_date: {next_date}")
        if not next_date:
            self._queue.pop()

    def write_to_store(self) -> None:
        """
        This is the only method that actually saves data.  All other
        'save' methods don't.
        """

        if self.is_done():
            return

        _data = self._current_container.flush_data()

        if _data is not None:
            data = self.data
            if data is None:
                data = pd.DataFrame()
            data = pd.concat([data, _data])
            version = self.store.write(self.contract, data)
            log.debug(
                f"{self!s} written to store "
                f"{self._current_container.from_date} - {self._current_container.to_date}"
                f"version {version}"
            )
            if version:
                self._current_container.clear()

    def backfill(self) -> Optional[datetime]:
        """
        Check if data earlier than earliest point in datastore available.
        Return the data at which backfill should start.
        """
        # prevent multiple calls to datastore
        from_date = self.from_date
        # data present in datastore
        if from_date:
            return from_date if from_date > self.head else None
        # data not in datastore yet
        else:
            return min(self.expiry, self.now) if self.expiry else self.now

    def update(self) -> Optional[datetime]:
        """
        Check if data newer than endpoint in datastore available for download.
        Return current date if yes, None if not.
        """
        # prevent multiple calls to datastore
        to_date = self.to_date
        if to_date:
            dt = min(self.expiry, self.now) if self.expiry else self.now

            if dt > to_date:
                return dt

        return None

    def gap_filler(self) -> list[NamedTuple]:
        if self.data is None:
            return []
        data = self.data.copy()
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
        return list(non_standard_gaps[["start", "stop"]].itertuples(index=False))

    @property
    def params(self) -> Params:
        return {
            "contract": self.contract,
            "endDateTime": self.next_date,
            "durationStr": self.duration,
            "barSizeSetting": self.barSize,
            "whatToShow": self.wts,
            "useRTH": False,
        }

    @property
    def duration(self) -> str:
        duration = helpers.barSize_to_duration(self.barSize, self.aggression)
        # this gets string and datetime error TODO !!!!!!!!!!!!!!!
        try:
            delta = self.next_date - self._current_container.from_date
        except Exception as e:
            log.error(
                f"next date: {self.next_date}, "
                f"from_date: {self._current_container.from_date}",
                e,
            )
            raise

        if delta < helpers.duration_to_timedelta(duration):
            # requests for periods shorter than 30s don't work
            duration = helpers.duration_str(
                max(delta.total_seconds(), 30), self.aggression, False
            )
        return duration

    @property
    def expiry(self) -> Optional[datetime]:  # this maybe an error
        """Expiry date for expirable contracts or ''"""
        e = self.contract.lastTradeDateOrContractMonth
        return (
            None
            if not e
            else datetime.strptime(e, "%Y%m%d").replace(tzinfo=timezone.utc)
        )

    @property
    def data(self) -> Optional[pd.DataFrame]:
        """Available data in datastore for contract or None"""
        return self.store.read(self.contract)

    @property
    def from_date(self) -> Optional[datetime]:
        """Earliest point in datastore"""
        # second point in the df to avoid 1 point gap
        return self.data.index[1] if self.data is not None else None  # type: ignore

    @property
    def to_date(self) -> Optional[datetime]:
        """Latest point in datastore"""
        date = self.data.index.max() if self.data is not None else None
        return date

    def str(self) -> str:
        return (
            f"{self.contract.localSymbol} {self.duration} {self.barSize} "
            f"next date: {self.next_date} "
        )


@dataclass
class DownloadContainer:
    """
    Hold downloaded data before it is saved to datastore.

    Object is initiated with desided from and to dates, for which data
    is to to be loaded, subsequently it is used to store any data
    before it's permanently saved.  It also keeps track of which data
    is still missing.


    Public interface:

    * next_date

    * save_chunk

    * clear

    * data

    """

    from_date: Union[datetime, date]
    to_date: Union[datetime, date]
    _next_date: Union[datetime, date, None] = field(init=False, repr=False)
    bars: list[ibi.BarDataList] = field(default_factory=list, repr=False)

    def __post_init__(self):
        self.next_date = self.to_date

    @property
    def next_date(self):
        return self._next_date

    @next_date.setter
    def next_date(self, date: Union[date, datetime, None]) -> None:
        if date is None:
            self._next_date = None
        elif date <= self.from_date:
            self._next_date = None
        else:
            self._next_date = date

    def save_chunk(self, bars: Optional[ibi.BarDataList]) -> None:
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
        """Return df ready to be written to datastore or date of end
        point for additional downloads.

        TODO: Not true now, do I want to make it true?
        Make sure to safe whatever
        comes out of this method because this data is being deleted.
        """
        if self.bars:
            df = ibi.util.df(
                [b for bars in reversed(self.bars) for b in bars]
            ).set_index("date")
            return df
        else:
            return None

    def clear(self):
        self.bars.clear()


class _Manager:
    """
    Helper class, whose only purpose is running validators, which
    would be otherwise difficult in a dataclass.
    """

    barSize: ClassVar = Validator(bar_size_validator)
    wts: ClassVar = Validator(wts_validator)


@dataclass
class Manager(_Manager):
    ib: ibi.IB
    barSize: str = CONFIG["barSize"]  # type: ignore
    wts: str = CONFIG["wts"]  # type: ignore
    aggression: int = CONFIG["aggression"]
    store: AbstractBaseStore = CONFIG["datastore"]
    fill_gaps: bool = CONFIG.get("fill_gaps", True)

    @functools.cached_property
    def sources(self) -> list[dict]:
        return pd.read_csv(CONFIG["source"], keep_default_na=False).to_dict("records")

    @functools.cached_property
    def contracts(self) -> list[ibi.Contract]:
        return ibi.util.run(self._contracts())

    async def _contracts(self) -> list[ibi.Contract]:
        ContractSelector.set_ib(self.ib)
        contracts = []
        for s in self.sources:
            contracts.extend(ContractSelector.from_kwargs(**s).objects())
        await self.ib.qualifyContractsAsync(*contracts)
        log.debug(f"{contracts=}")
        return contracts

    @functools.cached_property
    def headstamps(self):
        return ibi.util.run(self._headstamps())

    async def _headstamps(self) -> dict[ibi.Contract, datetime]:
        headstamps = {}
        for c in self.contracts:
            if c_ := await self.headstamp(c):
                headstamps[c] = c_
        return headstamps

    @functools.cached_property
    def writers(self) -> list[DataWriter]:
        return [
            self.init_writer(
                self.store,
                contract,
                headstamp,
                self.barSize,
                self.wts,
                self.aggression,
                self.fill_gaps,
            )
            for contract, headstamp in self.headstamps.items()
        ]

    @staticmethod
    def init_writer(
        store: AbstractBaseStore,
        contract: ibi.Contract,
        headTimeStamp: datetime,
        barSize: str,
        wts: str,
        aggression: int,
        fill_gaps: bool,
    ):
        DataWriter(
            store,
            contract,
            headTimeStamp,
            barSize=barSize,
            wts=wts,
            aggression=aggression,
            fill_gaps=fill_gaps,
        )

    async def headstamp(self, contract: ibi.Contract):
        try:
            headTimeStamp = await self.ib.reqHeadTimeStampAsync(
                contract, whatToShow=self.wts, useRTH=False, formatDate=2
            )

            if headTimeStamp == []:
                log.warning(
                    (
                        f"Unavailable headTimeStamp for {contract}. "
                        f"No data will be downloaded"
                    )
                )
        except Exception:
            log.exception(f"Exception while getting headTimeStamp for {contract}")
        return headTimeStamp


def validate_age(writer: DataWriter) -> bool:
    """
    IB doesn't permit to request data for bars < 30secs older than 6
    months.  Trying to push it here with 30secs.
    """
    if helpers.duration_in_secs(writer.barSize) < 30 and writer.next_date:
        assert isinstance(writer.next_date, datetime)
        # TODO: not sure if correct or necessary
        if (datetime.now(timezone.utc) - writer.next_date).days > 180:
            return False
    return True


async def worker(name: str, queue: asyncio.Queue, pacer: Pacer, ib: ibi.IB) -> None:
    while True:
        # TODO: questionable if this is necessary, as workers are cancelled eventually
        if queue.empty():
            break

        writer = await queue.get()
        log.debug(f"{name} loading {writer!s} ")
        async with pacer:
            chunk = await ib.reqHistoricalDataAsync(
                **writer.params, formatDate=2, timeout=0
            )

        writer.save_chunk(chunk)
        if writer.next_date:
            if validate_age(writer):
                await queue.put(writer)
            else:
                writer.save_chunk(None)
                log.debug(f"{writer!s} dropped on age validation")
        else:
            log.debug(f"{writer!s} done!")

        queue.task_done()


async def main(manager: Manager, ib: ibi.IB) -> None:

    writers = manager.writers
    log.debug(f"{writers=}")
    number_of_workers = min(len(writers), MAX_NUMBER_OF_WORKERS)

    log.debug(f"main function started, retrieving data for {len(writers)} instruments")

    queue: asyncio.Queue[DataWriter] = asyncio.LifoQueue()
    for writer in writers:
        await queue.put(writer)
    p = pacer(
        writer.barSize,
        writer.wts,
        restrictions=(
            []
            if CONFIG.get("pacer_no_restriction", False)
            else CONFIG["pacer_restrictions"]
        ),
    )
    log.debug(f"Pacer initialized: {p}")
    workers: list[asyncio.Task] = [
        create_task(
            worker(f"worker {i}", queue, p, ib),
            logger=log,
            message="asyncio error",
            message_args=(f"worker {i}",),
        )
        for i in range(number_of_workers)
    ]
    """
    workers = [asyncio.create_task(worker(f'worker {i}', queue, pacer))
               for i in range(number_of_workers)]

    """
    await queue.join()

    # cancel all workers
    log.debug("cancelling workers")
    for w in workers:
        w.cancel()

    # wait until all worker tasks are cancelled
    await asyncio.gather(*workers)


def start():

    ibi.util.patchAsyncio()
    ib = ibi.IB()
    manager = Manager(ib)
    asyncio.get_event_loop().set_debug(True)
    # util.logToConsole(logging.ERROR)
    log.debug("Will start...")

    Connection(ib, partial(main, manager, ib), watchdog=CONFIG.get("watchdog", True))

    log.debug("script finished, about to disconnect")
    ib.disconnect()
    log.debug("disconnected")
