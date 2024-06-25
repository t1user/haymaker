from __future__ import annotations

import asyncio
import functools
import logging
from collections import deque
from dataclasses import dataclass, field, fields
from datetime import date, datetime, timedelta, timezone
from functools import partial
from typing import ClassVar, NamedTuple, Optional, Self, Type, Union

import ib_insync as ibi
import pandas as pd

from ib_tools.config import CONFIG
from ib_tools.datastore import AbstractBaseStore
from ib_tools.logging import setup_logging
from ib_tools.validators import Validator, bar_size_validator, wts_validator

from .connect import Connection
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


class ContractSelector:
    """
    Based on passes contract attributes and config parameters
    determine instrument(s) that should be added to data download
    queue.  The output of any computations is accessible via
    :meth:`objects`

    :meth:`from_kwargs` will return instance of correct
    (sub)class based on instrument type and config

    :meth:`objects` contains a list of :class:`ibi.Contract` objects
    for which historical data will be loaded; those objects are not
    guaranteed to be qualified.
    """

    ib: ibi.IB
    sec_types = {
        "STK",  # Stock
        "OPT",  # Option
        "FUT",  # Future
        "CONTFUT",  # ContFuture
        "CASH",  # Forex
        "IND",  # Index
        "CFD",  # CFD
        "BOND",  # Bond
        "CMDTY",  # Commodity
        "FOP",  # FuturesOption
        "FUND",  # MutualFund
        "WAR",  # Warrant
        "IOPT",  # Warrant
        "BAG",  # Bag
        "CRYPTO",  # Crypto
    }
    contract_fields = {i.name for i in fields(ibi.Contract)}

    @classmethod
    def set_ib(cls, ib: ibi.IB) -> Type[Self]:
        cls.ib = ib
        return cls

    @classmethod
    def from_kwargs(cls, **kwargs) -> Self:
        secType = kwargs.get("secType")
        if secType not in cls.sec_types:
            raise TypeError(f"secType must be one of {cls.sec_types} not: {secType}")
        elif secType in {"FUT", "CONTFUT"}:
            return cls.from_future_kwargs(**kwargs)
        # TODO: specific cases for other asset classes
        else:
            return cls(**kwargs)

    @classmethod
    def from_future_kwargs(cls, **kwargs):
        return FutureContractSelector.create(**kwargs)

    def __init__(self, **kwargs) -> None:
        try:
            self.ib
        except AttributeError:
            raise AttributeError(
                f"ib attribute must be set on {self.__class__.__name__} "
                f"before class is instantiated."
            )
        self.kwargs = self.clean_fields(**kwargs)

    def clean_fields(self, **kwargs) -> dict:
        if diff := (set(kwargs.keys()) - self.contract_fields):
            for k in diff:
                del kwargs[k]
            log.warning(
                f"Removed incorrect contract parameters: {diff}, "
                f"will attemp to get Contract anyway"
            )
        return kwargs

    def objects(self) -> list[ibi.Contract]:
        try:
            return ibi.util.run(self._objects())  # type: ignore
        except AttributeError:
            return [ibi.Contract.create(**self.kwargs)]

    def repr(self) -> str:
        kwargs_str = ", ".join([f"{k}={v}" for k, v in self.kwargs.items()])
        return f"{self.__class__.__qualname__}({kwargs_str})"


class FutureContractSelector(ContractSelector):

    @classmethod
    def create(cls, **kwargs):
        # in case of any ambiguities just go for contfuture
        klass = {
            "contfuture": ContfutureFutureContractSelector,
            "fullchain": FullchainFutureContractSelector,
            "current": CurrentFutureContractSelector,
            "exact": ExactFutureContractSelector,
        }.get(
            CONFIG.get("futures_selector", "contfuture"),
            ContfutureFutureContractSelector,
        )
        return klass(**kwargs)

    @functools.cached_property
    def _fullchain(self) -> list[ibi.Contract]:
        kwargs = self.kwargs.copy()
        kwargs["secType"] = "FUT"
        kwargs["includeExpired"] = True
        details = ibi.util.run(self.ib.reqContractDetailsAsync(ibi.Contract(**kwargs)))
        return sorted(
            [c.contract for c in details if c.contract is not None],
            key=lambda x: x.lastTradeDateOrContractMonth,
        )

    @functools.cached_property
    def _contfuture_index(self) -> int:
        return self._fullchain.index(self._contfuture_qualified)

    @functools.cached_property
    def _contfuture_qualified(self) -> ibi.Contract:
        contfuture = self._contfuture
        ibi.util.run(self.ib.qualifyContractsAsync(contfuture))
        future_kwargs = contfuture.nonDefaults()  # type: ignore
        del future_kwargs["secType"]
        return ibi.Contract.create(**future_kwargs)

    @functools.cached_property
    def _contfuture(self) -> ibi.Contract:
        kwargs = self.kwargs.copy()
        kwargs["secType"] = "CONTFUT"
        return ibi.Contract.create(**kwargs)


class ContfutureFutureContractSelector(FutureContractSelector):

    def objects(self) -> list[ibi.Contract]:
        return [self._contfuture]


class FullchainFutureContractSelector(FutureContractSelector):

    async def _objects(self) -> list[ibi.Contract]:  # type: ignore
        spec = CONFIG.get("futures_fullchain_spec", "full")
        today = date.today()
        if spec == "full":
            return self._fullchain
        elif spec == "active":
            return [
                c
                for c in self._fullchain
                if datetime.fromisoformat(c.lastTradeDateOrContractMonth) > today
            ]
        elif spec == "expired":
            return [
                c
                for c in self._fullchain
                if datetime.fromisoformat(c.lastTradeDateOrContractMonth) <= today
            ]
        else:
            raise ValueError(
                f"futures_fullchain_spec must be one of: `full`, `active`, `expired`, "
                f"not {spec}"
            )


class CurrentFutureContractSelector(FutureContractSelector):
    async def _objects(self) -> list[ibi.Contract]:
        desired_index = CONFIG.get("futures_current_index", 0)
        if desired_index == 0:
            return [self._contfuture_qualified]
        else:
            return [self._fullchain[self._contfuture_index + int(desired_index)]]


class ExactFutureContractSelector(FutureContractSelector):
    pass


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

    public methods:

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
    now: datetime = field(default_factory=partial(datetime.now, timezone.utc))
    fill_gaps: bool = True
    next_date: Union[datetime, date, str] = ""  # Fucking hate this.TODO
    _queue: list[DownloadContainer] = field(default_factory=list, init=False)
    _current_object: Optional[DownloadContainer] = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.c = self.contract.localSymbol
        if asi := CONFIG.get("auto_save_interval", 0):
            pulse = ibi.Event().timerange(asi, None, asi)
            pulse += self.onPulse
        self.schedule_tasks()
        log.info(f"Object initialized: {self}")
        self.schedule_next()

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

    def schedule_next(self):
        if self._current_object:
            self.write_to_store()
        try:
            self._current_object = self._queue.pop()
        except IndexError:
            self.write_to_store()
            self.next_date = ""  # this should be None; TODO
            log.debug(f"{self.c} done!")
            return
        self.next_date = self._current_object.to_date
        log.debug(f"scheduling {self.c}: {self._current_object}")

    def save_chunk(self, data: ibi.BarDataList):
        assert self._current_object is not None
        # TODO
        # next data sometimes becomes None and subsequenty throws error in line 362
        next_date = self._current_object.save(data)
        log.debug(f"{self.c}: chunk saved, next_date: {next_date}")
        if next_date:
            self.next_date = next_date
        else:
            self.schedule_next()

    def write_to_store(self):
        try:
            _data = self._current_object.data
        except AttributeError:
            log.warning("Ignoring data...")
            _data = None

        if _data is not None:
            data = self.data
            if data is None:
                data = pd.DataFrame()
            data = pd.concat([data, _data])
            version = self.store.write(self.contract, data)
            log.debug(
                f"{self.c} written to store "
                f"{self._current_object.from_date} - {self._current_object.to_date}"
                f"version {version}"
            )
            if version:
                self._current_object.clear()

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
    def params(
        self,
    ) -> dict[
        str, Union[ibi.Contract, str, bool, date, datetime]
    ]:  # this is fucked. TODO
        return {
            "contract": self.contract,
            "endDateTime": self.next_date,
            "durationStr": self.duration,
            "barSizeSetting": self.barSize,
            "whatToShow": self.wts,
            "useRTH": False,
        }

    @property
    def duration(self):
        duration = barSize_to_duration(self.barSize, self.aggression)
        # this gets string and datetime error TODO !!!!!!!!!!!!!!!
        try:
            delta = self.next_date - self._current_object.from_date
        except Exception as e:
            log.error(
                f"next date: {self.next_date}, "
                f"from_date: {self._current_object.from_date}",
                e,
            )
            raise

        if delta < duration_to_timedelta(duration):
            # requests for periods shorter than 30s don't work
            duration = duration_str(
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

    @functools.cached_property
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


@dataclass
class DownloadContainer:
    """Hold downloaded data before it is saved to datastore"""

    from_date: datetime
    to_date: datetime
    current_date = None
    update: bool = False
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    bars: list[ibi.BarDataList] = field(default_factory=list)
    retries: int = 0
    nodata_retries: int = 0

    def save(self, bars: ibi.BarDataList) -> Optional[Union[datetime, date]]:
        """Store downloaded data and if more data needed return
        endpoint for next download"""

        if bars:
            log.debug(f"Received bars from: {bars[0].date} to {bars[-1].date}")
            self.bars.append(bars)
            self.current_date = bars[0].date
        elif self.current_date:
            log.warning(f"Cannot download data past {self.current_date}")
            # might be a bank holiday (TODO: this needs to be tested)
            # self.current_date -= timedelta(days=1)
            return None
        else:
            if self.ok_to_write:
                return None
            else:
                log.debug(f"Attempt {self.retries + 1} to fill in update gap")
                self.current_date = (
                    self.df.index.min() - timedelta(days=1) * self.retries
                )
                self.retries += 1
                if self.retries > 5:
                    self.retries = 0
                    return None
        # this is likely irrelevant. Check. TODO.
        if self.current_date:
            if self.from_date < self.current_date < self.to_date:
                return self.current_date
        return None

    @property
    def ok_to_write(self) -> bool:
        """Updated data should be written only if complete, otherwise
        difficult to find gaps would possibly occur in datastore."""

        if self.update:
            # TODO: this throws errors occationally
            try:
                return self.df.index.min() <= self.from_date
            except Exception as e:
                log.error(
                    f"ERROR index.min: {self.df.index.min()}, "
                    f"from_date: {self.from_date}",
                    e,
                )
                raise
        else:
            return True

    @property
    def data(self) -> Optional[Union[pd.DataFrame, datetime]]:
        """Return df ready to be written to datastore or date of end point
        for additional downloads"""
        if self.bars:
            self.df = ibi.util.df([b for bars in reversed(self.bars) for b in bars])
            self.df.set_index("date", inplace=True)
            if not self.ok_to_write:
                log.warning(
                    f"Writing update with gap between: "
                    f" {self.from_date} and {self.df.index.min()}"
                )

            df = self.df
            self.df = pd.DataFrame()
            return df
        else:
            return None

    def clear(self):
        self.bars = []

    def __repr__(self):
        return f"{self.from_date} - {self.to_date}, update: {self.update}"


def duration_in_secs(barSize: str):
    """Given duration string return duration in seconds int"""
    number, time = barSize.split(" ")
    time = time[:-1] if time.endswith("s") else time
    multiplier = {
        "sec": 1,
        "min": 60,
        "mins": 60,
        "hour": 3600,
        "day": 3600 * 23,
        "week": 3600 * 23 * 5,
    }
    return int(number) * multiplier[time]


def duration_str(duration_in_secs: int, aggression: float, from_bar: bool = True):
    """
    Given duration in seconds return acceptable duration str.

    :from_bar:
    if True it's assumed that the duration_in_secs number comes from barSize
    and appropriate multiplier is used to get to optimal duration. Otherwise
    duration_in_secs is converted into duration_str directly without
    any multiplication.
    """
    if from_bar:
        multiplier = 2000 if duration_in_secs < 30 else 15000 * aggression
    else:
        multiplier = 1
    duration = int(duration_in_secs * multiplier)
    days = int(duration / 60 / 60 / 23)
    if days:
        years = int(days / 250)
        if years:
            return f"{years} Y"
        months = int(days / 20)
        if months:
            return f"{months} M"
        return f"{days} D"
    return f"{duration} S"


def barSize_to_duration(s, aggression):
    """
    Given bar size str return optimal duration str,

    :aggression: how many data points will be pulled at a time,
                 should be between 0.5 and 3,
                 larger numbers might result in more throttling,
                 requires research what's optimal number for fastest
                 downloads
    """
    return duration_str(duration_in_secs(s), aggression)


def duration_to_timedelta(duration):
    """Convert duration string of reqHistoricalData into datetime.timedelta"""
    number, time = duration.split(" ")
    number = int(number)
    if time == "S":
        return timedelta(seconds=number)
    if time == "D":
        return timedelta(days=number)
    if time == "W":
        return timedelta(weeks=number)
    if time == "M":
        return timedelta(days=31)
    if time == "Y":
        return timedelta(days=365)
    raise ValueError(f"Unknown duration string: {duration}")


@dataclass
class Restriction:
    holder: ClassVar[deque[datetime]] = deque(maxlen=100)
    seconds: float
    requests: int

    def check(self) -> bool:
        """Return True if pacing restriction neccessary"""
        holder_ = deque(self.holder, maxlen=self.requests)
        if len(holder_) < self.requests:
            return False
        elif (datetime.now(timezone.utc) - holder_[0]) <= timedelta(
            seconds=self.seconds
        ):
            return True
        else:
            return False


@dataclass
class NoRestriction(Restriction):
    seconds: float = 0
    requests: int = 0

    def check(self) -> bool:
        return False


@dataclass
class Pacer:
    restrictions: list[Restriction] = field(
        default_factory=partial(list, NoRestriction())
    )

    async def __aenter__(self):
        while any([timer.check() for timer in self.timers]):
            await asyncio.sleep(0.1)
        # register request time right before exiting the context
        Restriction.holder.append(datetime.now(timezone.utc))

    async def __aexit__(self, *args):
        pass


def pacer(
    barSize,
    wts,
    *,
    restrictions: list[tuple[float, int]] = [],
    restriction_threshold: int = 30,  # barSize in secs above which restrictions apply
) -> Pacer:
    """
    Factory function returning correct pacer preventing (or rather
    limiting -:)) data pacing restrictions by Interactive Brokers.
    """

    if (not restrictions) or (duration_in_secs(barSize) > restriction_threshold):
        return Pacer()

    else:
        # 'BID_ASK' requests counted as double by ib
        if wts == "BID_ASK":
            restrictions = [
                (restriction[0], int(restriction[1] / 2))
                for restriction in restrictions
            ]
    return Pacer([Restriction(*res) for res in restrictions])


def validate_age(writer: DataWriter) -> bool:
    """
    IB doesn't permit to request data for bars < 30secs older than 6
    months.  Trying to push it here with 30secs.
    """
    if duration_in_secs(writer.barSize) < 30 and writer.next_date:
        assert isinstance(writer.next_date, datetime)
        # TODO: not sure if correct or necessary
        if (datetime.now(timezone.utc) - writer.next_date).days > 180:
            return False
    return True


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


async def worker(name: str, queue: asyncio.Queue, pacer: Pacer, ib: ibi.IB) -> None:
    while True:
        # TODO: questionable if this is necessary, as workers are cancelled eventually
        if queue.empty():
            break

        writer = await queue.get()
        log.debug(
            f"{name} loading {writer.contract.localSymbol} "
            f"ending {writer.next_date} "
            f'Duration: {writer.params["durationStr"]}, '
            f'Bar size: {writer.params["barSizeSetting"]} '
        )
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
                log.debug(f"{writer.contract.localSymbol} dropped on age validation")
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
