import asyncio
import itertools
import logging
import os
import random
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from functools import partial
from typing import Any, Deque, Dict, List, NamedTuple, Optional, Tuple, Union

import pandas as pd
import pytz
from ib_insync import IB, BarDataList, ContFuture, Contract, Event, Future, util
from typing_extensions import Protocol

from ib_tools.config import CONFIG
from ib_tools.connect import Connection
from ib_tools.datastore import AbstractBaseStore
from ib_tools.logging import setup_logging
from ib_tools.task_logger import create_task

# from logbook import DEBUG, INFO


"""
Async queue implementation modelled (loosely) on example here:
https://docs.python.org/3/library/asyncio-queue.html#examples
and here:
https://realpython.com/async-io-python/#using-a-queue
"""
setup_logging()

log = logging.getLogger(__name__)

# TODO: change this to factory
MAX_NUMBER_OF_WORKERS = CONFIG.get("max_number_of_workers", 40)


class ContractObjectSelector:
    """
    Given a csv file with parameters return appropriate Contract objects.
    For futures return all available contracts or current ContFuture only.
    """

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, ib: IB, file: str, directory: Union[str, None] = None) -> None:
        self.symbols = pd.read_csv(file, keep_default_na=False).to_dict("records")
        self.ib = ib
        self.contracts: List[Contract] = []
        log.debug("ContractObjectSelector about to create objects")
        self.create_objects()
        log.debug("Objects created")

    def create_objects(self) -> None:
        self.objects = [Contract.create(**s) for s in self.symbols]  # type: ignore
        self.non_futures = [obj for obj in self.objects if not isinstance(obj, Future)]
        log.debug(f"non-futures: {self.non_futures}")
        self.futures = [obj for obj in self.objects if isinstance(obj, Future)]
        log.debug(f"futures: {self.futures}")

        # converting Futures to ContFutures
        self.contFutures = []
        for obj in self.futures:
            params = obj.nonDefaults()  # type: ignore
            del params["secType"]
            self.contFutures.append(ContFuture(**params))
        log.debug(f"contfutures: {self.contFutures}")

    def lookup_futures(self, obj: List[Future]) -> List[Future]:
        futures = []
        for o in obj:
            o.update(includeExpired=True)  # type: ignore
            futures.append(
                [
                    Contract.create(**c.contract.dict())  # type: ignore
                    for c in self.ib.reqContractDetails(o)
                ]
            )
        return list(itertools.chain(*futures))  # type: ignore

    @property
    def list(self) -> List[Contract]:
        if not self.contracts:
            self.update()
        return self.contracts

    def update(self) -> List[Contract]:
        qualified = self.contFutures + self.non_futures  # type: ignore
        self.ib.qualifyContracts(*qualified)
        self.contracts = self.lookup_futures(self.futures) + qualified  # type: ignore
        return self.contracts

    @property
    def cont_list(self) -> List[ContFuture]:
        self.ib.qualifyContracts(*self.contFutures)
        return self.contFutures


@dataclass
class DataWriter:
    """Interface between dataloader and datastore"""

    store: AbstractBaseStore
    contract: Contract
    head: datetime
    barSize: str
    wts: str
    aggression: float = 2
    # !!!!! it was pytz.timezone("Europe/Berlin") here, INVESTIGATE!!
    now: datetime = field(default_factory=partial(datetime.now, pytz.utc))
    fill_gaps: bool = CONFIG.get("fill_gaps", True)

    def __post_init__(self) -> None:
        self.barSize = bar_size_validator(self.barSize)
        self.wts = wts_validator(self.wts)
        # !!!! REVIEW everything below
        self.c = self.contract.localSymbol
        # start, stop, step in seconds, ie. every 15min
        pulse = Event().timerange(900, None, 900)
        pulse += self.onPulse

        self.next_date: Union[datetime, date, str] = ""  # Fucking hate this.TODO
        self._objects: List[DownloadContainer] = []
        self._queue: List[DownloadContainer] = []
        self._current_object: Optional[DownloadContainer] = None
        self.schedule_tasks()
        log.info(f"Object initialized: {self}")

    def onPulse(self, time: datetime):
        self.write_to_store()

    def schedule_tasks(self):
        try:
            update = self.update()
            log.debug(f"update for {self.c}: {update}")
        except Exception as e1:
            log.error(f"Exception for update, line 139: {e1} ignored, {self.c}")
            update = None

        try:
            backfill = self.backfill()
            log.debug(f"backfill for {self.c}: {backfill}")
        except Exception as e2:
            log.error(f"Exception for backfill, line 144: {e2} ignored, {self.c}")
            backfill = None

        if self.fill_gaps:
            try:
                fill_gaps = self.gap_filler()
                log.debug(f"fill_gaps for {self.c}: {fill_gaps}")
            except Exception as e3:
                log.error(f"Exception for fill_gaps, line 149: {e3} ignored, {self.c}")
                fill_gaps = None
        else:
            fill_gaps = None

        if backfill:
            log.debug(f"{self.c} queued for backfill")
            self._objects.append(
                DownloadContainer(from_date=self.head, to_date=backfill)
            )

        if update:
            log.debug(f"{self.c} queued for update")
            self._objects.append(
                DownloadContainer(from_date=self.to_date, to_date=update, update=True)
            )

        if fill_gaps is not None:
            for gap in fill_gaps:
                log.debug(f"{self.c} queued gap from {gap.start} to {gap.stop}")
                self._objects.append(
                    DownloadContainer(from_date=gap.start, to_date=gap.stop)
                )

        self._queue = self._objects.copy()
        self.schedule_next()

    def schedule_next(self):
        if self._current_object:
            self.write_to_store()
            log.debug(
                f"{self.c} written to store "
                f"{self._current_object.from_date} - "
                f"{self._current_object.to_date}"
            )
        try:
            self._current_object = self._queue.pop()
        except IndexError:
            self.write_to_store()
            self.next_date = ""  # this should be None; TODO
            log.debug(f"{self.c} done!")
            return
        self.next_date = self._current_object.to_date
        log.debug(f"scheduling {self.c}: {self._current_object}")

    def save_chunk(self, data: BarDataList):
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
            log.warning("Ignoring data, line 210")
            _data = None

        if _data is not None:
            data = self.data
            if data is None:
                data = pd.DataFrame()
            # TODO: replace append with concat
            data = pd.concat([data, _data])
            # data = data.append(_data)
            version = self.store.write(self.contract, data)
            log.debug(f"data written to datastore as {version}")
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

    def gap_filler(self) -> List[NamedTuple]:
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
    ) -> Dict[str, Union[Contract, str, bool, date, datetime]]:  # this is fucked. TODO
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
        # if data.tzinfo is None or data.tzinfo.utcoffset(data) is None:
        return date

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            + "("
            + ", ".join([f"{k}={v}" for k, v in self.__dict__.items()])
            + ")"
        )


@dataclass
class DownloadContainer:
    """Hold downloaded data before it is saved to datastore"""

    from_date: datetime
    to_date: datetime
    current_date = None
    update: bool = False
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    bars: List[BarDataList] = field(default_factory=list)
    retries: int = 0
    nodata_retries: int = 0

    def save(self, bars: BarDataList) -> Optional[Union[datetime, date]]:
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
            self.df = util.df([b for bars in reversed(self.bars) for b in bars])
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


class ContractHolder:
    """Singleton class ensuring contract list kept after re-connect"""

    @dataclass
    class __ContractHolder:
        ib: IB
        source: str  # csv file name with contract list
        store: AbstractBaseStore
        wts: str  # whatToShow ib api parameter
        barSize: str  # ib api parameter
        cont_only: bool = False  # retrieve continuous contracts only
        # how big series request at each call (1 = normal, 2 = double, etc.)
        aggression: int = 1
        items: Optional[List[DataWriter]] = None

        def get_items(self):
            log.debug("getting items")
            objects = ContractObjectSelector(self.ib, self.source)
            log.debug("ContractObjectSelector ok")
            if self.cont_only:
                objects = objects.cont_list
            else:
                objects = objects.list
            log.debug(f"objects: {objects}")

            self.items = []
            for o in objects:
                try:
                    headTimeStamp = self.ib.reqHeadTimeStamp(
                        o, whatToShow=self.wts, useRTH=False, formatDate=2
                    )

                    if headTimeStamp == []:
                        log.warning(
                            (
                                f"Unavailable headTimeStamp for {o.localSymbol}. "
                                f"No data will be downloaded"
                            )
                        )
                        continue
                except Exception:
                    log.exception("Exception while getting headTimeStamp")
                    continue

                try:
                    self.items.append(
                        DataWriter(
                            self.store,
                            o,
                            headTimeStamp,
                            barSize=self.barSize,
                            wts=self.wts,
                            aggression=self.aggression,
                        )
                    )
                except Exception as e:
                    log.exception(f"Error ignored for object {o}", e)
                    # raise

        def __call__(self):
            log.debug("holder called")
            log.debug(f"items: {self.items}")
            if self.items is None:
                self.get_items()
                log.debug(f"items obtained: {self.items}")
            return self.items

    __instance = None

    def __new__(cls, *args, **kwargs):
        if not ContractHolder.__instance:
            ContractHolder.__instance = ContractHolder.__ContractHolder(*args, **kwargs)
        return ContractHolder.__instance

    def __getattr__(self, name: str) -> Any:
        return getattr(self.instance, name)

    def __setattr__(self, name: str, value: Any) -> None:
        return setattr(self.instance, name, value)

    def __call__(self):
        return self.instance()


def bar_size_validator(s):
    """Verify if given string is a valid IB api bar size str"""
    ok_str = [
        "1 secs",
        "5 secs",
        "10 secs",
        "15 secs",
        "30 secs",
        "1 min",
        "2 mins",
        "3 mins",
        "5 mins",
        "10 mins",
        "15 mins",
        "20 mins",
        "30 mins",
        "1 hour",
        "2 hours",
        "3 hours",
        "4 hours",
        "8 hours",
        "1 day",
        "1 week",
        "1 month",
    ]
    if s in ok_str:
        return s
    else:
        raise ValueError(f"bar size : {s} is invalid, must be one of {ok_str}")


def wts_validator(s: str):
    """Verify if given string is a valide IB api whatToShow str"""
    ok_str = [
        "TRADES",
        "MIDPOINT",
        "BID",
        "ASK",
        "BID_ASK",
        "ADJUSTED_LAST",
        "HISTORICAL_VOLATILITY",
        "OPTION_IMPLIED_VOLATILITY",
        "REBATE_RATE",
        "FEE_RATE",
        "YIELD_BID",
        "YIELD_ASK",
        "YIELD_BID_ASK",
        "YIELD_LAST",
    ]
    if s in ok_str:
        return s
    else:
        raise ValueError(f"{s} is a wrong whatToShow value, must be one of {ok_str}")


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


class TimerProtocol(Protocol):
    def check(self) -> bool: ...

    def register(self) -> None: ...


class NoTimer:
    def check(self) -> bool:
        return False

    def register(self) -> None:
        pass

    def __repr__(self):
        return "<NoTimer>"


class Timer:
    def __init__(self, seconds: int, requests: int) -> None:
        self.holder: Deque[datetime] = deque(maxlen=requests)
        self.seconds = seconds
        self._seconds = timedelta(seconds=seconds)
        self.requests = requests
        self.restriction_until = datetime.now()
        self.restriction = False

    def start(self) -> datetime:
        return self.holder[0]

    def register(self) -> None:
        """
        Register data request made for future tracking of number of
        requests made in the unit of time.  To be called by
        Pacer.__aexit__ method after every request made to ib.
        """
        self.holder.append(datetime.now())
        # log.debug(
        #    f'{self} registered request {len(self.holder)} at: {self.holder[-1]} '
        #    f'(elapsed: {self.elapsed_time()}sec)')

    def elapsed_time(self) -> timedelta:
        return datetime.now() - self.start()

    def _sleep_time(self) -> timedelta:
        """Time till lock release as timedelta"""
        return timedelta(seconds=self.seconds) - self.elapsed_time()
        # return timedelta(seconds=self.seconds)

    def sleep_time(self) -> float:
        """Time till lock release in fractional seconds"""
        return self.td_sec(self._sleep_time())
        # return self.seconds

    @staticmethod
    def td_sec(td: timedelta) -> float:
        """Convert timedelta to fractional seconds."""
        return td.seconds + td.microseconds / 1e6

    def check(self) -> bool:
        """Return True if pacing restriction neccessary"""
        if len(self.holder) < self.requests:
            return False
        elif (datetime.now() - self.start()) <= self._seconds:
            return True
        else:
            # log.debug(f'Checked against: {self.start()} elapsed: '
            #          f'{self.elapsed_time().seconds}sec returning False')
            return False

    def __repr__(self):
        return f"<Timer: {self.requests}req per {self.seconds}sec>"


class Pacer:
    def __init__(self, timers: List[TimerProtocol]) -> None:
        self.timers = timers

    async def __aenter__(self):
        for timer in self.timers:
            while _ := timer.check():
                log.debug(
                    f"restriction by timer: {timer} "
                    f"for {timer.sleep_time()}sec "
                    f"until {datetime.now() + timer._sleep_time()}"
                )
                # additional random up to 2 secs sleep to avoid all workers
                # exiting from sleep at exactly the same time
                await asyncio.sleep(timer.sleep_time() + 2 * random.random())
                # await asyncio.sleep(1)
        # register request time right before exiting the context
        for timer in self.timers:
            timer.register()

    async def __aexit__(self, exc_type, exc_value, exc_tb):
        pass

    def __repr__(self):
        return f"Pacer with timers: {self.timers}"


def pacer(
    barSize,
    wts,
    *,
    restrictions: List[Tuple[int, int]] = [(2, 6), (600, 60)],
    restriction_threshold: int = 30,
    norestriction: bool = False,
    timers: Optional[Union[List[TimerProtocol], TimerProtocol]] = None,
) -> Pacer:
    """
    Factory function returning correct pacer preventing (or rather
    limiting -:)) data pacing restrictions by Interactive Brokers.
    """

    log.debug(
        f"INSIDE PACER"
        f"duration in secs: {duration_in_secs(barSize)}"
        f"restriction threshold: {restriction_threshold}"
    )

    # 'BID_ASK' requests counted as double by ib
    if wts == "BID_ASK":
        restrictions = [
            (restriction[0], int(restriction[1] / 2)) for restriction in restrictions
        ]

    if timers:
        if not isinstance(timers, list):
            timers = [timers]
        for timer in timers:
            assert isinstance(timer, Timer), "timers must be a Timer or list of Timer"

    elif norestriction or (duration_in_secs(barSize) > restriction_threshold):
        timers = [NoTimer()]
    else:
        timers = [Timer(*res) for res in restrictions]

    return Pacer(timers)


def validate_age(contract: DataWriter) -> bool:
    """
    IB doesn't permit to request data for bars < 30secs older than 6
    monts.  Trying to push it here with 30secs.
    """
    if duration_in_secs(contract.barSize) < 30 and contract.next_date:
        assert isinstance(contract.next_date, datetime)
        if (datetime.now() - contract.next_date).days > 180:
            return False
        # log.debug(
        #    f'validating: {contract.contract.localSymbol} {contract.next_date}'
        #    f' duration: {duration_in_secs(contract.barSize)}sec '
        #    f'age: {(datetime.now() - contract.next_date).days} '
        #    f'validation: {(datetime.now() - contract.next_date).days > 180}')

    return True


async def worker(name: str, queue: asyncio.Queue, pacer: Pacer, ib: IB) -> None:
    while True:
        contract = await queue.get()
        log.debug(
            f"{name} loading {contract.contract.localSymbol} "
            f"ending {contract.next_date} "
            f'Duration: {contract.params["durationStr"]}, '
            f'Bar size: {contract.params["barSizeSetting"]} '
        )
        async with pacer:
            chunk = await ib.reqHistoricalDataAsync(
                **contract.params, formatDate=2, timeout=0
            )

        contract.save_chunk(chunk)
        if contract.next_date:
            if validate_age(contract):
                await queue.put(contract)
            else:
                contract.save_chunk(None)
                log.debug(
                    f"{contract.contract.localSymbol} " f"dropped on age validation"
                )
        queue.task_done()


async def main(holder: ContractHolder, ib: IB) -> None:

    contracts = holder()
    log.debug(f"Holder: {contracts}")
    number_of_workers = min(len(contracts), MAX_NUMBER_OF_WORKERS)

    log.debug(
        f"main function started, " f"retrieving data for {len(contracts)} instruments"
    )

    queue: asyncio.Queue[DataWriter] = asyncio.LifoQueue()
    for contract in contracts:
        await queue.put(contract)
    p = pacer(
        holder.barSize,
        holder.wts,
        restrictions=[(2, 6), (1200, 60 - number_of_workers)],
    )
    log.debug(f"Pacer initialized: {p}")
    workers: List[asyncio.Task] = [
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

    file = CONFIG.get("source")
    barSize = CONFIG.get("barSize")
    wts = CONFIG.get("wts")
    aggression = CONFIG.get("aggression", 2)
    continuous_futures_only = CONFIG.get("continuous_futures_only", True)
    watchdog = CONFIG.get("watchdog", True)

    util.patchAsyncio()
    ib = IB()

    # object where data is stored
    # store = ArcticStore(f"{wts}_{barSize}")

    store = CONFIG.get("datastore")

    # the bool is for cont_only
    holder = ContractHolder(
        ib,
        file,
        store,
        wts,
        barSize,
        continuous_futures_only,
        aggression=aggression,
    )

    asyncio.get_event_loop().set_debug(True)
    # util.logToConsole(logging.ERROR)
    log.debug("Will start...")

    Connection(ib, partial(main, holder, ib), watchdog=watchdog)

    log.debug("script finished, about to disconnect")
    ib.disconnect()
    log.debug("disconnected")
