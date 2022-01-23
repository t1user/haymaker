from __future__ import annotations

import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
import itertools
import asyncio
from typing import List, Union, Optional, Any, Dict, NamedTuple
from functools import partial


import pandas as pd
from ib_insync import (IB, Contract, Future, ContFuture, BarDataList, util,
                       Event)
# from logbook import DEBUG, INFO

from logger import logger
from connect import Connection
from config import max_number_of_workers
from datastore import ArcticStore, AbstractBaseStore
from task_logger import create_task


"""
Async queue implementation modelled (loosely) on example here:
https://docs.python.org/3/library/asyncio-queue.html#examples
and here:
https://realpython.com/async-io-python/#using-a-queue
"""

log = logger(__file__[:-3])


class ContractObjectSelector:
    """
    Given a csv file with parameters return appropriate Contract objects.
    For futures return all available contracts or current ContFuture only.
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, ib: IB, file: str,
                 directory: Union[str, None] = None):
        if directory:
            self.BASE_DIR = directory
        self.symbols = pd.read_csv(
            os.path.join(self.BASE_DIR, file),
            keep_default_na=False
        ).to_dict('records')
        self.ib = ib
        self.contracts: List[Contract] = []
        log.debug('ContractObjectSelector about to create objects')
        self.create_objects()
        log.debug('Objects created')

    def create_objects(self) -> None:
        self.objects = [Contract.create(**s) for s in self.symbols]
        self.non_futures = [
            obj for obj in self.objects if not isinstance(obj, Future)]
        log.debug(f'non-futures: {self.non_futures}')
        self.futures = [obj for obj in self.objects if isinstance(obj, Future)]
        log.debug(f'futures: {self.futures}')

        # converting Futures to ContFutures
        self.contFutures = []
        for obj in self.futures:
            params = obj.nonDefaults()
            del params['secType']
            self.contFutures.append(ContFuture(**params))
        log.debug(f'contfutures: {self.contFutures}')

    def lookup_futures(self, obj: List[Future]) -> List[Future]:
        futures = []
        for o in obj:
            o.update(includeExpired=True)
            futures.append(
                [Contract.create(**c.contract.dict())
                 for c in self.ib.reqContractDetails(o)]
            )
        return list(itertools.chain(*futures))

    @property
    def list(self) -> List[Contract]:
        if not self.contracts:
            self.update()
        return self.contracts

    def update(self) -> List[Contract]:
        qualified = self.contFutures + self.non_futures
        self.ib.qualifyContracts(*qualified)
        self.contracts = self.lookup_futures(self.futures) + qualified
        return self.contracts

    @property
    def cont_list(self) -> List[ContFuture]:
        self.ib.qualifyContracts(*self.contFutures)
        return self.contFutures


class DataWriter:
    """Interface between dataloader and datastore"""

    def __init__(self, store: AbstractBaseStore, contract: Contract,
                 head: datetime,
                 barSize: str, wts: str, aggression: float = 2,
                 now: datetime = datetime.now()) -> None:
        self.store = store
        self.contract = contract
        self.head = head
        self.barSize = bar_size_validator(barSize)
        self.wts = wts_validator(wts)
        self.now = now
        self.aggression = aggression

        self.c = self.contract.localSymbol
        # start, stop, step in seconds, ie. every 15min
        pulse = Event().timerange(900, None, 900)
        pulse += self.onPulse

        self.next_date = ''
        self._objects: List[DownloadContainer] = []
        self._queue: List[DownloadContainer] = []
        self._current_object: Optional[DownloadContainer] = None
        self.schedule_tasks()
        log.info(f'Object initialized: {self}')

    def onPulse(self, time: datetime):
        self.write_to_store()

    def schedule_tasks(self):
        update = self.update()
        backfill = self.backfill()
        fill_gaps = self.fill_gaps()

        log.debug(f'update for {self.c}: {update}')
        log.debug(f'backfill for {self.c}: {backfill}')
        log.debug(f'fill_gaps for {self.c}: {fill_gaps}')

        if backfill:
            log.debug(f'{self.c} queued for backfill')
            self._objects.append(
                DownloadContainer(from_date=self.head, to_date=backfill))

        if update:
            log.debug(f'{self.c} queued for update')
            self._objects.append(
                DownloadContainer(from_date=self.to_date, to_date=update,
                                  update=True))

        if fill_gaps is not None:
            for gap in fill_gaps:
                log.debug(
                    f'{self.c} queued gap from {gap.start} to {gap.stop}')
                self._objects.append(DownloadContainer(from_date=gap.start,
                                                       to_date=gap.stop))

        self._queue = self._objects.copy()
        self.schedule_next()

    def schedule_next(self):
        if self._current_object:
            self.write_to_store()
        try:
            self._current_object = self._queue.pop()
        except IndexError:
            self.write_to_store()
            self.next_date = None
            log.debug(f'{self.c} done!')
            return
        self.next_date = self._current_object.to_date
        log.debug(f'scheduling {self.c}: {self._current_object}')

    def save_chunk(self, data: BarDataList):
        next_date = self._current_object.save(data)
        log.debug(f'{self.c}: chunk saved, next_date: {next_date}')
        if next_date:
            self.next_date = next_date
        else:
            self.schedule_next()

    def write_to_store(self):
        _data = self._current_object.data
        if _data is not None:
            data = self.data
            if data is None:
                data = pd.DataFrame()
            data = data.append(_data)
            version = self.store.write(self.contract, data)
            log.debug(f'data written to datastore as {version}')
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

    def fill_gaps(self) -> List[NamedTuple]:
        data = self.data.copy()
        data['timestamp'] = data.index
        data['gap'] = data['timestamp'].diff()
        inferred_frequency = data['gap'].mode()[0]
        log.debug(f'inferred frequency: {inferred_frequency}')
        data['gap_bool'] = data['gap'] > inferred_frequency
        data['start'] = data.timestamp.shift()
        data['stop'] = data.timestamp.shift(-1)
        gaps = data[data['gap_bool']]
        out = pd.DataFrame({'start': gaps['start'], 'stop': gaps['stop']}
                           ).reset_index(drop=True)
        out = out[1:]
        out['start_time'] = out['start'].apply(lambda x: x.time())
        cutoff_time = out['start_time'].mode()[0]
        log.debug(f'inferred cutoff time: {cutoff_time}')
        non_standard_gaps = out[out['start_time'] != cutoff_time].reset_index(
            drop=True)
        return list(non_standard_gaps[['start', 'stop']].itertuples(
            index=False))

    @property
    def params(self) -> Dict[str: Union[Contract, str, bool]]:
        return {
            'contract': self.contract,
            'endDateTime': self.next_date,
            'durationStr': self.duration,
            'barSizeSetting': self.barSize,
            'whatToShow': self.wts,
            'useRTH': False,
        }

    @property
    def duration(self):
        duration = barSize_to_duration(self.barSize, self.aggression)
        delta = self.next_date - self._current_object.from_date
        if delta < duration_to_timedelta(duration):
            # requests for periods shorter than 30s don't work
            duration = duration_str(
                max(delta.total_seconds(), 30), self.aggression, False)
        return duration

    @property
    def expiry(self) -> Union[datetime, str]:
        """Expiry date for expirable contracts or '' """
        e = self.contract.lastTradeDateOrContractMonth
        return e if not e else datetime.strptime(e, '%Y%m%d')

    @property
    def data(self) -> Optional[pd.DataFrame]:
        """Available data in datastore for contract or None"""
        return self.store.read(self.contract)

    @property
    def from_date(self) -> Optional[pd.datetime]:
        """Earliest point in datastore"""
        # second point in the df to avoid 1 point gap
        return None or self.data.index[1]

    @property
    def to_date(self) -> Optional[pd.datetime]:
        """Latest point in datastore"""
        return None or self.data.index.max()

    def __repr__(self):
        return (f'{self.__class__.__name__}' + '(' + ', '.join(
            [f'{k}={v}' for k, v in self.__dict__.items()]) + ')')


@dataclass
class DownloadContainer:
    """Hold downloaded data before it is saved to datastore"""
    from_date: pd.datetime
    to_date: pd.datetime
    current_date: Optional[pd.datetime] = None
    update: bool = False
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    bars: List[BarDataList] = field(default_factory=list)
    retries: int = 0
    nodata_retries: int = 0

    def save(self, bars: BarDataList) -> Optional[datetime]:
        """Store downloaded data and if more data needed return
        endpoint for next download"""

        if bars:
            log.debug(f'Received bars from: {bars[0].date} to {bars[-1].date}')
            self.bars.append(bars)
            self.current_date = bars[0].date
        elif self.current_date:
            log.error(f'Cannot download data past {self.current_date}')
            # might be a bank holiday (TODO: this needs to be tested)
            # self.current_date -= timedelta(days=1)
            return
        else:
            if self.ok_to_write:
                return
            else:
                log.debug(f'Attempt {self.retires + 1} to fill in update gap')
                self.current_date = (
                    self.df.index.min() - timedelta(days=1) * self.retries)
                self.retries += 1
                if self.retries > 5:
                    self.retries = 0
                    return
        if self.from_date < self.current_date < self.to_date:
            return self.current_date

    @property
    def ok_to_write(self) -> bool:
        """Updated data should be written only if complete, otherwise
        difficult to find gaps would possibly occur in datastore."""

        if self.update:
            return self.df.index.min() <= self.from_date
        else:
            return True

    @property
    def data(self) -> Union[pd.DataFrame, pd.datetime]:
        """Return df ready to be written to datastore or date of end point
        for additional downloads"""
        if self.bars:
            self.df = util.df([b for bars in reversed(self.bars)
                               for b in bars])
            self.df.set_index('date', inplace=True)

            if not self.ok_to_write:
                log.warning(f'Writing update with gap between: '
                            f' {self.from_date} and {self.df.index.min()}')

            df = self.df
            self.df = pd.DataFrame()
            return df

    def clear(self):
        self.bars = []

    def __repr__(self):
        return f'{self.from_date} - {self.to_date}, update: {self.update}'


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
            log.debug('getting items')
            objects = ContractObjectSelector(self.ib, self.source)
            if self.cont_only:
                objects = objects.cont_list
            else:
                objects = objects.list
            log.debug(f'objects: {objects}')

            self.items = []
            for o in objects:
                headTimeStamp = self.ib.reqHeadTimeStamp(
                    o, whatToShow=self.wts, useRTH=False)
                if headTimeStamp == []:
                    log.warning(
                        (f'Unavailable headTimeStamp for {o.localSymbol}. '
                         f'No data will be downloaded')
                    )
                    continue
                self.items.append(
                    DataWriter(store,
                               o,
                               headTimeStamp,
                               barSize=self.barSize,
                               wts=self.wts,
                               aggression=self.aggression)
                )

        def __call__(self):
            log.debug('holder called')
            log.debug(f'items: {self.items}')
            if self.items is None:
                self.get_items()
            return self.items

    __instance = None

    def __new__(cls, *args, **kwargs):
        if not ContractHolder.__instance:
            ContractHolder.__instance = ContractHolder.__ContractHolder(
                *args, **kwargs)
        return ContractHolder.__instance

    def __getattr__(self, name: str) -> Any:
        return getattr(self.instance, name)

    def __setattr__(self, name: str, value: Any) -> None:
        return setattr(self.instance, name, value)

    def __call__(self):
        return self.instance()


def bar_size_validator(s):
    """Verify if given string is a valid IB api bar size str"""
    ok_str = ['1 secs', '5 secs', '10 secs', '15 secs', '30 secs',
              '1 min', '2 mins', '3 mins', '5 mins', '10 mins', '15 mins',
              '20 mins', '30 mins',
              '1 hour', '2 hours', '3 hours', '4 hours', '8 hours',
              '1 day', '1 week', '1 month']
    if s in ok_str:
        return s
    else:
        raise ValueError(f'bar size : {s} is invalid, must be one of {ok_str}')


def wts_validator(s: str):
    """Verify if given string is a valide IB api whatToShow str"""
    ok_str = ['TRADES', 'MIDPOINT', 'BID', 'ASK', 'BID_ASK', 'ADJUSTED_LAST',
              'HISTORICAL_VOLATILITY', 'OPTION_IMPLIED_VOLATILITY',
              'REBATE_RATE', 'FEE_RATE', 'YIELD_BID', 'YIELD_ASK',
              'YIELD_BID_ASK', 'YIELD_LAST']
    if s in ok_str:
        return s
    else:
        raise ValueError(
            f'{s} is a wrong whatToShow value, must be one of {ok_str}')


def duration_in_secs(barSize: str):
    """Given duration string return duration in seconds int"""
    number, time = barSize.split(' ')
    time = time[:-1] if time.endswith('s') else time
    multiplier = {'sec': 1, 'min': 60, 'mins': 60, 'hour': 3600,
                  'day': 3600*23, 'week': 3600*23*5}
    return int(number) * multiplier[time]


def duration_str(duration_in_secs: int, aggression: float,
                 from_bar: bool = True):
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
            return f'{years} Y'
        months = int(days / 20)
        if months:
            return f'{months} M'
        return f'{days} D'
    return f'{duration} S'


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
    number, time = duration.split(' ')
    number = int(number)
    if time == 'S':
        return timedelta(seconds=number)
    if time == 'D':
        return timedelta(days=number)
    if time == 'W':
        return timedelta(weeks=number)
    if time == 'M':
        return timedelta(days=31)
    if time == 'Y':
        return timedelta(days=365)
    raise ValueError(f'Unknown duration string: {duration}')


class SecondPacer:
    """    NOT IN USE   """

    def __init__(self, seconds: int = 2, request_limit: int = 6):
        self.seconds = seconds
        self.store = deque(maxlen=request_limit)

    def check(self):
        self.store.append(datetime.now())


class Pacer:
    """
    Keep track of requests made and check against IB restricions.

    check - returns True if requests have to stop to avoid data
    request pacing violation.
    """

    def __init__(self, barSize: str, wts: str, minutes=10) -> None:
        self.max_request_number = 60 if wts != 'BID_ASK' else 30
        self.time_limit = minutes * 60
        self.restriction = self.check_restriction(barSize)
        self.restriction_until = datetime.now()
        self.reset()

    def check_restriction(self, barSize: 'str') -> bool:
        return True if duration_in_secs(barSize) <= 60 else False

    def check(self) -> bool:
        self.request_number = next(self.counter)

        if not self.restriction:
            return False

        if datetime.now() < self.restriction_until:
            log.debug(
                f'Enforcing restriction in place until {self.restriction_until}')
            return True

        if self.timer == 0:
            self.start_timer()

        if (self.request_number >= self.max_request_number) and (
                (datetime.now() - self.timer).seconds <= self.time_limit):
            self.restrict()
            log.debug('Establishing restriction')
            return True
        else:
            return False

    def restrict(self):
        self.restriction_until = datetime.now() + timedelta(
            seconds=self.time_limit)

    def start_timer(self):
        self.timer = datetime.now()

    def reset(self) -> None:
        self.request_number = 0
        self.counter = itertools.count(1, 1)
        self.timer = 0


async def worker(name: str, queue: asyncio.Queue, pacer):
    while True:
        contract = await queue.get()
        log.debug(
            f'{name} loading {contract.contract.localSymbol} '
            f'ending {contract.next_date} '
            f'Duration: {contract.params["durationStr"]}, '
            f'Bar size: {contract.params["barSizeSetting"]} '
        )
        if pacer.check():
            log.debug(f'SLEEPING for {pacer.time_limit} seconds '
                      f'to avoid pacing violation.')
            util.sleep(pacer.time_limit)
            pacer.reset()
        else:
            log.debug(
                f'{datetime.now()} request number {pacer.request_number}')
            chunk = await ib.reqHistoricalDataAsync(**contract.params, timeout=0)
            contract.save_chunk(chunk)
            if contract.next_date:
                await queue.put(contract)
            queue.task_done()


async def main(holder: ContractHolder):

    contracts = holder()
    number_of_workers = min(len(contracts), max_number_of_workers)

    log.debug(f'main function started, '
              f'retrieving data for {len(contracts)} instruments')

    queue: asyncio.Queue[DataWriter] = asyncio.LifoQueue()
    for contract in contracts:
        await queue.put(contract)
    pacer = Pacer(holder.barSize, holder.wts)
    workers = [create_task(worker(f'worker {i}', queue, pacer),
                           logger=log, message='asyncio error',
                           message_args=f'worker {i}')
               for i in range(number_of_workers)]

    """
    workers = [asyncio.create_task(worker(f'worker {i}', queue, pacer))
               for i in range(number_of_workers)]

    """
    await queue.join()

    # cancel all workers
    log.debug('cancelling workers')
    for w in workers:
        w.cancel()

    # wait until all worker tasks are cancelled
    await asyncio.gather(*workers)


class ErrorState:
    """Not in use"""
    _state: bool = False

    def set(self, *args) -> None:
        self._state = True

    def clear(self) -> None:
        self._state = False

    def __call__(self) -> bool:
        state = self._state
        self.clear
        return state

    state = __call__


if __name__ == '__main__':
    util.patchAsyncio()
    ib = IB()
    barSize = '1 secs'
    wts = 'MIDPOINT'
    # object where data is stored
    store = ArcticStore(f'{wts}_{barSize}')

    # the bool is for cont_only
    holder = ContractHolder(ib, '_contracts.csv',
                            store, wts, barSize, True, aggression=1)

    asyncio.get_event_loop().set_debug(True)
    # util.logToConsole(INFO)
    Connection(ib, partial(main, holder), watchdog=True)

    log.debug('script finished, about to disconnect')
    ib.disconnect()
    log.debug('disconnected')
