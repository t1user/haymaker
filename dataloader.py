from __future__ import annotations

import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import itertools
import asyncio
from typing import List, Union, Type, Optional
from functools import partial

import pandas as pd
from ib_insync import IB, Contract, Future, ContFuture, BarDataList, util
from eventkit import Event

from logger import logger
from connect import Connection
from config import max_number_of_workers
from datastore import ArcticStore, AbstractBaseStore


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
        self.contracts = []
        self.create_objects()

    def create_objects(self) -> None:
        self.objects = [Contract.create(**s) for s in self.symbols]
        self.non_futures = [
            obj for obj in self.objects if not isinstance(obj, Future)]
        self.futures = [obj for obj in self.objects if isinstance(obj, Future)]
        self.contFutures = [ContFuture(**obj.nonDefaults()
                                       ).update(secType='CONTFUT')
                            for obj in self.futures]

    def lookup_futures(self, obj: List[Future]) -> List[Future]:
        futures = []
        for o in obj:
            o.update(includeExpired=True)
            futures.append(
                [Future(**c.contract.dict())
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
        self._objects = []
        self._queue = []
        self._current_object = None
        self.schedule_tasks()

    def onPulse(self, time: datetime):
        self.write_to_store()

    def schedule_tasks(self):
        update = self.update()
        backfill = self.backfill()

        if backfill:
            log.debug(f'{self.c} queued for backfill')
            self._objects.append(
                DownloadContainer(from_date=self.head, to_date=backfill))

        if update:
            log.debug(f'{self.c} queued for update')
            self._objects.append(
                DownloadContainer(from_date=self.to_date, to_date=update,
                                  update=True))

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

    @property
    def params(self):
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
            duration = duration_str(
                delta.total_seconds(), self.aggression, False)
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
        return self.data.index.min() if self.data is not None else None

    @property
    def to_date(self) -> Optional[pd.datetime]:
        """Latest point in datastore"""
        return self.data.index.max() if self.data is not None else None

    def __repr__(self):
        return (f'DataWriter for {self.contract.localSymbol} ')


@dataclass
class DownloadContainer:
    """Hold downloaded data before it is saved to datastore"""
    from_date: pd.datetime
    to_date: pd.datetime
    current_date: Optional[pd.datetime] = None
    update: bool = False
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    bars: List[BarDataList] = field(default_factory=list)

    def save(self, bars: BarDataList) -> Optional[datetime]:
        """Store downloaded data and if more data needed return
        endpoint for next download"""

        if bars:
            self.bars.append(bars)
            self.current_date = bars[0].date
        elif self.current_date:
            # might be a bank holiday
            self.current_date -= timedelta(days=1)
        else:
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
            if self.ok_to_write:
                df = self.df
                self.df = pd.DataFrame()
                return df
            log.debug(f'cannot write data')

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
            objects = ContractObjectSelector(self.ib, self.source)
            if self.cont_only:
                objects = objects.cont_list
            else:
                objects = objects.list

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
            if self.items is None:
                self.get_items()
            return self.items

    __instance = None

    def __new__(cls, *args, **kwargs):
        if not ContractHolder.__instance:
            ContractHolder.__instance = ContractHolder.__ContractHolder(
                *args, **kwargs)
        return ContractHolder.__instance

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def __setattr__(self, name):
        return setattr(self.instance, name)


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


async def worker(name: str, queue: asyncio.Queue):
    while True:
        contract = await queue.get()
        log.debug(
            f'{name} loading {contract.contract.localSymbol} '
            f'ending {contract.next_date} '
            # f'with params: {contract.params})'
        )
        chunk = await ib.reqHistoricalDataAsync(**contract.params)
        contract.save_chunk(chunk)
        if contract.next_date:
            await queue.put(contract)
        queue.task_done()


async def main(holder: ContractHolder):

    contracts = holder()
    number_of_workers = min(len(contracts), max_number_of_workers)

    log.debug(f'main function started, '
              f'retrieving data for {len(contracts)} instruments')

    queue = asyncio.LifoQueue()
    for contract in contracts:
        await queue.put(contract)
    workers = [asyncio.create_task(worker(f'worker {i}', queue))
               for i in range(number_of_workers)]

    await queue.join()

    # cancel all workers
    log.debug('cancelling workers')
    for w in workers:
        w.cancel()

    # wait until all worker tasks are cancelled
    await asyncio.gather(*workers)


if __name__ == '__main__':
    util.patchAsyncio()
    ib = IB()
    barSize = '30 secs'
    wts = 'MIDPOINT'
    # object where data is stored
    store = ArcticStore(f'{wts}_{barSize}')

    holder = ContractHolder(ib, 'contracts.csv',
                            store, wts, barSize, True)

    # asyncio.get_event_loop().set_debug(True)
    Connection(ib, partial(main, holder), watchdog=False)

    log.debug('script finished, about to disconnect')
    ib.disconnect()
    log.debug(f'disconnected')
