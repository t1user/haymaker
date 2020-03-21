from __future__ import annotations

import os
from typing import Tuple, List, Union, Type, Optional, Any
from dataclasses import dataclass, field
import itertools
from datetime import datetime

import pandas as pd
from ib_insync import IB, Contract, Future, ContFuture, BarDataList, util


@dataclass
class Params:
    contract: Tuple[str]  # contract given as tuple of params given to Future()
    periods: List[int]  # periods for breakout calculation
    ema_fast: int  # number of periods for moving average filter
    ema_slow: int  # number of periods for moving average filter
    sl_atr: int  # stop loss in ATRs
    atr_periods: int  # number of periods to calculate ATR on
    alloc: float  # fraction of capital to be allocated to instrument
    avg_periods: int = None  # candle volume to be calculated as average of x periods
    volume: int = None  # candle volume given directly


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

    def __init__(self, contract: Contract, head: datetime) -> None:
        self.contract = contract
        self.head = head
        self.bars = []

    @classmethod
    def create(cls, store) -> Type[DataWriter]:
        cls.store = store
        return cls

    @property
    def expiry(self) -> Union[datetime, str]:
        """Expiry date for expirable contracts or '' """
        e = self.contract.lastTradeDateOrContractMonth
        return e if e else datetime.strptime(e, '%Y%m%d')

    @property
    def data(self) -> Optional[pd.DataFrame]:
        """Available data in datastore for contract or None"""
        return self.store.read(self.contract)

    @property
    def from_date(self) -> Optional[pd.datetime]:
        """Earliest point in datastore"""
        return self.data.min() if self.data else None

    @property
    def to_date(self) -> Optional[pd.datetime]:
        """Latest point in datastore"""
        return self.data.max() if self.data else None

    def backfill(self,  now: datetime = datetime.now()) -> Optional[datetime]:
        """
        Check if data older than earliest point in datastore available.
        Return this start point if yes, return None if not.
        """
        # prevent multiple calls to datastore
        from_date = self.from_date
        # data present in datastore
        if from_date:
            return from_date if from_date > self.head else None
        # data not in datastore yet
        else:
            return min(self.expiry, now) if self.expiry else now

    def update(self, now: datetime = datetime.now()) -> Optional[datetime]:
        """
        Check if data newer than endpoint in datastore available for download.
        Return current date if yes, None if not.
        """
        # prevent multiple calls to datastore
        to_date = self.to_date
        if to_date:
            dt = min(self.expiry, now) if self.expiry else now
            if dt > to_date:
                return dt

    def schedule(self):
        pass

    def save_chunk(self):
        pass

    def write_to_store(self):
        pass


@dataclass
class DownloadContainer:
    """Hold downloaded data before it is saved to datastore"""
    from_date: pd.datetime
    to_date: pd.datetime
    update: bool = False
    df: pd.DataFrame = field(defaultfactory=pd.DataFrame)
    bars: List[BarDataList] = field(defaultfactory=list)

    def append(self, bar) -> None:
        self.bars.append(bar)

    @property
    def ok_to_write(self) -> bool:
        """Updated data should be written only if complete, otherwise
        difficult to find gaps would possibly occur in datastore."""

        if self.update:
            return self.df.index.min() <= self.from_date
        else:
            return True

    def data(self) -> Union[pd.DataFrame, pd.datetime]:
        """Return df ready to be written to datastore or date of end point
        for additional downloads"""
        self.df = util.df([b for bars in reversed(self.bars) for b in bars])
        self.df.set_index('date', inplace=True)
        if self.ok_to_save:
            return self.df
        else:
            new_to_date = self.df.index.min()
            self.to_date = new_to_date
            return new_to_date
