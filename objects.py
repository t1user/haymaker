import os
from typing import Tuple, List, Union
from dataclasses import dataclass
import itertools

import pandas as pd
from ib_insync import IB, Contract, Future, ContFuture


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


class ObjectSelector:
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

    def create_objects(self):
        self.objects = [Contract.create(**s) for s in self.symbols]
        self.non_futures = [
            obj for obj in self.objects if not isinstance(obj, Future)]
        self.futures = [obj for obj in self.objects if isinstance(obj, Future)]
        self.contFutures = [ContFuture(**obj.nonDefaults()
                                       ).update(secType='CONTFUT')
                            for obj in self.futures]

    def lookup_futures(self, obj: List[Future]):
        futures = []
        for o in obj:
            o.update(includeExpired=True)
            futures.append(
                [Future(**c.contract.dict())
                 for c in self.ib.reqContractDetails(o)]
            )
        return list(itertools.chain(*futures))

    def lookup_contfutures(self, symbols):
        return [ContFuture(**s) for s in symbols]

    @property
    def list(self):
        if not self.contracts:
            self.update()
        return self.contracts

    def update(self):
        qualified = self.contFutures + self.non_futures
        self.ib.qualifyContracts(*qualified)
        self.contracts = self.lookup_futures(self.futures) + qualified
        return self.contracts

    @property
    def cont_list(self):
        self.ib.qualifyContracts(*self.contFutures)
        return self.contFutures
