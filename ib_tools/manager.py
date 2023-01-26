from datetime import datetime
from typing import Optional, Union, List

from ib_insync import IB

from candle import Candle
from portfolio import Portfolio
from trader import Trader
from saver import AbstractBaseSaver, PickleSaver
from blotter import AbstractBaseBlotter, CsvBlotter
from execution_models import BaseExecModel, EventDrivenExecModel
from logger import Logger

log = Logger(__name__)


class Manager:

    def __init__(self, ib: IB, candles: List[Candle],
                 portfolio: Portfolio,
                 saver: AbstractBaseSaver = PickleSaver(),
                 blotter: AbstractBaseBlotter = CsvBlotter(),
                 exec_model: Optional[BaseExecModel] = None,
                 trader: Optional[Trader] = None,
                 contract_fields: Union[List[str], str] = 'contract',
                 keep_ref: bool = True):

        self.ib = ib
        self.candles = candles
        self.saver = saver
        self.blotter = blotter
        self.trader = Trader(ib, blotter)
        self.exec_model = exec_model or EventDrivenExecModel()
        self.connect_exec_model()
        self.keep_ref = keep_ref
        self.portfolio = portfolio
        self.connect_portfolio()
        log.debug(f'manager object initiated: {self}')

    def connect_exec_model(self):
        self.exec_model.connect_trader(self.trader)

    def connect_portfolio(self):
        self.portfolio.register(self.ib, self.candles)
        self.portfolio.entrySignal += self.exec_model.onEntry
        self.portfolio.closeSignal += self.exec_model.onClose

    def onStarted(self, *args, **kwargs):
        log.debug('manager onStarted')
        self.trader.onStarted()
        self.exec_model.onStarted()
        self.candles = get_contracts(self.candles, self.ib)
        # allow backtester to convey simulation time
        now = kwargs.get('now') or datetime.now()
        self.connect_candles(now)

    def onScheduledUpdate(self):
        self.freeze()

    def freeze(self):
        """Function called periodically to keep record of system data"""
        for candle in self.candles:
            candle.save(self.saver)
        log.debug('Freezed data saved')

    def connect_candles(self, now):
        for candle in self.candles:
            # make sure no previous events connected
            candle.entrySignal.clear()
            candle.closeSignal.clear()
            candle.signal.clear()
            # connect trade signals
            candle.entrySignal.connect(self.portfolio.onEntry,
                                       keep_ref=self.keep_ref)
            candle.closeSignal.connect(self.portfolio.onClose,
                                       keep_ref=self.keep_ref)
            candle.signal.connect(self.portfolio.onSignal,
                                  keep_ref=self.keep_ref)
            candle.set_now(now)
            # run candle logic
            candle(self.ib)

    def __str__(self):
        return (f'Manager: ib: {self.ib}, candles: {self.candles}, '
                f'portfolio: {self.portfolio}, '
                f'exec_model: {self.exec_model}, '
                f'trader: {self.exec_model._trader}, '
                f'saver: {self.saver}, '
                f'keep_ref: {self.keep_ref}')


def get_contracts(candles: List[Candle], ib: IB,
                  ) -> List[Candle]:
    contract_list = []
    for candle in candles:
        for contract in candle.contract_fields:
            contract_list.append(getattr(candle, contract))
    ib.qualifyContracts(*contract_list)
    log.debug(f'contracts qualified: {contract_list}')
    return candles
