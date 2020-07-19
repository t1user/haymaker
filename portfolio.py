from typing import List, Any, Dict
from abc import ABC

from ib_insync import IB, Contract, Event

from candle import Candle
from logger import Logger


log = Logger(__name__)


class Portfolio(ABC):

    def __init__(self, ib: IB, candles: List[Candle],
                 portfolio_params: Dict[Any, Any]):
        self.ib = ib
        self.candles = candles
        self.__dict__.update(portfolio_params)
        self.values = {}
        self._createEvents()

    def _createEvents(self):
        self.entrySignal = Event('entrySignal')
        self.closeSignal = Event('closeSignal')

    @property
    def account_value(self):
        self.update_value()
        return self.values['TotalCashBalance'] + min(
            self.values['UnrealizedPnL'], 0)

    @property
    def positions(self):
        positions = self.ib.positions()
        return {p.contract.symbol: p.position for p in positions}

    def update_value(self):
        tags = self.ib.accountValues()
        for item in tags:
            if item.currency == 'USD':
                try:
                    self.values[item.tag] = float(item.value)
                except ValueError:
                    pass

    def onEntry(self, signal):
        raise NotImplementedError

    def onClose(self, signal):
        raise NotImplementedError

    def onSignal(self, signal):
        raise NotImplementedError


class FixedPortfolio(Portfolio):

    def number_of_contracts(self, contract: Contract, price: float):
        return 1

    def onSignal(self, obj: Candle):
        position = self.positions.get(obj.contract.symbol)
        if obj.df.filtered_signal[-1] and not position:
            message = (f'entry signal emitted for {obj.contract.localSymbol},'
                       f' signal: {obj.df.filtered_signal[-1]},'
                       f' atr: {obj.df.atr[-1]}')
            log.debug(message)
            number_of_contracts = self.number_of_contracts(
                obj.contract, obj.df.price[-1])
            if number_of_contracts:
                self.entrySignal.emit(
                    obj.contract, obj.df.signal[-1], obj.df.atr[-1],
                    number_of_contracts)
            else:
                message = (f'Not enough equity to open position for: '
                           f'{obj.contract.localSymbol}')
                log.warning(message)
        elif obj.df.signal[-1] and position:
            if position * obj.df.signal[-1] < 0:
                log.debug(
                    f'close signal emitted for {obj.contract.localSymbol}')
                self.closeSignal.emit(obj.contract, obj.df.signal[-1],
                                      abs(self.positions[obj.contract.symbol]))


class AdjustedPortfolio(Portfolio):

    multiplier_dict = {2: 1.35, 3: 1.5, 4: 1.65}

    def __init__(self, ib, candles: List[Candle],
                 portfolio_params: Dict[Any, Any]):
        self.div_multiplier = self.multiplier_dict[len(candles)]
        super().__init__(ib, candles, portfolio_params)

    def alloc(self):
        return 1/len(self.candles)

    def number_of_contracts(self, contract: Candle):
        daily_vol = self.target_vol / 16
        daily_cash_alloc = daily_vol * self.account_value * self.alloc()
        cash_alloc_per_trade = daily_cash_alloc / \
            (contract.trades_per_day ** .5)
        points_alloc_per_trade = (cash_alloc_per_trade /
                                  float(contract.contract.multiplier))
        # 1.4 is atr to vol 'translation'
        contracts = (points_alloc_per_trade /
                     ((max(contract.df.atr[-1], contract.min_atr) / 1.4)
                      * contract.sl_atr))
        return round(contracts * self.div_multiplier, 1)

    def onSignal(self, obj: Candle):
        position = self.positions.get(obj.contract.symbol)
        if obj.df.filtered_signal[-1] and not position:
            message = (f'entry signal emitted for {obj.contract.localSymbol},'
                       f' signal: {obj.df.filtered_signal[-1]},'
                       f' atr: {obj.df.atr[-1]}')
            log.debug(message)
            number_of_contracts = self.number_of_contracts(obj)
            major_contracts = int(number_of_contracts)
            minor_contracts = int((number_of_contracts - major_contracts) * 10)
            log.debug(f'contracts will be traded: '
                      f'{obj.contract.symbol}: {major_contracts}, '
                      f'{obj.micro_contract.symbol}: {minor_contracts}')
            if major_contracts:
                log.debug(f'emitting signal for major contract: '
                          f'{obj.contract.symbol}: {major_contracts}')
                self.entrySignal.emit(
                    obj.contract, obj.df.signal[-1], obj.df.atr[-1],
                    major_contracts)
            # 'and not' part because minor contract might have not been
            # stopped out on previous position, even though major contract was
            if minor_contracts and not self.positions.get(
                    obj.micro_contract.symbol):
                log.debug(f'emitting signal for minor contract: '
                          f'{obj.micro_contract.symbol}: {minor_contracts}')
                self.entrySignal.emit(
                    obj.micro_contract, obj.df.signal[-1], obj.df.atr[-1],
                    minor_contracts)
            if not (major_contracts or minor_contracts):
                message = (f'Not enough equity to open position for: '
                           f'{obj.contract.localSymbol}')
                log.warning(message)
        elif obj.df.signal[-1] and position:
            if position * obj.df.signal[-1] < 0:
                log.debug(
                    f'close signal emitted for {obj.contract.localSymbol}')
                self.closeSignal.emit(obj.contract, obj.df.signal[-1],
                                      abs(self.positions[obj.contract.symbol]))
                if self.positions.get(obj.micro_contract.symbol):
                    self.closeSignal.emit(obj.micro_contract,
                                          obj.df.signal[-1],
                                          abs(self.positions[
                                              obj.micro_contract.symbol]))
