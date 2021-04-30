
from typing import List, Any
from abc import ABC

from ib_insync import IB, Contract, Event

from candle import Candle
from logger import Logger


log = Logger(__name__)


class Portfolio(ABC):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.kwargs = kwargs
        self.values = {}
        self._createEvents()

    def _createEvents(self):
        self.entrySignal = Event('entrySignal')
        self.closeSignal = Event('closeSignal')

    def register(self, ib: IB, candles: List[Candle]):
        self.ib = ib
        self.candles = candles

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

    def onEntry(self, signal: Any):
        raise NotImplementedError

    def onClose(self, signal: Any):
        raise NotImplementedError

    def onSignal(self, signal: Any):
        raise NotImplementedError

    def __str__(self):
        return f'{self.__class__.__name__} with args: {self.kwargs}'


class FixedPortfolio(Portfolio):

    def number_of_contracts(self, contract: Contract, price: float):
        return 1

    def onSignal(self, obj: Candle):
        error_message = f'Invalid signal: {obj.df.signal[-1]}'
        assert obj.df.signal[-1] in [-1, 1], error_message

        position = self.positions.get(obj.contract.symbol)
        if obj.df.filtered_signal[-1] and not position:
            error_message = f'Invalid filtered_signal: {obj.df.filtered_signal[-1]}'
            assert obj.df.filtered_signal[-1] in [-1, 1], error_message
            log.debug(f'entry signal emitted for {obj.contract.localSymbol},'
                      f' signal: {obj.df.filtered_signal[-1]},'
                      f' atr: {obj.df.atr[-1]}')
            number_of_contracts = self.number_of_contracts(
                obj.contract, obj.df.price[-1])
            if number_of_contracts:
                self.entrySignal.emit(
                    obj.contract, obj.df.signal[-1], number_of_contracts,
                    obj.df.atr[-1] * obj.sl_atr,  obj)
            else:
                log.warning(f'Not enough equity to open position for: '
                            f'{obj.contract.localSymbol}')
        elif obj.df.signal[-1] and position:
            if position * obj.df.signal[-1] < 0:
                log.debug(
                    f'close signal emitted for {obj.contract.localSymbol}')
                self.closeSignal.emit(obj.contract, obj.df.signal[-1],
                                      abs(self.positions[obj.contract.symbol]))


class DoubleSignalFixedPortfolio(FixedPortfolio):

    def onEntry(self, obj: Candle) -> None:
        FixedPortfolio.onSignal(self, obj)

    def onClose(self, obj: Candle) -> None:
        position = self.positions.get(obj.contract.symbol)
        if position and ((position * obj.df.close_signal[-1]) < 0):
            log.debug(f'close signal emitted for {obj.contract.localSymbol}')
            self.closeSignal.emit(obj.contract, obj.df.close_signal[-1],
                                  abs(position))


class AdjustedPortfolio(Portfolio):

    multiplier_dict = {2: 1.35, 3: 1.5, 4: 1.65}

    def register(self, ib: IB, candles: List[Candle]):
        super().register(ib, candles)
        self.div_multiplier = self.multiplier_dict[len(candles)]
        self.alloc = 1/len(self.candles)

    def number_of_contracts(self, contract: Candle):
        account_value = self.account_value
        log.debug(f'account value: {account_value}')
        daily_vol = self.target_vol / 16
        daily_cash_alloc = daily_vol * account_value * self.alloc
        cash_alloc_per_trade = (daily_cash_alloc /
                                (contract.trades_per_day ** .5))
        points_alloc_per_trade = (cash_alloc_per_trade /
                                  float(contract.contract.multiplier))
        # 1.4 is atr to vol 'translation'
        contracts = (points_alloc_per_trade /
                     ((max(contract.df.atr[-1], contract.min_atr) / 1.4)
                      * contract.sl_atr))
        return round(contracts * self.div_multiplier, 1)

    def onSignal(self, obj: Candle):
        #error_message = f'Invalid signal: {obj.df.filtered_signal[-1]}'
        #assert obj.df.filtered_signal[-1] in [-1, 1], error_message

        position = (self.positions.get(obj.contract.symbol)
                    or self.positions.get(obj.micro_contract.symbol))
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
                    obj.contract, obj.df.signal[-1], major_contracts,
                    obj.df.atr[-1] * obj.sl_atr,  obj)
            # 'and not' part because minor contract might have not been
            # stopped out on previous position, even though major contract was
            if minor_contracts and not self.positions.get(
                    obj.micro_contract.symbol):
                log.debug(f'emitting signal for minor contract: '
                          f'{obj.micro_contract.symbol}: {minor_contracts}')
                self.entrySignal.emit(
                    obj.micro_contract, obj.df.signal[-1], minor_contracts,
                    obj.df.atr[-1] * obj.sl_atr, obj)
            if not (major_contracts or minor_contracts):
                message = (f'Not enough equity to open position for: '
                           f'{obj.contract.localSymbol}')
                log.warning(message)
        elif obj.df.signal[-1] and position:
            if position * obj.df.signal[-1] < 0:
                log.debug(
                    f'close signal emitted for {obj.contract.localSymbol}')
                # try except to cover case when position is only in
                # micro_contract
                try:
                    self.closeSignal.emit(
                        obj.contract, obj.df.signal[-1],
                        abs(self.positions[obj.contract.symbol]))
                except KeyError:
                    pass
                if self.positions.get(obj.micro_contract.symbol):
                    self.closeSignal.emit(obj.micro_contract,
                                          obj.df.signal[-1],
                                          abs(self.positions[
                                              obj.micro_contract.symbol]),
                                          )


class WeightedAdjustedPortfolio(AdjustedPortfolio):

    def register(self, ib: IB, candles: List[Candle]):
        Portfolio.register(self, ib, candles)
        self.div_multiplier = self.multiplier_dict[len(candles)]
        allocs = [candle.alloc for candle in candles]
        assert round(sum(allocs), 4) == 1, "Allocations don't add-up to 1"

    def number_of_contracts(self, contract: Candle):
        daily_vol = self.target_vol / 16
        daily_cash_alloc = daily_vol * self.account_value * contract.alloc
        cash_alloc_per_trade = daily_cash_alloc / \
            (contract.trades_per_day ** .5)
        points_alloc_per_trade = (cash_alloc_per_trade /
                                  float(contract.contract.multiplier))
        # 1.4 is atr to vol 'translation'
        contracts = (points_alloc_per_trade /
                     ((max(contract.df.atr[-1], contract.min_atr) / 1.4)
                      * contract.sl_atr))
        return round(contracts * self.div_multiplier, 1)
