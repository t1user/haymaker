from logbook import Logger

import pandas as pd
import numpy as np
from ib_insync import Contract
from typing import List, Dict, Any


from streamers import VolumeStreamer
from trader import Candle, Portfolio
from params import contracts
from indicators import get_ATR, get_signals
from objects import Params


log = Logger(__name__)


class BreakoutCandle(Candle):

    def get_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['ema_fast'] = df.price.ewm(
            span=self.ema_fast, min_periods=int(self.ema_fast*.8)).mean()
        df['ema_slow'] = df.price.ewm(
            span=self.ema_slow, min_periods=int(self.ema_slow*.8)).mean()
        df['atr'] = get_ATR(df, self.atr_periods)
        df['signal'] = get_signals(df.price, self.periods)
        # df['signal'] = round(get_signals(
        #    df.price, self.periods).rolling(3).mean(), 0)
        df['filter'] = np.sign(df['ema_fast'] - df['ema_slow'])
        df['filtered_signal'] = df['signal'] * \
            ((df['signal'] * df['filter']) == 1)
        return df

    def process(self) -> None:
        message = (f"New candle for {self.contract.localSymbol} "
                   f"{self.df.index[-1]}: {self.df.iloc[-1].to_dict()}")
        log.debug(message)
        if self.df.signal[-1]:
            self.signal.emit(self)


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
        log.debug(f'daily cash allocation: {daily_cash_alloc}')
        cash_alloc_per_trade = daily_cash_alloc / \
            (contract.trades_per_day ** .5)
        log.debug(f'cash allocation per trade: {cash_alloc_per_trade}')
        points_alloc_per_trade = (cash_alloc_per_trade /
                                  float(contract.contract.multiplier))
        log.debug(f'points allocation per trade: {points_alloc_per_trade}')
        contracts = (points_alloc_per_trade /
                     ((contract.df.atr[-1] / 1.5) * contract.sl_atr))
        log.debug(f'contracts: {contracts}')
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
            # and part because minor contract might have not been stopped out
            # on previous position, even though major contract was
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


candles = [BreakoutCandle(VolumeStreamer(params.volume,
                                         params.avg_periods),
                          **params.__dict__)
           for params in contracts]
