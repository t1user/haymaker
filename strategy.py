from logbook import Logger

import pandas as pd
import numpy as np
from ib_insync import Contract


from streamers import VolumeStreamer
from trader import Candle, Portfolio
from params import contracts
from indicators import get_ATR, get_signals


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
        # self.account_value
        d = {'NQ': 1, 'ES': 1, 'GC': 1, 'CL': 1}
        return 1

        # return int((1e+5 * self.leverage *
        #            params.alloc) / (float(params.contract.multiplier) *
        #                             price))

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
                    obj, obj.df.signal[-1], obj.df.atr[-1],
                    number_of_contracts)
            else:
                message = (f'Not enough equity to open position for: '
                           f'{obj.contract.localSymbol}')
                log.warning(message)
        elif obj.df.signal[-1] and position:
            if position * obj.df.signal[-1] < 0:
                log.debug(
                    f'close signal emitted for {obj.contract.localSymbol}')
                self.closeSignal.emit(obj, obj.df.signal[-1],
                                      abs(self.positions[obj.contract.symbol]))


candles = [BreakoutCandle(VolumeStreamer(params.volume,
                                         params.avg_periods),
                          **params.__dict__)
           for params in contracts]
