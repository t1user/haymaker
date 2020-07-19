from abc import ABC, abstractmethod
from typing import Dict, Any

from ib_insync import Event, IB
import pandas as pd
import numpy as np

from logger import Logger, log_assert
import indicators


log = Logger(__name__)


class Candle(ABC):

    def __init__(self, streamer, **kwargs):
        self.__dict__.update(kwargs)
        self.streamer = streamer
        self.streamer.newCandle.connect(self.append, keep_ref=True)
        self.df = None
        log.debug(f'candle init for contract: {kwargs}')
        self.candles = []
        self._createEvents()

    def _createEvents(self):
        self.signal = Event('signal')
        self.entrySignal = Event('entrySignal')
        self.closeSignal = Event('closeSignal')

    def __call__(self, ib: IB):
        log.debug(
            f'Candle {self.contract.localSymbol} initializing data stream...')
        self.details = ib.reqContractDetails(self.contract)[0]
        self.streamer(ib, self.contract)

    def append(self, candle: Dict[str, Any]):
        self.candles.append(candle)
        if not candle['backfill']:
            df = pd.DataFrame(self.candles)
            df.set_index('date', inplace=True)
            self.df = self.get_indicators(df)
            log_assert(not self.df.iloc[-1].isna().any(), (
                f'Not enough data for indicators for instrument'
                f' {self.contract.localSymbol} '
                f' index: {df.index[-1]}'
                f' values: {self.df.iloc[-1].to_dict()}'
                f'{self.df}'), __name__)
            self.process()

    def save(self, saver):
        if self.df is not None:
            saver.save(self.df, 'candles', self.contract.localSymbol)
            saver.save(self.streamer.all_bars_df, 'all_bars',
                       self.contract.localSymbol)

    def set_now(self, now):
        self.streamer.now = now

    @abstractmethod
    def get_indicators(self, df):
        return df

    @abstractmethod
    def process(self):
        self.signal.emit(self)
        self.entrySignal.emit(self)
        self.closeSignal.emit(self)


class SingleSignalMixin:
    def process(self) -> None:
        log.debug(f'New candle for {self.contract.localSymbol} '
                  f'{self.df.index[-1]}: {self.df.iloc[-1].to_dict()}')
        if self.df.signal[-1]:
            self.signal.emit(self)


class BreakoutCandle(SingleSignalMixin, Candle):

    def get_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['ema_fast'] = df.price.ewm(
            span=self.ema_fast, min_periods=int(self.ema_fast*.6)).mean()
        df['ema_slow'] = df.price.ewm(
            span=self.ema_slow, min_periods=int(self.ema_slow*.6)).mean()
        df['atr'] = indicators.atr(df, self.atr_periods)
        df['signal'] = indicators.min_max_signal(df.price, self.periods)
        df['filter'] = np.sign(df['ema_fast'] - df['ema_slow'])
        df['filtered_signal'] = df['signal'] * \
            ((df['signal'] * df['filter']) == 1)
        return df


class RsiCandle(SingleSignalMixin, Candle):

    def get_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['rsi'] = indicators.rsi(df.price, 24)
        df['carver_rsi'] = indicators.carver(df['rsi'], 90)
        df['signal'] = indicators.range_crosser(df['carver_rsi'], 50)
        df['filtered_signal'] = df['signal']
        df['atr'] = indicators.atr(df, self.atr_periods)
        return df
