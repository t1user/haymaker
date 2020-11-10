from abc import ABC, abstractmethod
from typing import Dict, Any, Union, List

from ib_insync import Event, IB
import pandas as pd
import numpy as np

from logger import Logger, log_assert
import indicators


log = Logger(__name__)


class Candle(ABC):

    def __init__(self, streamer,
                 contract_fields: Union[List[str], str] = 'contract',
                 **kwargs):
        if isinstance(contract_fields, str):
            self.contract_fields = [contract_fields]
        else:
            self.contract_fields = contract_fields
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

    def __repr__(self):
        return f'Candle: {self.contract}'


class SingleSignalMixin:
    def process(self) -> None:
        log.debug(f'New candle for {self.contract.localSymbol} '
                  f'{self.df.index[-1]}: {self.df.iloc[-1].to_dict()}')
        if self.df.signal[-1]:
            self.signal.emit(self)


class FilterMixin:
    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        df['ema_fast'] = df.price.ewm(
            span=self.ema_fast, min_periods=int(self.ema_fast*.6)).mean()
        df['ema_slow'] = df.price.ewm(
            span=self.ema_slow, min_periods=int(self.ema_slow*.6)).mean()
        df['filter'] = np.sign(df['ema_fast'] - df['ema_slow'])
        df['filtered_signal'] = df['signal'] * \
            ((df['signal'] * df['filter']) == 1)
        return df


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


class RepeatBreakoutCandle(SingleSignalMixin, Candle):

    def get_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['ema_fast'] = df.price.ewm(
            span=self.ema_fast, min_periods=int(self.ema_fast*.6)).mean()
        df['ema_slow'] = df.price.ewm(
            span=self.ema_slow, min_periods=int(self.ema_slow*.6)).mean()
        df['atr'] = indicators.atr(df, self.atr_periods)
        df['signal'] = indicators.min_max_signal(df.price, self.periods)
        df['repeat_signal'] = df.signal.rolling(2).mean().round()
        df['filter'] = np.sign(df['ema_fast'] - df['ema_slow'])
        df['filtered_signal'] = df['repeat_signal'] * \
            ((df['signal'] * df['filter']) == 1)
        return df


class RsiCandle(SingleSignalMixin, Candle):

    def get_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['rsi'] = indicators.rsi(df.price, 24)
        df['carver_rsi'] = indicators.carver(df['rsi'], 100).rolling(15).mean()
        df['signal'] = indicators.range_crosser(df['carver_rsi'], 60)
        df['ema_fast'] = df.price.ewm(
            span=self.ema_fast, min_periods=int(self.ema_fast*.6)).mean()
        df['ema_slow'] = df.price.ewm(
            span=self.ema_slow, min_periods=int(self.ema_slow*.6)).mean()
        df['atr'] = indicators.atr(df, self.atr_periods)
        df['filter'] = np.sign(df['ema_fast'] - df['ema_slow'])
        df['filtered_signal'] = df['signal'] * \
            ((df['signal'] * df['filter']) == 1)
        return df


class CarverCandle(SingleSignalMixin, Candle):

    def get_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['carver'] = indicators.carver(
            df.price, self.periods).rolling(15).mean()
        df['signal'] = indicators.range_crosser(df['carver'], 1)
        df['ema_fast'] = df.price.ewm(
            span=self.ema_fast, min_periods=int(self.ema_fast*.6)).mean()
        df['ema_slow'] = df.price.ewm(
            span=self.ema_slow, min_periods=int(self.ema_slow*.6)).mean()
        df['atr'] = indicators.atr(df, self.atr_periods)
        df['filter'] = np.sign(df['ema_fast'] - df['ema_slow'])
        df['filtered_signal'] = df['signal'] * \
            ((df['signal'] * df['filter']) == 1)
        return df


class BreakoutRsiCandle(SingleSignalMixin, Candle):

    def get_indicators(self, df: pd.DataFrame) -> pd.DataFrame:

        # atr
        df['atr'] = indicators.atr(df, self.atr_periods)

        # breakout signal
        df['breakout_s'] = indicators.min_max_signal(
            df.price, self.periods)
        # rsi signal
        df['rsi'] = indicators.rsi(df.price, self.rsi_periods)
        df['carver_rsi'] = indicators.carver(
            df['rsi'], 100).rolling(self.rsi_smooth).mean()
        df['rsi_s'] = indicators.range_crosser(
            df['carver_rsi'], self.rsi_threshold)
        # combined signal
        df['signal'] = df['breakout_s'] + df['rsi_s']

        # moving average filter
        df['ema_fast'] = df.price.ewm(
            span=self.ema_fast, min_periods=int(self.ema_fast*.6)).mean()
        df['ema_slow'] = df.price.ewm(
            span=self.ema_slow, min_periods=int(self.ema_slow*.6)).mean()
        df['filter'] = np.sign(df['ema_fast'] - df['ema_slow'])
        df['filtered_signal'] = df['signal'] * \
            ((df['signal'] * df['filter']) == 1)
        return df


class MultipleBreakoutCandle(SingleSignalMixin, Candle):

    def get_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['ema_fast'] = df.price.ewm(
            span=self.ema_fast, min_periods=int(self.ema_fast*.6)).mean()
        df['ema_slow'] = df.price.ewm(
            span=self.ema_slow, min_periods=int(self.ema_slow*.6)).mean()
        df['atr'] = indicators.atr(df, self.atr_periods)
        df['signal'] = indicators.any_signal(df.price, self.periods)
        df['filter'] = np.sign(df['ema_fast'] - df['ema_slow'])
        df['filtered_signal'] = df['signal'] * \
            ((df['signal'] * df['filter']) == 1)
        return df


class BreakoutLockCandle(SingleSignalMixin, Candle):

    def get_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['ema_fast'] = df.price.ewm(
            span=self.ema_fast, min_periods=int(self.ema_fast*.6)).mean()
        df['ema_slow'] = df.price.ewm(
            span=self.ema_slow, min_periods=int(self.ema_slow*.6)).mean()
        df['atr'] = indicators.atr(df, self.atr_periods)
        df['signal'] = indicators.min_max_signal(df.price, self.periods)
        df['filter'] = np.sign(df['ema_fast'] - df['ema_slow'])
        df['lock'] = -1 * (df.signal.shift().rolling(self.lock_periods).max()
                           - df.signal.shift().rolling(self.lock_periods).min())
        df['filtered_signal'] = ((df['signal'] - df['lock']) *
                                 ((df['signal'] * df['filter']) == 1))
        return df
