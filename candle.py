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
            # log_assert(not self.df.iloc[-1].isna().any(), (
            #    f'Not enough data for indicators for instrument'
            #    f' {self.contract.localSymbol} '
            #    f' index: {df.index[-1]}'
            #    f' values: {self.df.iloc[-1].to_dict()}'
            #    f'{self.df}'), __name__)
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


class DoubleSignalMixin:
    def process(self) -> None:
        log.debug(f'New candle for {self.contract.localSymbol} '
                  f'{self.df.index[-1]}: {self.df.iloc[-1].to_dict()}')
        if self.df.signal[-1]:
            self.entrySignal.emit(self)
        elif self.df.close_signal[-1]:
            self.closeSignal.emit(self)


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
        df['pre_lock'] = df['signal'] * ((df['signal'] * df['filter']) == 1)

        df['lock'] = (((df.pre_lock.shift().rolling(self.lock_periods).min()
                        + df.pre_lock.shift().rolling(self.lock_periods).max()))
                      + df.pre_lock.shift()).clip(-1, 1)

        df['filtered_signal'] = (
            df['pre_lock'] * ~((df['pre_lock'] * df['lock']) == 1))
        return df


class BreakoutBufferCandle(SingleSignalMixin, Candle):

    def get_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['ema_fast'] = df.price.ewm(
            span=self.ema_fast, min_periods=int(self.ema_fast*.6)).mean()
        df['ema_slow'] = df.price.ewm(
            span=self.ema_slow, min_periods=int(self.ema_slow*.6)).mean()
        df['atr'] = indicators.atr(df, self.atr_periods)
        df['signal'] = indicators.min_max_buffer_signal(
            df.price, self.periods, df.atr * 2)
        df['filter'] = np.sign(df['ema_fast'] - df['ema_slow'])
        df['filtered_signal'] = df['signal'] * \
            ((df['signal'] * df['filter']) == 1)
        return df


class BreakoutLockBufferCandle(SingleSignalMixin, Candle):

    def get_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['ema_fast'] = df.price.ewm(
            span=self.ema_fast, min_periods=int(self.ema_fast*.6)).mean()
        df['ema_slow'] = df.price.ewm(
            span=self.ema_slow, min_periods=int(self.ema_slow*.6)).mean()
        df['atr'] = indicators.atr(df, self.atr_periods)
        df['signal'] = indicators.min_max_buffer_signal(
            df.price, self.periods, df.atr)
        df['filter'] = np.sign(df['ema_fast'] - df['ema_slow'])
        df['pre_lock'] = df['signal'] * ((df['signal'] * df['filter']) == 1)

        df['lock'] = (((df.pre_lock.shift().rolling(self.lock_periods).min()
                        + df.pre_lock.shift().rolling(self.lock_periods).max()))
                      + df.pre_lock.shift()).clip(-1, 1)

        df['filtered_signal'] = (
            df['pre_lock'] * ~((df['pre_lock'] * df['lock']) == 1))
        return df


class RocCandle(SingleSignalMixin, Candle):

    def get_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['atr'] = indicators.atr(df, self.atr_periods)
        df['roc'] = df.price.pct_change(self.roc)
        df['signal'] = np.sign(df['roc'])
        df['filtered_signal'] = df['signal']
        return df


class RocCandleFiltered(FilterMixin, RocCandle):

    def get_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.filter(RocCandle.get_indicators(self, df))


class BollingerCandle(DoubleSignalMixin, Candle):
    """
    Entry signal when price brakes out of Bollinger band, close signal
    when price crosses mean.
    """

    def get_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['atr'] = indicators.atr(df, self.atr_periods)
        df['mid'] = df.price.ewm(span=self.bollinger_periods).mean()
        df['std'] = df['price'].ewm(span=self.bollinger_periods).std()
        df['upper'] = df['mid'] + 1 * df['std']
        df['lower'] = df['mid'] - 1 * df['std']
        df['up'] = (df['price'] > df['upper'].shift()) * 1
        df['down'] = (df['price'] < df['lower'].shift()) * -1
        df['signal'] = df['up'] + df['down']
        df['filtered_signal'] = df['signal']
        df['direction'] = np.sign(df['price'] - df['mid'])
        df['close_signal'] = ((df['direction'] * df['direction'].shift()) < 0
                              ) * df['direction']
        return df


class BreakoutStrenthFilteredCandle(SingleSignalMixin, Candle):

    def get_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['atr'] = indicators.atr(df, self.atr_periods)

        df['ema_fast'] = df.price.ewm(
            span=self.ema_fast, min_periods=int(self.ema_fast*.6)).mean()
        df['ema_slow'] = df.price.ewm(
            span=self.ema_slow, min_periods=int(self.ema_slow*.6)).mean()
        df['ema_filter'] = np.sign(df['ema_fast'] - df['ema_slow'])

        df['ema_slow_diff'] = df['ema_slow'].diff().abs()
        df['vol_slow'] = df['price'].ewm(span=self.ema_slow).std()
        df['strength_slow'] = (df['ema_slow_diff'] / df['vol_slow']) * 100
        df['strength_slow_median'] = df['strength_slow'].expanding().median()
        df['strength_filter'] = (df['strength_slow'] >=
                                 df['strength_slow_median'])

        df['signal'] = indicators.min_max_signal(df.price, self.periods)

        df['pre_strength'] = df['signal'] * \
            ((df['signal'] * df['ema_filter']) == 1)
        df['filtered_signal'] = df['pre_strength'] * df['strength_filter']
        return df


class DonchianCandle(DoubleSignalMixin, Candle):

    def get_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['atr'] = indicators.atr(df, self.atr_periods)

        df['ema_fast'] = df.price.ewm(
            span=self.ema_fast, min_periods=int(self.ema_fast*.6)).mean()
        df['ema_slow'] = df.price.ewm(
            span=self.ema_slow, min_periods=int(self.ema_slow*.6)).mean()
        df['ema_filter'] = np.sign(df['ema_fast'] - df['ema_slow'])

        df['signal'] = indicators.min_max_signal(
            df.price, self.periods)
        df['filtered_signal'] = df['signal'] * \
            ((df['signal'] * df['ema_filter']) == 1)

        df['close_signal'] = indicators.min_max_signal(
            df.price, self.close_periods)

        return df


class BreakoutCandleVolFilter(SingleSignalMixin, Candle):

    def get_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['ema_fast'] = df.price.ewm(
            span=self.ema_fast, min_periods=int(self.ema_fast*.6)).mean()
        df['ema_slow'] = df.price.ewm(
            span=self.ema_slow, min_periods=int(self.ema_slow*.6)).mean()
        df['atr'] = indicators.atr(df, self.atr_periods)
        df['signal'] = indicators.min_max_signal(df.price, self.periods)

        df['ema_filter'] = np.sign(df['ema_fast'] - df['ema_slow'])
        df['ema_filtered_signal'] = df['signal'] * \
            ((df['signal'] * df['ema_filter']) == 1)

        df['vol_filter'] = df['atr'] < (df['close'] * self.lock_filter)

        df['filtered_signal'] = df['vol_filter'] * df['ema_filtered_signal']

        return df


class BreakoutLockCandleVolFilter(SingleSignalMixin, Candle):

    def get_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['ema_fast'] = df.price.ewm(
            span=self.ema_fast, min_periods=int(self.ema_fast*.6)).mean()
        df['ema_slow'] = df.price.ewm(
            span=self.ema_slow, min_periods=int(self.ema_slow*.6)).mean()
        df['atr'] = indicators.atr(df, self.atr_periods)
        df['signal'] = indicators.min_max_signal(df.price, self.periods)
        df['ema_filter'] = np.sign(df['ema_fast'] - df['ema_slow'])
        df['ema_filtered_signal'] = df['signal'] * \
            ((df['signal'] * df['ema_filter']) == 1)

        df['vol_filter'] = df['atr'] < (df['close'] * self.lock_filter)

        df['pre_lock'] = df['vol_filter'] * df['ema_filtered_signal']

        df['lock'] = (df.pre_lock.shift().rolling(self.lock_periods).min()
                      + df.pre_lock.shift().rolling(self.lock_periods).max()
                      ).clip(-1, 1)

        df['filtered_signal'] = (
            df['pre_lock'] * ~((df['pre_lock'] * df['lock']) == 1))
        return df
