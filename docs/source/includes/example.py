import ib_insync as ibi
import numpy as np
import pandas as pd

from haymaker import indicators
from haymaker.components import (
    BarAggregator,
    BinarySignalProcessor,
    BracketExecutionModel,
    FixedSizeAllocator,
    HistoricalDataStreamer,
    NoFilter,
    PandasSignalModel,
    PortfolioWrapper,
    SignalType,
    TrailingStop,
)


class EMACrossSignalModel(PandasSignalModel):
    def __init__(self, source_key, contract):
        self.fast_lookback = 12
        self.slow_lookback = 48
        self.atr_lookback = 24
        super().__init__(source_key, contract, SignalType.STATE)

    def df(self, df: pd.DataFrame) -> pd.DataFrame:
        df["fast_ema"] = df["close"].ewm(self.fast_lookback).mean()
        df["slow_ema"] = df["close"].ewm(self.slow_lookback).mean()
        df["signal"] = np.sign(df["fast_ema"] - df["slow_ema"])
        df["atr"] = indicators.atr(df, self.atr_lookback)
        return df


es_contract = ibi.ContFuture("ES", "CME")

pipe = HistoricalDataStreamer(
    es_contract, "10 D", "1 hour", "TRADES"
).pipe(
    BarAggregator(NoFilter()),
    EMACrossSignalModel(
        "ema_cross_ES",
        es_contract,
    ),
    BinarySignalProcessor(),
    PortfolioWrapper(FixedSizeAllocator(1)),
    BracketExecutionModel(
        "ema_cross_ES",
        name="ema_cross_brackets",
        stop=TrailingStop(3, vol_field="atr"),
    ),
)
