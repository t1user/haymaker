from dataclasses import dataclass

import ib_insync as ibi
import numpy as np
import pandas as pd

from haymaker import (
    app,
    base,
    bracket_legs,
    brick,
    execution_models,
    indicators,
    portfolio,
    processors,
    signals,
    streamers,
)


@dataclass
class EMACrossStrategy(brick.AbstractDfBrick):
    strategy: str
    contract: ibi.Contract
    fast_lookback: int
    slow_lookback: int
    atr_lookback: int

    def df(self, df: pd.DataFrame) -> pd.DataFrame:
        df["fast_ema"] = df["close"].ewm(self.fast_lookback).mean()
        df["slow_ema"] = df["close"].ewm(self.slow_lookback).mean()
        df["signal"] = np.sign(df["fast_ema"] - df["slow_ema"])
        df["atr"] = indicators.atr(df, self.atr_lookback)
        return df


es_contract = ibi.ContFuture("ES", "CME")

portfolio.FixedPortfolio(1)

pipe = base.Pipe(
    streamers.HistoricalDataStreamer(es_contract, "10 D", "1 hours", "TRADES"),
    processors.BarAggregator(processors.NoFilter()),
    EMACrossStrategy("ema_cross_ES", es_contract, 12, 48, 24),
    signals.BinarySignalProcessor(),
    portfolio.PortfolioWrapper(),
    execution_models.EventDrivenExecModel(
        stop=bracket_legs.TrailingStop(3, vol_field="atr")
    ),
)

if __name__ == "__main__":
    app.App().run()
