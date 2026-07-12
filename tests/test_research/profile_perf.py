import time
import numpy as np
import pandas as pd
from haymaker.research.backtester.pipeline import (
    no_stop,
    _PerfCalculator,
)


def _make_test_data(seed: int = 42, length: int = 1_000_000) -> pd.DataFrame:
    np.random.seed(seed)
    returns = np.random.normal(0, 0.001, length)
    prices = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame(
        {
            "open": prices,
        },
        index=pd.date_range("2020-01-01", periods=length, freq="1min"),
    )

    indicator = np.random.randn(length)
    signal = np.zeros(length)
    signal[indicator > 1.5] = 1
    signal[indicator < -1.5] = -1
    signal = pd.Series(signal).ffill().fillna(0).to_numpy()

    position = pd.Series(signal, index=df.index)
    df["position"] = position
    return df


if __name__ == "__main__":
    df = _make_test_data()

    print("--- Profiling Wrapper Overhead ---")

    t0 = time.perf_counter()
    tx = no_stop(df, price_column="open")
    t1 = time.perf_counter()
    print(f"1. TransactionFrame Creation: {t1 - t0:.4f} sec")

    from haymaker.research.backtester import _TransactionFrame

    calc = _PerfCalculator(
        _TransactionFrame(tx),
        slippage=1.5,
        use_numba=True,
        skip_last_open=False,
        capital=None,
        min_tick=None,
        sunday_to_monday=True,
    )

    t0 = time.perf_counter()
    arrs = calc._prepare_arrays()
    t1 = time.perf_counter()
    print(f"2. Prepare Arrays: {t1 - t0:.4f} sec")

    # Compile
    from haymaker.research.backtester.engine import _perf_engine

    z = np.zeros(10)
    _perf_engine(z, z, z, z, 1.5)

    t0 = time.perf_counter()
    pnl, trades = _perf_engine(*arrs)
    t1 = time.perf_counter()
    print(f"3. Numba Engine Execution: {t1 - t0:.4f} sec")

    t0 = time.perf_counter()
    bar_df = calc._build_bar_df(pnl, arrs[-1])
    positions = calc._build_positions(trades, df.index)
    if calc._skip_last_open and len(positions) > 0:
        bar_df = bar_df.iloc[: int(trades[-1, 0])]
        positions = positions.iloc[:-1]
    t1 = time.perf_counter()
    print(f"4. Build DataFrames: {t1 - t0:.4f} sec")

    t0 = time.perf_counter()
    from haymaker.research.backtester.metrics import (
        build_performance_frames,
        build_stats,
    )

    bar_df, daily = build_performance_frames(
        bar_df, capital=calc._capital, sunday_to_monday=True
    )
    stats = build_stats(
        positions,
        daily,
        bar_df,
        capital=calc._capital,
        min_tick=calc._min_tick,
        warnings=[],
    )
    t1 = time.perf_counter()
    print(f"5. Build Stats: {t1 - t0:.4f} sec")
