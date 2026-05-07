import time
import numpy as np
import pandas as pd

from haymaker.research.backtester.vector_backtester import perf as perf_old
from haymaker.research.backtester import perf as perf_new, no_stop
from haymaker.research.signal_converters import sig_pos


def _make_test_data(seed: int = 42, length: int = 500_000) -> pd.DataFrame:
    np.random.seed(seed)
    returns = np.random.normal(0, 0.001, length)
    prices = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices * (1 + np.random.uniform(0, 0.002, length)),
            "low": prices * (1 - np.random.uniform(0, 0.002, length)),
            "close": prices * (1 + np.random.normal(0, 0.0005, length)),
        },
        index=pd.date_range("2020-01-01", periods=length, freq="1min"),
    )

    # Generate random signals
    indicator = np.random.randn(length)
    signal = np.zeros(length)
    signal[indicator > 1.5] = 1
    signal[indicator < -1.5] = -1

    # Fill forward
    signal = pd.Series(signal).ffill().fillna(0).to_numpy()
    df["position"] = sig_pos(pd.Series(signal, index=df.index))

    return df


if __name__ == "__main__":
    print("Generating 1,000,000 bars of data...")
    df = _make_test_data(length=1_000_000)
    print(f"Data generated. Rows: {len(df):,}")

    # Pre-compile Numba
    print("Compiling Numba engine...")
    tx_small = no_stop(df.iloc[:1000], price_column="open")
    perf_new(tx_small, slippage=1.5, use_numba=True)

    print("\n--- Running Benchmarks ---")

    # 1. Old Vector Backtester
    t0 = time.perf_counter()
    res_old = perf_old(df["open"], df["position"], slippage=1.5)
    t1 = time.perf_counter()
    old_time = t1 - t0
    print(f"Old Implementation: {old_time:.4f} seconds")

    # 2. New Numba Backtester
    t0 = time.perf_counter()
    tx = no_stop(df, price_column="open")
    res_new = perf_new(tx, slippage=1.5, use_numba=True)
    t1 = time.perf_counter()
    new_time = t1 - t0
    print(f"New Implementation: {new_time:.4f} seconds")

    # 3. New Python Reference Backtester
    t0 = time.perf_counter()
    tx = no_stop(df, price_column="open")
    res_new_py = perf_new(tx, slippage=1.5, use_numba=False)
    t1 = time.perf_counter()
    new_py_time = t1 - t0
    print(f"New Python Ref:     {new_py_time:.4f} seconds")

    print(f"\nSpeedup (Old vs New Numba): {old_time / new_time:.2f}x faster")
