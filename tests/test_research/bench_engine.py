import time

import numpy as np
import pandas as pd

from haymaker.research.backtester.engine import _perf_engine as perf_engine_new
from haymaker.research.backtester.vector_backtester import _perf as perf_engine_old


def _make_test_data(seed: int = 42, length: int = 1_000_000) -> pd.DataFrame:
    np.random.seed(seed)
    returns = np.random.normal(0, 0.001, length)
    prices = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame(
        {
            "open": prices,
        }
    )

    indicator = np.random.randn(length)
    signal = np.zeros(length)
    signal[indicator > 1.5] = 1
    signal[indicator < -1.5] = -1
    signal = pd.Series(signal).ffill().fillna(0).to_numpy(dtype=np.float64)

    # Calculate transaction
    position = pd.Series(signal, index=df.index)
    transaction = position - position.shift().fillna(0)

    df["position"] = position
    df["transaction"] = transaction
    return df


if __name__ == "__main__":
    df = _make_test_data()

    # Pre-compile Numba
    print("Compiling Numba engine...")
    z = np.zeros(10)
    perf_engine_new(z, z, z, z, 1.5)

    print("\n--- Running Core Engine Benchmarks (1,000,000 bars) ---")

    # 1. Old Engine (Pandas Vectorized)
    t0 = time.perf_counter()
    res_old = perf_engine_old(df["open"], df["position"], cost=1.5)
    t1 = time.perf_counter()
    old_engine_time = t1 - t0
    print(f"Old Pandas Engine: {old_engine_time:.4f} seconds")

    # 2. New Engine (Numba)
    price_arr = df["open"].to_numpy(dtype=np.float64)
    # Simulate open_price
    open_price = np.where(df["transaction"] != 0, price_arr, 0.0)
    z_arr = np.zeros(len(df), dtype=np.float64)

    t0 = time.perf_counter()
    res_new = perf_engine_new(price_arr, open_price, z_arr, z_arr, 1.5)
    t1 = time.perf_counter()
    new_engine_time = t1 - t0
    print(f"New Numba Engine:  {new_engine_time:.4f} seconds")

    print(
        f"\nSpeedup (Old Pandas vs New Numba core): "
        f"{old_engine_time / new_engine_time:.2f}x faster"
    )
