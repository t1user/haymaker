import numpy as np
import pandas as pd
import pytest

# Old implementation
from haymaker.research.backtester.retired.vector_backtester import perf as perf_old
from haymaker.research.signal_converters import sig_pos

# New implementation
from haymaker.research.backtester import perf as perf_new, no_stop
from haymaker.research.stop import stop_loss


def _tx_frame(
    bar_price: list[float],
    position: list[int],
    open_price: list[float],
    close_price: list[float],
    stop_price: list[float],
) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=len(bar_price), freq="h", name="date")
    return pd.DataFrame(
        {
            "bar_price": bar_price,
            "position": position,
            "open_price": open_price,
            "close_price": close_price,
            "stop_price": stop_price,
        },
        index=index,
    )


def _assert_old_new_perf_equal(tx: pd.DataFrame, slippage: float = 0.0) -> None:
    old = perf_old(
        tx["bar_price"],
        tx[["position", "open_price", "close_price", "stop_price"]],
        slippage=slippage,
        raise_exceptions=False,
    )
    new = perf_new(tx, slippage=slippage, use_numba=True)

    old_positions = old.positions[
        ["date_o", "open", "date_c", "close", "g_pnl", "pnl", "duration"]
    ].reset_index(drop=True)
    new_positions = new.positions[
        ["date_o", "open", "date_c", "close", "g_pnl", "pnl", "duration"]
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(old_positions, new_positions)
    np.testing.assert_allclose(old.df["g_pnl"].sum(), new.df["g_pnl"].sum())
    np.testing.assert_allclose(old.df["pnl"].sum(), new.df["pnl"].sum())
    np.testing.assert_allclose(
        np.asarray(old.daily["returns"], dtype=float),
        np.asarray(new.daily["returns"], dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )


def _make_test_data(seed: int = 42, length: int = 500) -> pd.DataFrame:
    np.random.seed(seed)

    # Generate prices (random walk)
    returns = np.random.normal(0, 0.001, length)
    prices = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices * (1 + np.random.uniform(0, 0.002, length)),
            "low": prices * (1 - np.random.uniform(0, 0.002, length)),
            "close": prices * (1 + np.random.normal(0, 0.0005, length)),
        },
        index=pd.date_range("2020-01-01", periods=length, freq="1h"),
    )

    # Generate random signals (e.g., MACD crossover logic simplified)
    indicator = np.random.randn(length)
    signal = np.zeros(length)
    signal[indicator > 1.5] = 1
    signal[indicator < -1.5] = -1

    # Fill forward signal to simulate holding
    signal = pd.Series(signal).replace(0, np.nan).ffill().fillna(0).to_numpy()
    df["position"] = sig_pos(pd.Series(signal, index=df.index))

    return df


def test_parity_simple_positions() -> None:
    """Compare old and new implementations on a basic position series."""
    df = _make_test_data(seed=42)

    # 1. Run old perf
    # old perf expects a Series of prices and a Series of positions
    res_old = perf_old(df["open"], df["position"], slippage=1.5)

    # 2. Run new perf
    # new perf expects a TransactionFrame created via no_stop
    tx = no_stop(df, price_column="open")
    res_new = perf_new(tx, slippage=1.5, use_numba=True)

    # 3. Compare basic totals
    assert len(res_old.positions) == len(res_new.positions)

    old_total_pnl = res_old.positions.pnl.sum()
    new_total_pnl = res_new.positions.pnl.sum()
    np.testing.assert_allclose(old_total_pnl, new_total_pnl, rtol=1e-5, atol=1e-5)

    # 4. Compare daily returns
    np.testing.assert_allclose(
        np.asarray(res_old.daily["returns"], dtype=float),
        np.asarray(res_new.daily["returns"], dtype=float),
        rtol=1e-5,
        atol=1e-5,
    )


def test_parity_always_on_reversals() -> None:
    """Compare on an always-on strategy that flips from long to short immediately."""
    df = _make_test_data(seed=101)

    # Force always-on: +1 or -1 only (no flats)
    always_on_signal = np.where(np.random.randn(len(df)) > 0, 1, -1)
    df["position"] = sig_pos(pd.Series(always_on_signal, index=df.index))

    res_old = perf_old(df["open"], df["position"], slippage=1.5)

    tx = no_stop(df, price_column="open")
    res_new = perf_new(tx, slippage=1.5, use_numba=True)

    old_total_pnl = res_old.positions.pnl.sum()
    new_total_pnl = res_new.positions.pnl.sum()
    np.testing.assert_allclose(old_total_pnl, new_total_pnl, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "tx",
    [
        _tx_frame(
            [99.0, 100.0, 101.0],
            [0, 0, 0],
            [0.0, 100.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, -95.0, 0.0],
        ),
        _tx_frame(
            [99.0, 100.0, 101.0],
            [0, 0, 0],
            [0.0, -100.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 105.0, 0.0],
        ),
        _tx_frame(
            [99.0, 100.0, 100.0],
            [0, 1, 0],
            [0.0, 100.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -95.0],
        ),
        _tx_frame(
            [101.0, 100.0, 100.0],
            [0, -1, 0],
            [0.0, -100.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 105.0],
        ),
        _tx_frame(
            [99.0, 100.0, 102.0, 101.0],
            [0, 1, 0, 0],
            [0.0, 100.0, -102.0, 0.0],
            [0.0, 0.0, -102.0, 0.0],
            [0.0, 0.0, 107.0, 0.0],
        ),
        _tx_frame(
            [101.0, 100.0, 98.0, 99.0],
            [0, -1, 0, 0],
            [0.0, -100.0, 98.0, 0.0],
            [0.0, 0.0, 98.0, 0.0],
            [0.0, 0.0, -93.0, 0.0],
        ),
        _tx_frame(
            [99.0, 100.0],
            [0, 0],
            [0.0, 100.0],
            [0.0, 0.0],
            [0.0, -95.0],
        ),
    ],
)
def test_parity_stop_transaction_frames_with_slippage(tx: pd.DataFrame) -> None:
    _assert_old_new_perf_equal(tx, slippage=1.0)


@pytest.mark.parametrize(
    "stop_kwargs",
    [
        {"mode": "fixed"},
        {"mode": "trail"},
        {"mode": "fixed", "tp_multiple": 1.0},
        {"mode": "trail", "time_stop": 3},
        {"mode": "fixed", "adjust": ("trail", 1.0, 0.5)},
    ],
)
def test_parity_stop_loss_generated_transaction_frames(stop_kwargs: dict) -> None:
    df = _make_test_data(seed=303, length=300)
    df.index.name = "date"
    tx = stop_loss(
        df,
        distance=pd.Series(0.15, index=df.index),
        price_column="open",
        **stop_kwargs,
    )

    _assert_old_new_perf_equal(tx, slippage=0.0)
