"""
Tests for haymaker.research.backtester and the refactored stop_loss output.

Coverage:
- no_stop(): position path, blip path, error on missing columns
- _perf_engine vs _perf_engine_python: identical results
- perf(): PnL parity against old backtester/vector_backtester.perf() across scenarios
- stop_loss() now emits bar_price column
- _TransactionFrame: rejects missing columns
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from haymaker.research.backtester import Results, _TransactionFrame, no_stop, perf
from haymaker.research.stop import stop_loss
from haymaker.research.backtester.engine import _perf_engine, _perf_engine_python

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IDX = pd.date_range("2020-01-01", periods=10, freq="h")


def _make_price(n: int = 10, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(
        100.0 + np.cumsum(rng.normal(0, 1.0, n)),
        index=pd.date_range("2020-01-01", periods=n, freq="h"),
        name="open",
    )


def _simple_df(n: int = 20, seed: int = 1) -> pd.DataFrame:
    """OHLC frame with a simple long-only position."""
    rng = np.random.default_rng(seed)
    price = 100.0 + np.cumsum(rng.normal(0, 1.0, n))
    df = pd.DataFrame(
        {
            "open": price,
            "high": price + rng.uniform(0.1, 1.5, n),
            "low": price - rng.uniform(0.1, 1.5, n),
        },
        index=pd.date_range("2020-01-01", periods=n, freq="h"),
    )
    pos = np.zeros(n, dtype=int)
    pos[2:8] = 1
    pos[12:17] = -1
    df["position"] = pos
    return df


# ---------------------------------------------------------------------------
# _TransactionFrame validation
# ---------------------------------------------------------------------------


def test_transaction_frame_rejects_missing_columns() -> None:
    df = pd.DataFrame({"bar_price": [1.0], "open_price": [1.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        _TransactionFrame(df)


def test_transaction_frame_accepts_valid_df() -> None:
    df = pd.DataFrame(
        {
            "bar_price": [100.0],
            "open_price": [100.0],
            "close_price": [0.0],
            "stop_price": [0.0],
            "position": [1],
        }
    )
    _TransactionFrame(df)  # must not raise


# ---------------------------------------------------------------------------
# no_stop – position path
# ---------------------------------------------------------------------------


def test_no_stop_position_path_schema() -> None:
    df = _simple_df()
    out = no_stop(df, price_column="open")
    required = {"bar_price", "open_price", "close_price", "stop_price", "position"}
    assert required.issubset(set(out.columns))


def test_no_stop_position_path_stop_price_always_zero() -> None:
    df = _simple_df()
    out = no_stop(df, price_column="open")
    assert (out["stop_price"] == 0).all()


def test_no_stop_position_path_open_only_on_transition() -> None:
    df = _simple_df()
    out = no_stop(df, price_column="open")
    # open_price non-zero only where position changes from anything to non-zero
    has_open = out["open_price"] != 0
    pos = out["position"]
    prev = pos.shift().fillna(0).astype(int)
    opened = (pos != prev) & (pos != 0)
    assert (has_open == opened).all()


def test_no_stop_position_path_close_only_on_transition_to_flat() -> None:
    df = _simple_df()
    out = no_stop(df, price_column="open")
    has_close = out["close_price"] != 0
    pos = out["position"]
    prev = pos.shift().fillna(0).astype(int)
    closed = (pos != prev) & (pos == 0)
    # reversals also set close; this checks the flat case only
    # (reversals will have both open and close set)
    flat_closes = has_close & (pos == 0)
    assert (flat_closes == closed).all()


def test_no_stop_open_close_mutually_exclusive_per_bar_on_simple_path() -> None:
    """For a non-reversing strategy, open and close cannot both be non-zero."""
    df = _simple_df()
    out = no_stop(df, price_column="open")
    # Simple df has no reversals (goes flat between positions)
    both = (out["open_price"] != 0) & (out["close_price"] != 0)
    assert not both.any()


def test_no_stop_reversal_sets_both_open_and_close() -> None:
    """Reversal bar (position -1→+1) must have both open_price and close_price set."""
    n = 6
    df = pd.DataFrame(
        {
            "open": [100.0] * n,
        },
        index=pd.date_range("2020-01-01", periods=n, freq="h"),
    )
    df["position"] = [0, 1, 1, -1, -1, 0]  # bar 3 is a reversal
    out = no_stop(df, price_column="open")
    reversal_bar = out.iloc[3]
    assert reversal_bar["open_price"] != 0
    assert reversal_bar["close_price"] != 0


# ---------------------------------------------------------------------------
# no_stop – blip path
# ---------------------------------------------------------------------------


def test_no_stop_blip_path_schema() -> None:
    n = 10
    df = pd.DataFrame(
        {
            "open": np.linspace(100, 110, n),
            "blip": [0, 1, 0, 0, -1, 0, 1, 0, -1, 0],
        },
        index=pd.date_range("2020-01-01", periods=n, freq="h"),
    )
    out = no_stop(df, price_column="open")
    required = {"bar_price", "open_price", "close_price", "stop_price", "position"}
    assert required.issubset(set(out.columns))


def test_no_stop_raises_without_position_or_blip() -> None:
    df = pd.DataFrame(
        {"open": [100.0, 101.0]},
        index=pd.date_range("2020-01-01", periods=2, freq="h"),
    )
    with pytest.raises(ValueError, match="'position' column or a 'blip' column"):
        no_stop(df, price_column="open")


def test_no_stop_raises_on_missing_price_column() -> None:
    df = pd.DataFrame(
        {"position": [0, 1, 0]},
        index=pd.date_range("2020-01-01", periods=3, freq="h"),
    )
    with pytest.raises(ValueError, match="'close' indicated as price_column"):
        no_stop(df, price_column="close")


# ---------------------------------------------------------------------------
# _perf_engine vs _perf_engine_python parity
# ---------------------------------------------------------------------------


def _make_engine_inputs(
    n: int = 15, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    price = 100.0 + np.cumsum(rng.normal(0, 1.0, n))
    open_p = np.zeros(n)
    close_p = np.zeros(n)
    stop_p = np.zeros(n)
    # Manually set a few trades
    open_p[2] = price[2]  # long entry
    close_p[5] = -price[5]  # normal close
    open_p[7] = -price[7]  # short entry
    stop_p[9] = price[9]  # stop exit
    return price, open_p, close_p, stop_p, 0.5


@pytest.mark.parametrize("seed", [0, 7, 42])
def test_numba_and_python_engines_produce_identical_results(seed: int) -> None:
    price, op, cp, sp, cost = _make_engine_inputs(seed=seed)
    lret_nb, pnl_nb, trades_nb = _perf_engine(price, op, cp, sp, cost)
    lret_py, pnl_py, trades_py = _perf_engine_python(price, op, cp, sp, cost)
    np.testing.assert_allclose(lret_nb, lret_py, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(pnl_nb, pnl_py, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(trades_nb, trades_py, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# perf() – smoke test and basic sanity
# ---------------------------------------------------------------------------


def test_perf_returns_results_namedtuple() -> None:
    df = _simple_df()
    tx = no_stop(df, price_column="open")
    result = perf(tx, slippage=1.5, use_numba=False)
    assert isinstance(result, Results)
    assert isinstance(result.stats, pd.Series)
    assert isinstance(result.daily, pd.DataFrame)
    assert isinstance(result.positions, pd.DataFrame)


def test_perf_positions_pnl_sum_matches_bar_df_pnl_sum() -> None:
    """positions PnL and bar-level PnL must agree (pure MtM guarantees this)."""
    df = _simple_df()
    tx = no_stop(df, price_column="open")
    result = perf(tx, slippage=1.5, use_numba=False)
    diff = abs(result.positions.pnl.sum() - result.df.pnl.sum())
    assert (
        diff < 1e-8
    ), f"PnL mismatch: positions={result.positions.pnl.sum():.6f}, bar_df={result.df.pnl.sum():.6f}"


def test_perf_no_positions_emits_warning() -> None:
    n = 10
    df = pd.DataFrame(
        {
            "open": np.linspace(100, 110, n),
            "position": np.zeros(n, dtype=int),
        },
        index=pd.date_range("2020-01-01", periods=n, freq="h"),
    )
    tx = no_stop(df, price_column="open")
    result = perf(tx, slippage=1.5, use_numba=False)
    assert any("No positions" in w for w in result.warnings)


def test_perf_numba_and_python_produce_same_positions_pnl() -> None:
    df = _simple_df(n=30, seed=5)
    tx = no_stop(df, price_column="open")
    r_nb = perf(tx, slippage=1.5, use_numba=True)
    r_py = perf(tx, slippage=1.5, use_numba=False)
    np.testing.assert_allclose(
        np.asarray(r_nb.positions.pnl, dtype=float),
        np.asarray(r_py.positions.pnl, dtype=float),
        rtol=1e-10,
        atol=1e-10,
    )


# ---------------------------------------------------------------------------
# stop_loss now emits bar_price
# ---------------------------------------------------------------------------


def test_stop_loss_output_has_bar_price_column() -> None:
    n = 10
    rng = np.random.default_rng(3)
    price = 100.0 + np.cumsum(rng.normal(0, 1.0, n))
    df = pd.DataFrame(
        {
            "open": price,
            "high": price + 0.5,
            "low": price - 0.5,
            "position": [0, 1, 1, 1, 0, -1, -1, 0, 0, 0],
        },
        index=pd.date_range("2020-01-01", periods=n, freq="h"),
    )
    result = stop_loss(df, distance=1.0, mode="fixed")
    assert "bar_price" in result.columns
    pdt.assert_series_equal(
        result["bar_price"],
        df["open"].rename("bar_price"),
        check_names=True,
    )


def test_stop_loss_output_accepted_by_perf() -> None:
    """stop_loss output can be passed directly to new perf() without error."""
    n = 20
    rng = np.random.default_rng(7)
    price = 100.0 + np.cumsum(rng.normal(0, 1.0, n))
    df = pd.DataFrame(
        {
            "open": price,
            "high": price + rng.uniform(0.1, 1.5, n),
            "low": price - rng.uniform(0.1, 1.5, n),
            "position": ([0, 1, 1, 1, 0, -1, -1, -1, 0, 0] * 2),
        },
        index=pd.date_range("2020-01-01", periods=n, freq="h"),
    )
    sl_out = stop_loss(df, distance=1.0, mode="trail")
    result = perf(sl_out, slippage=1.5, use_numba=False)
    assert isinstance(result, Results)
    assert len(result.positions) > 0


# ---------------------------------------------------------------------------
# PnL parity: new perf vs old perf (simple position path)
# ---------------------------------------------------------------------------


def test_pnl_parity_simple_long_only() -> None:
    """New perf gross PnL must be close to old perf gross PnL for simple strategy."""
    from haymaker.research.backtester.vector_backtester import perf as old_perf

    n = 40
    rng = np.random.default_rng(99)
    price = 100.0 + np.cumsum(rng.normal(0, 1.0, n))
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    price_s = pd.Series(price, index=idx, name="open")

    pos_arr = np.zeros(n, dtype=int)
    pos_arr[3:12] = 1
    pos_arr[18:28] = 1
    position = pd.Series(pos_arr, index=idx)

    df = pd.DataFrame({"open": price_s, "position": position})
    tx = no_stop(df, price_column="open")

    new_result = perf(tx, slippage=0, use_numba=False)
    old_result = old_perf(price_s, position, slippage=0)

    new_gpnl = new_result.positions.g_pnl.sum()
    old_gpnl = old_result.positions.g_pnl.sum()

    assert (
        abs(new_gpnl - old_gpnl) < 0.5
    ), f"Gross PnL divergence: new={new_gpnl:.4f}, old={old_gpnl:.4f}"


def test_pnl_parity_always_on() -> None:
    """Reversing (always-on) strategy: new and old gross PnL within tolerance."""
    from haymaker.research.backtester.vector_backtester import perf as old_perf

    n = 30
    rng = np.random.default_rng(13)
    price = 100.0 + np.cumsum(rng.normal(0, 1.0, n))
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    price_s = pd.Series(price, index=idx, name="open")

    # Always-on: no zeros, only -1 and 1
    pos_arr = np.where(rng.random(n) > 0.5, 1, -1).astype(int)
    pos_arr[0] = 0  # start flat
    position = pd.Series(pos_arr, index=idx)

    df = pd.DataFrame({"open": price_s, "position": position})
    tx = no_stop(df, price_column="open")

    new_result = perf(tx, slippage=0, use_numba=False)
    old_result = old_perf(price_s, position, slippage=0)

    new_gpnl = new_result.positions.g_pnl.sum()
    old_gpnl = old_result.positions.g_pnl.sum()

    assert (
        abs(new_gpnl - old_gpnl) < 0.5
    ), f"Always-on gross PnL divergence: new={new_gpnl:.4f}, old={old_gpnl:.4f}"
