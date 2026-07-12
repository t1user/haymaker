"""Formula tests for backtester performance metrics."""

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from haymaker.research.backtester.metrics import (
    build_performance_frames,
    build_stats,
    reporting_dates,
    return_stream_metrics,
)


def _bar_frame(pnl: list[float], index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "position": 0,
            "open_price": 0.0,
            "close_price": 0.0,
            "stop_price": 0.0,
            "pnl": pnl,
            "g_pnl": pnl,
        },
        index=index,
    )


def _empty_positions() -> pd.DataFrame:
    return pd.DataFrame(
        columns=("date_o", "open", "date_c", "close", "g_pnl", "pnl", "duration")
    )


def test_session_returns_use_beginning_equity_not_bar_compounding() -> None:
    index = pd.to_datetime(
        [
            "2026-01-05 09:00",
            "2026-01-05 10:00",
            "2026-01-06 09:00",
            "2026-01-06 10:00",
        ]
    )
    bar_df, daily = build_performance_frames(
        _bar_frame([10.0, -5.0, 20.0, -10.0], index),
        capital=100.0,
        sunday_to_monday=True,
    )

    expected = pd.DataFrame(
        {
            "pnl": [5.0, 10.0],
            "returns": [0.05, 10.0 / 105.0],
            "lreturn": [np.log1p(0.05), np.log1p(10.0 / 105.0)],
            "equity": [105.0, 115.0],
            "balance": [1.05, 1.15],
            "fixed_return": [0.05, 0.10],
        },
        index=pd.DatetimeIndex(["2026-01-05", "2026-01-06"], name="session"),
    )
    pdt.assert_frame_equal(daily, expected)
    assert bar_df["equity"].tolist() == [110.0, 105.0, 125.0, 115.0]
    assert np.prod(1.0 + daily["returns"]) == pytest.approx(1.15)


def test_return_result_is_invariant_to_bar_pnl_partitioning() -> None:
    one_bar = _bar_frame([6.0, -2.0], pd.to_datetime(["2026-01-05", "2026-01-06"]))
    many_bars = _bar_frame(
        [1.0, 2.0, 3.0, -3.0, 1.0],
        pd.to_datetime(
            [
                "2026-01-05 09:00",
                "2026-01-05 10:00",
                "2026-01-05 11:00",
                "2026-01-06 09:00",
                "2026-01-06 10:00",
            ]
        ),
    )

    _, one_daily = build_performance_frames(
        one_bar, capital=100.0, sunday_to_monday=True
    )
    _, many_daily = build_performance_frames(
        many_bars, capital=100.0, sunday_to_monday=True
    )

    pdt.assert_frame_equal(one_daily, many_daily)


def test_sunday_is_optionally_combined_with_monday_without_synthetic_dates() -> None:
    index = pd.to_datetime(["2026-01-02 12:00", "2026-01-04 18:00", "2026-01-05 12:00"])
    combined = reporting_dates(index, sunday_to_monday=True)
    separate = reporting_dates(index, sunday_to_monday=False)

    assert combined.tolist() == list(
        pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-05"])
    )
    assert separate.tolist() == list(
        pd.to_datetime(["2026-01-02", "2026-01-04", "2026-01-05"])
    )

    _, daily = build_performance_frames(
        _bar_frame([1.0, 2.0, 3.0], index),
        capital=100.0,
        sunday_to_monday=True,
    )
    assert daily.index.tolist() == list(pd.to_datetime(["2026-01-02", "2026-01-05"]))
    assert daily["pnl"].tolist() == [1.0, 5.0]


def test_drawdown_uses_intraday_equity_and_constant_fixed_denominator() -> None:
    index = pd.to_datetime(["2026-01-05 09:00", "2026-01-05 10:00", "2026-01-05 11:00"])
    bar_df, daily = build_performance_frames(
        _bar_frame([10.0, -20.0, 15.0], index),
        capital=100.0,
        sunday_to_monday=True,
    )
    warnings: list[str] = []
    stats = build_stats(
        _empty_positions(),
        daily,
        bar_df,
        capital=100.0,
        min_tick=1.0,
        warnings=warnings,
    )

    assert stats["max_drawdown_pnl"] == -20.0
    assert stats["max_drawdown"] == pytest.approx(-20.0 / 110.0)
    assert stats["fixed_max_drawdown"] == -0.20
    assert daily["pnl"].iloc[0] == 5.0


def test_nonpositive_equity_keeps_fixed_metrics_and_warns() -> None:
    index = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
    bar_df, daily = build_performance_frames(
        _bar_frame([-110.0, 20.0, 10.0], index),
        capital=100.0,
        sunday_to_monday=True,
    )
    warnings: list[str] = []
    stats = build_stats(
        _empty_positions(),
        daily,
        bar_df,
        capital=100.0,
        min_tick=1.0,
        warnings=warnings,
    )

    assert stats["total_return"] == -0.8
    assert stats["fixed_annual_return"] == pytest.approx((-80.0 / 3) * 252 / 100)
    assert np.isnan(stats["annual_return"])
    assert daily["returns"].iloc[0] == -1.1
    assert daily["returns"].iloc[1:].isna().all()
    assert any("nonpositive" in warning for warning in warnings)


def test_trade_metrics_use_strict_wins_losses_and_point_pnl_run_rate() -> None:
    dates = pd.date_range("2026-01-05", periods=3, freq="D")
    bar_df, daily = build_performance_frames(
        _bar_frame([10.0, -5.0, 0.0], dates),
        capital=100.0,
        sunday_to_monday=True,
    )
    positions = pd.DataFrame(
        {
            "date_o": dates,
            "open": [100.0, -100.0, 100.0],
            "date_c": dates,
            "close": [-110.0, 105.0, -100.0],
            "g_pnl": [10.0, -5.0, 0.0],
            "pnl": [10.0, -5.0, 0.0],
            "duration": pd.to_timedelta([1, 2, 3], unit="h"),
        }
    )
    stats = build_stats(
        positions,
        daily,
        bar_df,
        capital=100.0,
        min_tick=0.25,
        warnings=[],
    )

    assert stats["trade_count"] == 3
    assert stats["win_rate"] == 1 / 3
    assert stats["avg_win"] == 10.0
    assert stats["avg_loss"] == -5.0
    assert stats["trade_expectancy"] == 5.0 / 3
    assert stats["trade_expectancy_ticks"] == (5.0 / 3) / 0.25
    assert stats["monthly_pnl"] == daily["pnl"].mean() * 21
    assert stats["net_pnl_ex_best"] == -5.0


def test_complete_21_session_window_has_one_monthly_drawdown() -> None:
    index = pd.bdate_range("2026-01-05", periods=21)
    pnl = [10.0, -20.0] + [0.0] * 19
    bar_df, daily = build_performance_frames(
        _bar_frame(pnl, index), capital=100.0, sunday_to_monday=True
    )
    stats = build_stats(
        _empty_positions(),
        daily,
        bar_df,
        capital=100.0,
        min_tick=1.0,
        warnings=[],
    )

    assert stats["median_21d_drawdown"] == pytest.approx(-20.0 / 110.0)
    assert stats["fixed_median_21d_drawdown"] == pytest.approx(-0.20)


def test_return_stream_metrics_match_empyrical_core_formulas() -> None:
    from empyrical import (  # type: ignore[import-untyped]
        annual_return,
        annual_volatility,
        sharpe_ratio,
    )

    returns = pd.Series([0.01, -0.02, 0.005, 0.03, -0.01])
    stats = return_stream_metrics(returns)

    expected_total = float(np.prod(1.0 + returns.to_numpy()) - 1.0)
    assert stats["total_return"] == pytest.approx(expected_total)
    assert stats["annual_return"] == pytest.approx(
        annual_return(returns, annualization=252)
    )
    assert stats["annual_volatility"] == pytest.approx(
        annual_volatility(returns, annualization=252)
    )
    assert stats["sharpe_ratio"] == pytest.approx(
        sharpe_ratio(returns, annualization=252)
    )
