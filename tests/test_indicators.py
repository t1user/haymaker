from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from haymaker.indicators import (
    adx,
    breakout,
    chande_momentum_indicator,
    downsampled_func,
    extreme_reversal_blip,
    min_max_blip,
    min_max_buffer_signal,
    range_blip,
)


def _ohlc(close: list[float]) -> pd.DataFrame:
    index = pd.date_range("2026-01-01 09:00", periods=len(close), freq="min")
    price = pd.Series(close, index=index)
    return pd.DataFrame(
        {
            "open": price,
            "high": price + 0.5,
            "low": price - 0.5,
            "close": price,
        }
    )


def test_min_max_blip_uses_previous_rolling_window() -> None:
    price = pd.Series([10.0, 11.0, 12.0, 11.0, 9.0, 13.0])

    actual = min_max_blip(price, period=3)

    expected = pd.Series([0, 0, 0, 0, -1, 1])
    pd.testing.assert_series_equal(actual, expected)


def test_min_max_buffer_signal_requires_move_beyond_buffer() -> None:
    price = pd.Series([10.0, 11.0, 12.0, 12.4, 13.0, 9.0])

    actual = min_max_buffer_signal(price, period=3, buff=0.5)

    expected = pd.Series([0, 0, 0, 0, 1, -1])
    pd.testing.assert_series_equal(actual, expected, check_names=False)


def test_breakout_opposite_blip_closes_by_default() -> None:
    price = pd.Series([1.0, 2.0, 3.0, 2.0, 1.0])

    actual = breakout(price, lookback=1, stop_frac=1)

    expected = pd.Series([0, 1, 1, 0, -1], dtype=np.int8, name="break")
    pd.testing.assert_series_equal(actual, expected)


def test_breakout_always_on_reverses_on_opposite_blip() -> None:
    price = pd.Series([1.0, 2.0, 3.0, 2.0, 1.0])

    actual = breakout(price, lookback=1, stop_frac=1, always_on=True)

    expected = pd.Series([0, 1, 1, -1, -1], dtype=np.int8, name="break")
    pd.testing.assert_series_equal(actual, expected)


def test_adx_detects_directional_movement_in_downtrend() -> None:
    data = pd.DataFrame(
        {
            "open": [10.0, 9.0, 8.0, 7.0, 6.0],
            "high": [10.5, 9.5, 8.5, 7.5, 6.5],
            "low": [9.5, 8.5, 7.5, 6.5, 5.5],
            "close": [10.0, 9.0, 8.0, 7.0, 6.0],
        }
    )

    actual = adx(data, lookback=2)

    assert actual.dropna().iloc[-1] == pytest.approx(100.0)


def test_chande_momentum_indicator_uses_absolute_losses_in_denominator() -> None:
    price = pd.Series([1.0, 2.0, 3.0, 2.0, 1.0])

    actual = chande_momentum_indicator(price, lookback=2)

    expected = pd.Series([np.nan, np.nan, 100.0, 0.0, -100.0])
    pd.testing.assert_series_equal(actual, expected, check_names=False)


def test_extreme_reversal_blip_marks_reentry_from_extreme_zones() -> None:
    indicator = pd.Series([2.0, 1.0, 0.5, -2.0, -1.0, -0.5, 2.0])

    actual = extreme_reversal_blip(indicator, threshold=1.0)

    expected = pd.Series([0, -1, 0, 0, 1, 0, 0])
    pd.testing.assert_series_equal(actual, expected)


def test_range_blip_inside_signals_entries_into_range() -> None:
    indicator = pd.Series([2.0, 0.5, 0.25, -2.0, -0.5, 0.0, 2.0])

    actual = range_blip(indicator, threshold=1.0, inout="inside")

    expected = pd.Series([0, 1, 0, 0, -1, 0, 0])
    pd.testing.assert_series_equal(actual, expected)


def test_range_blip_outside_signals_entries_outside_range() -> None:
    indicator = pd.Series([0.0, 2.0, 0.0, -2.0, 0.0])

    actual = range_blip(indicator, threshold=1.0, inout="outside")

    expected = pd.Series([0, 1, 0, -1, 0])
    pd.testing.assert_series_equal(actual, expected)


def test_downsampled_func_first_exposes_value_when_resampled_bar_completes() -> None:
    data = _ohlc([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

    actual = downsampled_func(data, "3min", lambda frame: frame["close"])

    assert actual.iloc[:3].isna().all()
    assert actual.iloc[3:].tolist() == [2.0, 2.0, 2.0]


def test_downsampled_func_hourly_indicator_is_filled_over_next_hour() -> None:
    index = pd.date_range("2026-01-01 09:00", periods=181, freq="min")
    close = pd.Series(np.arange(len(index), dtype=float), index=index)
    data = pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
        }
    )

    actual = downsampled_func(
        data,
        "1h",
        lambda hourly: hourly["close"].rolling(2).mean(),
    )

    assert actual.loc[: pd.Timestamp("2026-01-01 10:59")].isna().all()
    assert (
        actual.loc[
            pd.Timestamp("2026-01-01 11:00") : pd.Timestamp("2026-01-01 11:59")
        ]
        == 89.0
    ).all()
    assert actual.loc[pd.Timestamp("2026-01-01 12:00")] == 149.0
