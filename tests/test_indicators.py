from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

import haymaker.indicators as indicator_facade
from haymaker.research.indicators import (
    adx,
    atr,
    breakout,
    chande_momentum_indicator,
    chande_ranking,
    combine_signals,
    crosser,
    divergence_index,
    downsampled_func,
    extreme_reversal_blip,
    inout_range,
    macd,
    mmean,
    momentum,
    min_max_blip,
    min_max_index,
    range_blip,
    resample,
    rolling_weighted_mean,
    rolling_weighted_std,
    rsi,
    signal_generator,
    strength_oscillator,
    true_range,
    tsi,
    weighted_resample,
    weighted_zscore,
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


def test_legacy_indicators_facade_reexports_new_modules() -> None:
    assert indicator_facade.atr is atr
    assert indicator_facade.combine_signals is combine_signals


def test_min_max_blip_uses_previous_rolling_window() -> None:
    price = pd.Series([10.0, 11.0, 12.0, 11.0, 9.0, 13.0])

    actual = min_max_blip(price, period=3)

    expected = pd.Series([0, 0, 0, 0, -1, 1])
    pd.testing.assert_series_equal(actual, expected)


def test_min_max_blip_requires_move_beyond_scalar_buffer() -> None:
    price = pd.Series([10.0, 11.0, 12.0, 12.4, 13.0, 9.0])

    actual = min_max_blip(price, period=3, buff=0.5)

    expected = pd.Series([0, 0, 0, 0, 1, -1])
    pd.testing.assert_series_equal(actual, expected, check_names=False)


def test_min_max_blip_accepts_same_index_series_buffer() -> None:
    price = pd.Series([10.0, 11.0, 12.0, 12.4, 13.0, 9.0])
    buff = pd.Series([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], index=price.index)

    actual = min_max_blip(price, period=3, buff=buff)

    expected = pd.Series([0, 0, 0, 0, 1, -1])
    pd.testing.assert_series_equal(actual, expected, check_names=False)


def test_min_max_blip_rejects_series_buffer_with_different_index() -> None:
    price = pd.Series([10.0, 11.0, 12.0])
    buff = pd.Series([0.5, 0.5, 0.5], index=[1, 2, 3])

    with pytest.raises(ValueError, match="buff index"):
        min_max_blip(price, period=2, buff=buff)


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


def test_atr_default_remains_span_based_exponential_mean() -> None:
    data = pd.DataFrame(
        {
            "open": [0.0, 0.0, 0.0, 0.0],
            "high": [1.0, 2.0, 3.0, 4.0],
            "low": [0.0, 0.0, 0.0, 0.0],
            "close": [0.0, 0.0, 0.0, 0.0],
        }
    )

    actual = atr(data, periods=3)

    expected = pd.Series([1.0, 1.6666666667, 2.4285714286, 3.2666666667], name="ATR")
    pd.testing.assert_series_equal(actual, expected, check_exact=False, rtol=1e-10)


def test_atr_can_use_wilder_average_off_smoothing() -> None:
    data = pd.DataFrame(
        {
            "open": [0.0, 0.0, 0.0, 0.0, 0.0],
            "high": [1.0, 2.0, 3.0, 4.0, 5.0],
            "low": [0.0, 0.0, 0.0, 0.0, 0.0],
            "close": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )

    actual = atr(data, periods=3, smooth_type="wilder")

    expected = pd.Series([np.nan, np.nan, 2.0, 2.6666666667, 3.4444444444], name="ATR")
    pd.testing.assert_series_equal(actual, expected, check_exact=False, rtol=1e-10)


def test_atr_can_use_simple_smoothing() -> None:
    data = pd.DataFrame(
        {
            "open": [0.0, 0.0, 0.0, 0.0],
            "high": [1.0, 2.0, 3.0, 4.0],
            "low": [0.0, 0.0, 0.0, 0.0],
            "close": [0.0, 0.0, 0.0, 0.0],
        }
    )

    actual = atr(data, periods=3, smooth_type="simple")

    expected = pd.Series([np.nan, np.nan, 2.0, 3.0], name="ATR")
    pd.testing.assert_series_equal(actual, expected)


def test_mmean_rejects_invalid_periods() -> None:
    with pytest.raises(ValueError, match="periods"):
        mmean(pd.Series([1.0, 2.0]), periods=0)


def test_true_range_can_use_multi_bar_comparison() -> None:
    data = pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0, 13.0],
            "high": [11.0, 12.0, 14.0, 13.5],
            "low": [9.0, 10.5, 11.0, 12.0],
            "close": [10.0, 11.5, 12.5, 13.0],
        }
    )

    actual = true_range(data, bar=2)

    expected = pd.Series([np.nan, 3.0, 4.0, 3.0], name="TR")
    pd.testing.assert_series_equal(actual, expected)


def test_chande_momentum_indicator_uses_absolute_losses_in_denominator() -> None:
    price = pd.Series([1.0, 2.0, 3.0, 2.0, 1.0])

    actual = chande_momentum_indicator(price, lookback=2)

    expected = pd.Series([np.nan, np.nan, 100.0, 0.0, -100.0])
    pd.testing.assert_series_equal(actual, expected, check_names=False)


def test_chande_ranking_normalizes_lookback_log_return_by_volatility() -> None:
    price = pd.Series([100.0, 103.0, 101.0, 108.0, 111.0])

    actual = chande_ranking(price, lookback=2)

    one_period_returns = (price.pct_change() + 1).apply(np.log)
    expected = (price.pct_change(2) + 1).apply(np.log) / (
        one_period_returns.rolling(2).std() * np.sqrt(2)
    )
    pd.testing.assert_series_equal(actual, expected)


def test_rsi_uses_wilder_average_off_smoothing() -> None:
    price = pd.Series([1.0, 2.0, 3.0, 2.0, 4.0, 3.0])

    actual = rsi(price, lookback=3)

    expected = pd.Series(
        [np.nan, np.nan, np.nan, 66.6666666667, 83.3333333333, 60.6060606061],
        name="rsi",
    )
    pd.testing.assert_series_equal(actual, expected, check_exact=False, rtol=1e-10)


def test_rsi_can_be_rescaled_to_symmetric_oscillator() -> None:
    price = pd.Series([1.0, 2.0, 3.0, 2.0, 4.0, 3.0])

    actual = rsi(price, lookback=3, rescale=True)

    expected = pd.Series(
        [np.nan, np.nan, np.nan, 0.3333333333, 0.6666666667, 0.2121212121],
        name="rsi",
    )
    pd.testing.assert_series_equal(actual, expected, check_exact=False, rtol=1e-10)


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
        actual.loc[pd.Timestamp("2026-01-01 11:00") : pd.Timestamp("2026-01-01 11:59")]
        == 89.0
    ).all()
    assert actual.loc[pd.Timestamp("2026-01-01 12:00")] == 149.0


def test_resample_aggregates_ohlc_volume_barcount_and_custom_fields() -> None:
    index = pd.date_range("2026-01-01 09:00", periods=4, freq="min")
    data = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0, 4.0],
            "high": [2.0, 3.0, 5.0, 6.0],
            "low": [0.5, 1.5, 2.5, 3.5],
            "close": [1.5, 2.5, 3.5, 4.5],
            "volume": [10.0, 20.0, 30.0, 40.0],
            "barCount": [1, 2, 3, 4],
            "average": [1.25, 2.25, 3.25, 4.25],
        },
        index=index,
    )

    actual = resample(data, "2min", how={"average": "mean"})

    expected = pd.DataFrame(
        {
            "open": [1.0, 3.0],
            "high": [3.0, 6.0],
            "low": [0.5, 2.5],
            "close": [2.5, 4.5],
            "volume": [30.0, 70.0],
            "barCount": [3, 7],
            "average": [1.75, 3.75],
        },
        index=pd.date_range("2026-01-01 09:00", periods=2, freq="2min"),
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_resample_series_uses_last_value_and_drops_empty_groups() -> None:
    index = pd.to_datetime(["2026-01-01 09:00", "2026-01-01 09:01", "2026-01-01 09:04"])
    price = pd.Series([1.0, 2.0, 5.0], index=index)

    actual = cast(pd.Series, resample(price, "2min"))

    expected = pd.Series(
        [2.0, 5.0],
        index=pd.to_datetime(["2026-01-01 09:00", "2026-01-01 09:04"]),
    )
    pd.testing.assert_series_equal(actual, expected)


def test_weighted_resample_keeps_zero_volume_average_missing() -> None:
    index = pd.date_range("2026-01-01 09:00", periods=4, freq="min")
    data = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0, 4.0],
            "high": [1.0, 2.0, 3.0, 4.0],
            "low": [1.0, 2.0, 3.0, 4.0],
            "close": [1.0, 2.0, 3.0, 4.0],
            "volume": [0.0, 0.0, 10.0, 30.0],
            "average": [1.0, 2.0, 3.0, 5.0],
        },
        index=index,
    )

    actual = weighted_resample(data, "2min")

    assert pd.isna(actual.loc[pd.Timestamp("2026-01-01 09:00"), "average"])
    assert actual.loc[pd.Timestamp("2026-01-01 09:02"), "average"] == 4.5


def test_weighted_resample_respects_resampling_label_and_closed_kwargs() -> None:
    index = pd.date_range("2026-01-01 09:00", periods=4, freq="min")
    data = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0, 4.0],
            "high": [1.0, 2.0, 3.0, 4.0],
            "low": [1.0, 2.0, 3.0, 4.0],
            "close": [1.0, 2.0, 3.0, 4.0],
            "volume": [10.0, 30.0, 10.0, 30.0],
            "average": [1.0, 3.0, 5.0, 7.0],
        },
        index=index,
    )

    actual = weighted_resample(data, "2min", label="right", closed="right")

    assert actual.loc[pd.Timestamp("2026-01-01 09:00"), "average"] == 1.0
    assert actual.loc[pd.Timestamp("2026-01-01 09:02"), "average"] == 3.5


def test_rolling_weighted_mean_zero_weight_window_returns_nan() -> None:
    price = pd.Series([1.0, 2.0, 3.0])
    weights = pd.Series([0.0, 0.0, 1.0])

    actual = rolling_weighted_mean(price, weights, periods=2)

    assert np.isnan(actual.iloc[1])
    assert actual.iloc[2] == 3.0


def test_rolling_weighted_mean_rejects_negative_weights() -> None:
    price = pd.Series([1.0, 2.0])
    weights = pd.Series([1.0, -1.0])

    with pytest.raises(ValueError, match="non-negative"):
        rolling_weighted_mean(price, weights, periods=2)


def test_rolling_weighted_std_uses_population_denominator() -> None:
    price = pd.Series([1.0, 3.0])
    weights = pd.Series([1.0, 1.0])

    actual = rolling_weighted_std(price, weights, periods=2)

    assert actual.iloc[1] == pytest.approx(1.0)


def test_weighted_zscore_returns_volume_weighted_indicator() -> None:
    data = pd.DataFrame(
        {
            "close": [1.0, 2.0, 3.0],
            "volume": [1.0, 1.0, 1.0],
        }
    )

    actual = weighted_zscore(data, lookback=2)

    expected = pd.Series([1.0, 1.0], index=[1, 2])
    pd.testing.assert_series_equal(actual, expected)


def test_weighted_zscore_requires_close_and_volume_columns() -> None:
    with pytest.raises(ValueError, match="volume"):
        weighted_zscore(pd.DataFrame({"close": [1.0, 2.0]}), lookback=2)


def test_signal_generator_can_choose_missing_value_policy() -> None:
    indicator = pd.Series([np.nan, -1.0, 0.0, 1.0])

    ignored = signal_generator(indicator)
    dropped = signal_generator(indicator, handle_na="drop")

    pd.testing.assert_series_equal(ignored, pd.Series([0, -1, 0, 1]))
    pd.testing.assert_series_equal(dropped, pd.Series([-1, 0, 1], index=[1, 2, 3]))
    with pytest.raises(ValueError, match="NaN"):
        signal_generator(indicator, handle_na="raise")


def test_signal_helpers_preserve_expected_event_and_filter_semantics() -> None:
    primary = pd.Series([1, -1, 1, -1])
    filter_ = pd.Series([1, 1, 0, -1])
    crossing_indicator = pd.Series([0.0, 0.0, 1.0, 1.0, -1.0, -1.0, 0.0, 1.0])
    range_indicator = pd.Series([-2.0, -0.5, 0.0, 0.5, 2.0])

    combined = combine_signals(primary, filter_)
    zero_crossed = crosser(crossing_indicator)
    inside = inout_range(range_indicator, threshold=-1.0, inout="inside")
    outside = inout_range(range_indicator, threshold=1.0, inout="outside")

    pd.testing.assert_series_equal(combined, pd.Series([1, 0, 0, -1]))
    pd.testing.assert_series_equal(zero_crossed, pd.Series([0, 0, 1, 0, -1, 0, 0, 1]))
    pd.testing.assert_series_equal(
        inside, pd.Series([False, True, True, True, False], name="inside")
    )
    pd.testing.assert_series_equal(
        outside, pd.Series([True, False, False, False, True], name="outside")
    )


def test_crosser_marks_only_side_changes_and_treats_threshold_as_neutral() -> None:
    indicator = pd.Series([-1.0, 0.0, 0.5, -0.1, 0.0])

    actual = crosser(indicator, threshold=0.0)

    expected = pd.Series([0, 0, 1, -1, 0])
    pd.testing.assert_series_equal(actual, expected)


def test_crosser_does_not_turn_missing_values_into_crossings() -> None:
    indicator = pd.Series([-1.0, np.nan, 1.0, -1.0])

    actual = crosser(indicator, threshold=0.0)

    expected = pd.Series([0, 0, 0, -1])
    pd.testing.assert_series_equal(actual, expected)


def test_strength_oscillator_preserves_original_index() -> None:
    data = _ohlc([1.0, 2.0, 3.0, 4.0])

    actual = strength_oscillator(data, periods=3)

    assert actual.index.equals(data.index)
    assert actual.iloc[:2].isna().all()
    assert actual.iloc[-1] == pytest.approx(1.0)


def test_momentum_accepts_separate_smoothing_periods() -> None:
    price = pd.Series([1.0, 2.0, 4.0, 7.0, 11.0])

    actual = momentum(price, periods=2, smooth_periods=(3, 4))

    expected = price.diff(2).ewm(span=3).mean().ewm(span=4).mean()
    pd.testing.assert_series_equal(actual, expected)


def test_macd_returns_pandas_ewm_components() -> None:
    price = pd.Series([1.0, 2.0, 4.0, 3.0, 5.0])

    actual = macd(price, fastperiod=2, slowperiod=3, signalperiod=2)

    expected = pd.DataFrame(index=price.index)
    expected["fast_trendline"] = price.ewm(span=2).mean()
    expected["slow_trendline"] = price.ewm(span=3).mean()
    expected["macd"] = expected["fast_trendline"] - expected["slow_trendline"]
    expected["macdsignal"] = expected["macd"].ewm(span=2).mean()
    expected["macdhist"] = expected["macd"] - expected["macdsignal"]
    pd.testing.assert_frame_equal(actual, expected[["macd", "macdsignal", "macdhist"]])


def test_tsi_returns_raw_ratio_not_percentage_scale() -> None:
    price = pd.Series([1.0, 2.0, 4.0, 7.0])

    actual = tsi(price, lookback1=2, lookback2=3)

    expected = pd.Series([np.nan, 1.0, 1.0, 1.0])
    pd.testing.assert_series_equal(actual, expected)


def test_divergence_index_uses_one_period_diff_volatility() -> None:
    price = pd.Series(np.linspace(10.0, 25.0, 25) + np.sin(np.arange(25)))
    data = pd.DataFrame({"close": price})

    actual = divergence_index(data, fast=2, factor=1.5)
    actual_from_series = divergence_index(price, fast=2, factor=1.5)

    slow = 20
    fast_ema = price.ewm(span=2).mean()
    slow_ema = price.ewm(span=slow).mean()
    numerator = fast_ema - slow_ema
    denominator = price.diff().rolling(slow).std()
    di = numerator / denominator
    band = di.rolling(slow).std()
    expected = pd.DataFrame(
        {
            "di": di,
            "upper": 1.5 * band,
            "lower": -1.5 * band,
        }
    )
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(actual_from_series, expected)


def test_min_max_index_marks_more_recent_extreme_direction() -> None:
    price = pd.Series([1.0, 2.0, 3.0, 2.0, 1.0, 4.0])

    actual_raw = min_max_index(price, lookback=3, binary=False)
    actual_binary = min_max_index(price, lookback=3)

    expected_raw = pd.Series([np.nan, np.nan, 67.0, -33.0, -67.0, 33.0], name="ind")
    expected_binary = pd.Series([np.nan, np.nan, 1.0, -1.0, -1.0, 1.0], name="ind")
    pd.testing.assert_series_equal(actual_raw, expected_raw)
    pd.testing.assert_series_equal(actual_binary, expected_binary)
