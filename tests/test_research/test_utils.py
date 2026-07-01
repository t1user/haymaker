from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from haymaker.research.backtester import Results, no_stop, perf
from haymaker.research import upsample
from haymaker.research.stop import stop_loss
from haymaker.research.utils import (
    always_on,
    gap_tracer,
    long_short_returns,
    paths,
    round_tick,
    sampler,
    true_sharpe,
)


def _raw_frame() -> pd.DataFrame:
    index = pd.date_range("2026-01-01 09:00", periods=6, freq="min")
    price = np.arange(100.0, 106.0)
    return pd.DataFrame(
        {
            "open": price,
            "high": price + 0.25,
            "low": price - 0.25,
            "close": price,
        },
        index=index,
    )


def test_true_sharpe_uses_positional_last_return_with_integer_index() -> None:
    returns = pd.Series([0.1, -0.05], index=[10, 20])

    actual = true_sharpe(returns)

    assert actual["cumulative_return"] == pytest.approx(0.045)
    assert "cummulative_return" not in actual.index


def test_true_sharpe_rejects_empty_returns() -> None:
    with pytest.raises(ValueError, match="at least one return"):
        true_sharpe(pd.Series(dtype=float))


def test_sampler_returns_seeded_business_day_windows() -> None:
    data = pd.DataFrame(
        {"close": np.arange(8.0)},
        index=pd.bdate_range("2026-01-01", periods=8),
    )

    first = sampler(data, period_length=3, paths=2, seed=42)
    second = sampler(data, period_length=3, paths=2, seed=42)

    assert len(first) == 2
    assert all(len(path) == 3 for path in first)
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])


def test_sampler_rejects_too_short_data() -> None:
    data = pd.DataFrame(
        {"close": np.arange(3.0)},
        index=pd.bdate_range("2026-01-01", periods=3),
    )

    with pytest.raises(ValueError, match="not enough"):
        sampler(data, period_length=3)


def test_long_short_returns_uses_log1p_trade_returns() -> None:
    result = Results(
        stats=pd.Series(dtype=float),
        daily=pd.DataFrame(),
        positions=pd.DataFrame(
            {
                "date_c": pd.date_range("2026-01-01", periods=4, freq="D"),
                "open": [100.0, -200.0, 100.0, -100.0],
                "pnl": [10.0, -20.0, -50.0, 10.0],
            }
        ),
        df=pd.DataFrame(),
        warnings=[],
    )

    actual = long_short_returns(result)

    expected = pd.DataFrame(
        {
            "long": [1.1, 1.1, 0.55, 0.55],
            "short": [1.0, 0.9, 0.9, 0.99],
        },
        index=pd.date_range("2026-01-01", periods=4, freq="D", name="date_c"),
    )
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        ([1, 1, 1], True),
        ([-1, -1, -1], True),
        ([0, 0, 0], False),
        ([0, 1, 1, -1], True),
        ([0, 1, 0, -1], False),
        ([], False),
    ],
)
def test_always_on_ignores_leading_flat_rows(
    position: list[int], expected: bool
) -> None:
    assert always_on(pd.Series(position)) is expected


def test_round_tick_rounds_by_tick_size_not_whole_points() -> None:
    series = pd.Series([1.1, 1.2, 1.3])

    actual = round_tick(series)

    pd.testing.assert_series_equal(actual, series)


def test_round_tick_returns_copy_for_constant_series() -> None:
    series = pd.Series([1.0, 1.0, 1.0])

    actual = round_tick(series)

    pd.testing.assert_series_equal(actual, series)
    assert actual is not series


def test_gap_tracer_empty_or_short_input_returns_empty_gap_frame() -> None:
    expected = pd.DataFrame(columns=["from", "to", "duration"])

    empty = pd.DataFrame(index=pd.DatetimeIndex([]))
    single = pd.DataFrame(index=pd.date_range("2026-01-01", periods=1, freq="min"))

    pd.testing.assert_frame_equal(gap_tracer(empty), expected)
    pd.testing.assert_frame_equal(gap_tracer(single), expected)


def test_gap_tracer_keeps_first_detected_real_gap() -> None:
    frame = pd.DataFrame(
        index=pd.to_datetime(
            [
                "2026-01-01 09:00",
                "2026-01-01 09:01",
                "2026-01-01 09:05",
            ]
        )
    )

    actual = gap_tracer(frame)

    assert len(actual) == 1
    assert actual.loc[0, "from"] == pd.Timestamp("2026-01-01 09:01")
    assert actual.loc[0, "to"] == pd.Timestamp("2026-01-01 09:05")
    assert actual.loc[0, "duration"] == pd.Timedelta(minutes=4)


def test_upsample_left_labeled_blip_is_placed_on_generation_bar() -> None:
    raw = _raw_frame()
    grouped = pd.DataFrame(
        {"blip": [1, -1]},
        index=raw.index[[0, 3]],
    )

    actual = upsample(raw, grouped, label="left")

    assert actual.loc["2026-01-01 09:00", "raw_blip"] == 1
    assert actual.loc["2026-01-01 09:00", "blip"] == 0
    assert actual.loc["2026-01-01 09:01", "blip"] == 0
    assert actual.loc["2026-01-01 09:02", "blip"] == 1
    assert actual.loc["2026-01-01 09:03", "raw_blip"] == -1
    assert actual.loc["2026-01-01 09:03", "blip"] == 0
    assert actual.loc["2026-01-01 09:04", "blip"] == 0
    assert actual.loc["2026-01-01 09:05", "blip"] == -1


def test_upsample_left_labeled_features_start_when_group_is_known() -> None:
    raw = _raw_frame()
    grouped = pd.DataFrame(
        {"blip": [1, -1], "feature": [10.0, 20.0]},
        index=raw.index[[0, 3]],
    )

    actual = upsample(raw, grouped, label="left")

    assert raw.index[0] not in actual.index
    assert raw.index[1] not in actual.index
    assert actual.loc["2026-01-01 09:02", "feature"] == 10.0
    assert actual.loc["2026-01-01 09:03", "feature"] == 10.0
    assert actual.loc["2026-01-01 09:04", "feature"] == 10.0
    assert actual.loc["2026-01-01 09:05", "feature"] == 20.0


def test_upsample_rejects_position_column() -> None:
    raw = _raw_frame()
    grouped = pd.DataFrame(
        {"position": [1, -1]},
        index=raw.index[[0, 3]],
    )

    with pytest.raises(ValueError, match="Cannot upsample 'position'"):
        upsample(raw, grouped, label="left")


def test_upsample_warns_for_position_like_column_names() -> None:
    raw = _raw_frame()
    grouped = pd.DataFrame(
        {"outcome_position": [1.0, -1.0]},
        index=raw.index[[0, 3]],
    )

    with pytest.warns(UserWarning, match="outcome_position"):
        actual = upsample(raw, grouped, label="left")

    assert "outcome_position" in actual.columns


def test_upsample_raw_blip_preserves_original_grouped_label() -> None:
    raw = _raw_frame()
    grouped = pd.DataFrame(
        {"raw_blip": [1, -1]},
        index=raw.index[[0, 3]],
    )

    actual = upsample(raw, grouped, label="left")

    assert actual.loc["2026-01-01 09:00", "raw_blip"] == 1
    assert actual.loc["2026-01-01 09:01", "raw_blip"] == 0
    assert actual.loc["2026-01-01 09:02", "raw_blip"] == 0
    assert actual.loc["2026-01-01 09:03", "raw_blip"] == -1
    assert actual.loc["2026-01-01 09:04", "raw_blip"] == 0
    assert actual.loc["2026-01-01 09:05", "raw_blip"] == 0


def test_upsample_noncanonical_blip_named_columns_are_not_special() -> None:
    raw = _raw_frame()
    grouped = pd.DataFrame(
        {"custom_blip": [1, -1], "feature": [10.0, 20.0]},
        index=raw.index[[0, 3]],
    )

    actual = upsample(raw, grouped, label="left")

    assert actual.loc["2026-01-01 09:02", "custom_blip"] == 1
    assert actual.loc["2026-01-01 09:03", "custom_blip"] == 1
    assert actual.loc["2026-01-01 09:04", "custom_blip"] == 1
    assert actual.loc["2026-01-01 09:05", "custom_blip"] == -1


def test_upsample_raw_blip_is_always_sparse() -> None:
    raw = _raw_frame()
    grouped = pd.DataFrame(
        {"raw_blip": [1, -1], "feature": [10.0, 20.0]},
        index=raw.index[[0, 3]],
    )

    actual = upsample(raw, grouped, label="left")

    assert pd.Timestamp("2026-01-01 09:00") not in actual.index
    assert pd.Timestamp("2026-01-01 09:01") not in actual.index
    assert actual.loc["2026-01-01 09:02", "raw_blip"] == 0
    assert actual.loc["2026-01-01 09:03", "raw_blip"] == -1
    assert actual.loc["2026-01-01 09:04", "raw_blip"] == 0
    assert actual.loc["2026-01-01 09:05", "raw_blip"] == 0
    assert actual.loc["2026-01-01 09:03", "feature"] == 10.0


def test_upsample_sparse_keeps_custom_event_from_propagating() -> None:
    raw = _raw_frame()
    grouped = pd.DataFrame(
        {"custom_event": [1, -1], "feature": [10.0, 20.0]},
        index=raw.index[[0, 3]],
    )

    actual = upsample(raw, grouped, label="left", sparse=["custom_event"])

    assert actual.loc["2026-01-01 09:02", "custom_event"] == 1
    assert actual.loc["2026-01-01 09:03", "custom_event"] == 0
    assert actual.loc["2026-01-01 09:04", "custom_event"] == 0
    assert actual.loc["2026-01-01 09:05", "custom_event"] == -1
    assert actual.loc["2026-01-01 09:04", "feature"] == 10.0


def test_upsample_rejects_unknown_sparse_column() -> None:
    raw = _raw_frame()
    grouped = pd.DataFrame(
        {"feature": [10.0, 20.0]},
        index=raw.index[[0, 3]],
    )

    with pytest.raises(ValueError, match="sparse columns"):
        upsample(raw, grouped, label="left", sparse=["missing"])


def test_upsample_right_labeled_blip_remains_on_generation_bar() -> None:
    raw = _raw_frame()
    grouped = pd.DataFrame(
        {"blip": [1, -1]},
        index=raw.index[[2, 5]],
    )

    actual = upsample(raw, grouped, label="right")

    assert actual.loc["2026-01-01 09:00", "blip"] == 0
    assert actual.loc["2026-01-01 09:01", "blip"] == 0
    assert actual.loc["2026-01-01 09:02", "blip"] == 1
    assert actual.loc["2026-01-01 09:03", "blip"] == 0
    assert actual.loc["2026-01-01 09:04", "blip"] == 0
    assert actual.loc["2026-01-01 09:05", "blip"] == -1


def test_upsample_close_blip_gets_raw_provenance_column() -> None:
    raw = _raw_frame()
    grouped = pd.DataFrame(
        {"close_blip": [-1, 1]},
        index=raw.index[[0, 3]],
    )

    actual = upsample(raw, grouped, label="left")

    assert actual.loc["2026-01-01 09:00", "raw_close_blip"] == -1
    assert actual.loc["2026-01-01 09:02", "close_blip"] == -1
    assert actual.loc["2026-01-01 09:03", "raw_close_blip"] == 1
    assert actual.loc["2026-01-01 09:05", "close_blip"] == 1


def test_no_stop_executes_upsampled_left_labeled_blip_on_next_bar() -> None:
    raw = _raw_frame()
    grouped = pd.DataFrame(
        {"blip": [1, 0]},
        index=raw.index[[0, 3]],
    )
    upsampled = upsample(raw, grouped, label="left")

    actual = no_stop(upsampled, price_column="open")

    assert actual.loc["2026-01-01 09:02", "position"] == 0
    assert actual.loc["2026-01-01 09:02", "open_price"] == 0
    assert actual.loc["2026-01-01 09:03", "position"] == 1
    assert (
        actual.loc["2026-01-01 09:03", "open_price"]
        == raw.loc["2026-01-01 09:03", "open"]
    )


def test_stop_loss_executes_upsampled_left_labeled_blip_on_next_bar() -> None:
    raw = _raw_frame()
    grouped = pd.DataFrame(
        {"blip": [1, 0]},
        index=raw.index[[0, 3]],
    )
    upsampled = upsample(raw, grouped, label="left")

    actual = stop_loss(upsampled, distance=100.0, mode="fixed")

    assert actual.loc["2026-01-01 09:02", "position"] == 0
    assert actual.loc["2026-01-01 09:02", "open_price"] == 0
    assert actual.loc["2026-01-01 09:03", "position"] == 1
    assert (
        actual.loc["2026-01-01 09:03", "open_price"]
        == raw.loc["2026-01-01 09:03", "open"]
    )


@pytest.mark.parametrize("use_numba", [False, True])
def test_upsampled_stop_distance_uses_value_available_on_execution_bar(
    use_numba: bool,
) -> None:
    index = pd.date_range("2026-01-01 09:00", periods=6, freq="min")
    raw = pd.DataFrame(
        {
            "open": [100.0] * 6,
            "high": [100.0] * 6,
            "low": [100.0, 100.0, 100.0, 97.0, 100.0, 100.0],
            "close": [100.0] * 6,
        },
        index=index,
    )
    grouped = pd.DataFrame(
        {
            "blip": [1, 0],
            "atr": [2.0, 10.0],
        },
        index=index[[0, 3]],
    )
    upsampled = upsample(raw, grouped)

    actual = stop_loss(
        upsampled,
        upsampled["atr"],
        mode="fixed",
        use_numba=use_numba,
    )

    assert upsampled.loc[index[3], "atr"] == 2.0
    assert actual.loc[index[3], "open_price"] == 100.0
    assert actual.loc[index[3], "stop_price"] == -98.0


def test_pre_upsample_shifted_stop_distance_is_rejected() -> None:
    index = pd.date_range("2026-01-01 09:00", periods=6, freq="min")
    raw = pd.DataFrame(
        {
            "open": [100.0] * 6,
            "high": [100.0] * 6,
            "low": [100.0, 100.0, 100.0, 97.0, 100.0, 100.0],
            "close": [100.0] * 6,
        },
        index=index,
    )
    grouped = pd.DataFrame(
        {
            "blip": [1, 0],
            "atr": [2.0, 10.0],
        },
        index=index[[0, 3]],
    )
    upsampled = upsample(raw, grouped)

    with pytest.raises(ValueError, match="same index as df"):
        stop_loss(
            upsampled,
            grouped["atr"].shift(),
            mode="fixed",
        )


def test_unshifted_pre_upsample_stop_distance_is_rejected() -> None:
    index = pd.date_range("2026-01-01 09:00", periods=6, freq="min")
    raw = pd.DataFrame(
        {
            "open": [100.0] * 6,
            "high": [100.0] * 6,
            "low": [100.0, 100.0, 100.0, 97.0, 100.0, 100.0],
            "close": [100.0] * 6,
        },
        index=index,
    )
    grouped = pd.DataFrame(
        {
            "blip": [1, 0],
            "atr": [2.0, 10.0],
        },
        index=index[[0, 3]],
    )
    upsampled = upsample(raw, grouped)

    with pytest.raises(ValueError, match="same index as df"):
        stop_loss(
            upsampled,
            grouped["atr"],
            mode="fixed",
        )


def test_paths_accepts_current_perf_bar_price_schema() -> None:
    raw = _raw_frame()
    raw["position"] = [0, 1, 1, 0, -1, 0]
    tx = no_stop(raw, price_column="open")
    result = perf(tx, slippage=0, use_numba=False)

    actual = paths(result, cumsum=False, log_return=False)

    assert list(actual.columns) == ["price", "longs", "shorts", "strategy"]
    assert not actual.isna().any().any()
    assert actual["strategy"].sum() == result.df["pnl"].sum()


def test_paths_assigns_stop_loss_to_stopped_position_direction() -> None:
    raw = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 100.0, 100.0],
            "low": [100.0, 99.0, 98.0, 100.0],
            "close": [100.0, 100.0, 100.0, 100.0],
            "blip": [0, 1, 0, 0],
        },
        index=pd.date_range("2026-01-01 09:00", periods=4, freq="min"),
    )
    tx = stop_loss(raw, distance=1.0, mode="fixed")
    result = perf(tx, slippage=0, use_numba=False)

    actual = paths(result, cumsum=False, log_return=False)

    assert actual.loc["2026-01-01 09:02", "longs"] == -1.0
    assert actual.loc["2026-01-01 09:02", "shorts"] == 0.0
    assert actual["strategy"].sum() == result.df["pnl"].sum()


def test_paths_current_schema_reversal_bar_pnl_belongs_to_previous_position() -> None:
    raw = _raw_frame()
    raw["position"] = [1, 1, -1, -1, 1, 1]
    tx = no_stop(raw, price_column="open")
    result = perf(tx, slippage=0, use_numba=False)

    actual = paths(result, cumsum=False, log_return=False)

    assert (
        actual.loc["2026-01-01 09:02", "longs"]
        == result.df.loc["2026-01-01 09:02", "pnl"]
    )
    assert actual.loc["2026-01-01 09:02", "shorts"] == 0.0
    assert (
        actual.loc["2026-01-01 09:04", "shorts"]
        == result.df.loc["2026-01-01 09:04", "pnl"]
    )
    assert actual.loc["2026-01-01 09:04", "longs"] == 0.0
    pd.testing.assert_series_equal(
        actual["longs"] + actual["shorts"],
        actual["strategy"],
        check_names=False,
    )


def test_upsample_left_labeled_irregular_index_uses_previous_bar() -> None:
    index = pd.to_datetime(
        [
            "2026-01-01 09:00",
            "2026-01-01 09:01",
            "2026-01-01 09:04",
            "2026-01-01 09:07",
            "2026-01-01 09:08",
        ]
    )
    raw = pd.DataFrame(
        {
            "open": np.arange(100.0, 105.0),
            "high": np.arange(100.25, 105.25),
            "low": np.arange(99.75, 104.75),
            "close": np.arange(100.0, 105.0),
        },
        index=index,
    )
    grouped = pd.DataFrame(
        {"blip": [1, -1], "feature": [10.0, 20.0]},
        index=pd.to_datetime(["2026-01-01 09:00", "2026-01-01 09:05"]),
    )

    actual = upsample(raw, grouped, label="left")

    assert actual.loc["2026-01-01 09:04", "blip"] == 1
    assert actual.loc["2026-01-01 09:04", "feature"] == 10.0
    assert actual.loc["2026-01-01 09:07", "blip"] == 0
    assert actual.loc["2026-01-01 09:07", "feature"] == 10.0
    assert actual.loc["2026-01-01 09:08", "blip"] == -1
    assert actual.loc["2026-01-01 09:08", "feature"] == 20.0
