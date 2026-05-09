from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from haymaker.research.backtester import no_stop, perf
from haymaker.research.stop import stop_loss
from haymaker.research.utils import paths, upsample


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


def test_upsample_raw_blip_is_sparse_even_if_propagated() -> None:
    raw = _raw_frame()
    grouped = pd.DataFrame(
        {"raw_blip": [1, -1], "feature": [10.0, 20.0]},
        index=raw.index[[0, 3]],
    )

    actual = upsample(raw, grouped, label="left", propagate=["raw_blip", "feature"])

    assert pd.Timestamp("2026-01-01 09:00") not in actual.index
    assert pd.Timestamp("2026-01-01 09:01") not in actual.index
    assert actual.loc["2026-01-01 09:02", "raw_blip"] == 0
    assert actual.loc["2026-01-01 09:03", "raw_blip"] == -1
    assert actual.loc["2026-01-01 09:04", "raw_blip"] == 0
    assert actual.loc["2026-01-01 09:05", "raw_blip"] == 0
    assert actual.loc["2026-01-01 09:03", "feature"] == 10.0


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
    upsampled = upsample(raw, grouped, keep=["blip"])

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
    upsampled = upsample(raw, grouped, keep=["blip"])

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
    upsampled = upsample(raw, grouped, keep=["blip"])

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
