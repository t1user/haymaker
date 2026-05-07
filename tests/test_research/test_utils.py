from __future__ import annotations

import numpy as np
import pandas as pd

from haymaker.research.backtester import no_stop
from haymaker.research.stop import stop_loss
from haymaker.research.utils import upsample


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
