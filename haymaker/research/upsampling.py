"""
Utilities for aligning low-frequency research signals to high-frequency bars.

This module preserves the timing semantics used by the research backtester:
low-frequency values are made available only when their source bar is known,
ordinary values are propagated from that point, and event-like columns can be
kept sparse. The public entry point is :func:`upsample`.
"""

from __future__ import annotations

import warnings
from typing import Literal, Sequence

import numpy as np
import pandas as pd

SparseColumns = str | Sequence[str] | None

__all__ = ["upsample"]


def _as_unique_columns(columns: str | Sequence[str]) -> list[str]:
    column_list = [columns] if isinstance(columns, str) else list(columns)
    return list(dict.fromkeys(column_list))


def _difference(columns: Sequence[str], excluded: Sequence[str]) -> list[str]:
    excluded_set = set(excluded)
    return [column for column in columns if column not in excluded_set]


def _union(first: Sequence[str], second: Sequence[str]) -> list[str]:
    seen = set(first)
    combined = list(first)
    for column in second:
        if column not in seen:
            combined.append(column)
            seen.add(column)
    return combined


def _verify_sparse_columns(
    sparse: SparseColumns, upsampled_columns: Sequence[str]
) -> list[str]:
    if sparse is None:
        return []

    sparse_columns = _as_unique_columns(sparse)
    unknown = [column for column in sparse_columns if column not in upsampled_columns]
    if unknown:
        raise ValueError(
            f"Cannot upsample sparse columns: {unknown} - not in columns."
        )
    return sparse_columns


def _validate_indexes(
    hf_df: pd.DataFrame,
    lf_df: pd.DataFrame,
    label: Literal["left", "right"],
) -> None:
    if not hf_df.index.is_monotonic_increasing:
        raise ValueError("Higher frequency dataframe index must be sorted.")
    if not lf_df.index.is_monotonic_increasing:
        raise ValueError("Lower frequency dataframe index must be sorted.")
    if label not in ("left", "right"):
        raise ValueError(f"label must be 'left' or 'right', '{label}' given")


def _align_left_labeled(
    high_index: pd.Index, low_index: pd.Index
) -> tuple[np.ndarray, np.ndarray]:
    high_locs = np.empty(len(low_index), dtype=np.int64)
    if len(low_index) > 1:
        high_locs[:-1] = (
            high_index.searchsorted(low_index[1:].to_numpy(), side="left") - 1
        )
    high_locs[-1] = len(high_index) - 1
    rows = np.arange(len(low_index), dtype=np.int64)
    valid = (
        (low_index <= high_index[-1])
        & (high_locs >= 0)
        & (high_index.take(high_locs.clip(min=0)) >= low_index)
    )
    return rows[valid], high_locs[valid]


def _align_right_labeled(
    high_index: pd.Index, low_index: pd.Index
) -> tuple[np.ndarray, np.ndarray]:
    high_locs = high_index.searchsorted(low_index.to_numpy(), side="right") - 1
    rows = np.arange(len(low_index), dtype=np.int64)
    valid = (
        (low_index >= high_index[0])
        & (low_index <= high_index[-1])
        & (high_locs >= 0)
    )
    return rows[valid], high_locs[valid]


def _align_to_availability(
    hf_df: pd.DataFrame,
    lf_df: pd.DataFrame,
    upsampled_columns: Sequence[str],
    label: Literal["left", "right"],
) -> pd.DataFrame:
    """
    Align lower-frequency rows to the high-frequency row where they are known.
    """
    _validate_indexes(hf_df, lf_df, label)
    if not upsampled_columns:
        return lf_df[list(upsampled_columns)].iloc[:0]

    high_index = hf_df.index
    low_index = lf_df.index
    if len(high_index) == 0 or len(low_index) == 0:
        return lf_df[list(upsampled_columns)].iloc[:0]

    if label == "left":
        rows, high_locs = _align_left_labeled(high_index, low_index)
    else:
        rows, high_locs = _align_right_labeled(high_index, low_index)

    aligned = lf_df.iloc[rows][list(upsampled_columns)].copy()
    aligned.index = high_index.take(high_locs)
    if aligned.index.has_duplicates:
        raise ValueError(
            "Cannot upsample: multiple lower-frequency rows align to the "
            "same higher-frequency index point."
        )
    return aligned


def _validate_lower_frequency_values(lf_df: pd.DataFrame) -> None:
    if "position" in lf_df.columns:
        raise ValueError(
            "Cannot upsample 'position'. Position is already executable state; "
            "upsample generated signals or blips first, then derive position on "
            "the upsampled dataframe."
        )

    position_like_columns = [
        column for column in lf_df.columns if "position" in column.lower()
    ]
    if position_like_columns:
        warnings.warn(
            "Columns containing 'position' are not treated as executable state by "
            f"upsample: {position_like_columns}. Ensure these are generated "
            "values, not already-shifted positions.",
            UserWarning,
            stacklevel=3,
        )

    if lf_df.isna().any(axis=1).any():
        raise ValueError("Lower frequency dataframe (lf_df) must not have n/a values.")


def _raw_blip_map(
    hf_df: pd.DataFrame, lf_df: pd.DataFrame, blip_columns: Sequence[str]
) -> dict[str, str]:
    return {
        column: f"raw_{column}"
        for column in blip_columns
        if f"raw_{column}" not in hf_df.columns and f"raw_{column}" not in lf_df.columns
    }


def _join_raw_blip_columns(
    joined_df: pd.DataFrame,
    lf_df: pd.DataFrame,
    raw_source_columns: Sequence[str],
    raw_column_map: dict[str, str],
) -> pd.DataFrame:
    raw_frames = []
    if raw_source_columns:
        raw_frames.append(lf_df[list(raw_source_columns)])
    if raw_column_map:
        raw_frames.append(lf_df[list(raw_column_map)].rename(columns=raw_column_map))
    if not raw_frames:
        return joined_df
    return joined_df.join(pd.concat(raw_frames, axis=1))


def upsample(
    hf_df: pd.DataFrame,
    lf_df: pd.DataFrame,
    *,
    label: Literal["left", "right"] = "left",
    sparse: SparseColumns = None,
) -> pd.DataFrame:
    """
    Upsample time series by combining high-frequency and low-frequency data.

    Args:
        hf_df: High-frequency data. All columns are preserved.
        lf_df: Low-frequency data. Non-overlapping columns are aligned to
            ``hf_df`` and added to the output. ``position`` must not be passed
            here because it is already executable state; upsample generated
            signals or blips first, then derive ``position`` on the upsampled
            frame.
        label: How ``lf_df`` is labeled. ``"left"`` means each low-frequency
            row is labeled by the first high-frequency bar in its group.
            ``"right"`` means it is labeled by the bar where the
            low-frequency value becomes known. Values are first aligned to the
            high-frequency index point where they become known.
        sparse: Optional column or columns that should remain sparse events
            instead of being propagated. Sparse columns are filled with ``0``
            on high-frequency rows where they did not occur. Use this for
            event-like values, not for state or feature values.

            Canonical ``blip`` and ``close_blip`` columns are always sparse
            because they are events generated at a bar, not state values to
            forward-fill. When either canonical blip column is upsampled, its
            original low-frequency label is preserved as ``raw_blip`` or
            ``raw_close_blip`` if that column does not already exist. Raw blip
            provenance columns are also always sparse.

    Returns:
        ``hf_df`` plus aligned low-frequency columns. Ordinary low-frequency
        columns are forward-filled from their availability point onward.
    """
    _validate_lower_frequency_values(lf_df)

    upsampled_columns = [
        column for column in lf_df.columns if column not in hf_df.columns
    ]
    blip_columns = [
        column for column in ("blip", "close_blip") if column in upsampled_columns
    ]
    raw_source_columns = [
        column
        for column in ("raw_blip", "raw_close_blip")
        if column in upsampled_columns
    ]
    raw_column_map = _raw_blip_map(hf_df, lf_df, blip_columns)
    raw_blip_columns = list(raw_source_columns) + list(raw_column_map.values())
    aligned_columns = _difference(upsampled_columns, raw_source_columns)

    types = lf_df[upsampled_columns].dtypes.to_dict()
    types.update(
        {raw_column: lf_df[column].dtype for column, raw_column in raw_column_map.items()}
    )

    aligned = _align_to_availability(hf_df, lf_df, aligned_columns, label)
    joined_df = hf_df.join(aligned)
    joined_df = _join_raw_blip_columns(
        joined_df, lf_df, raw_source_columns, raw_column_map
    )

    sparse_columns = _verify_sparse_columns(sparse, upsampled_columns)
    sparse_columns = _union(sparse_columns, blip_columns + raw_blip_columns)
    propagated_columns = _difference(upsampled_columns, sparse_columns)

    joined_df[sparse_columns] = joined_df[sparse_columns].fillna(0)
    joined_df[propagated_columns] = joined_df[propagated_columns].ffill()
    return joined_df.dropna().astype(types)  # type: ignore
