"""
Regime-based empirical bootstrap generator.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Sequence

import numpy as np
import pandas as pd

from .data import (
    RAW_COLUMNS,
    RandomState,
    _reconstruct_path,
    _rng,
    prepare_bootstrap_frame,
)

__all__ = ["regime_bootstrap"]


def _align_states(states: pd.Series, index: pd.Index) -> pd.Series:
    aligned = states.reindex(index)
    missing = aligned.isna()
    if missing.any():
        missing_count = int(missing.sum())
        raise ValueError(
            "states must provide one non-null state for every generated bar; "
            f"{missing_count} missing state labels."
        )
    return aligned


def _state_codes(states: pd.Series) -> tuple[np.ndarray, list[Hashable]]:
    labels = list(pd.unique(states))
    for label in labels:
        if not isinstance(label, Hashable):
            raise ValueError(f"State labels must be hashable, got {label!r}.")
    state_to_code = {label: code for code, label in enumerate(labels)}
    codes = np.array([state_to_code[label] for label in states], dtype=np.int64)
    return codes, labels


def _validate_state_counts(
    state_counts: np.ndarray, labels: Sequence[Hashable], min_state_count: int
) -> None:
    too_small = [
        f"{labels[code]!r}: {count}"
        for code, count in enumerate(state_counts)
        if count < min_state_count
    ]
    if too_small:
        raise ValueError(
            "Every state must have at least "
            f"{min_state_count} rows; too few rows for {too_small}."
        )


def _transition_counts(codes: np.ndarray, n_states: int) -> np.ndarray:
    counts = np.zeros((n_states, n_states), dtype=np.int64)
    if len(codes) > 1:
        np.add.at(counts, (codes[:-1], codes[1:]), 1)
    return counts


def _validate_transition_counts(
    transition_counts: np.ndarray,
    labels: Sequence[Hashable],
    min_transition_count: int,
) -> None:
    outgoing = transition_counts.sum(axis=1)
    too_small = [
        f"{labels[code]!r}: {count}"
        for code, count in enumerate(outgoing)
        if count < min_transition_count
    ]
    if too_small:
        raise ValueError(
            "Every state must have at least "
            f"{min_transition_count} outgoing transitions; too few transitions "
            f"for {too_small}."
        )


def _generate_state_path(
    state_counts: np.ndarray,
    transition_counts: np.ndarray,
    length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    path = np.empty(length, dtype=np.int64)
    state_probabilities = state_counts / state_counts.sum()
    transition_probabilities = transition_counts / transition_counts.sum(
        axis=1, keepdims=True
    )
    current = int(rng.choice(len(state_counts), p=state_probabilities))

    for location in range(length):
        path[location] = current
        current = int(
            rng.choice(len(state_counts), p=transition_probabilities[current])
        )

    return path


def _sample_positions_by_state(
    state_path: np.ndarray,
    state_buckets: Sequence[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    sampled_positions = np.empty(len(state_path), dtype=np.int64)
    for code, bucket in enumerate(state_buckets):
        path_locations = np.flatnonzero(state_path == code)
        if len(path_locations) == 0:
            continue
        sampled_positions[path_locations] = rng.choice(
            bucket, size=len(path_locations), replace=True
        )
    return sampled_positions


def regime_bootstrap(
    data: pd.DataFrame,
    *,
    states: pd.Series,
    paths: int = 1,
    random_state: RandomState = None,
    raw_columns: Sequence[str] = RAW_COLUMNS,
    min_state_count: int = 30,
    min_transition_count: int = 20,
) -> list[pd.DataFrame]:
    """
    Generate synthetic OHLC paths with a Markov state process.

    Args:
        data: Source OHLC dataframe.
        states: Hard state labels whose index covers ``data.index[1:]``.
        paths: Number of paths to generate.
        random_state: Integer seed or numpy generator.
        raw_columns: Columns sampled as raw bar attributes instead of returns.
        min_state_count: Minimum number of historical rows required per state.
        min_transition_count: Minimum outgoing transitions required per state.

    Returns:
        List of synthetic OHLC dataframes. Each path has index ``data.index[1:]``
        and length ``len(data) - 1``.
    """
    if paths < 1:
        raise ValueError("paths must be at least 1.")
    if min_state_count < 1:
        raise ValueError("min_state_count must be at least 1.")
    if min_transition_count < 1:
        raise ValueError("min_transition_count must be at least 1.")

    prepared = prepare_bootstrap_frame(data, raw_columns=raw_columns)
    aligned_states = _align_states(states, prepared.index)
    codes, labels = _state_codes(aligned_states)
    state_counts = np.bincount(codes, minlength=len(labels))
    _validate_state_counts(state_counts, labels, min_state_count)

    transition_counts = _transition_counts(codes, len(labels))
    _validate_transition_counts(transition_counts, labels, min_transition_count)

    state_buckets = [np.flatnonzero(codes == code) for code in range(len(labels))]
    starting_price = float(data["close"].iloc[0])
    path_index = data.index[1:]
    generator = _rng(random_state)

    output = []
    for _ in range(paths):
        state_path = _generate_state_path(
            state_counts, transition_counts, len(prepared), generator
        )
        sampled_positions = _sample_positions_by_state(
            state_path, state_buckets, generator
        )
        sampled = prepared.iloc[sampled_positions]
        output.append(
            _reconstruct_path(
                sampled,
                starting_price=starting_price,
                index=path_index,
                raw_columns=raw_columns,
            )
        )

    return output
