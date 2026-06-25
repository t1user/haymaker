"""
Block bootstrap generators for synthetic OHLC research data.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence, cast

import numpy as np
import pandas as pd

from .data import RAW_COLUMNS, RandomState, _reconstruct_path, _rng
from .data import prepare_bootstrap_frame

BootstrapMethod = Literal["stationary", "moving", "circular"]
BlockLength = int | Literal["auto"]

__all__ = ["bootstrap", "optimal_block_length"]


def _acf(series: pd.Series, max_lag: int) -> list[float]:
    centered = series - series.mean()
    variance = float((centered * centered).sum())
    if variance == 0:
        return []
    return [
        float((centered.iloc[lag:] * centered.shift(lag).iloc[lag:]).sum() / variance)
        for lag in range(1, max_lag + 1)
    ]


def _fallback_optimal_block_length(series: pd.Series) -> int:
    clean = series.dropna()
    nobs = len(clean)
    if nobs < 2:
        raise ValueError("At least two observations are required.")

    max_lag = max(1, min(int(np.sqrt(nobs)), nobs - 1))
    noise_band = 2 / np.sqrt(nobs)
    acf_values = _acf(clean, max_lag)
    if not acf_values:
        return 1

    for lag, value in enumerate(acf_values, start=1):
        if abs(value) <= noise_band:
            return max(1, lag)
    return max_lag


def _extract_arch_block_length(
    result: pd.DataFrame, *, result_key: str, column: str
) -> float:
    """
    Extract a block length from supported ``arch`` result formats.

    Args:
        result: Dataframe returned by ``arch.bootstrap.optimal_block_length``.
        result_key: Bootstrap method key, ``"stationary"`` or ``"circular"``.
        column: Source column name used when ``arch`` returns variables on the
            index.

    Returns:
        Estimated block length as a finite float.

    Raises:
        ValueError: If the result format or value is unsupported.
    """
    if "block_length" in result.columns and result_key in result.index:
        block_length = float(cast(Any, result.loc[result_key, "block_length"]))
    elif result_key in result.columns:
        if column in result.index:
            block_length = float(cast(Any, result.loc[column, result_key]))
        elif len(result) == 1:
            block_length = float(cast(Any, result[result_key].iloc[0]))
        else:
            raise ValueError("Cannot select an arch block-length row.")
    else:
        raise ValueError("Unsupported arch block-length result format.")

    if not np.isfinite(block_length):
        raise ValueError("arch returned a non-finite block length.")
    return block_length


def optimal_block_length(
    data: pd.Series | pd.DataFrame,
    *,
    column: str = "close",
    method: BootstrapMethod = "stationary",
) -> int:
    """
    Estimate a block length for bootstrap sampling.

    If ``arch`` is installed, this delegates to
    ``arch.bootstrap.optimal_block_length`` and returns the stationary or
    circular recommendation. ``moving`` uses the circular recommendation because
    both methods use fixed-size blocks. Without ``arch``, a conservative
    ACF-cutoff fallback is used.

    Args:
        data: Series of returns, or a dataframe containing ``column``.
        column: Dataframe column to use when ``data`` is a dataframe.
        method: Bootstrap method the estimate will be used with.

    Returns:
        Positive integer block length.
    """
    if isinstance(data, pd.DataFrame):
        if column not in data.columns:
            raise ValueError(f"Column {column!r} is not in data.")
        series = data[column]
    else:
        series = data

    if method not in ("stationary", "moving", "circular"):
        raise ValueError(f"Unknown bootstrap method: {method!r}.")

    try:
        from arch.bootstrap import (  # type: ignore[import-not-found]
            optimal_block_length as arch_optimal_block_length,
        )
    except ImportError:
        return _fallback_optimal_block_length(series)

    try:
        result = arch_optimal_block_length(series.dropna())
    except (ValueError, ZeroDivisionError, FloatingPointError):
        return _fallback_optimal_block_length(series)

    result_key = "stationary" if method == "stationary" else "circular"
    try:
        block_length = _extract_arch_block_length(
            result, result_key=result_key, column=column
        )
    except (KeyError, TypeError, ValueError):
        return _fallback_optimal_block_length(series)
    return max(1, int(np.ceil(block_length)))


def _resolve_block_length(
    prepared: pd.DataFrame, block_length: BlockLength, method: BootstrapMethod
) -> int:
    if block_length == "auto":
        resolved = optimal_block_length(prepared, method=method)
        if method == "moving":
            return min(resolved, len(prepared))
        return resolved
    if block_length < 1:
        raise ValueError("block_length must be at least 1.")
    return int(block_length)


def _stationary_indices(
    nobs: int, length: int, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    reset_probability = 1 / block_length
    indices = np.empty(length, dtype=np.int64)
    current = int(rng.integers(0, nobs))

    for location in range(length):
        indices[location] = current
        if rng.random() < reset_probability:
            current = int(rng.integers(0, nobs))
        else:
            current = (current + 1) % nobs

    return indices


def _fixed_block_indices(
    nobs: int,
    length: int,
    block_length: int,
    rng: np.random.Generator,
    *,
    circular: bool,
) -> np.ndarray:
    if not circular and block_length > nobs:
        raise ValueError("moving block_length cannot exceed the data length.")

    indices: list[int] = []
    while len(indices) < length:
        max_start = nobs if circular else nobs - block_length + 1
        start = int(rng.integers(0, max_start))
        for offset in range(block_length):
            if circular:
                indices.append((start + offset) % nobs)
            else:
                indices.append(start + offset)
            if len(indices) == length:
                break

    return np.array(indices, dtype=np.int64)


def _sample_indices(
    nobs: int,
    length: int,
    block_length: int,
    method: BootstrapMethod,
    rng: np.random.Generator,
) -> np.ndarray:
    if method == "stationary":
        return _stationary_indices(nobs, length, block_length, rng)
    if method == "moving":
        return _fixed_block_indices(nobs, length, block_length, rng, circular=False)
    if method == "circular":
        return _fixed_block_indices(nobs, length, block_length, rng, circular=True)
    raise ValueError(f"Unknown bootstrap method: {method!r}.")


def bootstrap(
    data: pd.DataFrame,
    *,
    method: BootstrapMethod = "stationary",
    block_length: BlockLength = "auto",
    paths: int = 1,
    random_state: RandomState = None,
    raw_columns: Sequence[str] = RAW_COLUMNS,
) -> list[pd.DataFrame]:
    """
    Generate synthetic OHLC paths using block bootstrap sampling.

    The input dataframe is used as-is. The first row anchors reconstruction,
    so every generated path has index ``data.index[1:]`` and length
    ``len(data) - 1``.

    Args:
        data: Source OHLC dataframe. ``open``, ``high``, ``low``, and ``close``
            are required. ``volume`` and ``barCount`` are carried as raw sampled
            values when present. Unknown columns are dropped.
        method: How historical rows are resampled into synthetic paths. Use
            ``"stationary"`` for variable-length blocks, ``"moving"`` for
            fixed-length non-wrapping blocks, or ``"circular"`` for
            fixed-length wrapping blocks.
        block_length: Average block length for stationary bootstrap, fixed block
            length for moving/circular bootstrap, or ``"auto"``.
        paths: Number of paths to generate.
        random_state: Integer seed or numpy generator.
        raw_columns: Columns sampled as raw bar attributes instead of returns.

    Returns:
        List of synthetic OHLC dataframes. Each path has index ``data.index[1:]``
        and length ``len(data) - 1``.

    Notes:
        ``"stationary"`` continues to the next historical row unless it jumps
        to a new random row. The jump probability is ``1 / block_length``.
        ``"moving"`` samples fixed-length blocks whose full span fits inside
        the source data. ``"circular"`` samples fixed-length blocks that may
        wrap from the end of the source data back to the beginning.
    """
    if paths < 1:
        raise ValueError("paths must be at least 1.")

    prepared = prepare_bootstrap_frame(data, raw_columns=raw_columns)
    nobs = len(prepared)

    resolved_block_length = _resolve_block_length(prepared, block_length, method)
    starting_price = float(data["close"].iloc[0])
    path_index = data.index[1:]
    generator = _rng(random_state)

    output = []
    for _ in range(paths):
        sampled_positions = _sample_indices(
            nobs, nobs, resolved_block_length, method, generator
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
