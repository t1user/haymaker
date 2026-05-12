"""
Bootstrap utilities for generating synthetic OHLC research data.

The public entry point is :func:`bootstrap`, which supports stationary,
moving-block, and circular-block bootstrap paths. :func:`prepare_bootstrap_frame`
exposes the intermediate return representation used for sampling, and
:func:`optimal_block_length` provides an optional Politis-White style block
length helper using ``arch`` when it is installed.
"""

from __future__ import annotations

from typing import Literal, Sequence, cast

import numpy as np
import pandas as pd

BootstrapMethod = Literal["stationary", "moving", "circular"]
BlockLength = int | Literal["auto"]
RandomState = int | np.random.Generator | None

PRICE_COLUMNS = ("open", "high", "low", "close")
RAW_COLUMNS = ("volume", "barCount")

__all__ = ["bootstrap", "optimal_block_length", "prepare_bootstrap_frame"]


def _rng(random_state: RandomState) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _validate_ohlc(data: pd.DataFrame, price_columns: Sequence[str]) -> None:
    missing = [column for column in price_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required OHLC columns: {missing}.")
    if len(data) < 2:
        raise ValueError("At least two rows are required to bootstrap OHLC data.")
    if not data.index.is_monotonic_increasing:
        raise ValueError("Data index must be sorted in increasing order.")
    if (data[list(price_columns)] <= 0).any(axis=None):
        raise ValueError("OHLC values must be positive to compute log returns.")


def prepare_bootstrap_frame(
    data: pd.DataFrame,
    *,
    price_columns: Sequence[str] = PRICE_COLUMNS,
    raw_columns: Sequence[str] = RAW_COLUMNS,
) -> pd.DataFrame:
    """
    Convert OHLC data into the representation sampled by :func:`bootstrap`.

    OHLC values are expressed as log values relative to the previous close.
    This keeps each sampled row internally coherent when reconstructed: open,
    high, low, and close are all anchored to the same previous synthetic close.
    Raw columns such as volume and barCount are sampled as observed bar
    attributes. Unknown columns, including ``average``, are dropped.

    Args:
        data: Input OHLC dataframe.
        price_columns: Required OHLC columns. Defaults to
            ``("open", "high", "low", "close")``.
        raw_columns: Optional columns to carry through as raw sampled values.

    Returns:
        Dataframe indexed like ``data.iloc[1:]`` containing OHLC log values and
        any available raw columns.
    """
    _validate_ohlc(data, price_columns)

    previous_close = data["close"].shift()
    prepared = pd.DataFrame(index=data.index)
    for column in price_columns:
        prepared[column] = np.log(data[column] / previous_close)

    for column in raw_columns:
        if column in data.columns:
            prepared[column] = data[column]

    return prepared.iloc[1:].copy()


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
    block_length = cast(float, result.loc[result_key, "block_length"])
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


def _reconstruct_path(
    sampled: pd.DataFrame,
    *,
    starting_price: float,
    index: pd.Index,
    raw_columns: Sequence[str],
) -> pd.DataFrame:
    output = sampled[list(PRICE_COLUMNS)].set_axis(index)
    output["close"] = starting_price * np.exp(output["close"].cumsum())
    previous_closes = output["close"].shift(fill_value=starting_price)
    output["open"] = np.exp(output["open"]) * previous_closes
    output["high"] = np.exp(output["high"]) * previous_closes
    output["low"] = np.exp(output["low"]) * previous_closes

    raw_available = [column for column in raw_columns if column in sampled.columns]
    if raw_available:
        output[raw_available] = sampled[raw_available].set_axis(index)

    return output


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
