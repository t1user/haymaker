"""
Shared data preparation and reconstruction utilities for bootstrap generators.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

RandomState = int | np.random.Generator | None

PRICE_COLUMNS = ("open", "high", "low", "close")
RAW_COLUMNS = ("volume", "barCount")

__all__ = ["PRICE_COLUMNS", "RAW_COLUMNS", "RandomState", "prepare_bootstrap_frame"]

_DIFF_TOLERANCE = 1e-10
_TICK_DECIMALS = 10


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
    Convert OHLC data into the representation sampled by bootstrap generators.

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


def _estimate_tick_size(data: pd.DataFrame) -> float:
    """
    Estimate the source tick size from OHLC price columns.

    Args:
        data: Source OHLC dataframe.

    Returns:
        Most common positive adjacent price difference, or ``0.0`` when no
        tick size can be inferred.
    """
    values = data[list(PRICE_COLUMNS)].to_numpy(dtype=np.float64, copy=False).ravel()
    values = np.sort(values[np.isfinite(values)])
    diffs = np.diff(values)
    diffs = diffs[diffs > _DIFF_TOLERANCE]
    if len(diffs) == 0:
        return 0.0

    rounded_diffs = np.round(diffs, 8)
    ticks, counts = np.unique(rounded_diffs, return_counts=True)
    return float(ticks[np.argmax(counts)])


def _round_price_columns(output: pd.DataFrame, tick_size: float) -> pd.DataFrame:
    """
    Round reconstructed OHLC prices to the inferred tick grid.

    Args:
        output: Reconstructed synthetic OHLC dataframe.
        tick_size: Inferred source tick size.

    Returns:
        Dataframe with OHLC columns rounded to ``tick_size``. If ``tick_size``
        is not positive, ``output`` is returned unchanged.
    """
    if tick_size <= 0:
        return output

    rounded = output.copy()
    values = rounded[list(PRICE_COLUMNS)].to_numpy(dtype=np.float64, copy=True)
    values = np.rint(values / tick_size) * tick_size
    rounded[list(PRICE_COLUMNS)] = np.round(values, _TICK_DECIMALS)
    return rounded


def _reconstruct_path(
    sampled: pd.DataFrame,
    *,
    starting_price: float,
    index: pd.Index,
    raw_columns: Sequence[str],
    tick_size: float,
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

    return _round_price_columns(output, tick_size)
