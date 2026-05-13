"""
Simple helpers that produce state labels for regime bootstrap generation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "combine_states",
    "range_states",
    "return_states",
    "trend_states",
    "volatility_states",
]


def _validate_quantile(q: float) -> None:
    if not 0 < q < 1:
        raise ValueError("q must be between 0 and 1.")


def combine_states(*states: pd.Series) -> pd.Series:
    """
    Combine multiple state series into tuple-valued composite states.

    Rows where any input series is missing are dropped. The output is directly
    usable as the ``states`` input to ``regime_bootstrap``.
    """
    if not states:
        raise ValueError("At least one state series is required.")

    combined = pd.concat(states, axis=1).dropna()
    labels = list(map(tuple, combined.to_numpy(dtype=object)))
    return pd.Series(labels, index=combined.index, name="state")


def trend_states(data: pd.DataFrame, *, window: int = 50) -> pd.Series:
    """
    Label bars as ``"up"`` or ``"down"`` from close percentage change.
    """
    if window < 1:
        raise ValueError("window must be at least 1.")
    trend = data["close"].pct_change(window).fillna(0.0)
    labels = np.where(trend >= 0, "up", "down")
    return pd.Series(labels, index=data.index, name="trend_state")


def volatility_states(
    data: pd.DataFrame, *, window: int = 20, q: float = 0.5
) -> pd.Series:
    """
    Label bars as ``"low_vol"`` or ``"high_vol"`` from rolling volatility.
    """
    if window < 1:
        raise ValueError("window must be at least 1.")
    _validate_quantile(q)

    returns = pd.Series(
        np.log(data["close"] / data["close"].shift()), index=data.index
    )
    volatility = returns.rolling(window, min_periods=1).std().fillna(0.0)
    threshold = volatility.quantile(q)
    labels = np.where(volatility <= threshold, "low_vol", "high_vol")
    return pd.Series(labels, index=data.index, name="volatility_state")


def range_states(data: pd.DataFrame, *, q: float = 0.5) -> pd.Series:
    """
    Label bars as ``"narrow_range"`` or ``"wide_range"`` from log high/low.
    """
    _validate_quantile(q)

    ranges = pd.Series(np.log(data["high"] / data["low"]), index=data.index)
    threshold = ranges.quantile(q)
    labels = np.where(ranges <= threshold, "narrow_range", "wide_range")
    return pd.Series(labels, index=data.index, name="range_state")


def return_states(data: pd.DataFrame) -> pd.Series:
    """
    Label bars as ``"positive"`` or ``"negative"`` from close log returns.
    """
    returns = pd.Series(
        np.log(data["close"] / data["close"].shift()), index=data.index
    ).fillna(0.0)
    labels = np.where(returns >= 0, "positive", "negative")
    return pd.Series(labels, index=data.index, name="return_state")
