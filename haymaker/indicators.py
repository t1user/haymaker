"""Compatibility facade for research indicators and transformers.

The canonical implementations live in :mod:`haymaker.research.indicators` and
:mod:`haymaker.research.transformers`. Importing from ``haymaker.indicators`` is
kept for live strategies and older notebooks.
"""

from .research.indicators import (
    adx,
    atr,
    carver,
    chande_momentum_indicator,
    chande_ranking,
    divergence_index,
    downsampled_atr,
    downsampled_func,
    join_swing,
    macd,
    min_max_index,
    mmean,
    momentum,
    resample,
    rsi,
    spread,
    strength_oscillator,
    true_range,
    tsi,
    weighted_resample,
)
from .research.transformers import (
    breakout,
    breakout_blip,
    combine_signals,
    extreme_reversal_blip,
    inout_range,
    min_max_blip,
    range_blip,
    signal_generator,
    zero_crosser,
)

__all__ = [
    "adx",
    "atr",
    "breakout",
    "breakout_blip",
    "carver",
    "chande_momentum_indicator",
    "chande_ranking",
    "combine_signals",
    "divergence_index",
    "downsampled_atr",
    "downsampled_func",
    "extreme_reversal_blip",
    "inout_range",
    "join_swing",
    "macd",
    "min_max_blip",
    "min_max_index",
    "mmean",
    "momentum",
    "range_blip",
    "resample",
    "rsi",
    "signal_generator",
    "spread",
    "strength_oscillator",
    "true_range",
    "tsi",
    "weighted_resample",
    "zero_crosser",
]
