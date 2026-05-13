"""
Synthetic OHLC data generators for research workflows.
"""

from .block import bootstrap, optimal_block_length
from .data import PRICE_COLUMNS, RAW_COLUMNS, RandomState, prepare_bootstrap_frame
from .regime import regime_bootstrap
from .states import (
    combine_states,
    hmm_states,
    range_states,
    return_states,
    trend_states,
    volatility_states,
)

__all__ = [
    "PRICE_COLUMNS",
    "RAW_COLUMNS",
    "RandomState",
    "bootstrap",
    "combine_states",
    "hmm_states",
    "optimal_block_length",
    "prepare_bootstrap_frame",
    "range_states",
    "regime_bootstrap",
    "return_states",
    "trend_states",
    "volatility_states",
]
