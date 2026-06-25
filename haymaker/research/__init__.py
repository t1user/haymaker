"""
Research module for Haymaker.
"""

from .signal_converters import (
    blip_sig,
    pos_trans,
    pos_trans_array,
    pos_trans_numpy,
    sig_blip,
    sig_pos,
)
from .upsampling import upsample

__all__ = [
    "Results",
    "adverse_excursions",
    "auto_perf",
    "blip_extractor",
    "blip_sig",
    "bootstrap",
    "combine_states",
    "hmm_states",
    "excursions",
    "factor_extractor",
    "get_min_tick",
    "no_stop",
    "optimal_block_length",
    "perf",
    "pos_trans",
    "pos_trans_array",
    "pos_trans_numpy",
    "prepare_bootstrap_frame",
    "profitable_excursions",
    "range_states",
    "regime_bootstrap",
    "return_states",
    "sig_blip",
    "sig_pos",
    "summary",
    "trend_states",
    "upsample",
    "v_backtester",
    "volatility_states",
]
