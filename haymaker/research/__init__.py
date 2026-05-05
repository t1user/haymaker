"""
Research module for Haymaker.
"""

from .backtester import (
    Results,
    adverse_excursions,
    blip_extractor,
    excursions,
    factor_extractor,
    get_min_tick,
    no_stop,
    perf,
    profitable_excursions,
    summary,
    v_backtester,
)
from .signal_converters import (
    blip_sig,
    pos_trans,
    pos_trans_array,
    pos_trans_numpy,
    sig_blip,
    sig_pos,
)

__all__ = [
    "Results",
    "adverse_excursions",
    "blip_extractor",
    "blip_sig",
    "excursions",
    "factor_extractor",
    "get_min_tick",
    "no_stop",
    "perf",
    "pos_trans",
    "pos_trans_array",
    "pos_trans_numpy",
    "profitable_excursions",
    "sig_blip",
    "sig_pos",
    "summary",
    "v_backtester",
]
