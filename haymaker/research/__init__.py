"""
Research module for Haymaker.
"""

from .result_analysis import (
    excursions,
    factor_extractor,
    long_short_returns,
    paths,
    winning_trade_adverse_excursions,
)
from .signal_converters import (
    blip_sig,
    pos_trans,
    sig_blip,
    sig_pos,
)
from .upsampling import upsample
from .utils import always_on, gap_tracer, round_tick, sampler

__all__ = [
    "always_on",
    "blip_sig",
    "excursions",
    "factor_extractor",
    "gap_tracer",
    "long_short_returns",
    "pos_trans",
    "paths",
    "round_tick",
    "sampler",
    "sig_blip",
    "sig_pos",
    "upsample",
    "winning_trade_adverse_excursions",
]
