"""
Research module for Haymaker.
"""

from .result_analysis import always_on, long_short_returns, paths
from .signal_converters import (
    blip_sig,
    pos_trans,
    sig_blip,
    sig_pos,
)
from .upsampling import upsample
from .utils import gap_tracer, round_tick, sampler

__all__ = [
    "always_on",
    "blip_sig",
    "gap_tracer",
    "long_short_returns",
    "pos_trans",
    "paths",
    "round_tick",
    "sampler",
    "sig_blip",
    "sig_pos",
    "upsample",
]
