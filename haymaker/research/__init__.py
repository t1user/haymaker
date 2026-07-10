"""
Research module for Haymaker.
"""

from .result_analysis import paths
from .signal_converters import (
    blip_sig,
    pos_trans,
    sig_blip,
    sig_pos,
)
from .upsampling import upsample

__all__ = [
    "blip_sig",
    "pos_trans",
    "sig_blip",
    "sig_pos",
    "upsample",
    "paths",
]
