from __future__ import annotations

from typing import Tuple, List, Optional
from dataclasses import dataclass

from logbook import Logger


log = Logger(__name__)


@dataclass
class Params:
    contract: Tuple[str]  # contract given as tuple of params given to Future()
    micro_contract: Tuple[str]  # (1/10) contract corresponding to contract
    # periods for breakout calculation
    periods: Tuple[int] = (5, 10, 20, 40, 80, 160)
    ema_fast: int = 5  # number of periods for moving average filter
    ema_slow: int = 120  # number of periods for moving average filter
    sl_atr: int = 1  # stop loss in ATRs
    atr_periods: int = 180  # number of periods to calculate ATR on
    trades_per_day: int = 0
    alloc: float = 0  # fraction of capital to be allocated to instrument
    # candle volume to be calculated as average of x periods
    avg_periods: Optional[int] = None
    volume: Optional[int] = None  # candle volume given directly
