from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class Params:
    contract: Tuple[str]  # contract given as tuple of params given to Future()
    periods: List[int]  # periods for breakout calculation
    ema_fast: int  # number of periods for moving average filter
    ema_slow: int  # number of periods for moving average filter
    sl_atr: int  # stop loss in ATRs
    atr_periods: int  # number of periods to calculate ATR on
    alloc: float  # fraction of capital to be allocated to instrument
    avg_periods: int = None  # candle volume to be calculated as average of x periods
    volume: int = None  # candle volume given directly
