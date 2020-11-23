from typing import Tuple, Optional
from dataclasses import dataclass

from ib_insync import ContFuture
from logbook import Logger

from streamers import VolumeStreamer
from candle import MultipleBreakoutCandle, BreakoutLockCandle, BreakoutCandle
from portfolio import FixedPortfolio, AdjustedPortfolio, WeightedAdjustedPortfolio
from execution_models import EventDrivenExecModel


log = Logger(__name__)


@dataclass
class Params:
    contract: Tuple[str]  # contract given as tuple of params given to Future()
    micro_contract: Tuple[str]  # (1/10) contract corresponding to contract
    periods: int = 40  # periods for breakout calculation
    ema_fast: int = 5  # number of periods for moving average filter
    ema_slow: int = 120  # number of periods for moving average filter
    sl_atr: int = 1  # stop loss in ATRs
    atr_periods: int = 180  # number of periods to calculate ATR on
    trades_per_day: int = 0
    alloc: float = 0  # fraction of capital to be allocated to instrument
    # candle volume to be calculated as average of x periods
    avg_periods: Optional[int] = None
    volume: Optional[int] = None  # candle volume given directly
    min_atr: float = 0
    rsi_threshold: float = 70
    rsi_periods: float = 24
    rsi_smooth: float = 15

    def __post_init__(self) -> None:
        self.lock_periods = int(self.periods / 1)


nq = Params(
    contract=ContFuture('NQ', 'GLOBEX'),
    micro_contract=ContFuture('MNQ', 'GLOBEX'),
    trades_per_day=4.5,
    atr_periods=50,
    # avg_periods=60,
    volume=12000,
    min_atr=14,
    alloc=.4,
)

es = Params(
    contract=ContFuture('ES', 'GLOBEX'),
    micro_contract=ContFuture('MES', 'GLOBEX'),
    trades_per_day=1.5,
    ema_fast=120,
    ema_slow=320,
    sl_atr=3,
    # avg_periods=60,
    volume=43000,
    min_atr=5,
    alloc=.125,
)

gc = Params(
    contract=ContFuture('GC', 'NYMEX'),
    micro_contract=ContFuture('MGC', 'NYMEX'),
    trades_per_day=2.1,
    ema_fast=60,
    periods=60,
    sl_atr=2,
    # atr_periods=90,
    atr_periods=50,
    # avg_periods=60,
    volume=5500,
    min_atr=1.9,
    alloc=.225,
)

ym = Params(
    contract=ContFuture('YM', 'ECBOT'),
    micro_contract=ContFuture('MYM', 'ECBOT'),
    trades_per_day=1.5,
    atr_periods=50,
    ema_fast=60,
    ema_slow=120,
    sl_atr=2,
    # avg_periods=60,
    volume=8000,
    alloc=.25
)


contracts = [nq, es, ym, gc]

exec_model = EventDrivenExecModel()

candles = [BreakoutCandle(VolumeStreamer(params.volume,
                                         params.avg_periods),
                          contract_fields=[
    'contract', 'micro_contract'],
    **params.__dict__)
    for params in contracts]
portfolio = FixedPortfolio(target_vol=.55)

strategy_kwargs = {'candles': candles,
                   'portfolio': portfolio,
                   'exec_model': exec_model}
