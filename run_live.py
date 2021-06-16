import sys
from typing import Tuple, Optional
from dataclasses import dataclass

from ib_insync import IB, util, ContFuture
from ib_insync.ibcontroller import IBC, Watchdog

from handlers import Handlers
from saver import ArcticSaver
from blotter import MongoBlotter
from manager import Manager
from streamers import VolumeStreamer
from candle import BreakoutLockCandleVolFilter, BreakoutCandleVolFilter
from portfolio import AdjustedPortfolio
from execution_models import EventDrivenTakeProfitExecModel
from logger import rotating_logger_with_shell


log = rotating_logger_with_shell(__file__[:-3],
                                 folder='/home/tomek/ib_data/live_logs')

s = input('THIS IS LIVE. Continue? ').lower()
if s != 'yes' and s != 'y':
    sys.exit()


@dataclass
class Params:
    contract: Tuple[str]  # contract given as tuple of params given to Future()
    micro_contract: Tuple[str]  # (1/10) contract corresponding to contract
    periods: int = 40     # periods for breakout calculation
    ema_fast: int = 5  # number of periods for moving average filter
    ema_slow: int = 120  # number of periods for moving average filter
    sl_atr: int = 1  # stop loss in ATRs
    atr_periods: int = 50  # number of periods to calculate ATR on
    trades_per_day: int = 0
    alloc: float = 0  # fraction of capital to be allocated to instrument
    # candle volume to be calculated as average of x periods
    avg_periods: Optional[int] = None
    volume: Optional[int] = None  # candle volume given directly
    min_atr: float = 0  # minimum atr threshold used for for calculations
    lock_periods_multiple: float = 2
    tp_multiple: float = 2
    lock_filter: float = 0.005

    def __post_init__(self) -> None:
        self.lock_periods = int(self.periods / self.lock_periods_multiple)


nq = Params(
    contract=ContFuture('NQ', 'GLOBEX'),
    micro_contract=ContFuture('MNQ', 'GLOBEX'),
    trades_per_day=3.8,
    volume=13000,
    min_atr=14,
    lock_filter=0.004,
    tp_multiple=8,
)

es = Params(
    contract=ContFuture('ES', 'GLOBEX'),
    micro_contract=ContFuture('MES', 'GLOBEX'),
    trades_per_day=.5,
    ema_fast=120,
    ema_slow=320,
    atr_periods=180,
    sl_atr=3,
    volume=43000,
    min_atr=5,
    lock_periods_multiple=1,
    tp_multiple=2,
)

gc = Params(
    contract=ContFuture('GC', 'NYMEX'),
    micro_contract=ContFuture('MGC', 'NYMEX'),
    trades_per_day=1.0,
    ema_fast=60,
    periods=60,
    sl_atr=2,
    volume=5500,
    min_atr=1.9,
    tp_multiple=6,
)

ym = Params(
    contract=ContFuture('YM', 'ECBOT'),
    micro_contract=ContFuture('MYM', 'ECBOT'),
    trades_per_day=1,
    ema_fast=60,
    sl_atr=2,
    volume=8000,
    tp_multiple=6,
    min_atr=55,
)


exec_model = EventDrivenTakeProfitExecModel()

candles = [BreakoutLockCandleVolFilter(VolumeStreamer(params.volume,
                                                      params.avg_periods),
                                       contract_fields=[
                                           'contract', 'micro_contract'],
                                       **params.__dict__)
           for params in [es]]

candles.extend([BreakoutCandleVolFilter(VolumeStreamer(params.volume,
                                                       params.avg_periods),
                                        contract_fields=[
    'contract', 'micro_contract'],
    **params.__dict__)
    for params in [nq, ym]])

portfolio = AdjustedPortfolio(target_vol=.55)


class Start(Handlers):

    def __init__(self, ib, manager):
        util.patchAsyncio()
        # asyncio.get_event_loop().set_debug(True)
        # util.logToConsole()
        self.manager = manager
        ibc = IBC(twsVersion=978,
                  gateway=True,
                  ibcIni='/home/tomek/ibc/config_live.ini',
                  tradingMode='live',
                  )
        watchdog = Watchdog(ibc, ib,
                            port='4001',
                            clientId=0,
                            )
        log.debug('attaching handlers...')
        super().__init__(ib, watchdog)
        # this is the main entry point into strategy
        watchdog.startedEvent += manager.onStarted
        log.debug('initializing watchdog...')
        watchdog.start()
        log.debug('watchdog initialized')
        ib.run()


log.debug(f'candles: {candles}')
ib = IB()
blotter = MongoBlotter(collection='live_blotter')
saver = ArcticSaver(library='live_log')
manager = Manager(ib, saver=saver, blotter=blotter,
                  candles=candles, exec_model=exec_model, portfolio=portfolio)
start = Start(ib, manager)
