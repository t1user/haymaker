from typing import Tuple, List, Optional
from dataclasses import dataclass

from ib_insync import IB, util, ContFuture
from ib_insync.ibcontroller import IBC, Watchdog

from handlers import Handlers
from saver import ArcticSaver
from blotter import MongoBlotter
from trader import Manager
from streamers import VolumeStreamer
from candle import BreakoutCandle
from portfolio import AdjustedPortfolio
from logger import logger


log = logger(__file__[:-3])


@dataclass
class Params:
    contract: Tuple[str]  # contract given as tuple of params given to Future()
    micro_contract: Tuple[str]  # (1/10) contract corresponding to contract
    periods: int = 40     # periods for breakout calculation
    ema_fast: int = 5  # number of periods for moving average filter
    ema_slow: int = 120  # number of periods for moving average filter
    sl_atr: int = 1  # stop loss in ATRs
    atr_periods: int = 180  # number of periods to calculate ATR on
    trades_per_day: int = 0
    alloc: float = 0  # fraction of capital to be allocated to instrument
    # candle volume to be calculated as average of x periods
    avg_periods: Optional[int] = None
    volume: Optional[int] = None  # candle volume given directly
    min_atr: float = 0  # minimum atr threshold used for for calculations


nq = Params(
    contract=ContFuture('NQ', 'GLOBEX'),
    micro_contract=ContFuture('MNQ', 'GLOBEX'),
    sl_atr=1,
    trades_per_day=4.5,
    # avg_periods=60,
    volume=12000,
    min_atr=14,
)

es = Params(
    contract=ContFuture('ES', 'GLOBEX'),
    micro_contract=ContFuture('MES', 'GLOBEX'),
    trades_per_day=.8,
    ema_fast=120,
    ema_slow=320,
    sl_atr=3,
    # avg_periods=60,
    volume=43000,
    min_atr=5,
)

gc = Params(
    contract=ContFuture('GC', 'NYMEX'),
    micro_contract=ContFuture('MGC', 'NYMEX'),
    trades_per_day=1.9,  # 2.1
    ema_fast=60,
    ema_slow=120,
    periods=60,
    sl_atr=2,
    atr_periods=90,
    # avg_periods=60,
    volume=5500,
    min_atr=1.9,
)

ym = Params(
    contract=ContFuture('YM', 'ECBOT'),
    micro_contract=ContFuture('MYM', 'ECBOT'),
    trades_per_day=1.5,
    ema_fast=60,
    ema_slow=120,
    sl_atr=2,
    # avg_periods=60,
    volume=8000,
    min_atr=55,
)


contracts = [nq, es, ym, gc]


candles = [BreakoutCandle(VolumeStreamer(params.volume,
                                         params.avg_periods),
                          **params.__dict__)
           for params in contracts]


class Start(Handlers):

    def __init__(self, ib, manager):
        util.patchAsyncio()
        # asyncio.get_event_loop().set_debug(True)
        # util.logToConsole()
        self.manager = manager
        ibc = IBC(twsVersion=979,
                  gateway=True,
                  ibcIni='~/ibc/config_live.ini',
                  tradingMode='live',
                  )
        watchdog = Watchdog(ibc, ib,
                            port='4002',
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
manager = Manager(ib, candles, AdjustedPortfolio,
                  saver=saver, blotter=blotter,
                  contract_fields=['contract', 'micro_contract'],
                  portfolio_params={'target_vol': .7})
start = Start(ib, manager)
