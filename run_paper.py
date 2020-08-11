from ib_insync import IB, util
from ib_insync.ibcontroller import IBC, Watchdog
import asyncio

from handlers import Handlers
from saver import ArcticSaver
from blotter import MongoBlotter
from trader import Manager
from strategy import candles
from portfolio import AdjustedPortfolio
from logger import logger, rotating_logger_with_shell
from logbook import DEBUG

log = rotating_logger_with_shell(__file__[:-3])


class Start(Handlers):

    def __init__(self, ib, manager):
        util.patchAsyncio()
        self.manager = manager
        ibc = IBC(twsVersion=979,
                  gateway=False,
                  tradingMode='paper',
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

# util.logToConsole(DEBUG)

blotter = MongoBlotter()
saver = ArcticSaver()
manager = Manager(ib, candles, AdjustedPortfolio,
                  saver=saver, blotter=blotter,
                  contract_fields=['contract', 'micro_contract'],
                  portfolio_params={'target_vol': .5})
start = Start(ib, manager)
