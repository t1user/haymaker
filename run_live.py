from ib_insync import IB, util
from ib_insync.ibcontroller import IBC, Watchdog

from handlers import Handlers
from trader import Manager
from strategy import candles, FixedPortfolio
from logger import logger


log = logger(__file__[:-3])


class Start(Handlers):

    def __init__(self, ib, manager):
        util.patchAsyncio()
        # asyncio.get_event_loop().set_debug(True)
        # util.logToConsole()
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
manager = Manager(ib, candles, FixedPortfolio)
start = Start(ib, manager)
