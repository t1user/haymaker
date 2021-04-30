from ib_insync import IB, util
from ib_insync.ibcontroller import IBC, Watchdog

from handlers import Handlers
from saver import ArcticSaver
from blotter import MongoBlotter
from manager import Manager
from strategy import strategy_kwargs
from logger import rotating_logger_with_shell
from logbook import INFO, DEBUG

log = rotating_logger_with_shell(__file__[:-3], DEBUG, DEBUG)


class Start(Handlers):

    def __init__(self, ib, manager):
        util.patchAsyncio()
        self.manager = manager
        ibc = IBC(twsVersion=978,
                  gateway=True,
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


ib = IB()
# util.logToConsole(DEBUG)
blotter = MongoBlotter()
saver = ArcticSaver()
manager = Manager(ib, saver=saver, blotter=blotter, **strategy_kwargs)
start = Start(ib, manager)
