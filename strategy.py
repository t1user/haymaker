from pprint import pprint

from ib_insync import IB, util
from ib_insync.ibcontroller import IBC, Watchdog
from eventkit import Event

from trader import Manager, VolumeStreamer
from params import contracts
from logger import logger


log = logger(__file__[:-3])


class WatchdogHandlers:

    def __init__(self, dog):
        dog.startingEvent += self.onStartingEvent
        dog.startedEvent += self.onStartedEvent
        dog.stoppingEvent += self.onStoppingEvent
        dog.stoppedEvent += self.onStoppedEvent
        dog.softTimeoutEvent += self.onSoftTimeoutEvent
        dog.hardTimeoutEvent += self.onHardTimeoutEvent
        self.dog = dog

    @staticmethod
    def onStartingEvent(*args):
        log.debug(f'StartingEvent {args}')

    @staticmethod
    def onStartedEvent(*args):
        log.debug(f'StartedEvent {args}')

    @staticmethod
    def onStoppingEvent(*args):
        log.debug(f'StoppingEvent {args}')

    @staticmethod
    def onStoppedEvent(*args):
        log.debug(f'StoppedEvent {args}')

    @staticmethod
    def onSoftTimeoutEvent(*args):
        log.debug(f'SoftTimeoutEvent {args}')

    @staticmethod
    def onHardTimeoutEvent(*args):
        log.debug(f'HardTimeoutEvent {args}')


class Strategy(WatchdogHandlers):

    def __init__(self, ib, watchdog, manager):
        self.contracts = contracts
        ib.connectedEvent += self.onConnected
        ib.errorEvent += self.onError
        ib.updatePortfolioEvent += self.onUpdatePortfolioEvent
        ib.commissionReportEvent += self.onCommissionReportEvent
        ib.positionEvent += self.onPositionEvent
        ib.accountSummaryEvent += self.onAccountSummaryEvent
        update = Event().timerange(300, None, 600)
        update += self.onScheduledUpdate
        self.ib = ib
        self.manager = manager
        super().__init__(watchdog)
        self.portfolio_items = {}

    def onConnected(self):
        log.debug('connection established')
        self.account = ib.client.getAccounts()[0]
        self.ib.accountSummary()
        self.ib.reqPnL(self.account)

    def onStartedEvent(self, *args):
        log.debug('initializing strategy')
        self.manager.onConnected()

    def onError(self, *args):
        error = args[1]
        if error not in (2158, 2119, 2104, 2106, 165, 2108):
            log.error(f'ERROR: {args}')

    def onUpdatePortfolioEvent(self, i):
        realized = round(i.realizedPNL, 2)
        unrealized = round(i.unrealizedPNL, 2)
        total = round(realized + unrealized)
        report = (i.contract.localSymbol, realized, unrealized, total)
        log.info(f'Portfolio item: {report}')
        self.portfolio_items[i.contract.localSymbol] = (
            realized, unrealized, total)

    def onScheduledUpdate(self, time):
        log.info(f'pnl: {self.ib.pnl()}')
        summary = [0, 0, 0]
        for contract, value in self.portfolio_items.items():
            summary[0] += value[0]
            summary[1] += value[1]
            summary[2] += value[2]
        message = (f'realized: {summary[0]}, '
                   f'unrealized: {summary[1]}, total: {summary[2]}')
        log.info(message)
        positions = [(p.contract.localSymbol, p.position)
                     for p in self.ib.positions()]
        log.info(f'POSITIONS: {positions}')
        self.manager.freeze()

    def onAccountSummaryEvent(self, value):
        """
        tags = ['UnrealizedPnL', 'RealizedPnL', 'FuturesPNL',
                'NetLiquidationByCurrency']
        """
        tags = ['NetLiquidationByCurrency']
        if value.tag in tags:
            log.info(f'{value.tag}: {value.value}')

    def onCommissionReportEvent(self, trade, fill, report):
        log.debug(f'Commission report: {report}')

    def onPositionEvent(self, position):
        log.info(f'Position update: {position}')


if __name__ == '__main__':
    util.patchAsyncio()
    # util.logToConsole()
    ibc = IBC(twsVersion=978,
              gateway=False,
              tradingMode='paper',
              )
    ib = IB()

    watchdog = Watchdog(ibc, ib,
                        port='4002',
                        clientId=0,
                        )
    manager = Manager(ib, contracts, VolumeStreamer, leverage=15)
    # asyncio.get_event_loop().set_debug(True)
    strategy = Strategy(ib, watchdog, manager)
    watchdog.start()
    ib.run()
