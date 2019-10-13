import sys
import asyncio
from pprint import pprint

from ib_insync import IB, util
from ib_insync.contract import ContFuture, Future
from ib_insync.ibcontroller import IBC, Watchdog

from logger import logger
from trader import Candle, Trader
from credentials import creds

log = logger(__file__[:-3])

#ib.connect('127.0.0.1', 4002, clientId=0)


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

    contracts = [
        ('NQ', 'GLOBEX'),
        ('ES', 'GLOBEX'),
        ('NKD', 'GLOBEX'),
        ('CL', 'NYMEX'),
        ('GC', 'NYMEX'),
    ]

    def __init__(self, ib, watchdog, trader):
        ib.connectedEvent += self.onConnected
        ib.errorEvent += self.onError
        self.ib = ib
        self.trader = trader
        super().__init__(watchdog)

    def onConnected(self):
        log.debug('connection established')
        #contract = get_contract('NQ', 'GLOBEX')

    def onStartedEvent(self, *args):
        log.debug('initializing strategy')
        # trader = Trader(self.ib)
        #candle = Candle(contract, trader, ib)
        futures = self.get_contracts(self.contracts)
        candles = [Candle(contract, self.trader, self.ib)
                   for contract in futures]

    def get_contracts(self, contract_tuples):
        log.debug(f'initializing contract qualification')
        cont_contracts = [ContFuture(*contract)
                          for contract in contract_tuples]
        self.ib.qualifyContracts(*cont_contracts)
        ids = [contract.conId for contract in cont_contracts]
        contracts = [Future(conId=id) for id in ids]
        self.ib.qualifyContracts(*contracts)
        log.debug(f'Contracts qualified: {contracts}')
        return contracts

    def onError(self, *args):
        log.error(f'ERROR: {args}')


if __name__ == '__main__':
    util.patchAsyncio()
    util.logToConsole()
    ibc = IBC(twsVersion=978,
              gateway=True,
              tradingMode='paper',
              )
    ib = IB()

    watchdog = Watchdog(ibc, ib,
                        port='4002',
                        clientId=0,
                        )
    #handlers = WatchdogHandlers(watchdog)
    trader = Trader(ib)
    asyncio.get_event_loop().set_debug(True)
    strategy = Strategy(ib, watchdog, trader)
    watchdog.start()
    ib.run()
    print('enabling debug')
