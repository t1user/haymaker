import sys
from pprint import pprint
from ib_insync import IB, util
from ib_insync.contract import ContFuture, Future
from ib_insync.ibcontroller import IBC, Watchdog

from logger import logger
from trader import Candle, Trader
from credentials import creds

log = logger(__file__[:-3])

#ib.connect('127.0.0.1', 4002, clientId=0)


class Strategy:

    contracts = [
        ('NQ', 'GLOBEX'),
        ('ES', 'GLOBEX'),
        ('NKD', 'GLOBEX'),
        ('CL', 'NYMEX'),
        ('GC', 'NYMEX'),
    ]

    def __init__(self, ib):
        self.ib = ib

    def onConnected(self):
        log.debug('connection established')
        #contract = get_contract('NQ', 'GLOBEX')
        trader = Trader(self.ib)
        #candle = Candle(contract, trader, ib)

        candles = [Candle(self.get_contract(*contract), trader, self.ib)
                   for contract in self.contracts]

    def get_contract(self, *args):
        contract = ContFuture(*args)
        self.ib.qualifyContracts(contract)
        id = contract.conId
        contract = Future(conId=id)
        self.ib.qualifyContracts(contract)
        log.debug(f'Contract qualified: {contract}')
        return contract

    def onError(self, *args):
        log.debug(f'ERROR: {args}')


if __name__ == '__main__':

    ibc = IBC(twsVersion=978,
              gateway=True,
              tradingMode='paper',
              )
    ib = IB()
    strategy = Strategy(ib)
    ib.connectedEvent += strategy.onConnected
    watchdog = Watchdog(ibc, ib,
                        port='4002',
                        clientId=0,
                        )
    watchdog.start()
    util.patchAsyncio()
    ib.run()
