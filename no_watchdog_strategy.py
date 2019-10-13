import sys

from ib_insync import IB, util
from ib_insync.contract import ContFuture, Future

from logger import logger
from trader import Candle, Trader
from credentials import creds

log = logger(__file__[:-3])

ib = IB()
ib.connect('127.0.0.1', 4002, clientId=10)


def get_contracts(contract_tuples):
    log.debug(f'initializing contract qualification')
    cont_contracts = [ContFuture(*contract)
                      for contract in contract_tuples]
    ib.qualifyContracts(*cont_contracts)
    ids = [contract.conId for contract in cont_contracts]
    contracts = [Future(conId=id) for id in ids]
    ib.qualifyContracts(*contracts)
    log.debug(f'Contracts qualified: {contracts}')
    return contracts


contracts = [
    ('NQ', 'GLOBEX'),
    ('ES', 'GLOBEX'),
    ('NKD', 'GLOBEX'),
    ('CL', 'NYMEX'),
    ('GC', 'NYMEX'),
]

# util.patchAsyncio()
trader = Trader(ib)
futures = get_contracts(contracts)
candles = [Candle(contract, trader, ib)
           for contract in futures]


ib.run()
