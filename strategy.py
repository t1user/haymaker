import sys
from pprint import pprint
from ib_insync import IB, util
from ib_insync.contract import ContFuture, Future

from logger import logger
from trader import Candle, Trader


log = logger(__file__[:-3])
ib = IB()
ib.connect('127.0.0.1', 4002, clientId=0)


# volume = util.df(bars).volume.rolling(3).sum().mean().round()
# print(f'volume: {volume}')


contracts = [
    ('NQ', 'GLOBEX'),
    ('ES', 'GLOBEX'),
    ('NKD', 'GLOBEX'),
    ('CL', 'NYMEX'),
    ('GC', 'NYMEX'),
]


def get_contract(*args):
    contract = ContFuture(*args)
    ib.qualifyContracts(contract)
    id = contract.conId
    contract = Future(conId=id)
    ib.qualifyContracts(contract)
    log.debug(f'Contract qualified: {contract}')
    return contract


contract = get_contract('NQ', 'GLOBEX')
trader = Trader()
#candle = Candle(contract, trader, ib)

candles = [Candle(get_contract(*contract), trader, ib)
           for contract in contracts]


util.patchAsyncio()
ib.run()
