import pickle
import asyncio
from ib_insync import util
from logbook import ERROR

from backtester import IB
from logger import logger
from trader import Candle, Trader, Blotter, get_contracts


log = logger(__file__[:-3])  # , ERROR, ERROR)
#ib = IB()
ib = IB(start_date='20180201')
#ib.connect('127.0.0.1', 4001, clientId=10)


contracts = [
    ('NQ', 'GLOBEX'),
    ('ES', 'GLOBEX'),
    ('NKD', 'GLOBEX'),
    ('CL', 'NYMEX'),
    ('GC', 'NYMEX'),
]

# util.patchAsyncio()
util.logToConsole()
asyncio.get_event_loop().set_debug(True)
blotter = Blotter(False, 'backtest')
trader = Trader(ib, blotter)
#futures = get_contracts(contracts, ib)
with open('contracts.pickle', 'rb') as f:
    futures = pickle.load(f)
candles = [Candle(contract, trader, ib)
           for contract in futures]


ib.run()
blotter.save()
