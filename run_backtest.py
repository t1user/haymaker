import pickle
import asyncio
from ib_insync import util
from logbook import ERROR

from backtester import IB, DataSource
from logger import logger
from trader import Candle, Trader, Blotter, get_contracts
from datastore_pytables import Store

log = logger(__file__[:-3])  # , ERROR, ERROR)

start_date = '20180201'
store = Store()
source = DataSource.initialize(store, start_date)
ib = IB(source)


contracts = [
    ('NQ', 'GLOBEX'),
    ('ES', 'GLOBEX'),
    ('YM', 'ECBOT'),
    ('NKD', 'GLOBEX'),
    ('CL', 'NYMEX'),
    ('GC', 'NYMEX'),
    ('GE', 'GLOBEX'),
    ('ZB', 'ECBOT'),
    ('ZF', 'ECBOT'),
    ('ZN', 'ECBOT'),
]

# util.patchAsyncio()
util.logToConsole()
asyncio.get_event_loop().set_debug(True)

blotter = Blotter(False, 'backtest')
trader = Trader(ib, blotter)
futures = get_contracts(contracts, ib)
# with open('contracts.pickle', 'rb') as f:
#    futures = pickle.load(f)
candles = [Candle(contract, trader, ib)
           for contract in futures]


ib.run()
blotter.save()
for candle in candles:
    candle.freeze()
