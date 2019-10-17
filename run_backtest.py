import pickle
# from ib_insync import IB
from backtester import IB

from logger import logger
from trader import Candle, Trader, Blotter, get_contracts


log = logger(__file__[:-3])
#ib = IB()
ib = IB(start_date='20170901')
#ib.connect('127.0.0.1', 4001, clientId=10)


contracts = [
    ('NQ', 'GLOBEX'),
    ('ES', 'GLOBEX'),
    ('NKD', 'GLOBEX'),
    ('CL', 'NYMEX'),
    ('GC', 'NYMEX'),
]

# util.patchAsyncio()
blotter = Blotter()
trader = Trader(ib, blotter)
#futures = get_contracts(contracts, ib)
with open('contracts.pickle', 'rb') as f:
    futures = pickle.load(f)
candles = [Candle(contract, trader, ib)
           for contract in futures]


ib.run()
