import itertools

import pandas as pd
from ib_insync import IB, util
from ib_insync.contract import Future


ib = IB()
ib.connect('127.0.0.1', 4002, clientId=5)

symbols = ['ES', 'NQ', 'YM', 'CL', 'GC', 'NKD', 'DAX']


def lookup_contracts(symbols):
    return pd.concat(
        [util.df(ib.reqContractDetails(
            Future(s)
        )) for s in symbols])


df = lookup_contracts(symbols)
print(df)
df.to_csv('symbols.csv')
ib.disconnect()
