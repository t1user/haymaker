from pprint import pprint
import pandas as pd
from ib_insync import IB, util
from ib_insync.contract import ContFuture, Future
from rx import create, amb, Observable, subject, catch_with_iterable, of, from_iterable, just, combine_latest
import eventkit as ev
from collections import deque

from indicators import (get_ATR, get_signals)

ib = IB()
ib.connect('127.0.0.1', 4002, clientId=21)


contract = ContFuture('NQ', exchange='GLOBEX')
ib.qualifyContracts(contract)

print('starting to load bars')


bars = ib.reqHistoricalData(
    contract,
    endDateTime='',
    durationStr='180 S',
    barSizeSetting='5 secs',
    whatToShow='TRADES',
    useRTH=False,
    formatDate=1,
    keepUpToDate=True)


# volume = util.df(bars).volume.rolling(3).sum().mean()
# print(f'volume: {volume}')


def onBarUpdate(bars, hasNewBar):
    if hasNewBar:
        yield bars[-1]
    yield None


o = Observable()

o.subscribe(
    on_next=lambda x: print(x))


#source = Observable.from_iterable(onBarUpdate)
# source.subscribe(printer)


print('bars loaded')
print(util.df(bars))

#bars.updateEvent += create
# ib.pendingTickersEvent += onTick


class Bar:

    def __init__(self):
        self.bars = deque()
        bars.updateEvent += self.append

    def append(self, bar, hasBars):
        if hasBars:
            self.bars.append(bar)

    def __iter__(self):
        return self

    def __next__(self):
        return self.bars.popleft()


"""
def push_bars(observer, scheduler):
    bars.updateEvent += observer.on_next
"""

bar = Bar()
source = from_iterable(bar)

source.subscribe(
    on_next=lambda i: print(i),
    on_error=lambda e: print("Error Occurred: {0}".format(e)),
    on_completed=lambda: print("Done!"),
)

ib.run()
