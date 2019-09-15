from datetime import datetime
# from ib_insync.ib.connect import IB_connection
from ib_insync.contract import Future, ContFuture
from ib_insync import IB, util
import asyncio


ib = IB()
ib.connect('127.0.0.1', 4002, clientId=2)


async def get_chunk(contract,
                    endDateTime,
                    durationStr,
                    barSizeSetting,
                    whatToShow,
                    useRTH=False,
                    formatDate=1):

    return ib.reqHistoricalData(contract,
                                endDateTime,
                                durationStr,
                                barSizeSetting,
                                whatToShow,
                                useRTH,
                                formatDate,
                                )


async def get_history(contract):
    dt = ''
    barsList = []
    while True:
        bars = await get_chunk(contract,
                               endDateTime=dt,
                               durationStr='10 D',
                               barSizeSetting='1 min',
                               whatToShow='TRADES',
                               )

        if not bars:
            break
        barsList.append(bars)
        dt = bars[0].date - datetime.timedelta(minutes=1)
        print(contract.contract.symbol, dt)

        allBars = [b for bars in reversed(barsList) for b in bars]
        df = util.df(allBars)


async def main(*args):
    await asyncio.gather(*args)

"""

dt = ''
barsList = []
while True:
    if not bars:
        break
    barsList.append(bars)
    dt = bars[0].date - datetime.timedelta(minutes=1)
    print(dt)

allBars = [b for bars in reversed(barsList) for b in bars]
df = util.df(allBars)
"""
dt = ''
contracts = ['ES', 'NQ', 'YM']

tasks = (
    get_history(contract=Future(contract,
                                exchange='GLOBEX',
                                currency='USD',
                                lastTradeDateOrContractMonth='201909')
                ) for contract in contracts
)
# print(asyncio.get_event_loop())
asyncio.run(*tasks)
