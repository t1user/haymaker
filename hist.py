from pprint import pprint
from datetime import datetime, timedelta
import itertools
import asyncio
import os

import pandas as pd
from ib_insync import IB, util
from ib_insync.contract import Future, ContFuture

from connect import IB_connection


ib = IB_connection().ib
# ib.connect('127.0.0.1', 4002, clientId=2)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


async def get_history(contract):
    print('task started for: {}'.format(contract.localSymbol))
    dt = min(datetime.strptime(contract.lastTradeDateOrContractMonth, '%Y%m%d'),
             now)
    print(dt)
    chunks = []
    while True:
        try:
            # print('getting bars for {}'.format(contract.contract))
            chunk = await ib.reqHistoricalDataAsync(contract,
                                                    endDateTime=dt,
                                                    durationStr='2 D',
                                                    barSizeSetting='1 min',
                                                    whatToShow='TRADES',
                                                    useRTH=False,
                                                    formatDate=1,
                                                    keepUpToDate=False,
                                                    )
        except Exception as error:
            print('I caught error')
            print(error)

        if not chunk:
            print('finished downloading for ', contract.localSymbol)
            break
        chunks.append(chunk)
        dt = chunk[0].date  # - timedelta(minutes=1)
        print('downloaded data chunk for: ', contract.localSymbol, dt)

    all_chunks = [c for chunk in reversed(chunks) for c in chunks]
    df = util.df(all_chunks)


def lookup_contracts(symbols):
    futures = []
    for s in symbols:
        futures.append([c.contract for c in ib.reqContractDetails(
            Future(**s, includeExpired=True))])
    return list(itertools.chain(*futures))


def lookup_continuous_contracts(symbols):
    return [ContFuture(**s) for s in symbols]


def check_data_availability(symbol):
    return ib.reqHeadTimeStamp(symbol, whatToShow='TRADES', useRTH=False, formatDate=1)


async def schedule_tasks(contracts):
    print('schedulling tasks')
    tasks = []
    counter = 0
    for contract in contracts:
        assert await ib.qualifyContractsAsync(contract)
        # contract = await ib.reqContractDetailsAsync(contract)
        print('contract qualified: {}'.format(contract))
        task = asyncio.create_task(get_history(contract))
        tasks.append(task)
        counter += 1
        if counter > 49:
            print('sleeping for 1 sec')
            await asyncio.sleep(1)
            counter = 0
    print('task appended')
    return tasks


async def main(contracts):
    print('main function')
    tasks = await schedule_tasks(contracts)
    print('tasks scheduled')
    await asyncio.gather(*tasks, return_exceptions=True)
    print('main finished')


now = datetime.now()
symbols = pd.read_csv(os.path.join(
    BASE_DIR, '_contracts.csv')).to_dict('records')

contracts = [*lookup_continuous_contracts(symbols), *lookup_contracts(symbols)]
availability = {s.localSymbol: check_data_availability(
    s) for s in contracts}
pprint(availability)
print(len(contracts))
ib.run(main(contracts))


ib.disconnect()


"""
['ED',       'TY', 'FV', 'TU', 'US', 'FF', 'UL', 'NG',
           'RB', 'HO', 'BZ', 'EC', 'C', 'S', 'SI', 'PL', 'PA']

        contract = Future(contract,
                          exchange='GLOBEX',
                          currency='USD',
                          lastTradeDateOrContractMonth='201909',
                          )

"""
