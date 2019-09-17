from datetime import datetime, timedelta
import itertools
import asyncio
import os
import functools

import pandas as pd
from ib_insync import IB, util
from ib_insync.contract import Future, ContFuture
from datastore_pytables import Store


from connect import IB_connection
from config import max_number_of_workers

"""
Modelled on example here:
https://docs.python.org/3/library/asyncio-queue.html#examples

and here:
https://realpython.com/async-io-python/#using-a-queue
"""


ib = IB_connection().ib
# ib.connect('127.0.0.1', 4002, clientId=2)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


async def get_history(contract):
    await ib.qualifyContractsAsync(contract)
    print(f'contract {contract} qualified')
    dt = min(
        datetime.strptime(contract.lastTradeDateOrContractMonth, '%Y%m%d'),
        now)
    print(f'starting at {dt}')
    chunks = []
    while True:
        # print('getting bars for {}'.format(contract.contract))
        chunk = await ib.reqHistoricalDataAsync(contract,
                                                endDateTime=dt,
                                                durationStr='5 D',
                                                barSizeSetting='1 min',
                                                whatToShow='TRADES',
                                                useRTH=False,
                                                formatDate=1,
                                                keepUpToDate=False,
                                                )

        if not chunk:
            print('finished downloading for ', contract.localSymbol)
            break
        # chunks.append(chunk)
        dt = chunk[0].date  # - timedelta(minutes=1)
        print('downloaded data chunk for: ', contract.localSymbol, dt)
        store.write(util.df(chunk))
        print('saved data chunk for: ', contract.localSymbol)
    #all_chunks = [c for chunk in reversed(chunks) for c in chunks]
    #df = util.df(all_chunks)


def lookup_contracts(symbols):
    futures = []
    for s in symbols:
        futures.append([c.contract for c in ib.reqContractDetails(
            Future(**s, includeExpired=True))])
    return list(itertools.chain(*futures))


def lookup_continuous_contracts(symbols):
    return [ContFuture(**s) for s in symbols]


async def schedule_tasks(contract, queue):
    print(f'schedulling task for contract {contract.localSymbol}')
    await queue.put(functools.partial(get_history, contract))


async def worker(name, queue):
    while True:
        print(f'worker {name} started')
        task = await queue.get()
        await task()
        queue.task_done()
        print(f'worker {name} done')


async def main(contracts, number_of_workers):
    print('main function started')
    queue = asyncio.Queue()
    producers = [asyncio.create_task(schedule_tasks(c, queue))
                 for c in contracts]
    workers = [asyncio.create_task(worker(f'worker-{i}', queue))
               for i in range(number_of_workers)]
    await asyncio.gather(*producers)

    # wait until the queue is fully processed (implicitly awaits workers)
    await queue.join()

    # cancel all workers
    for w in workers:
        w.cancel()

    # wait until all worker tasks are cancelled
    await asyncio.gather(*workers, return_exceptions=True)


store = Store()
now = datetime.now()
symbols = pd.read_csv(os.path.join(
    BASE_DIR, 'contracts.csv')).to_dict('records')

contracts = [*lookup_continuous_contracts(symbols), *lookup_contracts(symbols)]

number_of_workers = min(len(contracts), max_number_of_workers)
ib.run(main(contracts, number_of_workers))
ib.disconnect()
