from datetime import datetime
import itertools
import asyncio
import os
import sys
import functools

import pandas as pd
from ib_insync import util
from ib_insync.contract import Future, ContFuture
from logbook import Logger, StreamHandler, FileHandler, set_datetime_format

from logger import logger
from connect import IB_connection
from config import max_number_of_workers
from datastore_pytables import Store


log = logger(__file__[:-3])

"""
Modelled (loosely) on example here:
https://docs.python.org/3/library/asyncio-queue.html#examples

and here:
https://realpython.com/async-io-python/#using-a-queue
"""


async def get_history(contract, dt):
    log.debug(f'loading {contract.localSymbol} ending {dt}')
    chunk = await ib.reqHistoricalDataAsync(contract,
                                            endDateTime=dt,
                                            durationStr='10 D',
                                            barSizeSetting='1 min',
                                            whatToShow='TRADES',
                                            useRTH=False,
                                            formatDate=1,
                                            keepUpToDate=False,
                                            )
    if not chunk:
        log.debug(f'Completed loading for {contract.localSymbol}')
        return None

    next_chunk = {'dt': chunk[0].date, 'contract': contract}
    log.debug(
        f'downloaded data chunk for: {contract.localSymbol} ending: {dt}')
    df = util.df(reversed(chunk))
    df.date = df.date.astype('datetime64')
    df.set_index('date', inplace=True)
    store.write(contract, df)
    log.debug(f'saved data chunk for: {contract.localSymbol}')
    return next_chunk


def lookup_contracts(symbols):
    futures = []
    for s in symbols:
        futures.append(
            [Future(**c.contract.dict())
             for c in ib.reqContractDetails(Future(**s, includeExpired=True))]
        )
    return list(itertools.chain(*futures))


def lookup_continuous_contracts(symbols):
    return [ContFuture(**s) for s in symbols]


async def schedule_task(contract, dt, queue):
    log.debug(f'schedulling task for contract {contract.localSymbol}')
    await queue.put(functools.partial(get_history, contract, dt))


def initial_schedule(contract, now):
    earliest = store.check_earliest(contract)
    if earliest:
        dt = earliest
    else:
        dt = min(
            datetime.strptime(contract.lastTradeDateOrContractMonth, '%Y%m%d'),
            now)
    return {'contract': contract, 'dt': dt}


async def worker(name, queue):
    while True:
        log.debug(f'{name} started')
        task = await queue.get()
        next_chunk = await task()
        if next_chunk:
            await schedule_task(**next_chunk, queue=queue)
        queue.task_done()
        log.debug(f'{name} done')


async def main(contracts, number_of_workers, now=datetime.now()):

    asyncio.get_event_loop().set_debug(True)

    log.debug('main function started')
    await ib.qualifyContractsAsync(*contracts)
    log.debug('contracts qualified')

    queue = asyncio.LifoQueue()
    producers = [asyncio.create_task(
        schedule_task(**initial_schedule(c, now), queue=queue))
        for c in contracts]
    workers = [asyncio.create_task(worker(f'worker {i}', queue))
               for i in range(number_of_workers)]
    await asyncio.gather(*producers, return_exceptions=True)

    # wait until the queue is fully processed (implicitly awaits workers)
    await queue.join()

    # cancel all workers
    log.debug('cancelling workers')
    for w in workers:
        w.cancel()

    # wait until all worker tasks are cancelled
    await asyncio.gather(*workers, return_exceptions=True)


if __name__ == '__main__':

    # logging
    # set_datetime_format("local")
    # StreamHandler(sys.stdout, bubble=True).push_application()
    # FileHandler(
    #    f'logs/{__file__[:-3]}_{datetime.today().strftime("%Y-%m-%d_%H-%M")}',
    #    bubble=True, delay=True).push_application()
    # log = Logger(__name__)

    # util.logToConsole()
    # get connection to Gateway

    ib = IB_connection().ib
    # ib.connect('127.0.0.1', 4002, clientId=2)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # object where data is stored
    store = Store()

    symbols = pd.read_csv(os.path.join(
        BASE_DIR, '_contracts.csv')).to_dict('records')

    # *lookup_contracts(symbols),
    contracts = [*lookup_continuous_contracts(symbols)]
    number_of_workers = min(len(contracts), max_number_of_workers)
    ib.run(main(contracts, number_of_workers))
    log.debug('script finished, about to disconnect')
    ib.disconnect()
