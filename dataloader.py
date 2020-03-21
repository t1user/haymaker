from datetime import datetime
import asyncio
import functools
from typing import List, Type

from ib_insync import util, Contract

from logger import logger
from connect import IB_connection
from config import max_number_of_workers
from datastore import Store
from objects import ObjectSelector


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


async def schedule_task(contract, dt, queue):
    log.debug(f'schedulling task for contract {contract.localSymbol}')
    await queue.put(functools.partial(get_history, contract, dt))


async def worker(name, queue):
    while True:
        log.debug(f'{name} started')
        task = await queue.get()
        next_chunk = await task()
        if next_chunk:
            await schedule_task(**next_chunk, queue=queue)
        queue.task_done()
        log.debug(f'{name} done')


async def main(contracts: List[Type[Contract]], number_of_workers: int,
               now: datetime = datetime.now()):

    asyncio.get_event_loop().set_debug(True)

    log.debug(f'main function started, '
              f'retrieving data for {len(contracts)} instruments')

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

    ib = IB_connection().ib

    # object where data is stored
    store = Store()

    contracts = ObjectSelector(ib, '_contracts.csv').cont_list
    number_of_workers = min(len(contracts), max_number_of_workers)
    ib.run(main(contracts, number_of_workers), debug=True)
    log.debug('script finished, about to disconnect')
    ib.disconnect()
    log.debug(f'disconnected')
