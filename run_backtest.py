import asyncio
from ib_insync import util
from logbook import ERROR

from backtester import IB, DataSource
from logger import logger
from trader import Blotter
from datastore_pytables import Store
from trader import Manager
from params import contracts


log = logger(__file__[:-3],)  # ERROR, ERROR)

start_date = '20190101'
end_date = '20190131'
cash = 1e+5
store = Store()
source = DataSource.initialize(store, start_date, end_date)
ib = IB(source, cash)

# util.patchAsyncio()
util.logToConsole()
asyncio.get_event_loop().set_debug(True)

blotter = Blotter(False, 'backtest')
manager = Manager(ib, contracts, leverage=15)
manager.onConnected()
ib.run()
blotter.save()
manager.freeze()
