import asyncio
from ib_insync import util
from logbook import ERROR, INFO, WARNING

from backtester import IB, DataSource
from logger import logger
from trader import Blotter
from datastore_pytables import Store
from trader import Manager
from params import contracts


log = logger(__file__[:-3], INFO, INFO)  # ERROR, ERROR)

start_date = '20190201'
end_date = '20191031'
cash = 1e+5
store = Store()
source = DataSource.initialize(store, start_date, end_date)
ib = IB(source, cash)

util.logToConsole()
asyncio.get_event_loop().set_debug(True)

blotter = Blotter(save_to_file=False, filename='backtest', path='backtests')
manager = Manager(ib, contracts, leverage=15, blotter=blotter)
manager.onConnected()
ib.run()
blotter.save()
manager.freeze()
