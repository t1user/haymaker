import asyncio
from ib_insync import util
from logbook import ERROR, INFO, WARNING

from backtester import IB, DataSourceManager, Market
from logger import logger
from trader import Blotter
from datastore import Store
from trader import Manager
from strategy import candles, FixedPortfolio


log = logger(__file__[:-3])

start_date = '20190101'
end_date = '20190131'
cash = 1e+5
store = Store()
source = DataSourceManager(store, start_date, end_date)
ib = IB(source)

util.logToConsole()
asyncio.get_event_loop().set_debug(True)

blotter = Blotter(save_to_file=False, filename='backtest', path='backtests',
                  note=f'_{start_date}_{end_date}')
manager = Manager(ib, candles, FixedPortfolio, blotter=blotter,
                  freeze_path='notebooks/freeze/backtest')
market = Market(cash, manager, reboot=True)
ib.run()
blotter.save()
manager.freeze()
