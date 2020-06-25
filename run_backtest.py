import asyncio
from ib_insync import util
from logbook import ERROR, INFO, WARNING

from backtester import IB, DataSourceManager, Market
from logger import logger
from trader import Blotter
from datastore import Store, ArcticStore
from trader import Manager
from strategy import candles, FixedPortfolio


log = logger(__file__[:-3])

start_date = '20180601'
end_date = '20191231'
cash = 2e+5
store = ArcticStore('TRADES_30_secs')
#store = Store()
source = DataSourceManager(store, start_date, end_date)
ib = IB(source, mode='db_only', index=-2)  # mode is: 'db_only' or 'use_ib'

util.logToConsole()
asyncio.get_event_loop().set_debug(True)

blotter = Blotter(save_to_file=False, filename='backtest', path='backtests',
                  note=f'_{start_date}_{end_date}')
manager = Manager(ib, candles, FixedPortfolio, blotter=blotter,
                  freeze_path='notebooks/freeze/backtest')
market = Market(cash, manager, reboot=False)
ib.run()
blotter.save()
manager.freeze()
