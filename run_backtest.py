import asyncio
from ib_insync import util
from logbook import ERROR, INFO, WARNING

from backtester import IB, DataSource
from logger import logger
from trader import Blotter
from datastore_pytables import Store
from trader import Manager, VolumeStreamer, ResampledStreamer
#from params import contracts
from params_backtest import contracts


log = logger(__file__[:-3])  # , ERROR, ERROR)

start_date = '20190101'
end_date = '20191031'
cash = 1e+5
store = Store()
source = DataSource.initialize(store, start_date, end_date)
ib = IB(source, cash)

util.logToConsole()
asyncio.get_event_loop().set_debug(True)

blotter = Blotter(save_to_file=False, filename='backtest', path='backtests',
                  note=f'_{start_date}_{end_date}')
manager = Manager(ib, contracts, VolumeStreamer,
                  leverage=15, blotter=blotter)
manager.onConnected()
ib.run()
blotter.save()
manager.freeze()
