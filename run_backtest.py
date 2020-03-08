import asyncio
from ib_insync import util
from logbook import ERROR, INFO, WARNING

from backtester import IB, DataSourceManager, Market
from logger import logger
from trader import Blotter
from datastore_pytables import Store
from trader import Manager, VolumeStreamer, ResampledStreamer
#from params import contracts
from params_backtest import contracts


log = logger(__file__[:-3])  # , INFO, INFO)

start_date = '20180401'
end_date = '20180414'
cash = 1e+5
store = Store()
source = DataSourceManager(store, start_date, end_date)
ib = IB(source)

util.logToConsole()
asyncio.get_event_loop().set_debug(True)

blotter = Blotter(save_to_file=False, filename='backtest', path='backtests',
                  note=f'_{start_date}_{end_date}')
manager = Manager(ib, contracts, VolumeStreamer,
                  leverage=15, blotter=blotter,
                  freeze_path='notebooks/freeze/backtest')
market = Market(cash, manager, True)
ib.run()
blotter.save()
manager.freeze()
