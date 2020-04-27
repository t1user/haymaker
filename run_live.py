from ib_insync import IB

from handlers import Start
from trader import Manager
from strategy import candles, FixedPortfolio
from logger import logger


log = logger(__file__[:-3])


log.debug(f'candles: {candles}')
ib = IB()
manager = Manager(ib, candles, FixedPortfolio)
start = Start(ib, manager)
