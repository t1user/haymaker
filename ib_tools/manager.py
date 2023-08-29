import logging
from typing import Final

import ib_insync as ibi

# from ib_tools.trader import Trader


IB: Final[ibi.IB] = ibi.IB()
# TRADER: Final[Trader] = Trader(IB)


log = logging.getLogger(__name__)
