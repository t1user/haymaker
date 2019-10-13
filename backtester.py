from ib_insync import IB as master_IB
from ib_insync.objects import BarList, BarDataList
from ib_insync.order import Order, OrderStatus, Trade, LimitOrder, StopOrder
from eventkit import Event
from datastore_pytables import Store


class IB:

    events = ('barUpdateEvent', )
    connect = master_IB.connect
    disconnect = master_IB.disconnect
    isConnected = master_IB.isConnected
    qualifyContracts = master_IB.qualifyContracts

    def __init__(self):
        self._createEvents()
        self.store = Store()

    def _createEvents(self):
        self.barUpdateEvent = Event('barUpdateEvent')

    def reqHistoricalData(self, contract, durationStr, barSizeSetting,
                          *args):

        bars = BarDataList()
        self.store.read(contract)
