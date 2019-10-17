from datetime import datetime
from itertools import count

import pandas as pd
from logbook import Logger

from ib_insync import IB as master_IB
import pickle
from ib_insync import IB as master_IB
from ib_insync.contract import Future, ContFuture
from ib_insync.objects import BarData, BarDataList, ContractDetails
from ib_insync.order import Order, OrderStatus, Trade, LimitOrder, StopOrder
from eventkit import Event

from datastore_pytables import Store


log = Logger(__name__)


def ib_decorator(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError:
            with master_IB().connect(port=4002) as ib:
                func = getattr(ib, func.__name__)
                return func(*args, **kwargs)
    return wrapper


class IB:
    path = 'b_temp'

    events = ('barUpdateEvent', 'newOrderEvent', 'orderModifyEvent',
              'cancelOrderEvent', 'orderStatusEvent')

    def __init__(self, start_date, end_date=datetime.today()):
        self._createEvents()
        self.start_date = pd.to_datetime(start_date, format='%Y%m%d')
        self.end_date = pd.to_datetime(end_date, format='%Y%m%d')
        self.store = Store()
        self.id = count(1, 1)

    def _createEvents(self):
        self.barUpdateEvent = Event('barUpdateEvent')
        self.newOrderEvent = Event('newOrderEvent')
        self.orderModifyEvent = Event('orderModifyEvent')
        self.orderStatusEvent = Event('orderStatusEvent')

    def reqContractDetails(self, contract):
        try:
            with open(f'{self.path}/details.pickle', 'rb') as f:
                c = pickle.load(f)
        except (FileNotFoundError, EOFError):
            c = dict()
        try:
            return c[contract]
        except KeyError:
            with master_IB().connect(port=4002) as ib:
                details = ib.reqContractDetails(contract)
            c[contract] = details
            with open(f'{self.path}/details.pickle', 'wb') as f:
                pickle.dump(c, f)
            return details

    def reqHistoricalData(self, contract, durationStr, barSizeSetting,
                          *args, **kwargs):
        duration = int(durationStr.split(' ')[0]) * 2
        source = DataSource(self.store, contract,
                            duration, self.start_date, self.end_date)
        return source.bars

    @staticmethod
    def translate_time_string(time_string):
        d = {
            'secs': 's',
            'mins': 'm',
            'hours': 'h',
            'day': 'D',
            # 'week': 'W',
            # 'month': 'M'
        }
        return d['time_string']

    def positions(self):
        # TODO
        return self.market.positions.values()

    def openTrades(self):
        return [v for v in self.market.trades.values()
                if v.orderStatus.status not in OrderStatus.DoneStates]

    def placeOrder(self, contract, order):
        """
        Trade(contract, order, orderStatus, fills, log)
        """

        orderId = order.orderId or next(self.id)
        now = self.market.date
        trade = self.market.trades.get(order)
        if trade:
            # this is a modification of an existing order
            assert trade.orderStatus.status not in OrderStatus.DoneStates
            logEntry = TradeLogEntry(now, trade.orderStatus.status, 'Modify')
            trade.log.append(logEntry)
            trade.modifyEvent.emit(trade)
            self.orderModifyEvent.emit(trade)
        else:
            order.orderId = orderId
            orderStatus = OrderStatus(status=OrderStatus.PendingSubmit)
            logEntry = TradeLogEntry(now, orderStatus, '')
            trade = Trade(contract, order, orderStatus, [], [logEntry])
            self.newOrderEvent.emit(trade)
        self.market.trades[order] = trade
        return trade

    def cancelOrder(self, order):
        now = self.market.date
        trade = self.market.get(order)
        if trade:
            if not trade.isDone():
                status = trade.orderStatus.status
                if (status == OrderStatus.PendingSubmit and not order.transmit
                        or status == OrderStatus.Inactive):
                    newStatus = OrderStatus.Cancelled
                else:
                    newStatus = OrderStatus.PendingCancel
                logEntry = TradeLogEntry(now, newStatus, '')
                trade.log.append(logEntry)
                trade.orderStatus.status = newStatus
                trade.cancelEvent.emit(trade)
                trade.statusEvent.emit(trade)
                self.canceOrderEvent.emit(trade)
                self.orderStatusEvent.emit(trade)
                if newStatus == OrderStatus.Cancelled:
                    trade.cancelledEvent.emit(trade)
        else:
            log.error(f'cancelOrder: Unknown orderId {order.orderId}')

    def run(self):
        self.market = Market()
        self.market.run()


class DataSource:

    start_date = None
    end_date = None

    def __init__(self, store, contract, duration, start_date, end_date):
        self.store = store
        self.start_date = start_date
        self.end_date = end_date
        self.contract = self.validate_contract(contract)
        self.df = self.get_df()
        self.startup_end_point = self.start_date + pd.Timedelta(duration, 'D')
        log.debug(f'startup end point: {self.startup_end_point}')
        data_df = self.df.loc[:self.startup_end_point]
        self.index = data_df.index
        self.data = self.get_BarDataList(data_df)
        self.bars = self.startup
        self.last_bar = None
        Market().register(self)

    @classmethod
    def set_dates(cls, start_date, end_date=datetime.now()):
        cls.start_date = start_date
        cls.end_date = end_date
        return cls

    def get_df(self):
        start_date = self.start_date.strftime('%Y%m%d')
        end_date = self.end_date.strftime('%Y%m%d')
        date_string = f'index > {start_date} & index < {end_date}'
        return self.store.read(self.contract, 'min', date_string)

    @property
    def startup(self):
        log.debug(f'generating startup data for {self.contract.localSymbol}')
        startup_chunk = self.df.loc[self.startup_end_point:].sort_index(
            ascending=True)
        log.debug(f'startup chunk length: {len(startup_chunk)}')
        return self.get_BarDataList(startup_chunk)

    def validate_contract(self, contract):
        if isinstance(contract, ContFuture):
            return contract
        elif isinstance(contract, Future):
            return ContFuture().update(**contract.dict())

    def get_BarDataList(self, chunk):
        bars = BarDataList()
        tuples = list(chunk.itertuples())
        for t in tuples:
            bar = BarData(date=t.Index,
                          open=t.open,
                          high=t.high,
                          low=t.low,
                          close=t.close,
                          average=t.average,
                          volume=t.volume,
                          barCount=t.barCount)
            bars.append(bar)
        return bars

    def emit(self, date):
        if self.last_bar is None:
            try:
                self.last_bar = self.data[-1]
            except IndexError:
                pass
        else:
            if self.last_bar.date == date:
                self.bars.append(self.last_bar)
                self.bars.updateEvent.emit(self.bars, True)
                self.data.pop()
            else:
                self.last_bar = None


class Market:
    class __Market:
        def __init__(self):
            self.objects = {}
            self.trades = []
            self.positions = []
            self.index = None
            self.date = None

        def register(self, source):
            if self.index is None:
                self.index = source.index
            else:
                self.index.join(source.index, how='outer')
            self.objects[source.contract] = source

        def run(self):
            self.index = self.index.sort_values()
            for date in self.index:
                log.debug(f'current date: {date}')
                self.date = date
                self.run_orders()
                for o in self.objects.values():
                    o.emit(date)

        def run_orders(self):
            for trade in trades:
                if not trade.isDone:
                    if trade.order.orderType == 'MKT':
                        self.execute_order(trade)
                    elif trade.order.orderType == 'STP':
                        self.validate_stop(trade)
                    elif trade.order.orderType == 'LMT':
                        self.validate_limit(trade)

        def validate_stop(self, trade):
            pass

        def validate_limit(self, trade):
            pass

        def execute_order(self, trade):
            p = Position(account='',
                         contract=trade.contract,
                         position=(trade.order.totalQuantity
                                   if trade.order.action == 'BUY'
                                   else -trade.order.totalQuantity)
                         avgCost=self.objects[trade.contract].last_bar.open)

    instance = None

    def __new__(cls):
        if not Market.instance:
            Market.instance = Market.__Market()
        return Market.instance

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def __setattr__(self, name):
        return setattr(self.instance, name)
