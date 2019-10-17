from datetime import datetime

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

    events = ('barUpdateEvent', )

    def __init__(self, start_date, end_date=datetime.today()):
        self._createEvents()
        self.start_date = pd.to_datetime(start_date, format='%Y%m%d')
        self.end_date = pd.to_datetime(end_date, format='%Y%m%d')
        self.store = Store()

    def _createEvents(self):
        self.barUpdateEvent = Event('barUpdateEvent')

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
        return Market().positions

    def openTrades(self):
        return Market().orders

    def placeOrder(self, contract, order):
        Market().orders.append(order)

    def cancelOrder(self, order):
        Market().orders.remove(order)

    def run(self):
        Market().run()


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
            self.last_bar = self.data.pop()
        if self.last_bar.date == date:
            self.bars.append(self.last_bar)
            self.bars.updateEvent.emit(self.bars, True)
            self.last_bar = None


class Market:
    class __Market:
        def __init__(self):
            self.objects = []
            self.orders = []
            self.positions = []
            self.index = None

        def register(self, source):
            if self.index is None:
                self.index = source.index
            else:
                self.index.join(source.index, how='outer')
            self.objects.append(source)

        def run(self):
            self.index = self.index.sort_values()
            for date in self.index:
                log.debug(f'current date: {date}')
                for o in self.objects:
                    o.emit(date)
                # execute orders

    instance = None

    def __new__(cls):
        if not Market.instance:
            Market.instance = Market.__Market()
        return Market.instance

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def __setattr__(self, name):
        return setattr(self.instance, name)
