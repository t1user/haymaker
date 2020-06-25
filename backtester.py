from __future__ import annotations

from datetime import datetime
from itertools import count
import pickle
from typing import (NamedTuple, List, Dict, Any, Union, Tuple, Type, Optional,
                    ClassVar)

import pandas as pd
import numpy as np
from logbook import Logger

from ib_insync import IB as master_IB
from ib_insync import util
from ib_insync.contract import Future, ContFuture, Contract
from ib_insync.objects import (BarData, BarDataList, ContractDetails,
                               TradeLogEntry, Position, CommissionReport,
                               Execution, Fill)
from ib_insync.order import OrderStatus, Trade, MarketOrder, Order
from eventkit import Event

from datastore import Store
from trader import Manager

log = Logger(__name__)
util.patchAsyncio()


class IB:
    path = 'b_temp'

    events = ('barUpdateEvent', 'newOrderEvent', 'orderModifyEvent',
              'cancelOrderEvent', 'orderStatusEvent')

    ib = master_IB()

    def __init__(self, datasource_manager: DataSourceManager,
                 mode: str = 'use_ib', index: int = -1,
                 field: str = 'symbol') -> None:
        self.datasource = datasource_manager.get_history
        self.store = datasource_manager.store
        self.mode = mode
        self.index = index
        self.field = field
        self.market = Market()
        self._createEvents()
        self.id = count(1, 1)

    def _createEvents(self) -> None:
        self.barUpdateEvent = Event('barUpdateEvent')
        self.newOrderEvent = Event('newOrderEvent')
        self.orderModifyEvent = Event('orderModifyEvent')
        self.orderStatusEvent = Event('orderStatusEvent')
        self.cancelOrderEvent = Event('cancelOrderEvent')

    def read_from_file_or_ib(self, filename: str, method: str, obj: Any,
                             *args):
        try:
            with open(f'{self.path}/{filename}.pickle', 'rb') as f:
                c = pickle.load(f)
        except (FileNotFoundError, EOFError):
            c = dict()
        try:
            details = c[repr(obj)]
            log.debug(f'{filename} for {repr(obj)} read from file')
            return details
        except KeyError:
            with self.ib.connect(port=4002, clientId=2) as conn:
                f = getattr(conn, method)
                if args:
                    details = f(obj, *args)
                else:
                    details = f(obj)
                log.debug(
                    f'{filename} for {repr(obj)} read from ib')
            c[repr(obj)] = details
            with open(f'{self.path}/{filename}.pickle', 'wb') as f:
                log.debug(f'{filename} saved to file')
                pickle.dump(c, f)
            return details

    def reqContractDetails(self, contract: Contract):
        if self.mode == 'use_ib':
            return self.read_from_file_or_ib('details', 'reqContractDetails',
                                             contract)
        elif self.mode == 'db_only':
            contfuture_object = self.store.contfuture_contract_object(
                contract.symbol, self.index, self.field)
            meta = self.store.read_metadata(contfuture_object)
            details = ContractDetails(**{'contract': contfuture_object,
                                         'minTick': meta['min_tick'],
                                         'longName': meta['name']})
            log.debug(f'details for {contfuture_object}: {details}')
            return [details]

    def qualifyContracts(self, *contracts):
        """
        Modified copy of:
        https://ib-insync.readthedocs.io/_modules/ib_insync/ib.html#IB.qualifyContractsAsync
        """
        log.debug(f'qualifying contracts: {contracts}')
        detailsLists = (self.reqContractDetails(c) for c in contracts)
        result = []
        for contract, detailsList in zip(contracts, detailsLists):
            if not detailsList:
                log.error(f'unknown contract {contract}')
            else:
                c = detailsList[0].contract
                expiry = c.lastTradeDateOrContractMonth
                if expiry:
                    # remove time and timezone part as it will cuse problems
                    expiry = expiry.split()[0]
                    c.lastTradeDateOrContractMonth = expiry
                if contract.exchange == 'SMART':
                    # overwriting 'SMART' exchange can create invalid contract
                    c.exchange = contract.exchange
                contract.update(**c.dict())
                result.append(contract)
        return result

    def reqCommissionsFromIB(self, contracts: List) -> Dict:
        order = MarketOrder('BUY', 1)
        commissions = {contract.symbol: self.read_from_file_or_ib(
            'commission',  'whatIfOrder', contract, order)
            for contract in contracts}
        missing_commissions = []
        for contract, commission in commissions.copy().items():
            if not commission:
                missing_commissions.append(contract)
                del commissions[contract]
        commissions.update(self.getCommissionBySymbol(missing_commissions))
        return commissions

    def getCommissionBySymbol(self, commissions: List) -> Dict:
        with open(f'{self.path}/commissions_by_symbol.pickle', 'rb') as f:
            c = pickle.load(f)
        return {comm: c.get(comm) for comm in commissions}

    def reqCommissionsFromDB(self, contracts: List) -> Dict:
        return {contract.symbol: self.store.read_metadata(contract
                                                          )['commission']
                for contract in contracts}

    def reqHistoricalData(self,
                          contract: Contract,
                          durationStr: str,
                          barSizeSetting: str,
                          whatToShow: str,
                          useRTH: str,
                          endDateTime: str = '',
                          formatDate: int = 1,
                          keepUpToDate: bool = True
                          ):
        return self.datasource(contract, durationStr, barSizeSetting)

    reqHistoricalDataAsync = reqHistoricalData

    def positions(self):
        return self.market.account.positions.values()

    def accountValues(self):
        log.info(f'cash: {self.market.account.cash}')
        return [
            AccountValue(tag='TotalCashBalance',
                         value=self.market.account.cash),
            AccountValue(tag='UnrealizedPnL',
                         value=self.market.account.unrealizedPnL)
        ]

    def openTrades(self):
        return [v for v in self.market.trades
                if v.orderStatus.status not in OrderStatus.DoneStates]

    def placeOrder(self, contract, order):
        orderId = order.orderId or next(self.id)
        now = self.market.date
        trade = self.market.get_trade(order)
        if trade:
            # this is a modification of an existing order
            assert trade.orderStatus.status not in OrderStatus.DoneStates
            logEntry = TradeLogEntry(now, trade.orderStatus.status, 'Modify')
            trade.log.append(logEntry)
            trade.modifyEvent.emit(trade)
            self.orderModifyEvent.emit(trade)
        else:
            # this is a new order
            assert order.totalQuantity != 0, 'Order quantity cannot be zero'
            order.orderId = orderId
            order.permId = orderId
            orderStatus = OrderStatus(status=OrderStatus.PendingSubmit,
                                      remaining=order.totalQuantity)
            logEntry = TradeLogEntry(now, orderStatus, '')
            trade = Trade(contract, order, orderStatus, [], [logEntry])
            self.newOrderEvent.emit(trade)
            self.market.append_trade(trade)
        return trade

    def cancelOrder(self, order):
        now = self.market.date
        trade = self.market.get_trade(order)
        if trade:
            if not trade.isDone():
                # this is a placeholder for implementation of further methods
                status = trade.orderStatus.status
                if (status == OrderStatus.PendingSubmit and not order.transmit
                        or status == OrderStatus.Inactive):
                    newStatus = OrderStatus.Cancelled
                else:
                    newStatus = OrderStatus.Cancelled
                logEntry = TradeLogEntry(now, newStatus, '')
                trade.log.append(logEntry)
                trade.orderStatus.status = newStatus
                trade.cancelEvent.emit(trade)
                trade.statusEvent.emit(trade)
                self.cancelOrderEvent.emit(trade)
                self.orderStatusEvent.emit(trade)
                if newStatus == OrderStatus.Cancelled:
                    trade.cancelledEvent.emit(trade)
        else:
            log.error(f'cancelOrder: Unknown orderId {order.orderId}')

    def run(self):
        # This is a monkey fucking patch, needs to be redone TODO
        if self.mode == 'use_ib':
            commissions = self.reqCommissionsFromIB(
                self.market.ib_objects.values())
            commissions = {k: v.commission for k, v in commissions.items()}
        elif self.mode == 'db_only':
            commissions = self.reqCommissionsFromDB(
                self.market.ib_objects.values())
        else:
            raise ValueError(f'Mode should be one of "use_ib" or "db_only"')

        self.market.commissions = commissions

        self.market.ticks = {
            cont.symbol: self.reqContractDetails(cont)[0].minTick
            for cont in self.market.ib_objects.values()}
        log.debug(f'market.object.keys: {self.market.objects.keys()}')
        log.debug(f'properties set on Market: {self.market}')
        log.debug(f'commissions: {self.market.commissions}')
        log.debug(f'ticks: {self.market.ticks}')

        self.market.run()

    def sleep(self, *args):
        # this makes sense only if run in asyncio
        util.sleep()


class DataSourceManager:
    """
    Holds initilized DataSource objects. Responds to history requests by
    providing appropriate source object.

    DataSource objects are initialized first time they are called. Every
    subsequent call is directed to the existing object. It's up to this
    object to determine date range of the data returned.

    DataSourceManager ensures DataSource for every contract is a singleton.
    It prevents unnecessary calls to the datastore - the call usually happens
    only at DataSource initialization, while at the same time simulating
    access to broker every time get_history is called by the strategy.

    TODO: multiple objects for the same contract for different barSizeSetting
    """

    def __init__(self, store: Store, start_date: datetime, end_date: datetime):
        self.sources = {}
        self.store = store
        self.DataSource = DataSource.initialize(store, start_date, end_date)

    def _source(self, contract: Contract, durationStr: str,
                barSizeSetting: str) -> dict:
        if not self.sources.get(repr(contract)):
            self.sources[repr(contract)] = self.DataSource(contract,
                                                           durationStr,
                                                           barSizeSetting)
        return self.sources[repr(contract)]

    def get_history(self, contract: Contract, durationStr: str,
                    barSizeSetting: str) -> None:
        return self._source(contract, durationStr, barSizeSetting).get_history(
            Market().date, durationStr, barSizeSetting)


class DataSource:

    start_date: ClassVar[Optional[str]] = None
    end_date: ClassVar[Optional[str]] = None
    store: ClassVar[Optional[Store]] = None
    _timedelta: ClassVar[Optional[pd.Timedelta]] = None
    _df: ClassVar[Optional[pd.DataFrame]] = None
    _data: ClassVar[Optional[Dict]] = None

    def __init__(self, contract: Contract, durationStr: str,
                 barSizeSetting: str) -> None:
        self.contract = self.validate_contract(contract)
        self._true_start = self.set_start(durationStr)
        self.handle_freq(barSizeSetting)
        # this is index of data available for emissions
        self.index = self.df.loc[self.start_date: self.end_date].index
        Market().register(self)

    def set_start(self, durationStr: str) -> pd.Timedelta:
        """
        Ensure appropriate back-data is available to start simulation
        at self.start_date.
        """
        return min(self.start_date - self.durationStr_to_timedelta(durationStr),
                   self.start_date)

    def handle_freq(self, barSizeSetting: str) -> None:
        """Warn (TODO: convert) of requested data frequency mismatch."""
        barSize_timedelta = self.barSizeSetting_to_timedelta(barSizeSetting)
        if self.freq_multiplier(barSize_timedelta, self.freq) != 1:
            log.warning(f'Requested data frequency for contract '
                        f'{self.contract.localSymbol}: '
                        f'{barSize_timedelta.seconds} secs '
                        f'is different than data frequency in data store: '
                        f'{self.freq.seconds} secs')

    @property
    def freq(self) -> pd.Timedelta:
        """
        Return frequency of df as pd.Timedelta object on which properties:
        'seconds' or 'days' should be used.

        Frequency will be inferred first time freq is called and stored for
        subsequent usage.
        """
        if not self._timedelta:
            s = pd.Series(self.df.index)
            self._timedelta = (s.shift(-1) - s).mode()[0]
        return self._timedelta

    @staticmethod
    def freq_multiplier(freq1: pd.Timedelta, freq2: pd.Timedelta) -> float:
        """Determine ratio between requested and available data freq."""
        return freq1 / freq2

    @staticmethod
    def durationStr_to_timedelta(durationStr: str) -> pd.Timedelta:
        duration, string = durationStr.split(' ')
        duration = int(duration)
        if string in 'SDWMY':
            return pd.Timedelta(duration, string)
        else:
            raise ValueError("Invalid durationStr, has to be one of: SDWMY")

    @staticmethod
    def barSizeSetting_to_timedelta(barSizeSetting: str) -> pd.Timedelta:
        """
        Warning: Method doesn't do 'week' and 'month'.
        """
        duration, string = barSizeSetting.split(' ')
        duration = int(duration)
        d = {'secs': 's',
             'mins': 'm',
             'min': 'm',
             'hour': 'h',
             'hours': 'h',
             'day': 'D',
             }
        return pd.Timedelta(duration, d[string])

    @property
    def df(self) -> pd.DataFrame:
        """
        Datastore accessed only first time df is requested. Df is the basis for
        all other data formats.
        """

        if self._df is None:
            self._df = self.store.read(self.contract,
                                       start_date=self._true_start,
                                       end_date=self.end_date).sort_index(
                ascending=True)
        return self._df

    @property
    def data(self) -> Dict[pd.Timestamp, BarData]:
        """
        Return dictionary of data available for emissions as bars list.
        Dictionary populated first time property accessed, which is
        accessed on subsequent calls.
        """
        if self._data is None:
            chunk = self.df.loc[self.start_date: self.end_date]
            source = chunk.to_dict('index')
            self._data = {k: BarData(date=k).update(**v)
                          for k, v in source.items()}
        return self._data

    def get_history(self, end_date: pd.Datetime,
                    durationStr: str, barSizeSetting: str) -> BarDataList:
        """
        Convert df into bars list that will be available as history.
        This method is called (indirectly) by ib.reqHistoricalData
        so it has to return historical bars in bulk and then keep track
        of bars available for subsequent emits if keepUpToDate == True.
        """
        start_date = end_date - self.durationStr_to_timedelta(durationStr)
        assert start_date >= self._true_start, \
            (f'Requested date {start_date} '
             f'is earlier than available: {self._true_start}')
        log.debug(f'history data from {start_date} to {end_date}')
        self.bars = self.get_BarDataList(self.df.loc[start_date: end_date])
        return self.bars

    @classmethod
    def initialize(cls, datastore: Store, start_date: str,
                   end_date: Union[str, datetime] = datetime.now()
                   ) -> Type[DataSource]:
        """
        Used to set class attributes ahead of instantiation.
        Start and end dates are dates of actual simulation.
        Instantiation will handle setting appropriate instrument
        and deal with any back-data requirements.
        """
        cls.start_date = pd.to_datetime(start_date, format='%Y%m%d')
        cls.end_date = pd.to_datetime(end_date, format='%Y%m%d')
        if cls.start_date > cls.end_date:
            message = (f'End date: {cls.end_date} is before start date: '
                       f'{cls.start_date}')
            raise(ValueError(message))
        cls.store = datastore
        return cls

    def validate_contract(self, contract):
        if isinstance(contract, ContFuture):
            return contract
        elif isinstance(contract, Future):
            return ContFuture(**contract.dict()).update(secType='CONTFUT')

    def get_BarDataList(self, chunk: pd.DataFrame) -> BarDataList:
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

    def __repr__(self):
        return f'data source for {self.contract.localSymbol}'

    def emit(self, date):
        bar = self.bar(date)
        if bar:
            self.bars.append(bar)
            self.bars.updateEvent.emit(self.bars, True)
        else:
            log.warning(
                f'missing data bar {date} for {self.contract.localSymbol}')

    def bar(self, date):
        return self.data.get(date)


class Market:
    """
    Wrap around _Market, ensuring that it is a singleton and set parameters
    on it.
    """

    instance = None

    def __new__(cls, cash: Optional[float] = None,
                manager: Optional[Manager] = None, reboot: bool = False):
        if not Market.instance:
            Market.instance = _Market()
        else:
            if manager:
                Market.instance.manager = manager
                Market.instance.reboot()
            if reboot:
                Market.instance._reboot = reboot
            if cash:
                Market.instance.account = Account(cash)
        return Market.instance

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def __setattr__(self, name):
        return setattr(self.instance, name)


class _Market:
    """
    Singleton providing market simulations.
    """

    def __init__(self, reboot=False):
        self._reboot = reboot
        self.objects = {}
        self.ib_objects = {}
        self.prices = {}
        self.trades = []
        self.index = None
        self.date = None
        self.exec_id = count(1, 1)

    def get_trade(self, order: Order) -> Union[Trade, None]:
        for trade in self.trades:
            if trade.order == order:
                return trade

    def register(self, source: DataSource) -> None:
        """
        All DataSource objects must be registered so that Market can
        preemptively get contract parameters like min tick, commission, etc.
        """
        if self.index is None:
            self.index = source.index
        else:
            self.index.join(source.index, how='outer')
        self.index = self.index.sort_values()
        duplicates = self.index[self.index.duplicated()]
        if len(duplicates) != 0:
            log.error(f'index duplicates: {duplicates}')
        self.date = self.index[0]
        self.objects[source.contract.tradingClass] = source
        self.ib_objects[source.contract.tradingClass] = source.contract

    def date_generator(self):
        """
        Keep track of last used date. Helper generator method used by self._run
        """
        for date in self.index:
            yield(date)

    def run(self) -> None:
        """Interface method, providing entry point to run simulations."""
        log.debug(f'Market initialized with reboot={self._reboot}')
        log.debug(f'commission levels for instruments: {self.commissions}')
        log.debug(f'minTicks for instruments: {self.ticks}')

        try:
            getattr(self, 'manager')
        except AttributeError:
            raise AttributeError('No manager object provided to Market')

        util.run(self._run())
        self.post_mortem()

    async def _run(self) -> None:
        """
        Main loop that iterates through all data points and provides
        prices to other components.

        Has to be a coroutine to allow Trader to release control.
        """
        date = self.date_generator()
        day = None
        while True:
            # TODO: why two different conditions?
            try:
                self.date = next(date)
                log.debug(f'next date: {self.date}')
            except StopIteration:
                log.debug('Stop iteration reached')
                return
            if self.date is None:
                # Test if this condition can be removed
                raise(AttributeError('Date is None'))
                log.error('Date is None. Breaking out of while loop')
                break

            self.simulate_data_point()

            # check for new day to simulate reboot
            if day:
                if self.date.day != day and self._reboot:
                    self.reboot()
                    day = None
            else:
                day = self.date.day

    def simulate_data_point(self):
        """
        Emit prices available at given data point.
        Test if any orders should be executed. Mark to market.
        """
        for o in self.objects.values():
            o.emit(self.date)
        self.extract_prices()
        self.account.mark_to_market(self.prices)
        self.run_orders()
        log.debug(f'current date: {self.date}')
        log.debug(f'current prices: {self.prices}')

    def post_mortem(self):
        """Summary after simulation"""
        log.debug(f'Final cash position: {self.account.cash}')
        log.debug(f'Open positions: {self.account.positions}')

    def reboot(self):
        """
        Simulate system reboot. Typically performed at the end of each day.
        All pending events are cancelled on reboot to mimick real app.
        """

        for trade in self.trades:
            for event in trade.events:
                getattr(trade, event).clear()
        self.manager.onStarted(now=self.date)

    def append_trade(self, trade: Trade) -> None:
        """Put new trade on the list of open trades for further processing."""
        if trade.order.orderType == 'TRAIL':
            trade = self.set_trail_price(trade)
        self.trades.append(trade)

    def set_trail_price(self, trade: Trade) -> Trade:
        """Keep track of trailing price for trailStop orders."""
        position = self.account.positions.get(trade.contract.symbol)
        if position:
            avgCost = position.avgCost
            if isinstance(trade.contract, (Future, ContFuture)):
                price = avgCost / int(trade.contract.multiplier)
            else:
                price = avgCost
        else:
            log.warning(f'trail order without corresponding position')
            price = self.prices[trade.contract.tradingClass].open

        if trade.order.action.upper() == 'BUY':
            trade.order.trailStopPrice = price + trade.order.auxPrice
        else:
            trade.order.trailStopPrice = price - trade.order.auxPrice
        log.debug(
            f'initial trail price set at {trade.order.trailStopPrice}')
        return trade

    def extract_prices(self) -> None:
        """Extract 'current' prices for every instrument."""
        for contract, bars in self.objects.items():
            if bars.bar(self.date) is not None:
                self.prices[contract] = bars.bar(self.date)

    def run_orders(self) -> None:
        open_trades = [trade for trade in self.trades if not trade.isDone()]
        for trade in open_trades.copy():
            price_or_bool = self.validate_order(
                trade.order,
                self.prices[trade.contract.tradingClass])
            if price_or_bool:
                self.execute_trade(trade, price_or_bool)

    def validate_order(self, order: Order, price: BarData
                       ) -> Union[bool, float]:
        """
        Validate order, ie. check if the order should be executed.
        Depending on order type, pick correct validation method
        and run it.

        Every validation method should return either False or execution
        price.

        Returns:
        False - order is not triggered (not executed)
        or
        Float - price at which order should be executed
        """

        funcs = {'MKT': lambda x, y: price.open,
                 'STP': self.validate_stop,
                 'LMT': self.validate_limit,
                 'TRAIL': self.validate_trail}
        return funcs[order.orderType](order, price)

    def execute_trade(self, trade: Trade, price: float) -> None:
        """
        After order is validated (ie. it should be executed), do the actual
        execution. Execution price is passed as argument.
        """

        price = self.apply_slippage(
            price,
            self.ticks[trade.contract.symbol],
            trade.order.action.upper())
        log.debug(
            f'executing trade: {trade}, price: {price}, date: {self.date}')
        executed_trade = self.fill_trade(trade,
                                         next(self.exec_id),
                                         self.date,
                                         price)
        contract = trade.contract.symbol
        quantity = trade.order.totalQuantity
        commission = self.commissions[contract] * quantity
        log.debug(
            f'inside execute_trade: quantity: {quantity}, commission: {commission}')
        self.account.update_cash(-commission)
        pnl, new_position = self.account.update_positions(executed_trade)
        self.account.update_cash(pnl)
        log.debug(f'pnl pre-net: {pnl}')
        net_pnl = pnl - (2 * commission) if not new_position else 0
        log.debug(f'net_pnl: {net_pnl}')
        log.debug(f'new_position: {new_position}')
        # trade events should be emitted after position is updated
        trade.statusEvent.emit(trade)
        trade.fillEvent.emit(trade, trade.fills[-1])
        trade.filledEvent.emit(trade)
        self.update_commission(executed_trade, net_pnl, commission)

    @staticmethod
    def apply_slippage(price: float, tick: float, action: str) -> float:
        return price + tick if action == 'BUY' else price - tick
        # return price

    @staticmethod
    def validate_stop(order: Order, price: BarData) -> bool:
        price = (price.open, price.high, price.low, price.close)
        if order.action.upper() == 'BUY' and order.auxPrice <= max(price):
            return max(price)
        if order.action.upper() == 'SELL' and order.auxPrice >= min(price):
            return min(price)
        return False

    @staticmethod
    def validate_limit(order: Order, price: BarData) -> Union[bool, float]:
        price = (price.open, price.high, price.low, price.close)
        if order.action.upper() == 'BUY' and order.lmtPrice <= min(price):
            return order.lmtPrice
        if order.action.upper() == 'SELL' and order.lmtPrice >= max(price):
            return order.lmtPrice
        return False

    @staticmethod
    def validate_trail(order: Order, price: BarData) -> Union[None, float]:
        price = (price.open, price.high, price.low, price.close)
        # check if BUY order hit
        if order.action.upper() == 'BUY':
            if order.trailStopPrice <= max(price):
                return order.trailStopPrice
            else:
                order.trailStopPrice = min(order.trailStopPrice,
                                           min(price) + order.auxPrice)
                return False

        # check if SELL order hit
        if order.action.upper() == 'SELL':
            if order.trailStopPrice >= min(price):
                return order.trailStopPrice
            else:
                order.trailStopPrice = max(order.trailStopPrice,
                                           max(price) - order.auxPrice)
                return False

    @staticmethod
    def fill_trade(trade: Trade, exec_id: int, date: pd.datetime,
                   price: float) -> Trade:
        quantity = trade.order.totalQuantity
        execution = Execution(execId=exec_id,
                              time=date,
                              acctNumber='',
                              exchange=trade.contract.exchange,
                              side=trade.order.action,
                              shares=quantity,
                              price=price,
                              permId=trade.order.permId,
                              orderId=trade.order.orderId,
                              cumQty=quantity,
                              avgPrice=price,
                              lastLiquidity=quantity)
        commission = CommissionReport()
        fill = Fill(time=date,
                    contract=trade.contract,
                    execution=execution,
                    commissionReport=commission)
        trade.fills.append(fill)
        trade.orderStatus = OrderStatus(status='Filled',
                                        filled=quantity,
                                        remaining=0,
                                        avgFillPrice=price,
                                        lastFillPrice=price)
        trade.log.append(TradeLogEntry(time=date,
                                       status=trade.orderStatus,
                                       message=f'Fill @{price}'))

        return trade

    @staticmethod
    def update_commission(trade: Trade, pnl: float,
                          commission: float) -> None:
        old_fill = trade.fills.pop()
        commission = CommissionReport(execId=old_fill.execution.execId,
                                      commission=commission,
                                      currency=trade.contract.currency,
                                      realizedPNL=round(pnl, 2)
                                      )
        new_fill = Fill(time=old_fill.time,
                        contract=trade.contract,
                        execution=old_fill.execution,
                        commissionReport=commission)
        trade.fills.append(new_fill)
        trade.commissionReportEvent.emit(trade, new_fill, commission)


class Account:
    def __init__(self, cash: float) -> None:
        self.cash = cash
        self.mtm = {}
        self.positions = {}

    def update_cash(self, cash: float) -> None:
        self.cash += cash

    @staticmethod
    def extract_params(trade: Trade) -> TradeParams:
        contract = trade.contract
        quantity = trade.fills[-1].execution.shares
        price = trade.fills[-1].execution.price
        side = trade.fills[-1].execution.side.upper()
        position = quantity if side == 'BUY' else -quantity
        avgCost = (price
                   if not isinstance(contract, (Future, ContFuture))
                   else price * int(contract.multiplier))
        return TradeParams(contract, quantity, price, side, position,
                           avgCost)

    def mark_to_market(self, prices) -> None:
        self.mtm = {}
        positions = [(contract, position.position, position.avgCost)
                     for contract, position in self.positions.items()]
        log.debug(f'positions: {positions}')
        for contract, position in self.positions.items():
            self.mtm[contract] = position.position * (
                prices[contract].average * int(position.contract.multiplier)
                - position.avgCost)
            log.debug(f'mtm: {contract} {self.mtm[contract]}')

    @property
    def unrealizedPnL(self) -> float:
        pnl = sum(self.mtm.values())
        log.debug(f'UnrealizedPnL: {pnl}')
        return pnl

    def update_positions(self, trade: Trade) -> Tuple[float, bool]:
        log.debug(f'Account updating positions by trade: {trade}')
        params = self.extract_params(trade)
        if trade.contract.symbol in self.positions:
            pnl = self.update_existing_position(params)
            new = False
        else:
            self.open_new_position(params)
            pnl = 0
            new = True
        log.debug(f'update_positions returning: {pnl}, {new}')
        return (pnl, new)

    def update_existing_position(self, params: TradeParams) -> float:
        old_position = self.positions[params.contract.symbol]
        # quantities are signed, avgCost is not
        # avgCost is a notional of one contract
        old_quantity = old_position.position
        new_quantity = old_quantity + params.position
        message = (f'updating existing position for {params.contract.symbol}, '
                   f'old: {old_position}, new: {new_quantity}')
        log.debug(message)
        if new_quantity != 0:
            if np.sign(new_quantity * old_quantity) == 1:
                # fraction of position has been liqidated
                log.error("we shouldn't be here: fraction liquidated")
                pnl = ((params.avgCost - old_position.avgCost) *
                       (new_quantity - old_quantity))
                position = Position(
                    account='',
                    contract=params.contract,
                    position=new_quantity,
                    avgCost=old_position.avgCost)
                self.positions[params.contract.symbol] = position
            else:
                # position has been reversed
                log.error("we shouldn't be here: position reversed")
                pnl = ((params.avgCost - old_position.avgCost) *
                       old_quantity)
                position = Position(
                    account='',
                    contract=params.contract,
                    position=new_quantity,
                    avgCost=params.avgCost)
                self.positions[params.contract.symbol] = position
        else:
            log.debug(f'closing position for {params.contract.symbol}')
            log.debug(f'params: {params}')
            # postion has been closed
            pnl = ((params.avgCost - old_position.avgCost) *
                   old_quantity)
            del self.positions[params.contract.symbol]
        log.debug(f'updating cash by: {pnl}')
        return pnl

    def open_new_position(self, params: TradeParams) -> None:
        position = Position(
            account='',
            contract=params.contract,
            position=params.position,
            avgCost=params.avgCost
        )
        self.positions[params.contract.symbol] = position


class TradeParams(NamedTuple):
    contract: Contract
    quantity: int
    price: float
    side: str
    position: float = 0
    avgCost: float = 0


class AccountValue(NamedTuple):
    tag: str
    value: str
    account: str = '0001'
    currency: str = 'USD'
    modelCode: str = 'whatever the fuck this is'
