from __future__ import annotations

from datetime import datetime
from itertools import count
import pickle
from typing import NamedTuple, List, Any, Union, Tuple, Type

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

from datastore_pytables import Store
from trader import Manager

log = Logger(__name__)
util.patchAsyncio()


class IB:
    path = 'b_temp'

    events = ('barUpdateEvent', 'newOrderEvent', 'orderModifyEvent',
              'cancelOrderEvent', 'orderStatusEvent')

    ib = master_IB()

    def __init__(self, datasource_manager: DataSourceManager,
                 ) -> None:
        self.datasource = datasource_manager.get_history
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

    def reqContractDetails(self, contract: Type[Contract]):
        return self.read_from_file_or_ib('details', 'reqContractDetails',
                                         contract)

    def qualifyContracts(self, *contracts):
        """
        Modified copy of:
        https://ib-insync.readthedocs.io/_modules/ib_insync/ib.html#IB.qualifyContractsAsync
        """
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

    def reqCommissions(self, contracts: List):
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

    def getCommissionBySymbol(self, commissions: List):
        with open(f'{self.path}/commissions_by_symbol.pickle', 'rb') as f:
            c = pickle.load(f)
        return {comm: c.get(comm) for comm in commissions}

    def reqHistoricalData(self, contract, durationStr, barSizeSetting,
                          *args, **kwargs):
        duration = int(durationStr.split(' ')[0]) * 2
        return self.datasource(contract, duration)

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
        commissions = self.reqCommissions(self.market.objects.keys())
        self.market.commissions = {cont: comm.commission
                                   for cont, comm in commissions.items()}
        self.market.ticks = {
            cont.symbol: self.reqContractDetails(cont)[0].minTick
            for cont in self.market.objects.keys()}
        log.debug(f'market.object.keys: {self.market.objects.keys()}')
        log.debug(f'properties set on Market: {self.market}')
        log.debug(f'commissions: {self.market.commissions}')
        log.debug(f'ticks: {self.market.ticks}')

        util.run(self.market.run())

    def sleep(self, *args):
        # this makes sense only if run in asyncio
        util.sleep()


class DataSourceManager:
    """
    Holds initilized DataSource objects. Responds to history requests by
    providing appropriate source object.
    """

    def __init__(self, store: Store, start_date: datetime, end_date: datetime):
        self.sources = {}
        self.DataSource = DataSource.initialize(store, start_date, end_date)

    def source(self, contract: Type[Contract], duration: int) -> dict:
        if not self.sources.get(repr(contract)):
            self.sources[repr(contract)] = self.DataSource(contract, duration)
        return self.sources[repr(contract)]

    def get_history(self, contract: Type[Contract], duration: int) -> None:
        return self.source(contract, duration).get_history(
            Market().date, duration)


class DataSource:

    start_date = None
    end_date = None
    store = None

    def __init__(self, contract: Type[Contract], duration: int) -> None:
        self.contract = self.validate_contract(contract)
        self.duration = duration
        start_date = self.start_date - pd.Timedelta(duration, 'D')
        self.pull_data(start_date=start_date)
        # this is index of data available for emissions
        self.index = self.df.loc[self.start_date: self.end_date].index
        Market().register(self)

    def pull_data(self, start_date: pd.Datetime = None) -> None:
        start_date = self.start_date if start_date is None else start_date
        self.df = self.store.read(self.contract,
                                  start_date=start_date,
                                  end_date=self.end_date).sort_index(
                                      ascending=True)

        # this is data for emissions as bars list
        self.data = self.get_dict(self.df.loc[self.start_date: self.end_date])

    def get_history(self, end_date: pd.Datetime, duration: int) -> BarDataList:
        """
        Convert df into bars list that will be available as history.
        """
        start_date = end_date - pd.Timedelta(duration, 'D')
        if start_date < self.df.index[0]:
            self.pull_data(start_date)
        log.debug(f'history data from {start_date} to {end_date}')
        self.bars = self.get_BarDataList(self.df.loc[start_date: end_date])
        return self.bars

    @classmethod
    def initialize(cls, datastore, start_date, end_date=datetime.now()):
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

    @staticmethod
    def get_dict(chunk):
        """
        Return a dict of:
        {'Timestamp': BarData(...)}
        """
        source = chunk.to_dict('index')
        return {k: BarData(date=k).update(**v) for k, v in source.items()}

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
    class __Market:
        def __init__(self, reboot=False):
            self._reboot = reboot
            self.objects = {}
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
            if self.index is None:
                self.index = source.index
            else:
                self.index.join(source.index, how='outer')
            self.index = self.index.sort_values()
            duplicates = self.index[self.index.duplicated()]
            if len(duplicates) != 0:
                log.error(f'index duplicates: {duplicates}')
            self.date = self.index[0]
            self.objects[source.contract] = source

        def date_generator(self):
            for date in self.index:
                yield(date)

        async def run(self) -> None:
            # has to be coroutine to allow Trader to release control
            log.debug(f'Market initialized with reboot={self._reboot}')
            log.debug(f'Market object: {self}')
            log.debug(f'commision levels for instruments: {self.commissions}')
            log.debug(f'minTicks for instruments: {self.ticks}')
            try:
                getattr(self, 'manager')
            except AttributeError:
                raise AttributeError(
                    'No manager object provided to the Market')
            date = self.date_generator()
            day = None

            while True:
                try:
                    self.date = next(date)
                except StopIteration:
                    return
                if self.date is None:
                    break
                if day:
                    if self.date.day != day and self._reboot:
                        self.reboot()
                        day = None
                else:
                    day = self.date.day
                log.debug(f'current date: {self.date}')
                for o in self.objects.values():
                    o.emit(self.date)
                self.extract_prices()
                self.account.mark_to_market(self.prices)
                self.run_orders()

        def reboot(self):
            # pending events must be cancelled on reboot to mimick real app
            for trade in self.trades:
                for event in trade.events:
                    getattr(trade, event).clear()
            self.manager.onConnected()

        def append_trade(self, trade: Trade) -> None:
            if trade.order.orderType == 'TRAIL':
                trade = self.set_trail_price(trade)
            self.trades.append(trade)

        def set_trail_price(self, trade: Trade) -> Trade:
            position = self.account.positions.get(trade.contract)
            if position:
                avgCost = position.avgCost
                if isinstance(trade.contract, (Future, ContFuture)):
                    price = avgCost / int(trade.contract.multiplier)
                else:
                    price = avgCost
            else:
                log.warning(f'trail order without corresponding position')
                price = self.prices[trade.contract.symbol].open

            if trade.order.action.upper() == 'BUY':
                trade.order.trailStopPrice = price + trade.order.auxPrice
            else:
                trade.order.trailStopPrice = price - trade.order.auxPrice
            log.debug(
                f'initial trail price set at {trade.order.trailStopPrice}')
            return trade

        def extract_prices(self) -> None:
            for contract, bars in self.objects.items():
                if bars.bar(self.date) is not None:
                    self.prices[contract.symbol] = bars.bar(self.date)

        def run_orders(self) -> None:
            open_trades = [
                trade for trade in self.trades if not trade.isDone()]
            for trade in open_trades.copy():
                price_or_bool = self.validate_order(
                    trade.order,
                    self.prices[trade.contract.symbol])
                if price_or_bool:
                    self.execute_trade(trade, price_or_bool)

        def validate_order(self, order: Order, price: BarData
                           ) -> Union[bool, float]:
            funcs = {'MKT': lambda x, y: True,
                     'STP': self.validate_stop,
                     'LMT': self.validate_limit,
                     'TRAIL': self.validate_trail}
            return funcs[order.orderType](order, price)

        def execute_trade(self, trade: Trade, price: float = None) -> None:
            """
            Execution price are passed as argument for limit, stop, stop-limit
            orders. For market orders True is passed and price has to be read
            from object properties.
            """
            if type(price) is bool:
                price = self.prices[trade.contract.symbol].open
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
            self.account.update_cash(-commission)
            pnl, new_position = self.account.update_positions(executed_trade)
            self.account.update_cash(pnl)
            net_pnl = pnl - (2 * commission) if not new_position else 0
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
                                  permId=exec_id,
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

    instance = None

    def __new__(cls, cash=None, manager=None, reboot=False):
        if not Market.instance:
            Market.instance = Market.__Market()
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


class Account:
    def __init__(self, cash: Union[int, float]) -> None:
        self.cash = cash
        self.mtm = {}
        self.positions = {}

    def update_cash(self, cash: Union[int, float]) -> None:
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
        positions = [(contract.symbol, position.position, position.avgCost)
                     for contract, position in self.positions.items()]
        log.debug(f'positions: {positions}')
        for contract, position in self.positions.items():
            self.mtm[contract.symbol] = position.position * (
                prices[contract.symbol].average * int(contract.multiplier)
                - position.avgCost)
            log.debug(f'mtm: {contract.symbol} {self.mtm[contract.symbol]}')

    @property
    def unrealizedPnL(self) -> float:
        pnl = sum(self.mtm.values())
        log.debug(f'UnrealizedPnL: {pnl}')
        return pnl

    def update_positions(self, trade: Trade) -> Tuple[float, bool]:
        log.debug(f'Account updating positions by trade: {trade}')
        params = self.extract_params(trade)
        if trade.contract in self.positions:
            pnl = self.update_existing_position(params)
            new = False
        else:
            self.open_new_position(params)
            pnl = 0
            new = True
        return (pnl, new)

    def update_existing_position(self, params: TradeParams) -> float:
        old_position = self.positions[params.contract]
        # quantities are signed avgCost is not
        # avgCost is a notional of one contract
        old_quantity = old_position.position
        new_quantity = old_quantity + params.position
        message = (f'updating existing position for {params.contract}, '
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
                self.positions[params.contract] = position
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
                self.positions[params.contract] = position
        else:
            log.debug(f'closing position for {params.contract}')
            log.debug(f'params: {params}')
            # postion has been closed
            pnl = ((params.avgCost - old_position.avgCost) *
                   old_quantity)
            del self.positions[params.contract]
        log.debug(f'updating cash by: {pnl}')
        return pnl

    def open_new_position(self, params: TradeParams) -> None:
        position = Position(
            account='',
            contract=params.contract,
            position=params.position,
            avgCost=params.avgCost
        )
        self.positions[params.contract] = position


class TradeParams(NamedTuple):
    contract: Type[Contract]
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
