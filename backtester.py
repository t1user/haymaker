from datetime import datetime
from itertools import count
import pickle

import pandas as pd
import numpy as np
from logbook import Logger

from ib_insync import IB as master_IB
from ib_insync import IB as master_IB
from ib_insync.contract import Future, ContFuture
from ib_insync.objects import (BarData, BarDataList, ContractDetails,
                               TradeLogEntry, Position, CommissionReport,
                               Execution, Fill)
from ib_insync.order import Order, OrderStatus, Trade, LimitOrder, StopOrder
from eventkit import Event

from datastore_pytables import Store


log = Logger(__name__)


class IB:
    path = 'b_temp'

    events = ('barUpdateEvent', 'newOrderEvent', 'orderModifyEvent',
              'cancelOrderEvent', 'orderStatusEvent')

    def __init__(self, datasource):
        self.datasource = datasource
        self._createEvents()
        self.id = count(1, 1)

    def _createEvents(self):
        self.barUpdateEvent = Event('barUpdateEvent')
        self.newOrderEvent = Event('newOrderEvent')
        self.orderModifyEvent = Event('orderModifyEvent')
        self.orderStatusEvent = Event('orderStatusEvent')
        self.cancelOrderEvent = Event('cancelOrderEvent')

    def reqContractDetails(self, contract):
        try:
            with open(f'{self.path}/details.pickle', 'rb') as f:
                c = pickle.load(f)
        except (FileNotFoundError, EOFError):
            c = dict()
        try:
            details = c[repr(contract)]
            log.debug(f'details for contract {repr(contract)} read from file')
            return details
        except KeyError:
            with master_IB().connect(port=4002) as conn:
                details = conn.reqContractDetails(contract)
                log.debug(f'details for contract {repr(contract)} read from ib')
            c[repr(contract)] = details
            with open(f'{self.path}/details.pickle', 'wb') as f:
                log.debug(f'contract details saved to file')
                pickle.dump(c, f)
            return details

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

    def reqHistoricalData(self, contract, durationStr, barSizeSetting,
                          *args, **kwargs):
        duration = int(durationStr.split(' ')[0]) * 2
        source = self.datasource(contract, duration)
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
        return self.market.positions.values()

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
            order.orderId = orderId
            orderStatus = OrderStatus(status=OrderStatus.PendingSubmit)
            logEntry = TradeLogEntry(now, orderStatus, '')
            trade = Trade(contract, order, orderStatus, [], [logEntry])
            self.newOrderEvent.emit(trade)
            self.market.trades.append(trade)
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
        self.market = Market()
        self.market.run()


class DataSource:

    start_date = None
    end_date = None
    store = None

    def __init__(self, contract, duration):
        self.contract = self.validate_contract(contract)
        # this is all available data as df
        df = self.store.read(self.contract, start_date=self.start_date,
                             end_date=self.end_date)
        self.startup_end_point = self.start_date + pd.Timedelta(duration, 'D')
        # this is data for emissions as df
        data_df = self.get_data_df(df)
        # this index will be available for emissions
        self.index = data_df.index
        # this is data for emissions as bars list
        self.data = self.get_dict(data_df)
        self.bars = self.startup(df)
        # self.last_bar = None
        Market().register(self)

    @classmethod
    def initialize(cls, datastore, start_date, end_date=datetime.now()):
        cls.start_date = pd.to_datetime(start_date, format='%Y%m%d')
        cls.end_date = pd.to_datetime(end_date, format='%Y%m%d')
        cls.store = datastore
        return cls

    def get_data_df(self, df):
        """
        Create bars list that will be available for emissions.
        Index is sorted: descending (but in current implementation, it doesn't matter)
        """
        log.debug(f'startup end point: {self.startup_end_point}')
        data_df = df.loc[:self.startup_end_point]
        return data_df

    def startup(self, df):
        """
        Create bars list that will be available pre-emissions (startup).
        Index is sorted: ascending.
        """
        log.debug(f'generating startup data for {self.contract.localSymbol}')
        startup_chunk = df.loc[self.startup_end_point:].sort_index(
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
        bar = self.data.get(date)
        if bar:
            self.bars.append(bar)
            self.bars.updateEvent.emit(self.bars, True)
        else:
            log.error(f'missing data bar {date} for {self.contract.localSymbol}')

    def bar(self, date):
        return self.data.get(date)


class Market:
    class __Market:
        def __init__(self):
            self.objects = {}
            self.trades = []
            self.positions = {}
            self.index = None
            self.date = None
            self.exec_id = count(1, 1)

        def get_trade(self, order):
            for trade in self.trades:
                if trade.order == order:
                    return trade

        def get_open_trades(self):
            trades = [trade for trade in self.trades if not trade.isDone]

        def register(self, source):
            if self.index is None:
                self.index = source.index
            else:
                self.index.join(source.index, how='outer')
            self.objects[source.contract] = source
            log.debug(f'object {source} registered with Market')

        def run(self):
            self.index = self.index.sort_values()
            log.info(f'index duplicates: {self.index[self.index.duplicated()]}')
            for date in self.index:
                log.debug(f'current date: {date}')
                self.date = date
                self.run_orders()
                for o in self.objects.values():
                    o.emit(date)

        def run_orders(self):
            for trade in self.trades:
                if self.objects[trade.contract].bar(self.date) is None:
                    continue
                self.price = self.objects[trade.contract].bar(
                    self.date).average
                if not trade.isDone():
                    if trade.order.orderType == 'MKT':
                        self.execute_order(trade)
                    elif trade.order.orderType == 'STP':
                        self.validate_stop(trade)
                    elif trade.order.orderType == 'LMT':
                        self.validate_limit(trade)

        def validate_stop(self, trade):
            order = trade.order
            if order.action.upper() == 'BUY' and order.auxPrice <= self.price:
                self.execute_order(trade)
            if order.action.upper() == 'SELL' and order.auxPrice >= self.price:
                self.execute_order(trade)

        def validate_limit(self, trade):
            order = trade.order
            if order.action.upper() == 'BUY' and order.lmtPrice >= self.price:
                self.execute_order(trade)
            if order.action.upper() == 'SELL' and order.lmtPrice <= self.price:
                self.execute_order(trade)

        def execute_order(self, trade):
            quantity = trade.order.totalQuantity
            exec_id = next(self.exec_id)
            execution = Execution(execId=exec_id,
                                  time=self.date,
                                  acctNumber='',
                                  exchange=trade.contract.exchange,
                                  side=trade.order.action,
                                  shares=quantity,
                                  price=self.price,
                                  permId=exec_id,
                                  orderId=trade.order.orderId,
                                  cumQty=quantity,
                                  avgPrice=self.price,
                                  lastLiquidity=quantity)
            commission = CommissionReport()
            fill = Fill(time=self.date,
                        contract=trade.contract,
                        execution=execution,
                        commissionReport=commission)
            trade.fills.append(fill)
            trade.orderStatus = OrderStatus(status='Filled',
                                            filled=quantity,
                                            avgFillPrice=self.price,
                                            lastFillPrice=self.price)
            trade.log.append(TradeLogEntry(time=self.date,
                                           status=trade.orderStatus,
                                           message=f'Fill @{self.price}'))

            trade.statusEvent.emit(trade)
            trade.fillEvent.emit(trade, fill)
            trade.filledEvent.emit(trade)
            self.update_positions(trade)

        def update_positions(self, trade):
            quantity = trade.fills[-1].execution.shares
            price = trade.fills[-1].execution.price
            side = trade.fills[-1].execution.side.upper()
            position = quantity if side == 'BUY' else -quantity
            avgCost = (price * quantity
                       if not isinstance(trade.contract, (Future, ContFuture))
                       else price * int(trade.contract.multiplier) * quantity)
            pnl = 0

            if trade.contract in self.positions:
                # trade corresponds to an open position
                old_position = self.positions[trade.contract]
                new_position = old_position.position + position
                if new_position != 0:
                    if np.sign(new_position * old_position.position) == 1:
                        # fraction of position has been liqidated
                        log.error("we shouldn't be here: fraction liquidated")
                        cost_base = (1 - new_position/old_position.position) \
                            * old_position.avgCost
                        pnl = (old_position.position - new_position) * price \
                            - cost_base * -np.sign(position)
                        position = Position(
                            account='',
                            contract=trade.contract,
                            position=new_position,
                            avgCost=old_position.avgCost - cost_base
                        )
                        self.positions[trade.contract] = position

                    else:
                        # position has been reversed
                        log.error("we shouldn't be here: position reversed")
                        closing_value = (old_position.position /
                                         position) * avgCost
                        pnl = (closing_value - old_position.avgCost) * \
                            -np.sign(position)
                        position = Position(
                            account='',
                            contract=trade.contract,
                            position=position,
                            avgCost=avgCost
                        )
                        self.positions[trade.contract] = position
                else:
                    # postion has been closed
                    pnl = (avgCost - old_position.avgCost) * -np.sign(position)
                    del self.positions[trade.contract]
            else:
                # this is a new position
                position = Position(
                    account='',
                    contract=trade.contract,
                    position=position,
                    avgCost=avgCost
                )
                self.positions[trade.contract] = position

            commission = CommissionReport(execId=trade.fills[-1].execution.execId,
                                          commission=1.3,
                                          currency=trade.contract.currency,
                                          realizedPNL=round(pnl, 2)
                                          )
            trade.commissionReportEvent.emit(
                trade, trade.fills[-1], commission)

    instance = None

    def __new__(cls):
        if not Market.instance:
            Market.instance = Market.__Market()
        return Market.instance

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def __setattr__(self, name):
        return setattr(self.instance, name)


class Positions:
    def __init__(self):
        self._positions = {}

    def update(self, trade):
        pass
