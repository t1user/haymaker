from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Any
from datetime import datetime
import csv

import pandas as pd
import numpy as np

from eventkit import Event
from ib_insync import util, Order, MarketOrder, StopOrder, IB
from ib_insync.objects import OrderStatus
from ib_insync.contract import ContFuture, Future

from logbook import Logger
from indicators import get_ATR, get_signals


log = Logger(__name__)


class BarStreamer:

    def __init__(self):
        self.bars = self.get_bars(self.contract)
        self.new_bars = []

    def get_bars(self, contract):
        return self.ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr='5 D',
            barSizeSetting='30 secs',
            whatToShow='TRADES',
            useRTH=False,
            formatDate=1,
            keepUpToDate=True)

    def subscribe(self):
        def onNewBar(bars, hasNewBar):
            if hasNewBar:
                self.aggregate(bars[-2])
        self.bars.updateEvent += onNewBar

    def process_back_data(self):
        self.backfill = True
        log.debug(f'generating startup data for {self.contract.localSymbol}')
        for bar in self.bars[:-2]:
            self.aggregate(bar)
        log.debug(f'startup data generated for {self.contract.localSymbol}')
        assert not self.df.iloc[-1].isna().any(), 'Not enough data for indicators'
        self.backfill = False
        self.freeze()

    def aggregate(self, bar):
        raise NotImplementedError

    def create_candle(self, bar):
        raise NotImplementedError

    def freeze(self):
        raise NotImplementedError


class VolumeCandle(BarStreamer):

    def __init__(self, avg_periods=60):
        super().__init__()
        self.aggregator = 0
        if not self.volume:
            if not self.avg_periods:
                self.avg_periods = avg_periods
            self.volume = self.reset_volume()
        self.process_back_data()
        self.subscribe()

    def reset_volume(self):
        return util.df(self.bars).volume \
            .rolling(self.avg_periods).sum() \
            .mean().round()

    def aggregate(self, bar):
        self.new_bars.append(bar)
        self.aggregator += bar.volume
        if not self.backfill:
            message = (f'{bar.date} {self.aggregator}/{self.volume}'
                       f' {self.contract.localSymbol}')
            log.debug(message)
        if self.aggregator >= self.volume:
            self.aggregator = 0
            # self.aggregator -= self.volume
            self.create_candle()

    def create_candle(self):
        df = util.df(self.new_bars)
        self.new_bars = []
        df.date = df.date.astype('datetime64')
        df.set_index('date', inplace=True)
        df['backfill'] = True
        # df['volume_weighted'] = (df.close + df.open)/2 * df.volume
        # df['volume_weighted'] = df.close * df.volume
        df['volume_weighted'] = df.average * df.volume
        weighted_price = df.volume_weighted.sum() / df.volume.sum()
        self.append({'backfill': self.backfill,
                     'date': df.index[-1],
                     'open': df.open[0],
                     'high': df.high.max(),
                     'low': df.low.min(),
                     'close': df.close[-1],
                     'price': weighted_price,
                     # 'price': df.close[-1],
                     'volume': df.volume.sum()})

    def append(self, candle):
        raise NotImplementedError


class Candle(VolumeCandle):

    def __init__(self, params, trader, portfolio, ib, keep_ref=True):
        self.__dict__.update(params.__dict__)
        self.params = params
        self.ib = ib
        self.portfolio = portfolio
        log.debug(f'candle init for contract {self.contract.localSymbol}')
        self.candles = []
        self.params.details = self.get_details()
        trader.register(self.params, self.contract.symbol)
        self._createEvents()
        self.entrySignal.connect(trader.onEntry, keep_ref=keep_ref)
        self.closeSignal.connect(trader.onClose, keep_ref=keep_ref)
        super().__init__()

    def _createEvents(self):
        self.entrySignal = Event('entrySignal')
        self.closeSignal = Event('closeSignal')

    def freeze(self):
        self.df.to_pickle(
            f'notebooks/freeze/freeze_df_{self.contract.localSymbol}.pickle')
        log.debug(f'freezed data saved for {self.contract.localSymbol}')

    def get_details(self):
        return self.ib.reqContractDetails(self.contract)[0]

    """
    def get_trading_hours(self):
        return ib.reqContractDetails(self.contract)[0].tradingHours.split(';')
    """

    def append(self, candle):
        self.candles.append(candle)
        self.get_indicators()
        if not self.backfill:
            self.process()
            self.freeze()

    def get_indicators(self):
        df = pd.DataFrame(self.candles)
        df.set_index('date', inplace=True)
        df['ema_fast'] = df.price.ewm(
            span=self.ema_fast, min_periods=int(self.ema_fast*.8)).mean()
        df['ema_slow'] = df.price.ewm(
            span=self.ema_slow, min_periods=int(self.ema_slow*.8)).mean()
        df['atr'] = get_ATR(df, self.atr_periods)
        df['signal'] = get_signals(df.price, self.periods)
        df['filter'] = np.sign(df['ema_fast'] - df['ema_slow'])
        df['filtered_signal'] = df['signal'] * \
            ((df['signal'] * df['filter']) == 1)
        self.df = df

    def process(self):
        message = (f'New candle for {self.contract.localSymbol} '
                   f'{self.df.iloc[-1].to_dict()}')
        log.debug(message)
        position = self.portfolio.positions.get(self.contract)
        if self.df.filtered_signal[-1] and not position:
            message = (f'entry signal emitted for {self.contract.localSymbol},'
                       f' signal: {self.df.filtered_signal[-1]},'
                       f' atr: {self.df.atr[-1]}')
            log.debug(message)
            self.entrySignal.emit(
                self.contract, self.df.signal[-1], self.df.atr[-1],
                self.portfolio.number_of_contracts(
                    self.params, self.df.price[-1]))
        elif self.df.signal[-1] and position:
            if position * self.df.signal[-1] < 0:
                log.debug(
                    f'close signal emitted for {self.contract.localSymbol}')
                self.closeSignal.emit(self.contract, self.df.signal[-1])


class Manager:

    def __init__(self, ib, contracts, leverage, trailing=True):
        self.ib = ib
        self.contracts = contracts
        self.portfolio = Portfolio(ib, leverage)
        self.trader = Trader(ib, self.portfolio, trailing)
        alloc = sum([c.alloc for c in contracts])
        assert alloc == 1, "Portfolio allocations don't add-up to 1"

    def onConnected(self):
        self.trader.reconcile_stops()
        contracts = get_contracts(self.contracts, self.ib)
        self.candles = [Candle(contract, self.trader, self.portfolio, self.ib)
                        for contract in contracts]

    def freeze(self):
        for candle in self.candles:
            candle.freeze()


class Portfolio:

    def __init__(self, ib, leverage):
        self.leverage = leverage
        self.ib = ib
        self.values = {}

    @property
    def account_value(self):
        self.update_value()
        return self.values['TotalCashBalance'] + min(
            self.values['UnrealizedPnL'], 0)

    @property
    def positions(self):
        positions = self.ib.positions()
        return {p.contract: p.position for p in positions}

    def update_value(self):
        tags = self.ib.accountValues()
        for item in tags:
            if item.currency == 'USD':
                try:
                    self.values[item.tag] = float(item.value)
                except ValueError:
                    pass

    def number_of_contracts(self, params, price):
        return int((self.account_value * self.leverage *
                    params.alloc) / (float(params.contract.multiplier) *
                                     price))


class Trader:

    def __init__(self, ib, portfolio, trailing=True, blotter=None):
        if blotter is None:
            blotter = Blotter()
        self.ib = ib
        self.portfolio = portfolio
        self.blotter = blotter
        self.trailing = trailing
        self.atr_dict = {}
        self.contracts = {}
        log.debug('Trader initialized')

    def onEntry(self, contract, signal, atr, amount):
        log.debug(
            f'entry signal handled for: {contract.localSymbol} {signal} {atr}')
        self.atr_dict[contract.symbol] = atr
        trade = self.trade(contract, signal, amount)
        trade.filledEvent += self.attach_sl
        log.debug('entry order placed')

    def onClose(self, contract, signal):
        message = (f'close signal handled for: {contract.localSymbol}'
                   f' signal: {signal}')
        log.debug(message)
        if contract in self.portfolio.positions:
            self.remove_sl(contract)
            # make sure sl didn't close the position before being removed
            if contract in self.portfolio.positions:
                trade = self.trade(contract, signal,
                                   abs(self.portfolio.positions[contract]))
                trade.filledEvent += self.report_close
                log.debug('closing order placed')

    def trade(self, contract, signal, amount):
        if signal == 1:
            log.debug(
                f'entering buy order for {amount} {contract.localSymbol}')
            order = MarketOrder('BUY', amount)
        elif signal == -1:
            log.debug(
                f'entering sell order for {amount} {contract.localSymbol}')
            order = MarketOrder('SELL', amount)
        trade = self.ib.placeOrder(contract, order)
        return trade

    def attach_sl(self, trade):
        self.blotter.log_trade(trade, 'entry')
        contract = trade.contract
        action = trade.order.action
        assert action in ('BUY', 'SELL')
        reverseAction = 'BUY' if action == 'SELL' else 'SELL'
        direction = 1 if reverseAction == 'BUY' else -1
        amount = trade.orderStatus.filled
        price = trade.orderStatus.avgFillPrice
        log.info(
            f'TRADE EXECUTED: {contract.localSymbol} {action} {amount} @{price}')
        sl_points = self.atr_dict[contract.symbol]
        if not self.trailing:
            sl_price = self.round_tick(
                price + sl_points * direction *
                self.contracts[contract.symbol].sl_atr,
                self.contracts[contract.symbol].details.minTick)
            log.info(f'STOP LOSS PRICE: {sl_price}')
            order = StopOrder(reverseAction, amount, sl_price,
                              outsideRth=True, tif='GTC')
        else:
            distance = self.round_tick(
                sl_points * self.contracts[contract.symbol].sl_atr,
                self.contracts[contract.symbol].details.minTick)
            log.info(f'TRAILING STOP LOSS DISTANCE: {distance}')
            order = Order(orderType='TRAIL', action=reverseAction,
                          totalQuantity=amount, auxPrice=distance,
                          outsideRth=True, tif='GTC')
        trade = self.ib.placeOrder(contract, order)
        log.debug(f'stop loss attached for {trade.contract.localSymbol}')
        self.attach_events(trade)

    def attach_events(self, trade):
        trade.filledEvent += self.report_stopout
        trade.cancelledEvent += self.report_cancel
        log.debug(f'Reporting events attached for {trade.contract.localSymbol} '
                  f'{trade.order.action} {trade.order.totalQuantity} '
                  f'{trade.order.orderType}')

    def remove_sl(self, contract):
        log.debug('inside remove_sl')
        open_trades = self.ib.openTrades()
        orders = defaultdict(list)
        for t in open_trades:
            orders[t.contract].append(t.order)
        for order in orders[contract]:
            if order.orderType in ('STP', 'TRAIL'):
                self.ib.cancelOrder(order)
                log.debug(f'stop loss removed for {contract.localSymbol}')

    def reconcile_stops(self):
        """
        To be executed on restart. For all existing stop-outs attach reporting
        events for the blotter.
        """
        # required to fill open trades list
        self.ib.reqAllOpenOrders()
        trades = self.ib.openTrades()
        for trade in trades:
            if trade.order.orderType == 'STP' and trade.orderStatus.remaining != 0:
                self.attach_events(trade)

    @staticmethod
    def round_tick(price, tick_size):
        floor = price // tick_size
        remainder = price % tick_size
        if remainder > .5:
            floor += 1
        return floor * tick_size

    def register(self, params, symbol):
        self.contracts[symbol] = params

    def report_stopout(self, trade):
        message = (f'STOP-OUT for {trade.contract.localSymbol} '
                   f'{trade.order.action} @{trade.orderStatus.avgFillPrice}')
        log.info(message)
        self.blotter.log_trade(trade, 'stop-out')

    def report_cancel(self, trade):
        log.info(f'Stop loss order for {trade.contract.localSymbol} cancelled')

    def report_close(self, trade):
        message = (f'CLOSE for {trade.contract.localSymbol} '
                   f'{trade.order.action} @{trade.orderStatus.avgFillPrice}')
        log.info(message)
        self.blotter.log_trade(trade, 'close')


class Blotter:
    def __init__(self, save_to_file=True, filename=None, path='blotter'):
        if filename is None:
            filename = __file__.split('/')[-1][:-3]
        self.file = (f'{path}/{filename}_'
                     f'{datetime.today().strftime("%Y-%m-%d_%H-%M")}.csv')
        self.save_to_file = save_to_file
        self.fieldnames = ['sys_time', 'time', 'contract', 'action', 'amount',
                           'price', 'exec_ids', 'order_id', 'reason',
                           'com_exec_id', 'commission', 'currency',
                           'realizedPNL', 'com_sys_time']
        self.unsaved_trades = {}
        self.blotter = []
        if self.save_to_file:
            self.create_header()

    def create_header(self):
        with open(self.file, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log_trade(self, trade, reason=''):
        sys_time = str(datetime.now())
        time = trade.log[-1].time
        contract = trade.contract.localSymbol
        action = trade.order.action
        amount = trade.orderStatus.filled
        price = trade.orderStatus.avgFillPrice
        exec_ids = [fill.execution.execId for fill in trade.fills]
        order_id = trade.order.orderId
        reason = reason
        row = {'sys_time': sys_time,
               'time': time,
               'contract': contract,
               'action': action,
               'amount': amount,
               'price': price,
               'exec_ids': exec_ids,
               'order_id': order_id,
               'reason': reason}
        self.unsaved_trades[order_id] = row
        trade.commissionReportEvent += self.update_commission

    def update_commission(self, trade, fill, report):
        # commission report might be for partial fill, those are ignored
        if trade.orderStatus not in OrderStatus.DoneStates:
            return

        comm_report = defaultdict(int)
        for fill in trade.fills:
            comm_report['commission'] += fill.commissionReport.commision
            comm_report['realizedPNL'] += fill.commissionReport.realizedPNL
        self.unsaved_trades[trade.order.orderId].update(comm_report)

        if self.save_to_file:
            self.write_to_file(self.unsaved_trades[trade.order.orderId])
        else:
            self.blotter.append(self.unsaved_trades[trade.order.orderId])
        del self.unsaved_trades[trade.order.orderId]

    def write_to_file(self, data):
        with open(self.file, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(data)

    def save(self):
        self.create_header()
        with open(self.file, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            for item in self.blotter:
                writer.writerow(item)


@dataclass
class Params:
    # = [5, 10, 20, 40, 80, ]  # 160, ]
    contract: Tuple[str]
    periods: List[int]
    ema_fast: int  # = 120  # number of periods for moving average filter
    ema_slow: int
    sl_atr: int  # = 1  # stop loss in ATRs
    atr_periods: int  # = 80  # number of periods to calculate ATR on
    alloc: float
    avg_periods: int = None
    volume: int = None


def get_contracts(contracts: Any, ib: IB):
    def convert(contract_tuples: List[tuple]):
        cont_contracts = [ContFuture(*contract)
                          for contract in contract_tuples]
        ib.qualifyContracts(*cont_contracts)
        #log.debug(f'Contracts qualified: {cont_contracts}')

        # Converting to Futures potentially unnecessary
        # But removing below likely breaks the backtester
        # TODO
        ids = [contract.conId for contract in cont_contracts]
        contracts = [Future(conId=id) for id in ids]
        ib.qualifyContracts(*contracts)
        log.debug(f'Contracts qualified: {contracts}')
        return contracts

    if type(contracts[0]) is not tuple:
        conts = convert([contract.contract for contract in contracts])
        for contract, cont in zip(contracts, conts):
            contract.contract = cont
        return contracts
    else:
        return convert(contracts)
