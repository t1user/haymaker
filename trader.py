import sys
from collections import defaultdict
from functools import partialmethod
from datetime import datetime
import csv

import pandas as pd

from eventkit import Event
from ib_insync import IB, util, MarketOrder, StopOrder
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
            durationStr='4 D',
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
        self.backfill = False

    def aggregate(self, bar):
        raise NotImplementedError


class VolumeCandle(BarStreamer):

    def __init__(self, avg_periods=30, span=5500):
        super().__init__()
        self.span = span
        self.avg_periods = avg_periods
        self.aggregator = 0
        self.reset_volume()
        self.process_back_data()
        self.subscribe()

    def reset_volume(self):
        """
        self.volume = util.df(self.bars).volume \
            .rolling(self.avg_periods).sum() \
            .ewm(span=self.span).mean().iloc[-1].round()
        """

        self.volume = util.df(self.bars).volume \
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
            #self.aggregator -= self.volume
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
        if not self.backfill:
            log.debug(f'new candle created for {self.contract.localSymbol}')
        self.append({'backfill': self.backfill,
                     'date': df.index[-1],
                     'open': df.open[0],
                     'high': df.high.max(),
                     'low': df.low.min(),
                     'close': df.close[-1],
                     # 'price': weighted_price,
                     'price': df.close[-1],
                     'volume': df.volume.sum()})


class SignalProcessor:

    def __init__(self, ib):
        self.ib = ib
        self._createEvents()

    def positions(self):
        positions = self.ib.positions()
        return {p.contract: p.position for p in positions}

    def _createEvents(self):
        self.entrySignal = Event('entrySignal')
        self.closeSignal = Event('closeSignal')

    def entry(self, contract, signal, atr):
        positions = self.positions()
        if not positions.get(contract):
            message = (f'entry signal emitted for {contract.localSymbol},'
                       f'signal: {signal}, atr: {atr}')
            log.debug(message)
            self.entrySignal.emit(contract, signal, atr)
        else:
            self.breakout(contract, signal)

    def breakout(self, contract, signal):
        positions = self.positions()
        if positions.get(contract):
            if positions.get(contract) * signal < 0:
                log.debug('close signal emitted')
                self.closeSignal.emit(contract, signal)


class Candle(VolumeCandle):

    periods = [5, 10, 20, 40, 80, ]  # 160, ]
    ema_fast = 120  # number of periods for moving average filter
    sl_atr = 1  # stop loss in ATRs
    atr_periods = 80  # number of periods to calculate ATR on
    time_int = 30  # interval in minutes to be used to define volume candle

    def __init__(self, contract, trader, ib, keep_ref=True):
        self.contract = contract
        self.ib = ib
        log.debug(f'candle init for contract {self.contract.localSymbol}')
        self.candles = []
        self.counter = 0
        trader.register(self.get_details(), self.contract)
        self.processor = SignalProcessor(ib)
        self.processor.entrySignal.connect(trader.onEntry, keep_ref=keep_ref)
        self.processor.closeSignal.connect(trader.onClose, keep_ref=keep_ref)
        super().__init__()

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

    def get_indicators(self):
        self.df = pd.DataFrame(self.candles)
        self.df.set_index('date', inplace=True)
        self.df['ema_fast'] = self.df.price.ewm(span=self.ema_fast).mean()
        self.df['atr'] = get_ATR(self.df, self.atr_periods)
        self.df['signal'] = get_signals(self.df.price, self.periods)
        if not self.backfill:
            log.debug(
                f'{self.contract.localSymbol} {self.df.iloc[-1].to_dict()}')
        self.process()

    def process(self):
        if self.df.backfill[-1]:
            return
        else:
            if self.counter == 0:
                self.freeze()
                self.counter += 1

        if self.df.signal[-1]:
            if self.df.signal[-1] * (self.df.price[-1] - self.df.ema_fast[-1]) > 0:
                self.processor.entry(
                    self.contract, self.df.signal[-1], self.df.atr[-1])
            elif self.df.signal[-1]:
                self.processor.breakout(self.contract, self.df.signal[-1])


class Trader:

    def __init__(self, ib, blotter=None):
        if blotter is None:
            blotter = Blotter()
        self.ib = ib
        self.blotter = blotter
        self.atr_dict = {}
        self.contract_details = {}
        log.debug('Trader initialized')

    def onEntry(self, contract, signal, atr):
        log.debug(f'entry signal handled for: {contract.localSymbol} {signal} {atr}')
        self.atr_dict[contract] = atr
        trade = self.trade(contract, signal)
        trade.filledEvent += self.attach_sl
        log.debug('entry order placed')

    def onClose(self, contract, signal):
        log.debug(f'close signal handled for: {contract.localSymbol} {signal}')
        positions = {p.contract: p.position for p in self.ib.positions()}
        if contract in positions:
            trade = self.trade(contract, signal)
            trade.filledEvent += self.remove_sl
            log.debug('closing order placed')

    def trade(self, contract, signal):
        if signal == 1:
            log.debug(f'entering buy order for {contract.localSymbol}')
            order = MarketOrder('BUY', 1)
        elif signal == -1:
            log.debug(f'entering sell order for {contract.localSymbol}')
            order = MarketOrder('SELL', 1)
        trade = self.ib.placeOrder(contract, order)
        return trade

    def attach_sl(self, trade):
        self.blotter.log_trade(trade, 'entry')
        side = {'BUY': 'SELL', 'SELL': 'BUY'}
        direction = {'BUY': -1, 'SELL': 1}
        contract = trade.contract
        action = trade.order.action
        amount = trade.orderStatus.filled
        price = trade.orderStatus.avgFillPrice
        log.info(f'TRADE EXECUTED: {contract.localSymbol} {action} @{price}')
        sl_points = self.atr_dict[contract]
        # TODO round to the nearest tick
        sl_price = self.round_tick(price + sl_points * direction[action],
                                   self.contract_details[contract].minTick)
        # sl_price = round(price + sl_points * direction[action])
        log.info(f'STOP LOSS PRICE: {sl_price}')
        order = StopOrder(side[action], amount, sl_price,
                          outsideRth=True, tif='GTC')
        trade = self.ib.placeOrder(contract, order)
        self.attach_events(trade)

    def attach_events(self, trade):
        trade.filledEvent += self.report_stopout
        trade.cancelledEvent += self.report_cancel
        log.debug(f'{trade.contract.localSymbol} stop loss attached')

    def remove_sl(self, trade):
        self.blotter.log_trade(trade, 'close')
        contract = trade.contract
        open_trades = self.ib.openTrades()
        # open_orders = self.ib.reqAllOpenOrders()
        orders = defaultdict(list)
        for t in open_trades:
            orders[t.contract].append(t.order)
        for order in orders[contract]:
            if order.orderType == 'STP':
                self.ib.cancelOrder(order)
                log.debug(f'stop loss removed for {contract.localSymbol}')

    def reconcile_stops(self):
        """
        To be executed on restart. For all existing stop-outs attach reporting
        events for the blotter.
        """
        # required to fill open trades list
        ib.reqAllOpenOrders()
        trades = ib.openTrades()
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

    def register(self, details, contract):
        self.contract_details[contract] = details

    def report_stopout(self, trade):
        message = (
            f'STOP-OUT for {trade.contract.localSymbol} '
            f'{trade.order.action} @{trade.orderStatus.avgFillPrice}'
        )
        log.info(message)
        self.blotter.log_trade(trade, 'stop-out')

    def report_cancel(self, trade):
        log.info('Stop loss order for {trade.contract.localSymbol} cancelled')


class Blotter:
    def __init__(self, save_to_file=True, filename=None, path='blotter'):
        if filename is None:
            filename = __file__.split('/')[-1][:-3]
        self.file = (f'{path}/{filename}_'
                     f'{datetime.today().strftime("%Y-%m-%d_%H-%M")}.csv')
        self.save_to_file = save_to_file
        self.fieldnames = ['sys_time', 'time', 'contract', 'action', 'amount', 'price',
                           'exec_ids', 'order_id', 'reason', 'com_exec_id',
                           'commission', 'currency', 'realizedPNL', 'com_sys_time']
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
        self.unsaved_trades[trade.order.orderId].update(
            {'com_exec_id': report.execId,
             'commission': report.commission,
             'currency': report.currency,
             'realizedPNL': report.realizedPNL,
             'com_sys_time': str(datetime.now())}
        )
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


def get_contracts(contract_tuples, ib):
    cont_contracts = [ContFuture(*contract)
                      for contract in contract_tuples]
    ib.qualifyContracts(*cont_contracts)
    ids = [contract.conId for contract in cont_contracts]
    contracts = [Future(conId=id) for id in ids]
    ib.qualifyContracts(*contracts)
    log.debug(f'Contracts qualified: {contracts}')
    return contracts
