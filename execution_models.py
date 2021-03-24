from __future__ import annotations
from collections import defaultdict
from abc import ABC, abstractmethod
import string
import random

import numpy as np
from typing import NamedTuple, Optional, Dict, Any, List, Literal

from trader import Trader
from candle import Candle
from ib_insync import (Contract, Trade, Order, MarketOrder, LimitOrder,
                       StopOrder, BracketOrder, TagValue, Position)
from logger import Logger


log = Logger(__name__)


def round_tick(price: float, tick_size: float) -> float:
    floor = price // tick_size
    remainder = price % tick_size
    if remainder > (tick_size / 2):
        floor += 1
    return round(floor * tick_size, 4)


class ContractMemo(NamedTuple):
    sl_points: float
    min_tick: float


class AbstractBracketLeg(ABC):
    """
    For use by EventDrivenExecModel to create stop-loss and take-profit
    orders.

    Extract information from Trade object and create appropriate bracket
    order.
    """

    def __call__(self, trade: Trade, sl_points: int, min_tick: float,
                 *args, **kwargs) -> Order:
        self._extract_trade(trade)
        self.sl_points = sl_points
        self.min_tick = min_tick
        return self.order()

    def _extract_trade(self, trade: Trade) -> None:
        self.contract = trade.contract
        self.action = trade.order.action
        assert self.action in ('BUY', 'SELL')
        self.reverseAction = 'BUY' if self.action == 'SELL' else 'SELL'
        self.direction = 1 if self.reverseAction == 'BUY' else -1
        self.amount = trade.orderStatus.filled
        self.price = trade.orderStatus.avgFillPrice

    @abstractmethod
    def order(self) -> Order:
        raise NotImplementedError


class FixedStop(AbstractBracketLeg):
    """
    Stop-loss with fixed distance from the execution price of entry order.
    """

    def order(self) -> Order:
        sl_price = round_tick(
            self.price + self.sl_points * self.direction,
            self.minTick)
        log.info(f'STOP LOSS PRICE: {sl_price}')
        return StopOrder(self.reverseAction, self.amount, sl_price,
                         outsideRth=True, tif='GTC')


class TrailingStop(AbstractBracketLeg):
    """
    Stop loss trailing price by given distance.
    """

    def order(self) -> Order:
        distance = round_tick(self.sl_points, self.min_tick)
        log.info(f'TRAILING STOP LOSS DISTANCE: {distance}')
        return Order(orderType='TRAIL', action=self.reverseAction,
                     totalQuantity=self.amount, auxPrice=distance,
                     outsideRth=True, tif='GTC')


class TrailingFixedStop(TrailingStop):
    """
    Trailing stop loss that will adjust itself to fixed stop-loss after
    reaching specified trigger.
    """

    def __init__(self, multiple: float = 2) -> None:
        self.multiple = multiple

    def order(self) -> Order:
        sl = super().order()
        sl.adjustedOrderType = 'STP'
        sl.adjustedStopPrice = (self.price - self.direction
                                * self.multiple * sl.auxPrice)
        log.debug(f'adjusted stop price: {sl.adjustedStopPrice}')
        sl.triggerPrice = sl.adjustedStopPrice - self.direction * sl.auxPrice
        log.debug(f'stop loss for {self.contract.localSymbol} '
                  f'fixed at {sl.triggerPrice}')
        return sl


class TrailingAdjustableStop(TrailingStop):
    """
    Trailing stop-loss that will widen trailing distance after reaching
    pre-specified trigger.
    """

    def __call__(self, trade: Trade, sl_points: int, min_tick: float,
                 sl_trigger_multiple: float, sl_adjusted_multiple: float
                 ) -> Order:
        self.sl_trigger_multiple = sl_trigger_multiple
        self.sl_adjusted_multiple = sl_adjusted_multiple
        return super().__call__(trade, sl_points, min_tick)

    def order(self) -> Order:
        sl = super().order()
        # when trigger is penetrated
        sl.triggerPrice = (self.price - self.direction
                           * self.sl_trigger_multiple * sl.auxPrice)
        # sl order will remain trailing order
        sl.adjustedOrderType = 'TRAIL'
        # with a stop price of
        sl.adjustedStopPrice = (sl.triggerPrice + self.direction
                                * sl.auxPrice * self.sl_adjusted_multiple)
        # being trailed by fixed amount
        sl.adjustableTrailingUnit = 0
        # of:
        sl.adjustedTrailingAmount = sl.auxPrice * self.sl_adjusted_multiple
        return sl


class StopMultipleTakeProfit(AbstractBracketLeg):
    """
    Take-profit order with distance from entry price specified as multiple
    of stop-loss distance. The multiple is fixed, given on object
    initialization.
    """

    def __init__(self, multiple: float = 2) -> None:
        self.multiple = multiple

    def order(self) -> Order:
        tp_price = round_tick(
            self.price - self.sl_points * self.direction * self.multiple,
            self.min_tick)
        log.info(f'TAKE PROFIT PRICE: {tp_price}')
        return LimitOrder(self.reverseAction, self.amount, tp_price,
                          outsideRth=True, tif='GTC')


class StopFlexiMultipleTakeProfit(AbstractBracketLeg):
    """
    Take-profit order with distance from entry price specified as multiple
    of stop-loss distance. The multiple is flexible, passed every time object
    is called.
    """

    def __call__(self, trade: Trade, sl_points: int, min_tick: float,
                 tp_multiple: float) -> Order:
        self.tp_multiple = tp_multiple
        return super().__call__(trade, sl_points, min_tick)

    def order(self):
        tp_price = round_tick(
            self.price - self.sl_points * self.direction * self.tp_multiple,
            self.min_tick)
        log.info(f'TAKE PROFIT PRICE: {tp_price}')
        return LimitOrder(self.reverseAction, self.amount, tp_price,
                          tif='GTC')


class BaseExecModel:
    """
    Intermediary between Portfolio and Trader. Allows for fine tuning of order
    types used, order monitoring, post-order events, etc.

    Initiated object passed to Manager. Portfolio sends events to onEntry.
    onStarted executed on (re)start.
    """

    def __str__(self):
        return self.__class__.__name__

    def connect_trader(self, trader: Trader) -> None:
        """
        Used by Manager on initialization.
        """
        self.trade = trader.trade
        self._trader = trader

    def onStarted(self) -> None:
        pass

    @staticmethod
    def entry_order(action: str, quantity: int) -> Order:
        """
        Return order to be used for entry transactions.
        """
        return MarketOrder(action, quantity,
                           algoStrategy='Adaptive',
                           algoParams=[
                               TagValue('adaptivePriority', 'Normal')],
                           tif='Day')

    @staticmethod
    def close_order(action: str, quantity: int) -> Order:
        """
        Return order to be used for close transactions.
        """
        return MarketOrder(action, quantity, tif='GTC')

    @staticmethod
    def action(signal: int) -> str:
        """
        Convert numerical trade direction signal (-1, or 1) to string
        ('BUY' or 'SELL').
        """
        assert signal in (-1, 1), 'Invalid trade signal'
        return 'BUY' if signal == 1 else 'SELL'

    def onEntry(self, contract: Contract, signal: int, amount: int,
                *args, **kwargs) -> Trade:
        """
        Accept Portfolio event to execute position entry transaction.
        """
        return self.trade(contract, self.entry_order(self.action(signal),
                                                     amount), 'ENTRY')

    def onClose(self, contract: Contract, signal: int, amount: int,
                *args, **kwargs) -> Trade:
        """
        Accept Portfolio event to execute position close transaction.
        """
        return self.trade(contract, self.close_order(self.action(signal),
                                                     amount), 'CLOSE')


class EventDrivenExecModel(BaseExecModel):
    """
    Use events to attach stop-loss and optional take-profit orders after
    execution of entry order. After close transaction remove existing
    bracketing orders. On re(start) make sure that all positions have
    stop-losses and that there are no open orders not attached to any
    existing position.
    """

    contracts = {}

    def __init__(self, stop: AbstractBracketLeg = TrailingStop(),
                 take_profit: Optional[AbstractBracketLeg] = None):
        self.stop = stop
        self.take_profit = take_profit
        log.debug(f'execution model initialized {self}')

    def save_contract(self, contract, sl_points, obj):
        self.contracts[contract.symbol] = ContractMemo(
            sl_points=sl_points,
            min_tick=obj.details.minTick)

    def onEntry(self, contract: Contract, signal: int, amount: int,
                sl_points: float, obj: Candle) -> Trade:
        """
        Save information required for bracket orders and attach events
        that will attach brackets after order completion.
        """
        self.save_contract(contract, sl_points, obj)
        log.debug(
            f'{contract.localSymbol} entry signal: {signal} '
            f'sl_distance: {sl_points}')
        trade = super().onEntry(contract, signal, amount)
        trade.filledEvent += self.attach_bracket
        return trade

    def onClose(self, contract: Contract, signal: int, amount: int) -> Trade:
        """
        Attach events that will cancel any brackets on order completion.
        """
        log.debug(f'{contract.localSymbol} close signal: {signal}')
        trade = super().onClose(contract, signal, amount)
        trade.filledEvent += self.remove_bracket
        return trade

    def emergencyClose(self, contract: Contract, signal: int,
                       amount: int) -> None:
        """
        Used if on (re)start positions without associated stop-loss orders
        detected.
        """
        log.warning(f'Emergency close on restart: {contract.localSymbol} '
                    f'side: {signal} amount: {amount}')
        trade = self.trade(contract,
                           self.close_order(self.action(signal), amount),
                           'EMERGENCY CLOSE')
        trade.filledEvent += self.bracketing_action

    def order_kwargs(self):
        return {}

    def attach_bracket(self, trade: Trade) -> None:
        params = self.contracts[trade.contract.symbol]
        order_kwargs = self.order_kwargs()
        log.debug(f'attaching bracket with params: {params}, '
                  f'order_kwargs: {order_kwargs}')
        for bracket_order, label in zip(
                (self.stop, self.take_profit), ('STOP-LOSS', 'TAKE-PROFIT')):
            # take profit may be None
            if bracket_order:
                log.debug(f'bracket: {bracket_order}')
                order = bracket_order(trade, *params)
                order.update(**order_kwargs)
                log.debug(f'order: {order}')
                bracket_trade = self.trade(trade.contract, order, label)
                bracket_trade.filledEvent += self.bracketing_action

    def remove_bracket(self, trade: Trade) -> None:
        self.cancel_all_trades_for_contract(trade.contract)

    bracketing_action = remove_bracket

    def cancel_all_trades_for_contract(self, contract: Contract) -> None:
        open_trades = self._trader.trades()
        trades = defaultdict(list)
        for t in open_trades:
            trades[t.contract.localSymbol].append(t)
        for trade in trades[contract.localSymbol]:
            self._trader.cancel(trade)

    def onStarted(self) -> None:
        """
        Make sure all positions have corresponding stop-losses or
        emergency-close them. Check for stray orders (take profit or stop loss
        without associated position) and cancel them if any. Attach events
        canceling take-profit/stop-loss after one of them is hit.
        """
        # check for orphan positions
        trades = self._trader.trades()
        positions = self._trader.positions()
        positions_for_log = [(position.contract.symbol, position.position)
                             for position in positions]
        log.debug(f'positions on re-connect: {positions_for_log}')

        orphan_positions = self.check_for_orphan_positions(trades, positions)
        if orphan_positions:
            log.warning(f'orphan positions: {orphan_positions}')
            log.debug(f'len(orphan_positions): {len(orphan_positions)}')
            self.handle_orphan_positions(orphan_positions)

        orphan_trades = self.check_for_orphan_trades(trades, positions)
        if orphan_trades:
            log.warning(f'orphan trades: {orphan_trades}')
            self.handle_orphan_trades(orphan_trades)

        for trade in trades:
            trade.filledEvent += self.bracketing_action

    @staticmethod
    def check_for_orphan_positions(trades: List[Trade],
                                   positions: List[Position]) -> List[Trade]:
        trade_contracts = set([t.contract for t in trades
                               if t.order.orderType in ('STP', 'TRAIL', 'MKT')])
        position_contracts = set([p.contract for p in positions])
        orphan_contracts = position_contracts - trade_contracts
        orphan_positions = [position for position in positions
                            if position.contract in orphan_contracts]
        return orphan_positions

    @staticmethod
    def check_for_orphan_trades(trades, positions) -> List[Trade]:
        trade_contracts = set([t.contract for t in trades])
        position_contracts = set([p.contract for p in positions])
        orphan_contracts = trade_contracts - position_contracts
        orphan_trades = [trade for trade in trades
                         if trade.contract in orphan_contracts]
        return orphan_trades

    def handle_orphan_positions(self, orphan_positions: List[Position]
                                ) -> None:
        self._trader.ib.qualifyContracts(
            *[p.contract for p in orphan_positions])
        for p in orphan_positions:
            self.emergencyClose(
                p.contract, -np.sign(p.position), int(np.abs(p.position)))
            log.error(f'emergencyClose position: {p.contract}, '
                      f'{-np.sign(p.position)}, {int(np.abs(p.position))}')
            t = self._trader.ib.reqAllOpenOrders()
            log.debug(f'reqAllOpenOrders: {t}')
            trades_for_log = [(trade.contract.symbol, trade.order.action,
                               trade.order.orderType, trade.order.totalQuantity,
                               trade.order.lmtPrice, trade.order.auxPrice,
                               trade.orderStatus.status, trade.order.orderId)
                              for trade in self._trader.ib.openTrades()]
            log.debug(f'openTrades: {trades_for_log}')

    def handle_orphan_trades(self, trades: List[Trade]) -> None:
        for trade in trades:
            self._trader.cancel(trade)


class OcaExecModel(EventDrivenExecModel):
    """
    Use Interactive Brokers OCA (one cancells another) orders for stop-loss
    and take-profit.
    """

    oca_ids = []

    def oca_group(self):
        while (o := ''.join(random.choices(string.ascii_letters + string.digits,
                                           k=10)
                            )) in self.oca_ids:
            pass
        self.oca_ids.append(o)
        return o

    def order_kwargs(self):
        return {'ocaGroup': self.oca_group(), 'ocaType': 1}

    def report_bracket(self, trade: Trade) -> None:
        log.debug(f'Bracketing order filled: {trade.order}')

    bracketing_action = report_bracket


class AdjustableTrailingStopExecModel(EventDrivenExecModel):

    def __init__(self):
        self.stop = TrailingAdjustableStop()
        self.take_profit = None
        log.debug(f'Execution model initialized: {self}')

    def save_contract(self, contract, sl_points, obj):
        self.contracts[contract.symbol] = (
            sl_points,
            obj.details.minTick,
            obj.sl_trigger_multiple,
            obj.sl_adjusted_multiple)


class EventDrivenTakeProfitExecModel(OcaExecModel):
    """
    For every entry order obligatorily create stop-loss and take-profit
    orders. Use oca order mechanism.
    """

    def __init__(self):
        """
        Trailing stop-loss with trailing amount given in points.
        Take profit distance from entry price given as multiple of trailing
        distance.
        """
        self.stop = TrailingStop()
        self.take_profit = StopFlexiMultipleTakeProfit()
        log.debug(f'Excution model initialized: {self}')

    def save_contract(self, contract, sl_points, obj):
        self.contracts[contract.symbol] = (
            sl_points,
            obj.details.minTick,
            obj.tp_multiple)


class BracketExecModel(BaseExecModel):
    """
    Use Interactive Brokers bracket orders to implement stop-loss and
    take-profit.
    Work in progress - TODO
    """

    def getId(self):
        return self._trader.ib.client.getReqId()

    @staticmethod
    def entry_order(action: str, quantity: int, price: float, **kwargs
                    ) -> Order:
        return LimitOrder(action, quantity, price, **kwargs)

    @staticmethod
    def take_profit(action: str, quantity: int, price: float, **kwargs
                    ) -> Order:
        return LimitOrder(action, quantity, price, tif='GTC')

    @staticmethod
    def stop_loss(action: str, quantity: int, distance: float, **kwargs
                  ) -> Order:
        log.debug(f'stop_loss params: {action}, {quantity}, {distance}, '
                  f'kwargs: {kwargs}')
        return Order(action=action, totalQuantity=quantity, orderType='TRAIL',
                     auxPrice=distance, outsideRth=True, tif='GTC')

    @staticmethod
    def close_order(action: str, quantity: int) -> Order:
        return MarketOrder(action, quantity, tif='GTC')

    @staticmethod
    def algo_kwargs(priority: Literal['Normal', 'Urgent', 'Patient'] = 'Normal'
                    ) -> Dict[str, Any]:
        return {
            'algoStrategy': 'Adaptive',
            'algoParams': [TagValue('adaptivePriority', priority)]
        }

    def bracket(self, action: str, quantity: float,
                limitPrice: float, takeProfitPrice: float,
                stopLossDistance: float, **kwargs) -> BracketOrder:
        assert action in ('BUY', 'SELL')
        reverseAction = 'BUY' if action == 'SELL' else 'SELL'
        k = self.algo_kwargs()

        parent = self.entry_order(action, quantity, limitPrice, **k)
        parent.orderId = self.getId()
        parent.transmit = False

        takeProfit = self.take_profit(reverseAction, quantity, takeProfitPrice,
                                      **k)
        takeProfit.orderId = self.getId()
        takeProfit.transmit = False
        takeProfit.parentId = parent.orderId

        stopLoss = self.stop_loss(reverseAction, quantity, stopLossDistance,
                                  **k)
        stopLoss.orderId = self.getId()
        stopLoss.transmit = True
        stopLoss.parentId = parent.orderId

        return BracketOrder(parent, takeProfit, stopLoss)

    def price(self, action: str, contract: Contract) -> float:
        quote = self._trader.quote(contract)
        log.debug(f'quote for {contract.localSymbol}: {quote}')
        return quote.bid if action == 'SELL' else quote.ask

    def onEntry(self, contract: Contract, signal: int, amount: int,
                sl_points: int, obj: Candle) -> None:
        assert signal in (-1, 1)
        action = 'BUY' if signal == 1 else 'SELL'
        price = self.price(action, contract)
        takeProfitPrice = round_tick(
            price + sl_points * obj.tp_multiple * signal,
            obj.details.minTick)
        log.debug(f'sl_points: {sl_points}')
        bracket = self.bracket(action, amount, price, takeProfitPrice,
                               round_tick(sl_points, obj.details.minTick),
                               **self.algo_kwargs())
        _t = {}
        for order, label in zip(bracket, ('ENTRY', 'TAKE-PROFIT', 'STOP-LOSS')):
            _t[label] = self.trade(contract, order, label)
        if not self.monitor(_t['ENTRY']):
            log.info('Cannot fill entry order. Cancelling...')
            self._trader.cancel(_t['ENTRY'])

    def monitor(self, trade: Trade) -> bool:
        c = 0
        while c < 5:
            c += 1
            self._trader.ib.sleep(15)
            if trade.isDone():
                return True
            else:
                log.debug(f'{trade.contract.localSymbol} attempt {c} to amend '
                          f'{trade.order.orderType}')
                trade.order.limitPrice = self.price(trade.order.action,
                                                    trade.contract)
                self.trade(trade.contract, trade.order)
        return False

    def onClose(self, contract: Contract, signal: int, amount: int) -> None:
        log.debug(f'{contract.localSymbol} close signal: {signal}')
        action = 'BUY' if signal == 1 else 'SELL'
        self.trade(contract, self.close_order(action, amount), 'CLOSE')
