from __future__ import annotations
from collections import defaultdict
from itertools import count

import numpy as np
from typing import NamedTuple, Optional, Dict, Any, List, Literal

from trader import Trader, Quote
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


class BracketLeg:

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

    def order(self) -> Order:
        raise NotImplementedError


class FixedStop(BracketLeg):

    def order(self) -> Order:
        sl_price = round_tick(
            self.price + self.sl_points * self.direction,
            self.minTick)
        log.info(f'STOP LOSS PRICE: {sl_price}')
        return StopOrder(self.reverseAction, self.amount, sl_price,
                         outsideRth=True, tif='GTC')


class TrailingStop(BracketLeg):

    def order(self) -> Order:
        distance = round_tick(self.sl_points, self.min_tick)
        log.info(f'TRAILING STOP LOSS DISTANCE: {distance}')
        return Order(orderType='TRAIL', action=self.reverseAction,
                     totalQuantity=self.amount, auxPrice=distance,
                     outsideRth=True, tif='GTC')


class TrailingFixedStop(TrailingStop):

    def __init__(self, multiple: float = 2) -> None:
        self.multiple = multiple

    def order(self) -> Order:
        sl = super().order()
        log.debug(sl)
        sl.adjustedOrderType = 'STP'
        sl.adjustedStopPrice = (self.price - self.direction
                                * self.multiple * sl.auxPrice)
        log.debug(f'adjusted stop price: {sl.adjustedStopPrice}')
        sl.triggerPrice = sl.adjustedStopPrice - self.direction * sl.auxPrice
        log.debug(f'stop loss for {self.contract.localSymbol} '
                  f'fixed at {sl.triggerPrice}')
        return sl


class StopMultipleTakeProfit(BracketLeg):

    def __init__(self, multiple: float = 2) -> None:
        self.multiple = multiple

    def order(self) -> Order:
        tp_price = round_tick(
            self.price - self.sl_points * self.direction * self.multiple,
            self.min_tick)
        log.info(f'TAKE PROFIT PRICE: {tp_price}')
        return LimitOrder(self.reverseAction, self.amount, tp_price,
                          outsideRth=True, tif='GTC')


class StopFlexiMultipleTakeProfit(BracketLeg):

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

    def __str__(self):
        return self.__class__.__name__

    def connect_trader(self, trader: Trader) -> None:
        self.trade = trader.trade
        self._trader = trader

    def onStarted(self) -> None:
        pass

    @staticmethod
    def entry_order(signal: int, amount: int) -> Order:
        raise NotImplementedError

    @staticmethod
    def close_order(signal: int, amount: int) -> Order:
        raise NotImplementedError

    def action(self, signal: int) -> str:
        assert signal in (-1, 1), 'Invalid trade signal'
        return 'BUY' if signal == 1 else 'SELL'

    def onEntry(self, contract: Contract, signal: int, amount: int,
                *args, **kwargs) -> None:
        raise NotImplementedError

    def onClose(self, contract: Contract, signal: int, amount: int,
                *args, **kwargs) -> None:
        raise NotImplementedError


class EventDrivenExecModel(BaseExecModel):

    contracts = {}

    def __init__(self, stop: BracketLeg = TrailingStop(),
                 take_profit: Optional[BracketLeg] = None):
        self.stop = stop
        self.take_profit = take_profit
        log.debug(f'execution model initialized {self}')

    @staticmethod
    def entry_order(action: int, quantity: int) -> Order:
        return MarketOrder(action, quantity,
                           algoStrategy='Adaptive',
                           algoParams=[
                               TagValue('adaptivePriority', 'Normal')],
                           tif='Day')

    @staticmethod
    def close_order(action: int, quanity: int) -> Order:
        return MarketOrder(action, quanity, tif='GTC')

    def save_contract(self, contract, sl_points, obj):
        self.contracts[contract.symbol] = ContractMemo(
            sl_points=sl_points,
            min_tick=obj.details.minTick)

    def onEntry(self, contract: Contract, signal: int, amount: int,
                sl_points: float, obj: Candle) -> None:
        self.save_contract(contract, sl_points, obj)
        log.debug(
            f'{contract.localSymbol} entry signal: {signal} '
            f'sl_distance: {sl_points}')
        trade = self.trade(contract,
                           self.entry_order(self.action(signal), amount),
                           'ENTRY')
        trade.filledEvent += self.attach_bracket

    def onClose(self, contract: Contract, signal: int, amount: int) -> None:
        log.debug(f'{contract.localSymbol} close signal: {signal}')
        trade = self.trade(contract,
                           self.close_order(self.action(signal), amount),
                           'CLOSE')
        trade.filledEvent += self.remove_bracket

    def emergencyClose(self, contract: Contract, signal: int,
                       amount: int) -> None:
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
        log.debug(f'positions on re-connect: {positions}')

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
                               if t.order.orderType in ('STP', 'TRAIL')])
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
            log.debug(f'openTrades: {self._trader.ib.openTrades()}')

    def handle_orphan_trades(self, trades: List[Trade]) -> None:
        for trade in trades:
            self._trader.cancel(trade)


class OcaExecModel(EventDrivenExecModel):

    count = count(1, 1)

    def order_kwargs(self):
        return {'ocaGroup': f'oca_{next(self.count)}',
                'ocaType': 1}

    def report_bracket(self, trade: Trade) -> None:
        log.debug(f'Bracketing order filled: {trade.order}')

    bracketing_action = report_bracket


class EventDrivenTakeProfitExecModel(OcaExecModel):

    def __init__(self):
        self.stop = TrailingStop()
        self.take_profit = StopFlexiMultipleTakeProfit()
        log.debug(f'Excution model initialized: {self}')

    def save_contract(self, contract, sl_points, obj):
        self.contracts[contract.symbol] = (
            sl_points,
            obj.details.minTick,
            obj.tp_multiple)


class BracketExecModel(BaseExecModel):

    def getId(self):
        return self._trader.ib.client.getReqId()

    @staticmethod
    def entry_order(action: str, quantity: int, price: float, **kwargs
                    ) -> Order:
        return LimitOrder(action, quantity, price, **kwargs)

    take_profit = entry_order

    @staticmethod
    def stop_loss(action: str, quantity: int, distance: float, **kwargs
                  ) -> Order:
        return Order(action, quantity, orderType='TRAIL', auxPrice=distance,
                     outsideRth=True, tif='GTC')

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

    def price_limit(self, quote: Quote, min_tick: float) -> float:
        spread = quote.ask - quote.bid

    def onEntry(self, contract: Contract, signal: int, amount: int,
                sl_points: int, obj: Candle) -> None:
        assert signal in (-1, 1)
        quote = self._trader.quote(contract)
        log.debug(f'quote: {quote}')
        action = 'BUY' if signal == 1 else 'SELL'
        price = quote.bid if action == 'SELL' else quote.ask
        takeProfitPrice = round_tick(
            price - sl_points * obj.tp_multiple * signal,
            obj.details.minTick)
        bracket = self.bracket(action, amount, price, takeProfitPrice,
                               sl_points, **self.algo_kwargs())
        for order, label in zip(bracket, ('ENTRY', 'TAKE-PROFIT', 'STOP-LOSS')):
            self.trade(contract, order, label)

    def onClose(self, contract: Contract, signal: int, amount: int) -> None:
        log.debug(f'{contract.localSymbol} close signal: {signal}')
        action = 'BUY' if signal == 1 else 'SELL'
        self.trade(contract, self.close_order(action, amount), 'CLOSE')
