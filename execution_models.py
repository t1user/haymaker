from __future__ import annotations

import numpy as np
from typing import NamedTuple, Optional

from trader import Trader
from candle import Candle
from ib_insync import (Contract, Trade, Order, MarketOrder, LimitOrder,
                       StopOrder, TagValue)
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

    def __call__(self, trade: Trade, sl_points: int, min_tick: float) -> Order:
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

    def __init__(self, multiple):
        self.multiple = multiple

    def order(self) -> Order:
        tp_price = round_tick(self.price -
                              self.sl_points * self.direction * self.multiple,
                              self.min_tick)
        log.info(f'TAKE PROFIT PRICE: {tp_price}')
        return LimitOrder(self.reverseAction, self.amount, tp_price,
                          outsideRth=True, tif='GTC')


class BaseExecModel:

    contracts = {}

    def __init__(self, trader):
        self.trade = trader.trade
        self._trader = trader

    def __str__(self):
        return self.__class__.__name__

    @staticmethod
    def entry_order(signal: int, amount: int) -> Order:
        raise NotImplementedError

    @staticmethod
    def close_order(signal: int, amount: int) -> Order:
        raise NotImplementedError

    def onEntry(self, contract: Contract, signal: int, amount: int,
                *args, **kwargs) -> None:
        raise NotImplementedError

    def onClose(self, contract: Contract, signal: int, amount: int,
                *args, **kwargs) -> None:
        raise NotImplementedError


class EventDrivenExecModel(BaseExecModel):

    def __init__(self, trader: Optional[Trader] = None,
                 stop: BracketLeg = TrailingStop(),
                 take_profit: Optional[BracketLeg] = None):
        if trader is None:
            trader = Trader()
        super().__init__(trader)
        self.stop = stop
        self.take_profit = take_profit
        log.debug(f'execution model initialized {self}')

    @staticmethod
    def entry_order(signal: int, amount: int) -> Order:
        assert signal in [-1, 1], 'Invalid trade signal'
        return MarketOrder('BUY' if signal == 1 else 'SELL',
                           amount, algoStrategy='Adaptive',
                           algoParams=[
                               TagValue('adaptivePriority', 'Normal')],
                           tif='Day')

    @staticmethod
    def close_order(signal: int, amount: int) -> Order:
        assert signal in [-1, 1], 'Invalid trade signal'
        return MarketOrder('BUY' if signal == 1 else 'SELL',
                           amount, tif='GTC')

    def onEntry(self, contract: Contract, signal: int, amount: int,
                sl_points: float, obj: Candle) -> None:
        self.contracts[contract.symbol] = ContractMemo(
            sl_points=sl_points,
            min_tick=obj.details.minTick)
        log.debug(
            f'{contract.localSymbol} entry signal: {signal} '
            'sl_distance: {sl_distance}')
        trade = self.trade(contract, self.entry_order(signal, amount), 'ENTRY')
        trade.filledEvent += self.attach_bracket

    def onClose(self, contract: Contract, signal: int, amount: int) -> None:
        # TODO can sl close position before being removed?
        # make sure sl didn't close the position before being removed
        log.debug(f'{contract.localSymbol} close signal: {signal}')
        trade = self.trade(contract, self.close_order(signal, amount), 'CLOSE')
        trade.filledEvent += self.remove_bracket

    def emergencyClose(self, contract: Contract, signal: int,
                       amount: int) -> None:
        log.warning(f'Emergency close on restart: {contract.localSymbol} '
                    f'side: {signal} amount: {amount}')
        self.trade(contract, self.close_order(
            signal, amount), 'EMERGENCY CLOSE')

    def attach_bracket(self, trade: Trade) -> None:
        sl_points, min_tick = self.contracts[trade.contract.symbol]
        for bracket_order, label in zip(
                (self.stop, self.take_profit), ('STOP-LOSS', 'TAKE-PROFIT')):
            # take profit may be None
            if bracket_order:
                order = bracket_order(trade, sl_points, min_tick)
                bracket_trade = self.trade(trade.contract, order, label)
                bracket_trade.filledEvent += self.remove_bracket

    def remove_bracket(self, trade: Trade) -> None:
        self._trader.remove_bracket(trade.contract)

    def reconcile_stops(self) -> None:
        """
        To be executed on restart. Make sure all positions have corresponding
        stop-losses, if not send a closing order. For all existing
        bracket orders attach reporting events for the blotter.

        THIS LIKELY DOESN'T WORK. DISTINGUISH BETWEEN STP AND LIMIT FOR EMERGENCY
        CLOSE?????
        """

        trades = self._trader.trades()
        log.info(f'open trades on re-connect: {len(trades)} '
                 f'{[t.contract.localSymbol for t in trades]}')
        # attach reporting events
        for trade in trades:
            if trade.orderStatus.remaining != 0:
                if trade.order.orderType in ('STP', 'TRAIL'):
                    self.attach_events(trade, 'STOP-LOSS')
                else:
                    self.attach_events(trade, 'TAKE-PROFIT')

        # check for orphan positions
        positions = self._trader.positions()
        log.debug(f'positions on re-connect: {positions}')
        trade_contracts = set([t.contract for t in trades])
        position_contracts = set([p.contract for p in positions])
        orphan_contracts = position_contracts - trade_contracts
        orphan_positions = [position for position in positions
                            if position.contract in orphan_contracts]
        if orphan_positions:
            log.warning(f'orphan positions: {orphan_positions}')
            log.debug(f'len(orphan_positions): {len(orphan_positions)}')
            log.debug(f'orphan contracts: {orphan_contracts}')
            for p in orphan_positions:
                self._trader.contracts(p.contract)
                self.emergencyClose(
                    p.contract, -np.sign(p.position), int(np.abs(p.position)))
                log.error(f'emergencyClose position: {p.contract}, '
                          f'{-np.sign(p.position)}, {int(np.abs(p.position))}')
                t = self._trader.ib.reqAllOpenOrders()
                log.debug(f'reqAllOpenOrders: {t}')
                log.debug(f'openTrades: {self._trader.ib.openTrades()}')
