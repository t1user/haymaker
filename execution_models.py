from __future__ import annotations

from collection import defaultdict
import numpy as np
from typing import NamedTuple, Optional

from trader import Trader
from Candle import Candle
from ib_insync import Contract, Trade, Order, MarketOrder, StopOrder, TagValue
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


class Stop:

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


class FixedStop(Stop):

    def order(self):
        sl_price = round_tick(
            self.price + self.sl_points * self.direction,
            self.minTick)
        log.info(f'STOP LOSS PRICE: {sl_price}')
        return StopOrder(self.reverseAction, self.amount, sl_price,
                         outsideRth=True, tif='GTC')


class TrailingStop(Stop):

    def order(self):
        distance = round_tick(self.sl_points, self.min_tick)
        log.info(f'TRAILING STOP LOSS DISTANCE: {distance}')
        return Order(orderType='TRAIL', action=self.reverseAction,
                     totalQuantity=self.amount, auxPrice=distance,
                     outsideRth=True, tif='GTC')


class TrailingFixedStop(TrailingStop):

    def __init__(self, multiple):
        self.multiple = multiple

    def order(self):
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


class BaseExecModel:

    contracts = {}

    def __init__(self, trader):
        self.trade = trader.trade
        self._trader = trader

    def __str__(self):
        return self.__class__.__name__

    def entry_order(self) -> Order:
        raise NotImplementedError

    def close_order(self) -> Order:
        raise NotImplementedError

    def onEntry(self) -> None:
        raise NotImplementedError

    def onClose(self) -> None:
        raise NotImplementedError


class EventDrivenExecModel(BaseExecModel):

    # TODO
    # EACH ORDER TYPE PASSED TO INIT
    # ??? RECONCILE ON RE-CONNECT ????

    # is it done?

    def __init__(self, trader: Optional[Trader] = None,
                 stop: Stop = TrailingStop):
        if trader is None:
            trader = Trader()
        super().__init__(trader)
        self.stop = stop()
        log.debug(f'execution model initialized {self}')

    def entry_order(self, signal, amount):
        assert signal in [-1, 1], 'Invalid trade signal'
        return MarketOrder('BUY' if signal == 1 else 'SELL',
                           amount, algoStrategy='Adaptive',
                           algoParams=[
                               TagValue('adaptivePriority', 'Normal')],
                           tif='Day')

    close_order = entry_order

    def onEntry(self, contract: Contract, signal: int, sl_points: float,
                amount: int, obj: Candle) -> None:
        self.contracts[contract.symbol] = ContractMemo(
            sl_points=sl_points,
            min_tick=obj.details.min_tick)
        log.debug(
            f'{contract.localSymbol} entry signal: {signal} '
            'sl_distance: {sl_distance}')

        trade = self.trade(contract, self.entry_order(signal, amount),
                           'ENTRY')
        trade.filledEvent += self.attach_sl

    def onClose(self, contract: Contract, signal: int, atr: float,
                amount: int) -> Trade:
        # TODO can sl close position before being removed?
        # make sure sl didn't close the position before being removed
        log.debug(f'{contract.localSymbol} close signal: {signal}')
        self.remove_sl(contract)
        trade = self.trade(contract, self.close_order(signal, amount),
                           'CLOSE')
        return trade

    def emergencyClose(self, contract: Contract, signal: int,
                       amount: int) -> None:
        log.warning(f'Emergency close on restart: {contract.localSymbol} '
                    f'side: {signal} amount: {amount}')
        trade = self.trade(contract, signal, amount)
        self.attach_events(trade, 'EMERGENCY CLOSE')

    def attach_sl(self, trade: Trade) -> None:
        sl_points, min_tick = self.contracts[self.trade.contract.symbol]
        order = self.stop(trade, sl_points, min_tick)
        self.trade(trade.contract, order, 'STOP-LOSS')

    def remove_sl(self, contract: Contract) -> None:
        self._trader.remove_sl(contract)

    def reconcile_stops(self) -> None:
        """
        To be executed on restart. Make sure all positions have corresponding
        stop-losses, if not send a closing order. For all existing
        stop-losses attach reporting events for the blotter.
        """

        trades = self.ib.openTrades()
        log.info(f'open trades on re-connect: {len(trades)} '
                 f'{[t.contract.localSymbol for t in trades]}')
        # attach reporting events
        for trade in trades:
            if trade.order.orderType in ('STP', 'TRAIL'
                                         ) and trade.orderStatus.remaining != 0:
                self.attach_events(trade, 'STOP-LOSS')

        # check for orphan positions
        positions = self.ib.positions()
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
                self.ib.qualifyContracts(p.contract)
                self.emergencyClose(
                    p.contract, -np.sign(p.position), int(np.abs(p.position)))
                log.error(f'emergencyClose position: {p.contract}, '
                          f'{-np.sign(p.position)}, {int(np.abs(p.position))}')
                t = self.ib.reqAllOpenOrders()
                log.debug(f'reqAllOpenOrders: {t}')
                log.debug(f'openTrades: {self.ib.openTrades()}')
