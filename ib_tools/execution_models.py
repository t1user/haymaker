from __future__ import annotations

import random
import string
from abc import ABC, abstractmethod
from typing import NamedTuple, Optional, Protocol

import ib_insync as ibi

from ib_tools.logger import Logger
from ib_tools.trader import Trader

from . import misc

log = Logger(__name__)


# This is not true for every type of execution model, TODO
class ParamProtocol(Protocol):
    details: ibi.ContractDetails
    sl_triger_multiple: float
    sl_adjusted_multiple: float
    tp_multiple: float


class TraderProtocol(Protocol):
    def trade(self, conntract: ibi.Contract, order: ibi.Order, note: str) -> ibi.Trade:
        ...

    def cancel(self, trade: ibi.Trade) -> ibi.Trade:
        ...

    def trades_for_contract(self, contract: ibi.Contract) -> list[ibi.Trade]:
        ...


# =====================================================================
# Helpers
# =====================================================================


def round_tick(price: float, tick_size: float) -> float:
    floor = price // tick_size
    remainder = price % tick_size
    if remainder > (tick_size / 2):
        floor += 1
    return round(floor * tick_size, 4)


class ContractMemo(NamedTuple):
    sl_points: float
    min_tick: float


# =====================================================================
# Various  types of orders packaged in objects required by exec models
# =====================================================================


class AbstractBracketLeg(ABC):
    """
    For use by EventDrivenExecModel to create stop-loss and take-profit
    orders.

    Extract information from Trade object and create appropriate bracket
    order.
    """

    def __call__(
        self, trade: ibi.Trade, sl_points: int, min_tick: float, *args, **kwargs
    ) -> ibi.Order:
        self._extract_trade(trade)
        self.sl_points = sl_points
        self.min_tick = min_tick
        return self.order()

    def _extract_trade(self, trade: ibi.Trade) -> None:
        self.contract = trade.contract
        self.action = trade.order.action
        assert self.action in ("BUY", "SELL")
        self.reverseAction = "BUY" if self.action == "SELL" else "SELL"
        self.direction = 1 if self.reverseAction == "BUY" else -1
        self.amount = trade.orderStatus.filled
        self.price = trade.orderStatus.avgFillPrice

    @abstractmethod
    def order(self) -> ibi.Order:
        raise NotImplementedError


class FixedStop(AbstractBracketLeg):
    """
    Stop-loss with fixed distance from the execution price of entry order.
    """

    def order(self) -> ibi.Order:
        sl_price = round_tick(
            self.price + self.sl_points * self.direction, self.min_tick
        )
        log.info(f"STOP LOSS PRICE: {sl_price}")
        return ibi.StopOrder(
            self.reverseAction, self.amount, sl_price, outsideRth=True, tif="GTC"
        )


class TrailingStop(AbstractBracketLeg):
    """
    Stop loss trailing price by given distance.
    """

    def order(self) -> ibi.Order:
        distance = round_tick(self.sl_points, self.min_tick)
        log.info(f"TRAILING STOP LOSS DISTANCE: {distance}")
        return ibi.Order(
            orderType="TRAIL",
            action=self.reverseAction,
            totalQuantity=self.amount,
            auxPrice=distance,
            outsideRth=True,
            tif="GTC",
        )


class TrailingFixedStop(TrailingStop):
    """
    Trailing stop loss that will adjust itself to fixed stop-loss after
    reaching specified trigger.
    """

    def __init__(self, multiple: float = 2) -> None:
        self.multiple = multiple

    def order(self) -> ibi.Order:
        sl = super().order()
        sl.adjustedOrderType = "STP"
        sl.adjustedStopPrice = self.price - self.direction * self.multiple * sl.auxPrice
        log.debug(f"adjusted stop price: {sl.adjustedStopPrice}")
        sl.triggerPrice = sl.adjustedStopPrice - self.direction * sl.auxPrice
        log.debug(
            f"stop loss for {self.contract.localSymbol} " f"fixed at {sl.triggerPrice}"
        )
        return sl


class TrailingAdjustableStop(TrailingStop):
    """
    Trailing stop-loss that will widen trailing distance after reaching
    pre-specified trigger.
    """

    def __call__(  # type: ignore
        self,
        trade: ibi.Trade,
        sl_points: int,
        min_tick: float,
        sl_trigger_multiple: float,
        sl_adjusted_multiple: float,
    ) -> ibi.Order:
        self.sl_trigger_multiple = sl_trigger_multiple
        self.sl_adjusted_multiple = sl_adjusted_multiple
        return super().__call__(trade, sl_points, min_tick)

    def order(self) -> ibi.Order:
        sl = super().order()
        # when trigger is penetrated
        sl.triggerPrice = (
            self.price - self.direction * self.sl_trigger_multiple * sl.auxPrice
        )
        # sl order will remain trailing order
        sl.adjustedOrderType = "TRAIL"
        # with a stop price of
        sl.adjustedStopPrice = (
            sl.triggerPrice + self.direction * sl.auxPrice * self.sl_adjusted_multiple
        )
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

    def order(self) -> ibi.Order:
        tp_price = round_tick(
            self.price - self.sl_points * self.direction * self.multiple, self.min_tick
        )
        log.info(f"TAKE PROFIT PRICE: {tp_price}")
        return ibi.LimitOrder(
            self.reverseAction, self.amount, tp_price, outsideRth=True, tif="GTC"
        )


class StopFlexiMultipleTakeProfit(AbstractBracketLeg):
    """
    Take-profit order with distance from entry price specified as multiple
    of stop-loss distance. The multiple is flexible, passed every time object
    is called.
    """

    def __call__(  # type: ignore
        self, trade: ibi.Trade, sl_points: int, min_tick: float, tp_multiple: float
    ) -> ibi.Order:
        self.tp_multiple = tp_multiple
        return super().__call__(trade, sl_points, min_tick)

    def order(self):
        tp_price = round_tick(
            self.price - self.sl_points * self.direction * self.tp_multiple,
            self.min_tick,
        )
        log.info(f"TAKE PROFIT PRICE: {tp_price}")
        return ibi.LimitOrder(self.reverseAction, self.amount, tp_price, tif="GTC")


# =====================================================================
# Actual execution models
# =====================================================================


class BaseExecModel:
    """
    Intermediary between Portfolio and Trader. Allows for fine tuning of order
    types used, order monitoring, post-order events, etc.

    Initiated object passed to Manager. Portfolio sends events to onEntry.
    onStarted executed on (re)start.
    """

    def __init__(self, trader: TraderProtocol) -> None:
        """
        Used by Manager on initialization.
        """
        self.trade = trader.trade
        self._trader = trader

    def __repr__(self):
        return f"{self.__class__.__name__}({self._trader!r})"

    def execute(self, trade_params) -> None:
        """Figure out whether this is open or close order and call
        appropriate method."""
        pass

    @staticmethod
    def entry_order(action: str, quantity: int) -> ibi.Order:
        """
        Return order to be used for entry transactions.
        """
        return ibi.MarketOrder(
            action,
            quantity,
            algoStrategy="Adaptive",
            algoParams=[ibi.TagValue("adaptivePriority", "Normal")],
            tif="Day",
        )

    @staticmethod
    def close_order(action: str, quantity: int) -> ibi.Order:
        """
        Return order to be used for close transactions.
        """
        return ibi.MarketOrder(action, quantity, tif="GTC")

    def onEntry(
        self, contract: ibi.Contract, signal: int, amount: int, *args, **kwargs
    ) -> ibi.Trade:
        """
        Accept Portfolio event to execute position entry transaction.
        """
        return self.trade(
            contract, self.entry_order(misc.action(signal), amount), "ENTRY"
        )

    def onClose(
        self, contract: ibi.Contract, signal: int, amount: int, *args, **kwargs
    ) -> ibi.Trade:
        """
        Accept Portfolio event to execute position close transaction.
        """
        return self.trade(
            contract, self.close_order(misc.action(signal), amount), "CLOSE"
        )


class EventDrivenExecModel(BaseExecModel):
    """
    Use events to attach stop-loss and optional take-profit orders after
    execution of entry order. After close transaction remove existing
    bracketing orders.

    this has been moved out of this class:
    On re(start) make sure that all positions have
    stop-losses and that there are no open orders not attached to any
    existing position.
    """

    contracts: dict[str, ContractMemo] = {}

    def __init__(
        self,
        trader: TraderProtocol,
        stop: Optional[AbstractBracketLeg] = None,
        take_profit: Optional[AbstractBracketLeg] = None,
    ):
        super().__init__(trader)
        if stop:
            self.stop = stop
        else:
            self.stop = TrailingStop()
        self.take_profit = take_profit
        log.debug(f"execution model initialized {self}")

    def save_contract(self, contract, sl_points, obj):
        self.contracts[contract.symbol] = ContractMemo(
            sl_points=sl_points, min_tick=obj.details.minTick
        )

    def onEntry(  # type: ignore
        self,
        contract: ibi.Contract,
        signal: int,
        amount: int,
        sl_points: float,
        obj: ParamProtocol,
    ) -> ibi.Trade:
        """
        Save information required for bracket orders and attach events
        that will attach brackets after order completion.
        """
        self.save_contract(contract, sl_points, obj)
        log.debug(
            f"{contract.localSymbol} entry signal: {signal} "
            f"sl_distance: {sl_points}"
        )
        trade = super().onEntry(contract, signal, amount)
        trade.filledEvent += self.attach_bracket
        return trade

    def onClose(
        self, contract: ibi.Contract, signal: int, amount: int, *args, **kwargs
    ) -> ibi.Trade:
        """
        Attach events that will cancel any brackets on order completion.
        """
        log.debug(f"{contract.localSymbol} close signal: {signal}")
        trade = super().onClose(contract, signal, amount)
        trade.filledEvent += self.remove_bracket
        return trade

    def order_kwargs(self):
        return {}

    def attach_bracket(self, trade: ibi.Trade) -> None:
        params = self.contracts[trade.contract.symbol]
        order_kwargs = self.order_kwargs()
        log.debug(
            f"attaching bracket with params: {params}, " f"order_kwargs: {order_kwargs}"
        )
        for bracket_order, label in zip(
            (self.stop, self.take_profit), ("STOP-LOSS", "TAKE-PROFIT")
        ):
            # take profit may be None
            if bracket_order:
                log.debug(f"bracket: {bracket_order}")
                order = bracket_order(trade, *params)  # type: ignore
                ibi.util.dataclassUpdate(order, **order_kwargs)
                log.debug(f"order: {order}")
                bracket_trade = self.trade(trade.contract, order, label)
                bracket_trade.filledEvent += self.bracketing_action

    def remove_bracket(self, trade: ibi.Trade) -> None:
        self.cancel_all_trades_for_contract(trade.contract)

    bracketing_action = remove_bracket

    def cancel_all_trades_for_contract(self, contract: ibi.Contract) -> None:
        for trade in self._trader.trades_for_contract(contract):
            self._trader.cancel(trade)


class OcaExecModel(EventDrivenExecModel):
    """
    Use Interactive Brokers OCA (one cancells another) orders for stop-loss
    and take-profit.
    """

    oca_ids: list[str] = []

    def oca_group(self):
        while (
            o := "".join(random.choices(string.ascii_letters + string.digits, k=10))
        ) in self.oca_ids:
            pass
        self.oca_ids.append(o)
        return o

    def order_kwargs(self):
        return {"ocaGroup": self.oca_group(), "ocaType": 1}

    def report_bracket(self, trade: ibi.Trade) -> None:
        log.debug(f"Bracketing order filled: {trade.order}")

    bracketing_action = report_bracket


class AdjustableTrailingStopExecModel(EventDrivenExecModel):
    def __init__(self, trader: Trader):
        self.trader = trader
        self.stop = TrailingAdjustableStop()
        self.take_profit = None
        log.debug(f"Execution model initialized: {self}")

    def save_contract(self, contract, sl_points, obj):
        self.contracts[contract.symbol] = (
            sl_points,
            obj.details.minTick,
            obj.sl_trigger_multiple,
            obj.sl_adjusted_multiple,
        )


class EventDrivenTakeProfitExecModel(OcaExecModel):
    """
    For every entry order obligatorily create stop-loss and take-profit
    orders. Use oca order mechanism.
    """

    def __init__(self, trader: Trader):
        """
        Trailing stop-loss with trailing amount given in points.
        Take profit distance from entry price given as multiple of trailing
        distance.
        """
        self.trader = trader
        self.stop = TrailingStop()
        self.take_profit = StopFlexiMultipleTakeProfit()
        log.debug(f"Excution model initialized: {self}")

    def save_contract(self, contract, sl_points, obj):
        self.contracts[contract.symbol] = (
            sl_points,
            obj.details.minTick,
            obj.tp_multiple,
        )
