from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

import ib_insync as ibi

from .manager import CONTRACT_DETAILS
from .misc import round_tick

log = logging.getLogger(__name__)

# ====================================================================================
# Parameters for various types of orders packaged into objects required by exec models
# ====================================================================================


class AbstractBracketLeg(ABC):
    """
    For use by EventDrivenExecModel to create stop-loss and
    take-profit orders.

    Extract information from Trade object and create dictionary with
    parameters for appropriate bracket order.
    """

    details = CONTRACT_DETAILS
    vol_field: str = "atr"

    def __init__(self, stop_multiple: float, vol_field: Optional[str] = None) -> None:
        self.stop_multiple = stop_multiple
        if vol_field:
            self.vol_field = vol_field

    def __call__(self, params: dict, trade: ibi.Trade) -> dict[str, Any]:
        trade_params = self._extract_trade(trade)
        self.contract = trade_params["contract"]
        trade_params["sl_points"] = self.stop_multiple * params[self.vol_field]
        return self._order(trade_params)

    @property
    def min_tick(self):
        try:
            minTick = self.details[self.contract].minTick
        except KeyError:
            log.critical(
                f"No details for contract {self.contract}. "
                f"Will attempt to send bracket order with minTick 0.25 ",
                exc_info=True,
            )
            minTick = 0.25
        return minTick

    @staticmethod
    def _extract_trade(trade: ibi.Trade) -> dict[str, Any]:
        trade_params = {
            "contract": trade.contract,
            "action": trade.order.action,
            "amount": trade.orderStatus.filled,
            "price": trade.orderStatus.avgFillPrice,
        }
        trade_params["reverseAction"] = (
            "BUY" if trade_params["action"] == "SELL" else "SELL"
        )
        trade_params["direction"] = 1 if trade_params["reverseAction"] == "BUY" else -1
        return trade_params

    @abstractmethod
    def _order(self, params: dict[str, Any]) -> dict[str, Any]:
        ...

    def __repr__(self):
        attrs = ", ".join((f"{i}={j}" for i, j in self.__dict__.items()))

        return f"{__class__.__name__}({attrs})"


class FixedStop(AbstractBracketLeg):
    """
    Stop-loss with fixed distance from the execution price of entry order.
    """

    def _order(self, params: dict[str, Any]) -> dict[str, Any]:
        sl_price = round_tick(
            params["price"] + params["sl_points"] * params["direction"], self.min_tick
        )
        log.info(f"STOP LOSS PRICE: {sl_price}")
        return {
            "order_type": "STP",
            "action": params["reverseAction"],
            "totalQuantity": params["amount"],
            "stopPrice": sl_price,
            "outsideRth": True,
            "tif": "GTC",
        }


class TrailingStop(AbstractBracketLeg):
    """
    Stop loss trailing price by given distance.
    """

    def _order(self, params: dict[str, Any]) -> dict[str, Any]:
        distance = round_tick(params["sl_points"], self.min_tick)
        log.info(f"TRAILING STOP LOSS DISTANCE: {distance}")
        return {
            "orderType": "TRAIL",
            "action": params["reverseAction"],
            "totalQuantity": params["amount"],
            "auxPrice": distance,
            "outsideRth": True,
            "tif": "GTC",
        }


class TrailingFixedStop(TrailingStop):
    """
    Trailing stop loss that will adjust itself to fixed stop-loss after
    reaching specified trigger expressed as multiple of trailing points.
    """

    def __init__(self, stop_multiple: float, trigger_multiple: float = 2) -> None:
        super().__init__(stop_multiple)
        self.trigger_multiple = trigger_multiple

    def _order(self, params: dict[str, Any]) -> dict[str, Any]:
        k = super()._order(params)
        k["adjustedOrderType"] = "STP"
        k["adjustedStopPrice"] = (
            params["price"] - params["direction"] * self.stop_multiple * k["auxPrice"]
        )

        log.debug(f"adjusted stop price: {k['adjustedStopPrice']}")
        k["triggerPrice"] = k["adjustedStopPrice"] - k["direction"] * k["auxPrice"]
        log.debug(
            f"stop loss for {params['contract'].localSymbol} "
            f"fixed at {k['triggerPrice']}"
        )
        return k


class TrailingAdjustableStop(TrailingStop):
    """
    Trailing stop-loss that will widen trailing distance after reaching
    pre-specified trigger.
    """

    def __init__(
        self,
        stop_multiple: float,
        sl_trigger_multiple: float,
        sl_adjusted_multiple: float,
    ) -> None:
        super().__init__(stop_multiple)
        self.sl_trigger_multiple = sl_trigger_multiple
        self.sl_adjusted_multiple = sl_adjusted_multiple

    def _order(self, params: dict[str, Any]) -> dict[str, Any]:
        k = super()._order(params)

        # when trigger is penetrated
        k["triggerPrice"] = (
            params["price"]
            - params["direction"] * self.sl_trigger_multiple * k["auxPrice"]
        )
        # sl order will remain trailing order
        k["adjustedOrderType"] = "TRAIL"
        # with a stop price of
        k["adjustedStopPrice"] = (
            k["triggerPrice"]
            + params["direction"] * k["auxPrice"] * self.sl_adjusted_multiple
        )
        # being trailed by fixed amount
        k["adjustableTrailingUnit"] = 0
        # of:
        k["adjustedTrailingAmount"] = k["auxPrice"] * self.sl_adjusted_multiple
        return k


class TakeProfitAsStopMultiple(AbstractBracketLeg):
    """
    Take-profit order with distance from entry price specified as multiple
    of stop-loss distance. The multiple is fixed, given on object
    initialization.
    """

    def __init__(self, stop_multiple: float, tp_multiple: float = 2) -> None:
        super().__init__(stop_multiple)
        self.tp_multiple = tp_multiple

    def _order(self, params: dict[str, Any]) -> dict[str, Any]:
        tp_price = round_tick(
            params["price"]
            - params["sl_points"] * params["direction"] * self.tp_multiple,
            self.min_tick,
        )
        log.info(f"TAKE PROFIT PRICE: {tp_price}")
        return {
            "orderType": "LMT",
            "action": params["reverseAction"],
            "totalQuantity": params["amount"],
            "lmtPrice": tp_price,
            "outsideRth": True,
            "tif": "GTC",
        }


class FlexibleTakeProfitAsStopMultiple(AbstractBracketLeg):
    """
    Take-profit order with distance from entry price specified as multiple
    of stop-loss distance. The multiple is flexible, passed every time object
    is called.

    """

    def __init__(self, stop_multiple: float, tp_multiple: float) -> None:
        super().__init__(stop_multiple)
        self.tp_multiple = tp_multiple

    def _order(self, params: dict[str, Any]) -> dict[str, Any]:
        tp_price = round_tick(
            params["price"]
            - params["sl_points"] * params["direction"] * self.tp_multiple,
            self.min_tick,
        )
        log.info(f"TAKE PROFIT PRICE: {tp_price}")
        return {
            "orderType": "LMT",
            "action": params["reverseAction"],
            "totalQuantity": params["amount"],
            "lmtPrice": tp_price,
            "tif": "GTC",
        }
