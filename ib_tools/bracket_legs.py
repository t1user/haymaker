from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

import ib_insync as ibi

from .manager import INIT_DATA
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

    details = INIT_DATA.contract_details
    vol_field: str = "atr"

    def __init__(self, stop_multiple: float, vol_field: Optional[str] = None) -> None:
        self.stop_multiple = stop_multiple
        if vol_field:
            self.vol_field = vol_field

    def __call__(self, params: dict, trade: ibi.Trade) -> dict[str, Any]:
        trade_params = self._extract_trade(trade)
        trade_params["min_tick"] = self.min_tick(trade_params["contract"])
        trade_params["sl_points"] = self.stop_multiple * params[self.vol_field]
        return self._order(trade_params)

    def min_tick(self, contract):
        try:
            minTick = self.details[contract].minTick
        except KeyError:
            log.critical(
                f"No details for contract {contract}. "
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

        return f"{self.__class__.__name__}({attrs})"


class FixedStop(AbstractBracketLeg):
    """
    Stop-loss with fixed distance from the execution price of entry order.
    """

    def _order(self, params: dict[str, Any]) -> dict[str, Any]:
        sl_price = round_tick(
            params["price"] + params["sl_points"] * params["direction"],
            params["min_tick"],
        )
        log.info(f"STOP LOSS PRICE: {sl_price}")
        return {
            "orderType": "STP",
            "action": params["reverseAction"],
            "totalQuantity": params["amount"],
            "auxPrice": sl_price,
            "outsideRth": True,
            "tif": "GTC",
        }


class TrailingStop(AbstractBracketLeg):
    """
    Stop loss trailing price by given distance.
    """

    def _order(self, params: dict[str, Any]) -> dict[str, Any]:
        distance = round_tick(params["sl_points"], params["min_tick"])
        log.info(f"TRAILING STOP LOSS DISTANCE: {distance}")
        return {
            "orderType": "TRAIL",
            "action": params["reverseAction"],
            "totalQuantity": params["amount"],
            "auxPrice": distance,
            "outsideRth": True,
            "tif": "GTC",
        }


class AdjustableTrailingFixedStop(TrailingStop):
    """
    Trailing stop loss that will adjust itself to fixed stop-loss
    after reaching specified trigger expressed as multiple of trailing
    points.

    Args:
    =====

    stop_multiple: Multiple of ``vol_field`` at which the TRAIL will
    be trailing

    trigger_multiple: Multiple of ``stop_multiple`` at which order
    will be adjusted to STP

    fixed_stop_multiple: Multiple of ``stop_multiple`` to calculate stop
    price as.  Stop price distance from entry price is
    ``fixed_stop_multiple`` * ``stop_multiple`` * ``vol_field``

    vol_field: (default: ``atr``) Volatility measure used to calculate
    stop loss distance from entry price
    """

    def __init__(
        self,
        stop_multiple: float,
        trigger_multiple: float,
        fixed_stop_multiple: float,
        **kwargs,
    ) -> None:
        super().__init__(stop_multiple, **kwargs)
        self.trigger_multiple = trigger_multiple
        self.fixed_stop_multiple = fixed_stop_multiple

    def _order(self, params: dict[str, Any]) -> dict[str, Any]:
        k = super()._order(params)

        # k is from super order, params is from Trade object
        # k['auxPrice] is: stop_multiple * vol_field (a.k.a. self.sl_points)

        # when trigger price is penetrated
        k["triggerPrice"] = (
            k["price"] - k["direction"] * k["auxPrice"] * k["trigger_multiple"]
        )
        # the parent order will be turned into s STP order
        k["adjustedOrderType"] = "STP"
        # with the given STP price
        k["adjustedStopPrice"] = (
            k["triggerPrice"]
            + params["direction"] * self.fixed_stop_multiple * k["auxPrice"]
        )
        log.debug(
            f"{params['contract'].localSymbol} TRAIL of: {k['auxPrice']} with trigger:"
            f"{k['triggerPrice']} will be fixed to {k['adjustedStopPrice']}"
        )
        return k


class AdjustableFixedTrailingStop(FixedStop):
    """
    Fixed stop loss that will adjust itself to trailing stop-loss
    after reaching specified trigger expressed as multiple of trailing
    points.

    Args:
    =====

    stop_multiple: Multiple of ``vol_filed`` will be the distance of
    initial STP

    trigger_multiple: Multiple of ``stop_multiple`` at which
    adjustment will be triggered

    trail_multiple: Multiple of ``stop_multiple`` at which TRAIL will be
    trialing post adjustment

    vol_field: (default: ``atr``) Volatility measure used to calculate
    stop loss distance from entry price
    """

    def __init__(
        self,
        stop_multiple: float,
        trigger_multiple: float,
        trail_multiple: float,
        **kwargs,
    ) -> None:
        super().__init__(stop_multiple, **kwargs)
        self.trigger_multiple = trigger_multiple
        self.trail_multiple = trail_multiple

    def _order(self, params: dict[str, Any]) -> dict[str, Any]:
        k = super()._order(params)

        # k is from super order, params is from Trade object
        # k['auxPrice] is: stop_multiple * vol_field (a.k.a. self.sl_points)

        # when trigger price is penetrated
        k["triggerPrice"] = round_tick(
            params["price"]
            - params["sl_points"] * self.trigger_multiple * params["direction"],
            params["min_tick"],
        )
        # the parent order will be turned int a TRAIL order
        k["adjustedOrderType"] = "TRAIL"
        # trailing by an amount (0) or a percent (100)...
        k["adjustableTrailingUnit"] = 0
        # of ...
        k["adjustedTrailingAmount"] = round_tick(
            self.trail_multiple * params["sl_points"], params["min_tick"]
        )
        # with a stop price
        k["adjustedStopPrice"] = (
            k["triggerPrice"] + k["adjustedTrailingAmount"] * params["direction"]
        )

        log.debug(
            f"{params['contract'].localSymbol} STP at {k['auxPrice']} "
            f"with trigger: {k['triggerPrice']} will TRAIL at: "
            f"{k['adjustedTrailingAmount']}"
        )
        return k


class AdjustableTrailingStop(TrailingStop):
    """
    Trailing stop-loss that will widen trailing distance after
    reaching pre-specified trigger.

    Args:
    =====

    stop_multiple: Multiple of ``vol_field`` at which the TRAIL will
    initially be trailing

    trigger_multiple: Multiple of ``stop_multiple`` at which order
    trailing distance will be adjusted

    adjusted_multiple: Multiple of ``stop_multiple`` at which TRAIL
    will be trailing after adjustment; i.e. the trailing amount will be:
    ``vol_field`` * ``stop_multiple`` * ``adjusted_multiple``

    vol_field: (default: ``atr``) Volatility measure used to calculate
    stop loss distance from entry price
    """

    def __init__(
        self,
        stop_multiple: float,
        trigger_multiple: float,
        adjusted_multiple: float,
        **kwargs,
    ) -> None:
        super().__init__(stop_multiple, **kwargs)
        self.trigger_multiple = trigger_multiple
        self.adjusted_multiple = adjusted_multiple

    def _order(self, params: dict[str, Any]) -> dict[str, Any]:
        k = super()._order(params)

        # when trigger is penetrated
        k["triggerPrice"] = (
            params["price"]
            - params["direction"] * self.trigger_multiple * k["auxPrice"]
        )
        # sl order will remain trailing order
        k["adjustedOrderType"] = "TRAIL"
        # with a stop price of
        k["adjustedStopPrice"] = (
            k["triggerPrice"]
            + params["direction"] * k["auxPrice"] * self.adjusted_multiple
        )
        # being trailed by fixed amount
        k["adjustableTrailingUnit"] = 0
        # of:
        k["adjustedTrailingAmount"] = round_tick(
            k["auxPrice"] * self.adjusted_multiple, params["min_tick"]
        )
        return k


class TakeProfitAsStopMultiple(AbstractBracketLeg):
    """
    Take-profit order with distance from entry price specified as multiple
    of stop-loss distance. The multiple is fixed, given on object
    initialization.
    """

    def __init__(self, stop_multiple: float, tp_multiple: float, **kwargs) -> None:
        super().__init__(stop_multiple, **kwargs)
        self.tp_multiple = tp_multiple

    def _order(self, params: dict[str, Any]) -> dict[str, Any]:
        tp_price = round_tick(
            params["price"]
            - params["sl_points"] * params["direction"] * self.tp_multiple,
            params["min_tick"],
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

    def __init__(self, stop_multiple: float, tp_multiple: float, **kwargs) -> None:
        super().__init__(stop_multiple, **kwargs)
        self.tp_multiple = tp_multiple

    def _order(self, params: dict[str, Any]) -> dict[str, Any]:
        tp_price = round_tick(
            params["price"]
            - params["sl_points"] * params["direction"] * self.tp_multiple,
            params["min_tick"],
        )
        log.info(f"TAKE PROFIT PRICE: {tp_price}")
        return {
            "orderType": "LMT",
            "action": params["reverseAction"],
            "totalQuantity": params["amount"],
            "lmtPrice": tp_price,
            "tif": "GTC",
        }
