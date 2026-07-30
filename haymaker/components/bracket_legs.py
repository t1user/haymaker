from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

import ib_insync as ibi

from ..contract_registry import DetailsContainer
from ..misc import round_tick

log = logging.getLogger(__name__)

# ====================================================================================
# Parameters for various types of orders packaged into objects required by exec models
# ====================================================================================


class AbstractBracketLeg(ABC):
    """Build one protective-order leg after a complete entry fill.

    BracketExecutionModel calls the leg with validated PositionTarget metadata,
    the completely filled entry Trade, and Contract details. Subclasses return
    IB Order keyword arguments for a stop or take-profit order.

    Args:
        stop_multiple: Multiple applied to the configured volatility field.
        vol_field: Metadata field containing the distance basis. Defaults to
            ``"atr"``.

    Raises:
        KeyError: If the configured volatility field is absent.
    """

    vol_field: str = "atr"

    def __init__(self, stop_multiple: float, vol_field: Optional[str] = None) -> None:
        self.stop_multiple = stop_multiple
        if vol_field:
            self.vol_field = vol_field

    def __call__(
        self,
        params: dict,
        trade: ibi.Trade,
        memo: Optional[dict] = None,
        details: DetailsContainer | None = None,
    ) -> dict[str, Any]:
        # trade params are params extracted from trade
        # params are passed by the caller
        trade_params = self._extract_trade(trade)
        trade_params["min_tick"] = self.min_tick(trade_params["contract"], details)
        trade_params["vol_field_name"] = self.vol_field
        trade_params["vol_field_value"] = params[self.vol_field]
        trade_params["sl_points"] = self.stop_multiple * trade_params["vol_field_value"]
        order = self._order(trade_params)
        # any notes made on this object will be accessible for logging by caller
        # sub-classes can add keys to trade_params thus logging their parameters
        if memo is not None:
            memo.update(trade_params)
            memo.update(self.__dict__)
        return order

    def min_tick(self, contract, details: DetailsContainer | None = None):
        details = details or DetailsContainer()
        try:
            minTick = details[contract].minTick
        except KeyError:
            log.critical(
                f"No details for contract {contract}. "
                f"Will attempt to send bracket order with minTick 0.25 ",
                exc_info=True,
            )
            log.debug(f"Details: {details}")
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
    def _order(self, params: dict[str, Any]) -> dict[str, Any]: ...

    def __repr__(self):
        attrs = ", ".join((f"{i}={j}" for i, j in self.__dict__.items()))

        return f"{self.__class__.__name__}({attrs})"


class FixedStop(AbstractBracketLeg):
    """Create a fixed-price stop from entry price and volatility distance.

    ``stop_multiple * metadata[vol_field]`` determines the distance, rounded
    to Contract minimum tick. The generated GTC stop closes the completely
    filled entry quantity and is permitted outside regular trading hours.
    """

    def _order(self, params: dict[str, Any]) -> dict[str, Any]:
        params["sl_price"] = round_tick(
            params["price"] + params["sl_points"] * params["direction"],
            params["min_tick"],
        )
        log.info(f"STOP LOSS PRICE: {params['sl_price']}")
        return {
            "orderType": "STP",
            "action": params["reverseAction"],
            "totalQuantity": params["amount"],
            "auxPrice": params["sl_price"],
            "outsideRth": True,
            "tif": "GTC",
        }


class TrailingStop(AbstractBracketLeg):
    """Create a fixed-distance trailing stop for the filled entry quantity.

    The trailing distance is ``stop_multiple * metadata[vol_field]`` rounded to
    Contract minimum tick. The generated trailing order is GTC and active
    outside regular trading hours.
    """

    def _order(self, params: dict[str, Any]) -> dict[str, Any]:
        params["distance"] = round_tick(params["sl_points"], params["min_tick"])
        log.info(f"TRAILING STOP LOSS DISTANCE: {params['distance']}")
        return {
            "orderType": "TRAIL",
            "action": params["reverseAction"],
            "totalQuantity": params["amount"],
            "auxPrice": params["distance"],
            "outsideRth": True,
            "tif": "GTC",
        }


class AdjustableTrailingFixedStop(TrailingStop):
    """Create a trailing stop that later becomes a fixed stop.

    Args:
        stop_multiple: Initial trailing-distance multiple of ``vol_field``.
        trigger_multiple: Trailing-distance multiple from entry at which IB
            changes order type.
        fixed_stop_multiple: Trailing-distance multiple used to place the
            adjusted fixed stop relative to its trigger.
        **kwargs: Optional ``vol_field`` accepted by AbstractBracketLeg.
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
        # log.debug(f"super order: {k}")
        # k is from super order, params is from Trade object
        # k['auxPrice] is: stop_multiple * vol_field (a.k.a. self.sl_points)

        # when trigger price is penetrated
        k["triggerPrice"] = (
            params["price"]
            - params["direction"] * k["auxPrice"] * self.trigger_multiple
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
    """Create a fixed stop that later becomes a trailing stop.

    Args:
        stop_multiple: Initial stop-distance multiple of ``vol_field``.
        trigger_multiple: Stop-distance multiple from entry at which IB
            changes order type.
        trail_multiple: Stop-distance multiple used as the adjusted trailing
            amount.
        **kwargs: Optional ``vol_field`` accepted by AbstractBracketLeg.
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
    """Create a trailing stop whose distance widens after a trigger.

    Args:
        stop_multiple: Initial trailing-distance multiple of ``vol_field``.
        trigger_multiple: Initial-distance multiple from entry at which IB
            adjusts the order.
        adjusted_multiple: Initial-distance multiple used as the new trailing
            amount.
        **kwargs: Optional ``vol_field`` accepted by AbstractBracketLeg.
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
    """Create a take-profit limit as a multiple of stop distance.

    Args:
        stop_multiple: Multiple converting ``vol_field`` to stop distance.
        tp_multiple: Multiple converting stop distance to take-profit distance.
        **kwargs: Optional ``vol_field`` accepted by AbstractBracketLeg.

    The generated GTC limit closes the filled entry quantity and is permitted
    outside regular trading hours.
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
    """Create a GTC take-profit limit without forcing ``outsideRth``.

    Args:
        stop_multiple: Multiple converting ``vol_field`` to stop distance.
        tp_multiple: Multiple converting stop distance to take-profit distance.
        **kwargs: Optional ``vol_field`` accepted by AbstractBracketLeg.

    Use this variant when order defaults or model options should decide
    outside-regular-hours behavior.
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


__all__ = [
    "AbstractBracketLeg",
    "AdjustableFixedTrailingStop",
    "AdjustableTrailingFixedStop",
    "AdjustableTrailingStop",
    "FixedStop",
    "FlexibleTakeProfitAsStopMultiple",
    "TakeProfitAsStopMultiple",
    "TrailingStop",
]
