from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Optional, cast
from uuid import uuid4

import ib_insync as ibi

from .base import Atom
from .bracket_legs import AbstractBracketLeg
from .config import CONFIG
from .validators import Validator, order_field_validator

if TYPE_CHECKING:
    from .controller import Controller

from . import misc

log = logging.getLogger(__name__)

OrderKey = Literal[
    "open_order", "close_order", "stop_order", "tp_order", "reverse_order"
]

BracketLabel = Literal["STOP_LOSS", "TAKE_PROFIT"]


class Bracket(NamedTuple):
    label: BracketLabel
    order_key: OrderKey
    order_kwargs: dict
    trade: ibi.Trade


class AbstractExecModel(Atom, ABC):
    """
    Intermediary between Portfolio and Trader.  Allows for fine tuning
    of order types used, order monitoring, post-order events, etc.

    Object initialized by a :class:`Brick`, :class:`Controller`
    will call execute method.

    All execution models must inherit from this class.
    """

    _open_order: dict[str, Any]
    _close_order: dict[str, Any]
    open_order = Validator(order_field_validator)
    close_order = Validator(order_field_validator)

    def __init__(
        self,
        orders: Optional[dict[OrderKey, dict[str, Any]]] = None,
        *,
        controller: Optional[Controller] = None,
    ) -> None:
        super().__init__()
        self.strategy: str = ""  # placeholder, defined in onStart
        if controller:
            self.controller = controller
        else:
            # importing from .manager creates singleton instances
            # may screw up tests if test needs a specific mock
            from .manager import CONTROLLER

            self.controller = CONTROLLER

        if orders:
            for key, order_kwargs in orders.items():
                setattr(self, key, order_kwargs)

        self.connect_controller()

    def connect_controller(self):
        self += self.controller

    def onStart(self, data, *args) -> None:
        super().onStart(data, *args)
        # super just set strategy, only now subsequent is possible
        try:
            self.data.update(**data)
        except AttributeError:
            log.error(f"{self} doesn't have `strategy` attribute set.")

    def get_position_id(self, reset=False):
        if reset or not self.data.position_id:
            self.data.position_id = misc.COUNTER()  # int(uuid4())
        return self.data.position_id

    def _order(self, key: OrderKey, params: dict) -> ibi.Order:
        """
        Build order object from passed params and defaults stored as
        object attributes.

        `params` will override any order defaults stored as object
        attributes.

        Args:
        =====

        key: what kind of transaction it is.  Must be:

        * either one of pre-defined names: `open_order`,
        `close_order`

        * or one of keys in `orders` dict passed to
        :meth:`self.__init__`, examples of good orders potentially
        defined this way: `stop_order`, `tp_order`

        params: must be a :py:`dict` of keywords and values accepted by
        ibi.Order

        Returns:
        ========

        :class:`ibi.Order` object, which can be directly passed to
        the broker via :meth:`ibi.IB.send_order`.
        """
        order_kwargs = getattr(self, key)
        params.update(order_kwargs)
        return ibi.Order(**params)

    @abstractmethod
    def onData(self, data: dict, *args):
        """
        Must use :meth:`self.trade`(:class:`ibi.Contract`,
        :class:`ibi.Order`, strategy_key, reason) to send orders for
        execution, and subsequently link any :class:`ibi.Trade`
        events returned by :meth:`self.trade` to required callbacks.

        While openning position must set :attr:`self.contract` to
        :class:`ibi.Contract` that has been used.

        Must keep track of current position in the market by updating
        :attr:`self.data.position`.

        Args:
        =====

        data (dict): is a dict created by :class:`Brick`, updated by
        :class:`Portfolio`, which must contain all parameters required
        to execute transactions in line with this execution model.

        Returns:
        ========

        (trade, note), where:

        * trade: :class:`ibi.Trade` object for the issued order *
        note: info string for loggers and blotters about the character
        of the transaction (open, close, stop, etc.)
        """
        # TODO: what if more than one order issued????
        super().onData(data)
        self.dataEvent.emit(data)


class BaseExecModel(AbstractExecModel):
    """
    Orders generated based on data sent to :meth:`.onData`, of which
    following keys are required:

        - ``action``: must be one of: ``OPEN``, ``CLOSE``, ``REVERSE``

        - ``signal`` determines transaction direction, must be one of
          {-1, 1} for sell/buy respectively

        - ``contract`` - this :class:`ibi.Contract` instance will be
          traded

        - ``amount`` - quantity of ``contract``s that will be traded

        - ``target_position`` - one of {-1, 0, 1} determining
          direction AFTER transaction is executed; will be used by
          :class:`.StateMachine` to verify if the transaction's effect
          was as desired

    Enters and closes positions based on params sent to
    :meth:`.execute`.  Orders composed by :meth:`.open` and
    :meth:`.close`, which can be overridden or extended in subclasses
    to get more complex behaviour.
    """

    _open_order = {**cast(dict, CONFIG.get("open_order")), "orderType": "MKT"}
    _close_order = {**cast(dict, CONFIG.get("close_order")), "orderType": "MKT"}

    def trade(
        self,
        contract: ibi.Contract,
        order: ibi.Order,
        action: str,
    ) -> ibi.Trade:
        return self.controller.trade(self.strategy, contract, order, action, self.data)

    def cancel(self, trade: ibi.Trade) -> Optional[ibi.Trade]:
        return self.controller.cancel(trade)

    async def live_ticker(self):
        # NOT IN USE
        if self.ticker:
            n = 1
            while not self.ticker.hasBidAsk():
                await asyncio.sleep(0.1)
                n += 1
                if n > 50:
                    break
            else:
                return self.ticker
        else:
            log.info(f"No subscription for live ticker {self.data.active_contract}")

    def get_ticker(self, contract) -> Optional[ibi.Ticker]:
        for t in self.ib.tickers():
            if t.contract == contract:
                return t
        return None

    def onData(self, data: dict, *args):
        contract = data.get("contract")
        if contract and (ticker := self.get_ticker(data["contract"])):
            data["arrival_price"] = {
                "time": ticker.time,
                "bid": ticker.bid,
                "ask": ticker.ask,
            }

        try:
            action = data["action"]
        except KeyError:
            log.exception(f"Missing data for {self}")
            return

        if action == "OPEN":
            self.open(data)
        elif action == "CLOSE":
            self.close(data)
        elif action == "REVERSE":
            self.reverse(data)
        else:
            log.error(f"Ambiguous action: {action} for {self}")
            return
        self.dataEvent.emit(data)

    def open(
        self,
        data: dict,
        dynamic_order_kwargs: Optional[dict] = None,
    ) -> ibi.Trade:
        self.data.params["close"] = {}
        data["position_id"] = self.get_position_id(True)
        self.data.params["open"] = data
        try:
            contract = data["contract"]
            signal = data["signal"]
            amount = int(data["amount"])
        except KeyError:
            log.exception("Insufficient data to execute OPEN transaction")
        self.data.active_contract = contract
        order_kwargs = {"action": misc.action(signal), "totalQuantity": amount}
        if dynamic_order_kwargs:
            order_kwargs.update(dynamic_order_kwargs)
        order = self._order("open_order", order_kwargs)
        log.debug(
            f"{self.strategy} {contract.localSymbol} processing OPEN signal {signal}",
            extra={"data": data},
        )
        return self.trade(contract, order, "OPEN")

    def close(
        self,
        data: dict,
        dynamic_order_kwargs: Optional[dict] = None,
    ) -> Optional[ibi.Trade]:
        self.data.params["close"] = data
        data["position_id"] = self.get_position_id()
        # THIS IS TEMPORARY ----> FIX ---> TODO
        if self.data.position == 0:
            log.error(
                f"Abandoning CLOSE position for {self.data.active_contract} "
                f"(No position, but close signal)"
            )
            return None

        signal = -misc.sign(self.data.position)

        order_kwargs = {
            "action": misc.action(signal),
            "totalQuantity": abs(self.data.position),
        }
        if dynamic_order_kwargs:
            order_kwargs.update(dynamic_order_kwargs)
        order = self._order("close_order", order_kwargs)
        assert self.data.active_contract is not None
        log.debug(
            f"{self.strategy} {self.data.active_contract.localSymbol} "
            f"processing CLOSE signal {signal}, current position: {self.data.position},"
            f" order: {order_kwargs['action']} {order_kwargs['totalQuantity']}",
            extra={"data": data},
        )
        return self.trade(
            self.data.active_contract,
            order,
            "CLOSE",
        )

    def reverse(self, data: dict) -> None:
        close_trade = self.close(data)
        if close_trade:
            close_trade.filledEvent += partial(self.open, data=data)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class EventDrivenExecModel(BaseExecModel):
    """
    Use events to attach stop-loss and optional take-profit orders after
    execution of open order. After close transaction remove existing
    bracketing orders.

    this has been moved out of this class:
    On re(start) make sure that all positions have
    stop-losses and that there are no open orders not attached to any
    existing position.
    """

    _stop_order: dict[str, Any] = {}
    _tp_order: dict[str, Any] = {}
    stop_order = Validator(order_field_validator)
    tp_order = Validator(order_field_validator)

    def __init__(
        self,
        orders: Optional[dict[OrderKey, Any]] = None,
        *,
        stop: Optional[AbstractBracketLeg] = None,
        take_profit: Optional[AbstractBracketLeg] = None,
        controller: Optional[Controller] = None,
    ):
        if not stop:
            raise TypeError(
                f"{self.__class__.__name__} must be initialized with a stop bracket leg"
            )
        self.stop = stop
        self.take_profit = take_profit
        self.oca_group_generator = lambda: str(uuid4())
        super().__init__(orders, controller=controller)

    def onStart(self, data, *args):
        super().onStart(data, *args)
        self.data.brackets = {}

    def open(
        self,
        data: dict,
        dynamic_order_kwargs: Optional[dict] = None,
    ) -> ibi.Trade:
        """
        Save information required for bracket orders and attach events
        that will attach brackets after order completion.
        """
        attach_bracket = partial(self._attach_bracket, params=data)
        # reset any previous oca_group settings
        self.data.oca_group = None

        trade = super().open(data)

        trade.filledEvent += attach_bracket
        return trade

    def close(
        self,
        data: dict,
        dynamic_order_kwargs: Optional[dict] = None,
    ) -> Optional[ibi.Trade]:
        """
        Attach oca that will cancel any brackets after order execution.
        """

        return super().close(data, self._dynamic_bracket_kwargs())

    def _dynamic_bracket_kwargs(self) -> dict[str, Any]:
        self.data.oca_group = (
            self.data.get("oca_group", None) or self.oca_group_generator()
        )
        return {"ocaGroup": self.data.oca_group, "ocaType": CONFIG.get("oca_type") or 1}

    def _attach_bracket(self, trade: ibi.Trade, params: dict) -> None:
        # called once for sl/tp pair! Don't put inside the for loop!
        log.debug(f"Will attach bracket for {trade.order.orderId}")
        try:
            dynamic_bracket_kwargs = self._dynamic_bracket_kwargs()
            for bracket, order_key, label in zip(
                (self.stop, self.take_profit),
                ("stop_order", "tp_order"),
                ("STOP-LOSS", "TAKE-PROFIT"),
            ):
                # take profit may be None
                if bracket:
                    memo: dict[str, Any] = {}
                    bracket_kwargs = bracket(params, trade, memo)
                    bracket_kwargs.update(CONFIG.get(order_key))  # type: ignore
                    bracket_kwargs.update(dynamic_bracket_kwargs)
                    memo.update(
                        {
                            "strategy": self.strategy,
                            "position_id": self.get_position_id(),
                            "bracket_timestamp": datetime.now(timezone.utc),
                        }
                    )
                    self.data.params[label.lower()] = memo
                    order = self._order(cast(OrderKey, order_key), bracket_kwargs)

                    bracket_trade = self.trade(
                        trade.contract,
                        order,
                        label,
                    )
                    self.data.brackets[str(bracket_trade.order.orderId)] = Bracket(
                        cast(BracketLabel, label),
                        cast(OrderKey, order_key),
                        bracket_kwargs,
                        bracket_trade,
                    )
        except Exception as e:
            log.exception(f"Error while attaching bracket: {e}")

    def re_attach_brackets(self):
        """
        Possibly used by :class:`Controller` if it's determined that a bracket
        is missing.
        """

        for orderId, bracket in self.data.brackets.copy().items():
            log.info(f"attempt to re-attach bracket {bracket}")
            del self.data.brackets[orderId]
            self._place_bracket(
                bracket.trade.contract,
                bracket.order_key,
                bracket.label,
                bracket.bracket_kwargs,
            )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(stop={self.stop}, "
            f"take_profit={self.take_profit})"
        )
