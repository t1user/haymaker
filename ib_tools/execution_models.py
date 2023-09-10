from __future__ import annotations

import dataclasses
import itertools
import logging
import random
import string
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Literal, Optional, TypedDict

import ib_insync as ibi
import numpy as np

from ib_tools.base import Atom
from ib_tools.bracket_legs import AbstractBracketLeg
from ib_tools.controller import Controller
from ib_tools.manager import CONTROLLER

from . import misc

log = logging.getLogger(__name__)

OrderKey = Literal[
    "open_order", "close_order", "stop_order", "tp_order", "reverse_order"
]


class Params(TypedDict):
    open: dict[str, Any]
    close: dict[str, Any]


class OrderFieldValidator:
    allowed_keys: set[str] = set(dataclasses.asdict(ibi.Order()).keys())

    def __set_name__(self, owner, name) -> None:
        self.private_name = "_" + name

    def __get__(self, obj, objtype=None) -> dict[str, Any]:
        return getattr(obj, self.private_name)

    def __set__(self, obj, value: dict[str, Any]) -> None:
        if diff := self.validate(value):
            raise ValueError(f"Wrong fields for {self.private_name}: {diff}")
        setattr(obj, self.private_name, value)

    def validate(self, value: dict[str, Any]) -> set:
        return set(value.keys()) - self.allowed_keys


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
    open_order = OrderFieldValidator()
    close_order = OrderFieldValidator()

    def __init__(
        self,
        orders: Optional[dict[OrderKey, Any]] = None,
        *,
        controller: Optional[Controller] = None,
    ) -> None:
        self.active_contract: Optional[ibi.Contract] = None
        self.position: float = 0.0
        self.strategy: str = ""
        self.params: Params = {"open": {}, "close": {}}
        self.controller = controller or CONTROLLER

        if orders:
            for key, order_kwargs in orders.items():
                setattr(self, key, order_kwargs)

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

        params: must be a dict of keywords and values accepted by
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
    def onData(self, data: dict, *args) -> None:
        """
        Must use :meth:`self.trade`(:class:`ibi.Contract`,
        :class:`ibi.Order`, strategy_key, reason) to send orders for
        execution, and subsequently link any :class:`ibi.Trade`
        events returned by :meth:`self.trade` to required callbacks.

        While openning position must set :attr:`self.contract` to
        :class:`ibi.Contract` that has been used.

        Must keep track of current position in the market by updating
        :attr:`self.position`.

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
        data["exec_model"] = self


class BaseExecModel(AbstractExecModel):
    """
    Enters and closes positions based on params sent to
    :meth:`.execute`.  Orders composed by :meth:`.open` and
    :meth:`.close`, which can be overridden or extended in subclasses
    to get more complex behaviour.
    """

    _open_order = {
        "orderType": "MKT",
        "algoStrategy": "Adaptive",
        "algoParams": [ibi.TagValue("adaptivePriority", "Normal")],
        "tif": "Day",
    }
    _close_order = {"oderType": "MKT", "tif": "GTC"}
    _reverse_order = _open_order

    def __init__(
        self,
        orders: Optional[dict[OrderKey, Any]] = None,
        *,
        controller: Optional[Controller] = None,
    ) -> None:
        super().__init__(orders, controller=controller)
        self.position = 0
        self.active_contract = None

    def trade(
        self,
        contract: ibi.Contract,
        order: ibi.Order,
        action: str,
        callback: Optional[Callable[[ibi.Trade], None]] = None,
    ) -> None:
        self.controller.trade(
            contract,
            order,
            action,
            self,
            partial(self.trade_callback, callback=callback),
        )

    def cancel(
        self, trade: ibi.Trade, callback: Optional[misc.Callback] = None
    ) -> None:
        if callback is None:
            callback = self.cancel_callback
        self.controller.cancel(trade, self, callback)

    def trade_callback(
        self, trade: ibi.Trade, callback: Optional[misc.Callback] = None
    ) -> None:
        trade.fillEvent += self.register_position
        if callback is not None:
            callback(trade)

    def cancel_callback(self, trade, callback: Optional[misc.Callback] = None) -> None:
        # override this in subclass if needed
        pass

    def onData(self, data: dict, *args) -> None:
        data["exec_model"] = self
        try:
            action = data["action"]
        except KeyError as e:
            log.error(f"Missing data for {self}", e)
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

    def register_position(self, trade: ibi.Trade, fill: ibi.Fill) -> None:
        if fill.execution.side == "BOT":
            self.position += fill.execution.shares
        elif fill.execution.side == "SLD":
            self.position -= fill.execution.shares
        else:
            log.critical(
                f"Abiguous fill: {fill} for order: {trade.order} for "
                f"{trade.contract.localSymbol}"
            )

    def open(self, data: dict, callback: Optional[misc.Callback] = None) -> None:
        self.params["close"] = {}
        self.params["open"].update(data)
        try:
            contract = data["contract"]
            signal = data["signal"]
            amount = int(data["amount"])
        except KeyError as e:
            log.error("Insufficient data to execute OPEN transaction", e)
        self.active_contract = contract
        order_kwargs = {"action": misc.action(signal), "totalQuantity": amount}
        order = self._order("open_order", order_kwargs)
        strategy = data["strategy"]
        log.debug(
            f"{strategy} {contract.localSymbol} executing OPEN signal {signal}.", data
        )
        self.trade(contract, order, "OPEN", callback)

    def close(self, data: dict, callback: Optional[misc.Callback] = None) -> None:
        self.params["close"].update(data)
        signal = -np.sign(self.position)
        order_kwargs = {"action": misc.action(signal), "totalQuantity": self.position}
        order = self._order("close_order", order_kwargs)
        assert self.active_contract is not None
        strategy = data["strategy"]
        log.debug(
            f"{strategy} {self.active_contract.localSymbol} executing close signal"
            f" {signal}",
            data,
        )
        self.trade(self.active_contract, order, "CLOSE", callback)

    def reverse(self, data: dict, callback: Optional[misc.Callback] = None) -> None:
        def connect_open_after_close(trade):
            trade += partial(self.close, data)

        self.close(data, connect_open_after_close)

    def __repr__(self):
        items = (f"{k}={v}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({', '.join(items)})"


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

    _stop_order: dict[str, Any]
    _tp_order: dict[str, Any]
    stop_order = OrderFieldValidator()
    tp_order = OrderFieldValidator()

    contract: ibi.Contract  # or should it be a list of ibi.Contract ?
    brackets: list[ibi.Trade]

    def __init__(
        self,
        orders: Optional[dict[OrderKey, Any]] = None,
        *,
        stop: Optional[AbstractBracketLeg] = None,
        take_profit: Optional[AbstractBracketLeg] = None,
        controller: Optional[Controller] = None,
    ):
        super().__init__(orders, controller=controller)
        if not stop:
            log.error(
                f"{self.__class__.__name__} must be initialized with a stop order"
            )
        self.stop = stop
        self.take_profit = take_profit
        self.brackets = []
        log.debug(f"execution model initialized {self}")

    def open(self, data: dict, callback: Optional[misc.Callback] = None) -> None:
        """
        Save information required for bracket orders and attach events
        that will attach brackets after order completion.
        """
        attach_bracket = partial(self._attach_bracket, params=data)

        def callback_attach_bracket(trade):
            trade.filledEvent += attach_bracket

        super().open(data, callback_attach_bracket)

    def close(self, data: dict, callback: Optional[misc.Callback] = None) -> None:
        """
        Attach events that will cancel any brackets on order completion.
        """

        def callback_remove_bracket(trade: ibi.Trade) -> None:
            trade.filledEvent += self.remove_bracket

        super().close(data, callback_remove_bracket)

    def _dynamic_bracket_kwargs(self) -> dict[str, Any]:
        return {}

    def _attach_bracket(self, trade: ibi.Trade, params: dict) -> None:
        def callback_bracket_filled(trade):
            trade.filledEvent += self.bracket_filled_callback
            self.brackets.append(trade)

        for bracket, order_key, label in zip(
            (self.stop, self.take_profit),
            ("stop_order", "tp_order"),
            ("STOP-LOSS", "TAKE-PROFIT"),
        ):
            # take profit may be None
            if bracket:
                bracket_kwargs = bracket(params, trade)
                bracket_kwargs.update(self._dynamic_bracket_kwargs)  # type: ignore
                order = self._order(order_key, bracket_kwargs)  # type: ignore
                log.debug(f"bracket order: {order}")
                self.trade(trade.contract, order, label)

    def remove_bracket(self, trade: ibi.Trade) -> None:
        # trade is for a bracket order that has just been filled!!!
        # irrelevant here, because we want to cancel potential other bracket(s)
        # for the same position

        for bracket in self.brackets.copy():
            if not bracket.isDone():
                self.cancel(bracket)
                self.brackets.remove(bracket)

    bracket_filled_callback = remove_bracket


class OcaExecModel(EventDrivenExecModel):
    """
    Use Interactive Brokers OCA (one cancells another) orders for
    stop-loss and take-profit.
    """

    counter_seed = itertools.count(1, 1).__next__

    def __init__(
        self,
        orders: Optional[dict[OrderKey, Any]] = None,
        *,
        stop: Optional[AbstractBracketLeg] = None,
        take_profit: Optional[AbstractBracketLeg] = None,
        controller: Optional[Controller] = None,
    ):
        super().__init__(
            orders, stop=stop, take_profit=take_profit, controller=controller
        )
        self.counter = itertools.count(
            100000 * self.counter_seed(), 1  # type: ignore
        ).__next__
        self.character_string = "".join(random.choices(string.ascii_letters, k=5))

    def oca_group(self):
        return f"{self.character_string}{self.counter()}"

    def _dynamic_bracket_kwargs(self):
        return {"ocaGroup": self.oca_group(), "ocaType": 1}

    def report_bracket(self, trade: ibi.Trade) -> None:
        log.debug(f"Bracketing order filled: {trade.order}")

    bracket_filled_callback = report_bracket
