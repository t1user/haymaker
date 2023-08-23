from __future__ import annotations

import itertools
import logging
import random
import string
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, ClassVar, Literal, Optional, TypedDict

import ib_insync as ibi

from ib_tools.bracket_legs import AbstractBracketLeg
from ib_tools.trader import Trader

from . import misc

log = logging.getLogger(__name__)

OrderKey = Literal["open_order", "close_order", "stop_order", "tp_order"]


class Params(TypedDict):
    open: dict[str, Any]
    close: dict[str, Any]


class AbstractExecModel(ABC):
    """
    Intermediary between Portfolio and Trader.  Allows for fine tuning
    of order types used, order monitoring, post-order events, etc.

    Object initialized by a :class:``Brick``, :class:``Controller``
    will call execute method.

    All execution models must inherit from this class.
    """

    trader: ClassVar[Trader]

    open_order: dict[str, Any]
    close_order: dict[str, Any]

    @classmethod
    def inject_trader(cls, trader: Trader) -> None:
        cls.trader = trader

    def __init__(self, orders: Optional[dict[OrderKey, Any]] = None) -> None:
        self.contract: Optional[ibi.Contract] = None
        self.position: float = 0.0
        self.params: Params = {"open": {}, "close": {}}

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

        * or one of keys in ``orders`` dict passed to
        :meth:``self.__init__``, examples of good orders potentially
        defined this way: `stop_order`, `tp_order`

        params: must be a dict of keywords and values accepted by
        ibi.Order

        Returns:
        ========

        :class:``ibi.Order`` object, which can be directly passed to
        the broker via :meth:``ibi.IB.send_order``.
        """
        order_kwargs = getattr(self, key)
        params.update(order_kwargs)
        return ibi.Order(**params)

    @abstractmethod
    def execute(self, data: dict) -> tuple[ibi.Trade, str]:
        """
        Must use :meth:``self.trade``(:class:``ibi.Contract``,
        :class:``ibi.Order``, strategy_key, reason) to send orders for
        execution, and subsequently link any :class:``ibi.Trade``
        events returned by :meth:``self.trade`` to required callbacks.

        While openning position must set :attr:``self.contract`` to
        :class:``ibi.Contract`` that has been used.

        Must keep track of current position in the market by updating
        :attr:``self.position``.

        Args:
        =====

        data: is a dict created by :class:``Brick``, updated by
        :class:``Portfolio``, which must contain all parameters
        required to execute transactions in line with this execution
        model.

        Returns:
        ========

        (trade, note), where:

        * trade: :class:``ibi.Trade`` object for the issued order *
        note: info string for loggers and blotters about the character
        of the transaction (open, close, stop, etc.)
        """
        # TODO: what if more than one order issued????
        ...


class BaseExecModel(AbstractExecModel):
    """
    Enters and closes positions based on params sent to
    :meth:``execute``.  Orders composed by :meth:`open` and
    :meth:`close`, which can be overridden or extended in subclasses
    to get more complex behaviour.
    """

    def __init__(self, orders: Optional[dict[OrderKey, Any]] = None) -> None:
        super().__init__(orders)

    open_order = {
        "orderType": "MKT",
        "algoStrategy": "Adaptive",
        "algoParams": [ibi.TagValue("adaptivePriority", "Normal")],
        "tif": "Day",
    }
    close_order = {"oderType": "MKT", "tif": "GTC"}

    def execute(self, data: dict) -> tuple[ibi.Trade, str]:
        if data["signal"][2] == "open":
            self.params["open"].update(data)
            note = "OPEN"
            trade = self.open(data)
            trade.filledEvent += self.register_contract
        elif data["signal"][2] == "close":
            # double check if position exists / or is it checked by signal already?
            # make sure correct contract used (from params dict?)
            self.params["close"].update(data)
            note = "CLOSE"
            trade = self.close(data)
        trade.fillEvent += self.register_position

        return trade, note

    def register_position(self, trade: ibi.Fill) -> None:
        # TODO: if several different contracts, this is screwed
        self.position += trade.execution.shares

    def register_contract(self, trade: ibi.Trade) -> None:
        # TODO: handle situation if contract is a list of ibi.Contract ???
        self.contract = trade.contract

    def open(self, data: dict) -> ibi.Trade:
        contract = data["contract"]
        signal = data["signal"][0]  # TODO: this is a dummy
        amount = int(data["amount"])
        order_kwargs = {"action": misc.action(signal), "totalQuantity": amount}
        order = self._order("open_order", order_kwargs)
        key = data.get("key") or ""
        return self.trader.trade(contract, order, "OPEN", key)

    def close(self, data: dict) -> ibi.Trade:
        # TODO: this is a dummy, make it work
        signal = data["signal"]  # wtf???
        order_kwargs = {"action": misc.action(signal), "totalQuantity": self.position}
        order = self._order("close_order", order_kwargs)
        assert self.contract is not None
        key = data.get("key") or ""
        return self.trader.trade(self.contract, order, "CLOSE", key)

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

    stop_order: dict[str, Any] = {}
    tp_order: dict[str, Any] = {}

    contract: ibi.Contract  # or should it be a list of ibi.Contract ?
    brackets: list[ibi.Trade]

    def __init__(
        self,
        stop: Optional[AbstractBracketLeg] = None,
        take_profit: Optional[AbstractBracketLeg] = None,
    ):
        if not stop:
            log.error("EventDrivenExecModel must be initialized with a stop order")
        self.stop = stop
        self.take_profit = take_profit
        self.brackets = []
        log.debug(f"execution model initialized {self}")

    def open(self, data: dict) -> ibi.Trade:
        """
        Save information required for bracket orders and attach events
        that will attach brackets after order completion.
        """
        log.debug(f"{data['contract'].localSymbol} open signal: {data['signal']} ")
        trade = super().open(data)
        attach_bracket = partial(self._attach_bracket, params=data)
        trade.filledEvent += attach_bracket
        return trade

    def close(self, data: dict) -> ibi.Trade:
        """
        Attach events that will cancel any brackets on order completion.
        """
        log.debug(
            f"{data['contract'].localSymbol} close signal: "
            f"{data['close_params']['signal']}"
        )
        trade = super().close(data)
        trade.filledEvent += self.remove_bracket
        return trade

    def _dynamic_bracket_kwargs(self) -> dict[str, Any]:
        return {}

    def _attach_bracket(self, trade: ibi.Trade, params: dict) -> None:
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
                bracket_trade = self.trader.trade(
                    trade.contract, order, label, params["key"]
                )
                self.brackets.append(bracket_trade)
                bracket_trade.filledEvent += self.bracket_filled_callback

    def remove_bracket(self, trade: ibi.Trade) -> None:
        # trade is for an order that has just been filled!!!
        # irrelevant here
        for bracket in self.brackets.copy():
            if not bracket.isDone():
                self.trader.cancel(bracket)
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
        stop: Optional[AbstractBracketLeg] = None,
        take_profit: Optional[AbstractBracketLeg] = None,
    ):
        super().__init__(stop, take_profit)
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
