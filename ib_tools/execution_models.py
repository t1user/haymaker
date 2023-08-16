from __future__ import annotations

import random
import string
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Literal, NamedTuple, Optional, Protocol

import ib_insync as ibi

from ib_tools.bracket_legs import AbstractBracketLeg
from ib_tools.logger import Logger

from . import misc

# from ib_tools.trader import Trader


log = Logger(__name__)

OrderKey = Literal["entry_order", "close_order", "stop_order", "tp_order"]


class TraderProtocol(Protocol):
    def trade(self, conntract: ibi.Contract, order: ibi.Order, note: str) -> ibi.Trade:
        ...

    def cancel(self, trade: ibi.Trade) -> ibi.Trade:
        ...

    def modify(self, contract: ibi.Contract, order: ibi.Order) -> ibi.Trade:
        ...

    def trades_for_contract(self, contract: ibi.Contract) -> list[ibi.Trade]:
        ...


# =====================================================================
# Helpers
# =====================================================================


class ContractMemo(NamedTuple):
    sl_points: float
    min_tick: float


# =====================================================================
# Actual execution models
# =====================================================================


class AbstractExecModel(ABC):
    """
    Intermediary between Portfolio and Trader.  Allows for fine tuning
    of order types used, order monitoring, post-order events, etc.

    Object initialized by a ``Brick``, ``Controller`` will call
    execute method.

    All execution models must inherit from this class.
    """

    trader: TraderProtocol

    entry_order: dict[str, Any]
    close_order: dict[str, Any]

    @classmethod
    def inject_trader(cls, trader: TraderProtocol) -> None:
        cls.trader = trader

    def __init__(self, orders: Optional[dict[OrderKey, Any]] = None) -> None:
        if orders:
            for key, order_kwargs in orders.items():
                setattr(self, key, order_kwargs)

    def _order(self, key: OrderKey, params: dict) -> ibi.Order:
        """Builds order object from passed params.

        Args:
        =====

        key: what kind of transaction it is.  Must be one of:
        `entry_order`, `close_order`, `stop_order`, `tp_order`

        params: must be a dict of keywords and values accepted by
        ibi.Order
        """
        order_kwargs = getattr(self, key)
        params.update(order_kwargs)
        return ibi.Order(**params)

    @abstractmethod
    def execute(self, data: dict) -> tuple[ibi.Trade, str]:
        """
        Must use ``self.trade`` to send orders for execution, link any
        ``ibi.Trade`` events to required callbacks.

        Args:
        =====

        data: is a dict created by :class:``Brick``, updated by
        :class:``Portfolio``, which must contain all parameters
        required to execute transactions in line with this execution
        model.

        Returns:
        ========

        (trade, note), where:

        * trade: :class:``ibi.Trade`` object for the issued order
        * note: info string for loggers and blotters about the
        character of the transaction (open, close, stop, etc.)


        """
        # TODO: what if more than one order issued????
        ...


class BaseExecModel(AbstractExecModel):
    """
    Enters and closes positions based on params sent to
    :meth:``execute``.  Orders composed by :meth:`entry` and
    :meth:`close`, which can be overridden or extended in subclasses
    to get more complex behaviour.
    """

    params: dict
    contract: ibi.Contract
    position: float

    entry_order = {
        "orderType": "MKT",
        "algoStrategy": "Adaptive",
        "algoParams": [ibi.TagValue("adaptivePriority", "Normal")],
        "tif": "Day",
    }
    close_order = {"oderType": "MKT", "tif": "GTC"}

    def execute(self, data: dict) -> tuple[ibi.Trade, str]:
        if data["signal"][2] == "entry":
            self.params = data
            note = "ENTRY"
            trade = self.entry(data)
            trade.filledEvent += self.register_contract
        elif data["signal"][2] == "close":
            # double check if position exists / or is it checked by signal already?
            # make sure correct contract used (from params dict?)
            self.params.update({"close_params": data})
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

    def entry(self, data: dict) -> ibi.Trade:
        try:
            contract = data["contract"]
            signal = data["signal"][0]  # TODO: this is a dummy
            amount = int(data["amount"])
        except KeyError as e:
            log.error("Missing params. Cannot execute trade", e)
            # TODO: it's broken here, what do to?
        order_kwargs = {"action": misc.action(signal), "totalQuantity": amount}
        order = self._order("entry_order", order_kwargs)
        return self.trader.trade(contract, order, "ENTRY")

    def close(self, data: dict) -> ibi.Trade:
        # TODO: this is a dummy, make it work
        contract = self.contract
        signal = data["signal"]  # wtf???
        amount = self.position
        order_kwargs = {"action": misc.action(signal), "totalQuantity": amount}
        order = self._order("close_order", order_kwargs)
        return self.trader.trade(contract, order, "CLOSE")

    def __repr__(self):
        items = (f"{k}={v}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({', '.join(items)})"


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

    def entry(self, data: dict) -> ibi.Trade:
        """
        Save information required for bracket orders and attach events
        that will attach brackets after order completion.
        """
        log.debug(f"{data['contract'].localSymbol} entry signal: {data['signal']} ")
        trade = super().entry(data)
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
                bracket_trade = self.trader.trade(trade.contract, order, label)
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

    oca_ids: list[str] = []

    def oca_group(self):
        while (
            o := "".join(random.choices(string.ascii_letters + string.digits, k=10))
        ) in self.oca_ids:
            pass
        self.oca_ids.append(o)
        return o

    def _dynamic_bracket_kwargs(self):
        return {"ocaGroup": self.oca_group(), "ocaType": 1}

    def report_bracket(self, trade: ibi.Trade) -> None:
        log.debug(f"Bracketing order filled: {trade.order}")

    bracket_filled_callback = report_bracket
