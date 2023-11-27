from __future__ import annotations

import asyncio
import dataclasses
import itertools
import logging
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Literal, NamedTuple, Optional, TypedDict

import ib_insync as ibi

from ib_tools.base import Atom
from ib_tools.bracket_legs import AbstractBracketLeg
from ib_tools.controller import Controller
from ib_tools.manager import CONTROLLER, STATE_MACHINE

from . import misc

log = logging.getLogger(__name__)

OrderKey = Literal[
    "open_order", "close_order", "stop_order", "tp_order", "reverse_order"
]

BracketLabel = Literal["STOP_LOSS", "TAKE_PROFIT"]


class Params(TypedDict):
    open: dict[str, Any]
    close: dict[str, Any]


class Bracket(NamedTuple):
    label: BracketLabel
    order_key: OrderKey
    order_kwargs: dict
    trade: ibi.Trade


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
        orders: Optional[dict[OrderKey, dict[str, Any]]] = None,
        *,
        controller: Optional[Controller] = None,
    ) -> None:
        super().__init__()
        self.active_contract: Optional[ibi.Contract] = None
        self.position: float = 0.0
        self.strategy: str = ""
        self.params: Params = {"open": {}, "close": {}}
        self.controller = controller or CONTROLLER

        if orders:
            for key, order_kwargs in orders.items():
                setattr(self, key, order_kwargs)

        self.position_id_generator = misc.Counter()
        self._position_id = ""
        self.connect_state_machine()

    def connect_state_machine(self):
        self += STATE_MACHINE

    def position_id(self, reset=False):
        if reset or not self._position_id:
            self._position_id = self.position_id_generator()
        return self._position_id

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
        super().onData(data)
        data["exec_model"] = self
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

    _open_order = {
        "orderType": "MKT",
        "algoStrategy": "Adaptive",
        "algoParams": [ibi.TagValue("adaptivePriority", "Normal")],
        "tif": "Day",
    }
    _close_order = {
        "orderType": "MKT",
        "tif": "GTC",
    }

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
        trade.fillEvent -= self.register_position
        trade.fillEvent += self.register_position
        if callback is not None:
            callback(trade)

    def cancel_callback(self, trade, callback: Optional[misc.Callback] = None) -> None:
        # override this in subclass if needed
        pass

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
            log.info(f"No subscription for live ticker {self.active_contract}")

    def get_ticker(self, contract) -> Optional[ibi.Ticker]:
        for t in self.ib.tickers():
            if t.contract == contract:
                return t
        return None

    def onData(self, data: dict, *args):
        data["exec_model"] = self

        # if await (ticker := self.live_ticker()):
        #     data["arrival_price"] = {"bid": ticker.bid, "ask": ticker.ask}
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

    def register_position(self, trade: ibi.Trade, fill: ibi.Fill) -> None:
        if fill.execution.side == "BOT":
            self.position += fill.execution.shares
            log.debug(
                f"Registered {trade.order.orderId} BUY {trade.order.orderType} "
                f"for {self.strategy} --> position: {self.position}"
            )
        elif fill.execution.side == "SLD":
            self.position -= fill.execution.shares
            log.debug(
                f"Registered {trade.order.orderId} SELL {trade.order.orderType} "
                f"for {self.strategy} --> position: {self.position}"
            )
        else:
            log.critical(
                f"Abiguous fill: {fill} for order: {trade.order} for "
                f"{trade.contract.localSymbol} strategy: {self.strategy}"
            )

    def open(self, data: dict, callback: Optional[misc.Callback] = None) -> None:
        self.params["close"] = {}
        data["position_id"] = self.position_id(True)
        self.params["open"].update(data)
        try:
            contract = data["contract"]
            signal = data["signal"]
            amount = int(data["amount"])
        except KeyError:
            log.exception("Insufficient data to execute OPEN transaction")
        self.active_contract = contract
        order_kwargs = {"action": misc.action(signal), "totalQuantity": amount}
        order = self._order("open_order", order_kwargs)
        log.debug(
            f"{self.strategy} {contract.localSymbol} processing OPEN signal {signal}",
            extra={"data": data},
        )
        self.trade(contract, order, "OPEN", callback)

    def close(self, data: dict, callback: Optional[misc.Callback] = None) -> None:
        data["position_id"] = self.position_id()
        self.params["close"].update(data)
        # THIS IS TEMPORARY ----> FIX ---> TODO
        if self.position == 0:
            log.error(
                f"Abandoning CLOSE position for {self.active_contract} "
                f"(No position, but close signal)"
            )
            return

        signal = -misc.sign(self.position)

        order_kwargs = {
            "action": misc.action(signal),
            "totalQuantity": abs(self.position),
        }
        order = self._order("close_order", order_kwargs)
        assert self.active_contract is not None
        log.debug(
            f"{self.strategy} {self.active_contract.localSymbol} "
            f"processing CLOSE signal {signal}, current position: {self.position},"
            f" order: {order_kwargs['action']} {order_kwargs['totalQuantity']}",
            extra={"data": data},
        )
        self.trade(self.active_contract, order, "CLOSE", callback)

    def reverse(self, data: dict, callback: Optional[misc.Callback] = None) -> None:
        def open_after_close(trade, data):
            self.open(data)

        def connect_open_after_close(trade):
            trade.filledEvent += partial(open_after_close, data=data)

        self.close(data, connect_open_after_close)

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
    stop_order = OrderFieldValidator()
    tp_order = OrderFieldValidator()

    # contract: ibi.Contract  # or should it be a list of ibi.Contract ?

    def __init__(
        self,
        orders: Optional[dict[OrderKey, Any]] = None,
        *,
        stop: Optional[AbstractBracketLeg] = None,
        take_profit: Optional[AbstractBracketLeg] = None,
        controller: Optional[Controller] = None,
    ):
        if not stop:
            log.error(
                f"{self.__class__.__name__} must be initialized with a stop bracket leg"
            )
        self.stop = stop
        self.take_profit = take_profit
        self.brackets: dict[int, Bracket] = {}
        super().__init__(orders, controller=controller)

    def open(self, data: dict, callback: Optional[misc.Callback] = None) -> None:
        """
        Save information required for bracket orders and attach events
        that will attach brackets after order completion.
        """
        attach_bracket = partial(self._attach_bracket, params=data)

        def callback_open_trade(trade):
            trade.filledEvent -= attach_bracket
            trade.filledEvent += attach_bracket

        super().open(data, callback_open_trade)

    def close(self, data: dict, callback: Optional[misc.Callback] = None) -> None:
        """
        Attach events that will cancel any brackets on order completion.
        """

        def callback_close_trade(trade: ibi.Trade) -> None:
            trade.filledEvent -= self.remove_bracket
            trade.filledEvent += self.remove_bracket

        super().close(data, callback_close_trade)

    def _dynamic_bracket_kwargs(self) -> dict[str, Any]:
        # make sure to call only once for sl/tp pair!!!!
        return {}

    def _attach_bracket(self, trade: ibi.Trade, params: dict) -> None:
        # called once for sl/tp pair! Don't put inside the for loop!
        dynamic_bracket_kwargs = self._dynamic_bracket_kwargs()

        for bracket, order_key, label in zip(
            (self.stop, self.take_profit),
            ("stop_order", "tp_order"),
            ("STOP-LOSS", "TAKE-PROFIT"),
        ):
            # take profit may be None
            if bracket:
                bracket_kwargs = bracket(params, trade)
                bracket_kwargs.update(dynamic_bracket_kwargs)
                self._place_bracket(trade.contract, order_key, label, bracket_kwargs)

    def _place_bracket(self, contract, order_key, label, bracket_kwargs):
        def save_bracket(
            label: BracketLabel,
            bracket_kwargs: dict,
            trade: ibi.Trade,
        ):
            self.brackets[trade.order.orderId] = Bracket(
                label, order_key, bracket_kwargs, trade
            )

        def callback_bracket_trade(trade, label="", bracket_kwargs=None):
            trade.filledEvent += self.bracket_filled_callback
            trade.filledEvent += self.bracket_filled_callback
            save_bracket(label, bracket_kwargs, trade)

        order = self._order(order_key, bracket_kwargs)
        log.debug(f"bracket order: {order}")
        self.trade(
            contract,
            order,
            label,
            partial(
                callback_bracket_trade,
                label=label,
                bracket_kwargs=bracket_kwargs,
            ),
        )

    def re_attach_brackets(self):
        """
        Possibly used by :class:`Controller` if it's determined that a bracket
        is missing.
        """

        for orderId, bracket in self.brackets.copy().items():
            log.info(f"attempt to re-attach bracket {bracket}")
            del self.brackets[orderId]
            self._place_bracket(
                bracket.trade.contract,
                bracket.order_key,
                bracket.label,
                bracket.bracket_kwargs,
            )

    def remove_bracket(self, trade: ibi.Trade) -> None:
        # trade is for a bracket order that has just been filled!!!
        # it may be either a close transaction or one of the brackets
        # in either case position is closed and we need to remove
        # remaining brackets if any
        for orderId, bracket in self.brackets.copy().items():
            if not bracket.trade.isDone():
                # trade may be one of the brackets
                if orderId != trade.order.orderId:
                    log.debug(
                        f"Will attempt to remove: {bracket.label} "
                        f"{trade.order.orderId} {trade.contract.symbol} "
                        f"{trade.order.orderType} {trade.order.action}"
                        f"trade object id: {id(trade)}"
                    )
                    self.cancel(bracket.trade)
                del self.brackets[orderId]

    bracket_filled_callback = remove_bracket

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(stop={self.stop}, "
            f"take_profit={self.take_profit})"
        )


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
        self.oca_group = misc.Counter()

    def _dynamic_bracket_kwargs(self):
        return {"ocaGroup": self.oca_group(), "ocaType": 1}

    def report_bracket(self, trade: ibi.Trade) -> None:
        log.debug(f"Bracketing order filled: {trade.order}")

    bracket_filled_callback = report_bracket
