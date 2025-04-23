from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast
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


OPEN_ORDER = {**CONFIG.get("open_order", {}), "orderType": "MKT"}
CLOSE_ORDER = {**CONFIG.get("close_order", {}), "orderType": "MKT"}
STOP_ORDER = {**CONFIG.get("stop_order", {}), "orderType": "STP"}
TP_ORDER = {**CONFIG.get("tp_order", {}), "orderType": "LMT"}
OCA_TYPE = CONFIG.get("oca_type", 1)


class OrderKey(str, Enum):
    open_order = "open_order"
    close_order = "close_order"
    stop_order = "stop_order"
    tp_order = "tp_order"

    def __str__(self):
        return self.value.lower()


BracketLabel = Literal["STOP_LOSS", "TAKE_PROFIT"]


class Bracket(NamedTuple):
    label: BracketLabel
    order_key: OrderKey
    order_kwargs: dict
    trade: ibi.Trade


class AbstractExecModel(Atom, ABC):
    """
    Intermediary between Portfolio and Trader.  It translates strategy signals into
    orders acceptable by Interactive Brokers. May also implement market execution
    strategy, monitor post-order events or manage any other function related to order
    execution.
    """

    open_order = Validator(order_field_validator)
    close_order = Validator(order_field_validator)

    def __init__(
        self,
        *,
        open_order: dict[str, Any] = {},
        close_order: dict[str, Any] = {},
        controller: Controller | None = None,
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
        self.open_order = {**OPEN_ORDER, **open_order}
        self.close_order = {**CLOSE_ORDER, **close_order}
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

        key: what kind of transaction it is, must be a member of :class:`OrderKey`

        params: must be a :py:`dict` of keywords and values accepted by
                ibi.Order

        Returns:
        ========

        :class:`ibi.Order` object, which can be directly passed to
        the broker via :meth:`ibi.IB.send_order`.
        """
        defaults = getattr(self, str(key))
        # passed kwargs must override defaults, not the other way round
        order_kwargs = {**defaults, **params}
        return ibi.Order(**order_kwargs)

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
        super().onData(data)
        self.dataEvent.emit(data)


class BaseExecModel(AbstractExecModel):
    """
    Orders generated based on data sent to :meth:`.onData`, of which
    following keys are required:

        - ``action``: must be one of: ``OPEN``, ``CLOSE``, ``REVERSE``

        - ``signal`` determines transaction direction, must be one of
          {-1, 1} for sell/buy respectively; irrelevant for ``CLOSE``,
          where direction is determined by current position

        - ``contract`` - this :class:`ibi.Contract` instance will be
          traded

        - ``amount`` - quantity of ``contract``s that will be traded

        - ``target_position`` - one of {-1, 0, 1} determining
          direction AFTER transaction is executed; will be used by
          :class:`Controller` to verify if the transaction's effect
          was as desired

    Enters and closes positions based on params sent to
    :meth:`onData`.  Orders composed by :meth:`open` and
    :meth:`close`, which can be overridden or extended in subclasses
    to get more complex behaviour.
    """

    def trade(
        self,
        contract: ibi.Contract,
        order: ibi.Order,
        action: str,
    ) -> ibi.Trade:
        return self.controller.trade(self.strategy, contract, order, action, self.data)

    def cancel(self, trade: ibi.Trade) -> ibi.Trade | None:
        return self.controller.cancel(trade)

    def get_ticker(self, contract) -> ibi.Ticker | None:
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
        dynamic_order_kwargs: dict | None = None,
    ) -> ibi.Trade | None:
        self.data.params["close"] = {}
        data["position_id"] = self.get_position_id(True)
        self.data.params["open"] = data
        try:
            contract = data["contract"]
            signal = data["signal"]
            amount = int(data["amount"])
        except KeyError:
            log.exception("Insufficient data to execute OPEN transaction")
            return None
        self.data.active_contract = contract
        order_kwargs = {"action": misc.action(signal), "totalQuantity": amount}
        if dynamic_order_kwargs:
            order_kwargs.update(dynamic_order_kwargs)
        order = self._order(OrderKey.open_order, order_kwargs)
        log.debug(
            f"{self.strategy} {contract.localSymbol} processing OPEN signal {signal}",
            extra={"data": data},
        )
        return self.trade(contract, order, "OPEN")

    def close(
        self,
        data: dict,
        dynamic_order_kwargs: dict | None = None,
    ) -> ibi.Trade | None:
        self.data.params["close"] = data
        data["position_id"] = self.get_position_id()

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
        order = self._order(OrderKey.close_order, order_kwargs)
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
        log.debug(f"reverse method returned close trade: {close_trade}")
        if close_trade:
            log.debug(
                f"Reverse trade. OPEN will be issued after CLOSE "
                f"{close_trade.order.orderId} done"
            )
            close_trade.filledEvent += partial(self.open, data=data)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class EventDrivenExecModel(BaseExecModel):
    """
    Use events to attach stop-loss and optional take-profit orders after
    execution of open order. After close transaction remove existing
    bracketing orders.

    Parameters (keyword only):
    ===========
    open_order, close_order, stop_order, tp_order: dict, optional
            maybe used to define order type used for respective transactions;
            passed dicts will be used to extend and/or update any defaults
            from config

    stop: :class:`AbstractBracketLeg` instance, must be provided
            manages creation of stop-loss order

    take_profit: :class:`AbstractBracketLeg` instance, optional
            manages creation of take-profit order

    oca_type: int, default 1
            OCA group type as per Interactive Brokers definition

    controller: :class:`Controller` instance, optional
            passing :class:`Controller` is meant for testing;
            otherwise the system should be allowed to use its own
            mechanisms to create it
    """

    stop_order = Validator(order_field_validator)
    tp_order = Validator(order_field_validator)

    def __init__(
        self,
        *,
        open_order: dict[str, Any] = {},
        close_order: dict[str, Any] = {},
        stop_order: dict[str, Any] = {},
        tp_order: dict[str, Any] = {},
        stop: AbstractBracketLeg | None = None,
        take_profit: AbstractBracketLeg | None = None,
        oca_type: int = OCA_TYPE,
        controller: Controller | None = None,
    ):
        if not stop:
            raise TypeError(
                f"{self.__class__.__name__} must be initialized with a stop bracket leg"
            )
        self.stop = stop
        self.take_profit = take_profit
        self.stop_order = {**STOP_ORDER, **stop_order}
        self.tp_order = {**TP_ORDER, **tp_order}
        self.oca_type = oca_type
        self.oca_group_generator = lambda: str(uuid4())
        super().__init__(
            open_order=open_order, close_order=close_order, controller=controller
        )

    def onStart(self, data, *args):
        super().onStart(data, *args)
        self.data.brackets = {}

    def open(
        self,
        data: dict,
        dynamic_order_kwargs: dict | None = None,
    ) -> ibi.Trade:
        """
        On top of actions perfomed by base class, this method will:
        save information required for bracket orders and attach events
        that will attach brackets after order completion.
        """
        attach_bracket = partial(self._attach_bracket, params=data)
        # reset any previous oca_group settings
        self.data.oca_group = None

        trade = super().open(data)
        assert trade

        trade.filledEvent += attach_bracket
        return trade

    def close(
        self,
        data: dict,
        dynamic_order_kwargs: dict | None = None,
    ) -> ibi.Trade | None:
        """
        On top of actions perfomed by base class, this method will:
        attach oca that will cancel any brackets after order execution.
        """

        return super().close(data, self._dynamic_bracket_kwargs())

    def _dynamic_bracket_kwargs(self) -> dict[str, Any]:
        self.data.oca_group = (
            self.data.get("oca_group", None) or self.oca_group_generator()
        )
        return {"ocaGroup": self.data.oca_group, "ocaType": self.oca_type}

    def _attach_bracket(self, trade: ibi.Trade, params: dict) -> None:
        # called once for sl/tp pair! Don't put inside the for loop!
        log.debug(f"Will attach brackets for {trade.order.orderId}")
        try:
            dynamic_bracket_kwargs = self._dynamic_bracket_kwargs()
            for bracket, order_key, label in zip(
                (self.stop, self.take_profit),
                (OrderKey.stop_order, OrderKey.tp_order),
                ("STOP-LOSS", "TAKE-PROFIT"),
            ):
                # take profit may be None
                if bracket:
                    memo: dict[str, Any] = {}
                    bracket_kwargs = bracket(params, trade, memo)
                    order_kwargs = {
                        **bracket_kwargs,
                        **dynamic_bracket_kwargs,
                    }
                    memo.update(
                        {
                            "strategy": self.strategy,
                            "position_id": self.get_position_id(),
                            "bracket_timestamp": datetime.now(timezone.utc),
                        }
                    )
                    self.data.params[label.lower()] = memo

                    order = self._order(order_key, order_kwargs)

                    bracket_trade = self.trade(
                        trade.contract,
                        order,
                        label,
                    )
                    # this doesn't make it into database now, because it's updating
                    # a nested dict and only top one triggers a save
                    # so potentially it's obsolete
                    self.data.brackets[str(bracket_trade.order.orderId)] = Bracket(
                        cast(BracketLabel, label),
                        order_key,
                        bracket_kwargs,
                        bracket_trade,
                    )
        except Exception as e:
            log.exception(f"Error while attaching bracket: {e}")

    # def re_attach_brackets(self):
    #     """
    #     Possibly used by :class:`Controller` if it's determined that a bracket
    #     is missing.
    #     """

    #     for orderId, bracket in self.data.brackets.copy().items():
    #         log.info(f"attempt to re-attach bracket {bracket}")
    #         del self.data.brackets[orderId]
    #         self._place_bracket(
    #             bracket.trade.contract,
    #             bracket.order_key,
    #             bracket.label,
    #             bracket.bracket_kwargs,
    #         )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(stop={self.stop}, "
            f"take_profit={self.take_profit})"
        )
