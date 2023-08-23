from __future__ import annotations

import asyncio
import logging
from typing import NamedTuple, Optional

import ib_insync as ibi

from .base import Atom
from .execution_models import AbstractExecModel

log = logging.getLogger("__name__")


class OrderInfo(NamedTuple):
    key: str
    reason: str
    trade: ibi.Trade


class StateMachine(Atom):
    """
    This class provides information.  It should be other classes that
    act on this information.

    Answers questions:

        1. Is a strategy in the market?

            - Which portion of the position in a contract is allocated
              to this strategy

        2. Is strategy locked? # NOT SURE ABOUT THIS ONE. MAYBE SIGNAL SHOULD DO IT.

        3. After (re)start:

            - do all positions have stop-losses?

            - has anything happened during blackout that needs to be
              reported in blotter/logs?

            - are all active orders accounted for (i.e. linked to
              strategies)?

            - compare position records with live update (do we hold
              what we think we hold?)

            - attach events (or check if events attached after another
              object attaches them) for:

                - reporting

                - cancellation of redundand orders (tp after stop hit,
                  etc)

        4. Was execution successful (i.e. record desired position
           after signal sent to execution model, then record actual
           live transactions)?

        5. Are holdings linked to strategies?

        6. Are orders linked to strategies?

        7. Make sure order events connected to call-backs
    """

    _instance: Optional["StateMachine"] = None

    _data: dict[str, AbstractExecModel] = {}
    orders: dict[int, OrderInfo] = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            return cls._instance
        else:
            raise TypeError("Attampt to instantiate StateMachine more than once.")

    def __init__(self):
        self.ib.newOrderEvent.connect(self.handleNewOrderEvent)
        self.ib.orderStatusEvent.connect(self.handleOrderStatusEvent)

    def position(self, key: str) -> float:
        """
        Return information about position held by strategy identified
        by `key`.

        If this `key` hasn't been created, than there is no position
        for it.  Otherwise :class:``AbstractExecModel`` contains info
        about `key`'s position.
        """
        exec_model_or_none = self._data.get(key)
        if exec_model_or_none:
            return exec_model_or_none.position
        else:
            return 0.0

    def register_order(self, strategy_key: str, reason: str, trade: ibi.Trade) -> None:
        """
        Register `order` that has just been posted to the broker.
        Verify that position has been registered.

        This method is called by :class:``Trader``.
        """
        self.orders[trade.order.orderId] = OrderInfo(strategy_key, reason, trade)
        ibi.util.sleep(0.5)
        if not self._data.get(strategy_key):
            log.error(f"Unknown trade: {trade}")

    def onData(self, data, *args) -> None:
        """
        Save data sent by :class:``Controller`` about recently sent
        open/close order.
        """
        self._data[data["key"]] = data["exec_model"]

    async def handleNewOrderEvent(self, trade: ibi.Trade) -> None:
        """
        This is an event handler (callback).  Connected (subscribed)
        to :meth:``ibi.IB.newOrderEvent`` in :meth:``__init__``
        """
        await asyncio.sleep(0.1)
        if not self.orders.get(trade.order.orderId):
            log.critical(f"Unknown trade in the system {trade}")

    def handleOrderStatusEvent(self, trade: ibi.Trade) -> None:
        pass
