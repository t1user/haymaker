from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import Final, NamedTuple, Optional

import ib_insync as ibi
import numpy as np

from .base import Atom
from .execution_models import AbstractExecModel
from .misc import Lock

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

        2. Is strategy locked?  # NOT SURE ABOUT THIS ONE.  MAYBE
           SIGNAL SHOULD DO IT.

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

    Collect all information needed to restore state.  This info will
    be stored to database.  If the app crashes, after restart, all
    info about state required by any object must be found here.
    """

    _instance: Optional["StateMachine"] = None

    _data: dict[str, AbstractExecModel] = {}
    _locks: dict[str, Lock] = {}
    orders: dict[int, OrderInfo] = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            return cls._instance
        else:
            raise TypeError("Attampt to instantiate StateMachine more than once.")

    def __init__(self):
        super().__init__()
        self.ib.newOrderEvent.connect(self.handleNewOrderEvent)
        self.ib.orderStatusEvent.connect(self.handleOrderStatusEvent)

    def position(self, key: str) -> float:
        """
        Return information about position held by strategy identified
        by `key`.

        If this `key` hasn't been created, than there is no position
        for it.  Otherwise :class:`AbstractExecModel` contains info
        about `key`'s position.
        """
        exec_model_or_none = self._data.get(key)
        if exec_model_or_none:
            return exec_model_or_none.position
        else:
            return 0.0

    def locked(self, key: str) -> Lock:
        lock_or_none = self._locks.get(key)
        if lock_or_none:
            return lock_or_none
        else:
            return 0

    def register_lock(self, strategy_key: str, trade: ibi.Trade) -> None:
        self._locks[strategy_key] = np.sign(trade.filled())

    def new_position_callbacks(self, strategy_key: str, trade: ibi.Trade) -> None:
        """Additional methods may be added in sub-class"""

        self.register_lock(strategy_key, trade)

    def register_order(self, strategy_key: str, reason: str, trade: ibi.Trade) -> None:
        """
        Register order, register lock, verify that position has been registered.

        Register order that has just been posted to the broker.  If
        it's an order openning a new position register a lock on this
        strategy (the lock may or may not be used by strategy itself,
        it doesn't matter here, locks are registered for all
        positions).  Verify that position has been registered.

        This method is called by :class:`Trader`.
        """
        self.orders[trade.order.orderId] = OrderInfo(strategy_key, reason, trade)

        if reason.upper() in ("OPEN", "REVERSE"):
            trade.filledEvent += partial(self.new_position_callbacks, strategy_key)

        # What exactly is the purpose of this? Check if python dictionaries work???
        ibi.util.sleep(0.5)
        if not self._data.get(strategy_key):
            log.error(f"Unknown trade: {trade}")

    def onData(self, data, *args) -> None:
        """
        Save data sent by :class:`Controller` about recently sent
        open/close order.
        """
        self._data[data["key"]] = data["exec_model"]

    async def handleNewOrderEvent(self, trade: ibi.Trade) -> None:
        """
        Check if the system knows about the order that was just posted
        to the broker.

        This is an event handler (callback).  Connected (subscribed)
        to :meth:`ibi.IB.newOrderEvent` in
        :meth:`State_Machine.__init__`
        """
        await asyncio.sleep(0.1)
        if not self.orders.get(trade.order.orderId):
            log.critical(f"Unknown trade in the system {trade}")

    def handleOrderStatusEvent(self, trade: ibi.Trade) -> None:
        pass


STATE_MACHINE: Final[StateMachine] = StateMachine()
