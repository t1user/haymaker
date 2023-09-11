from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from functools import partial
from typing import TYPE_CHECKING, Any, NamedTuple, Optional

import ib_insync as ibi

from .base import Atom
from .misc import Lock

if TYPE_CHECKING:
    from .execution_models import AbstractExecModel


log = logging.getLogger("__name__")


class OrderInfo(NamedTuple):
    strategy: str
    action: str
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

        self._data: dict[str, AbstractExecModel] = {}
        self._locks: dict[str, Lock] = {}
        self._position_dict: defaultdict[ibi.Contract, list[str]] = defaultdict(list)
        self.orders: dict[int, OrderInfo] = {}

    def onStart(self, data: dict[str, Any], *args: AbstractExecModel) -> None:
        strategy = data["strategy"]
        exec_model, *_ = args
        self._data[strategy] = exec_model

    def onData(self, *args) -> None:
        pass

    def position(self, strategy: str) -> float:
        """
        Return information about position held by strategy identified
        by `strategy`.

        If this `strategy` hasn't been created, than there is no position
        for it.  Otherwise :class:`AbstractExecModel` contains info
        about `strategy`'s position.
        """
        exec_model_or_none = self._data.get(strategy)
        if exec_model_or_none:
            return exec_model_or_none.position
        else:
            return 0.0

    def locked(self, strategy: str) -> Lock:
        lock_or_none = self._locks.get(strategy)
        if lock_or_none:
            return lock_or_none
        else:
            return 0

    def register_lock(self, strategy: str, trade: ibi.Trade) -> None:
        self._locks[strategy] = 1 if trade.order.action == "BUY" else -1

    def new_position_callbacks(self, strategy: str, trade: ibi.Trade) -> None:
        """Additional methods may be added in sub-class"""

        self.register_lock(strategy, trade)

    def register_position(self, strategy: str, trade: ibi.Trade):
        self._position_dict[trade.contract].append(strategy)
        self.verify_positions()

    def total_positions(self, contract: ibi.Contract) -> float:
        total = 0.0
        for strategy_key in self._position_dict[contract]:
            total += self._data[strategy_key].position
        return total

    def verify_positions(self) -> list[ibi.Contract]:
        difference = []
        broker_positions = self.ib.positions()
        for position in broker_positions:
            if position.position != self.total_positions(position.contract):
                log.error(
                    f"Position discrepancy for {position.contract}, "
                    f"mine: {self.total_positions(position.contract)}, "
                    f"theirs: {position.position}"
                )
                difference.append(position.contract)
        return difference

    def register_order(self, strategy: str, action: str, trade: ibi.Trade) -> None:
        """
        Register order, register lock, verify that position has been registered.

        Register order that has just been posted to the broker.  If
        it's an order openning a new position register a lock on this
        strategy (the lock may or may not be used by strategy itself,
        it doesn't matter here, locks are registered for all
        positions).  Verify that position has been registered.

        This method is called by :class:`Controller`.
        """
        self.orders[trade.order.orderId] = OrderInfo(strategy, action, trade)

        if action.upper() == "OPEN":
            trade.filledEvent += partial(self.new_position_callbacks, strategy)

        trade.filledEvent += partial(self.register_position, strategy)

    def register_cancel(self, trade, exec_model):
        del self.order[trade.order.orderId]

    async def handleNewOrderEvent(self, trade: ibi.Trade) -> None:
        """
        Check if the system knows about the order that was just posted
        to the broker.

        This is an event handler (callback).  Connected (subscribed)
        to :meth:`ibi.IB.newOrderEvent` in :meth:`__init__`
        """
        await asyncio.sleep(0.1)
        if not self.orders.get(trade.order.orderId):
            log.critical(f"Unknown trade in the system {trade}")

    def handleOrderStatusEvent(self, trade: ibi.Trade) -> None:
        if trade.orderStatus.status == ibi.OrderStatus.Inactive:
            messages = ";".join([m.message for m in trade.log])
            log.error(f"Rejected order: {trade.order}, messages: {messages}")

    def handleErrorEvent(
        self, reqId: int, errorCode: int, errorString: str, contract: ibi.Contract
    ) -> None:
        # reqId is most likely orderId
        # order rejected is errorCode = 201
        pass
