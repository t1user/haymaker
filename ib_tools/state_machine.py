from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from functools import partial
from typing import TYPE_CHECKING, Any, NamedTuple, Optional

import ib_insync as ibi
import numpy as np

from .base import Atom
from .misc import Lock, Signal

if TYPE_CHECKING:
    from .execution_models import AbstractExecModel


log = logging.getLogger(__name__)


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

        2. Is strategy locked?

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

    def __init__(self) -> None:
        super().__init__()
        self.ib.newOrderEvent.connect(self.handleNewOrderEvent)
        self.ib.orderStatusEvent.connect(self.handleOrderStatusEvent)

        self.data: dict[str, AbstractExecModel] = {}
        self._locks: dict[str, Lock] = {}
        self._position_dict: defaultdict[ibi.Contract, list[str]] = defaultdict(list)
        self.orders: dict[int, OrderInfo] = {}
        # keeps open orders for every strategy
        self._orders_by_strategy: defaultdict[str, list[ibi.Order]] = defaultdict(list)

    def onStart(self, data: dict[str, Any], *args: AbstractExecModel) -> None:
        """
        Register all strategies that are in use.
        """

        strategy = data["strategy"]
        exec_model, *_ = args
        self.data[strategy] = exec_model
        log.info(f"Registered execution models {self.data}")

    async def onData(self, data, *args) -> None:
        try:
            strategy = data["strategy"]
            amount = data["amount"]
            target_position = data["target_position"]
            exec_model = data["exec_model"]
            await asyncio.sleep(10)
            self.verify_transaction_integrity(
                strategy, amount, target_position, exec_model
            )
        except KeyError:
            log.exception(
                "Unable to verify transaction integrity", extra={"data": data}
            )

    def verify_transaction_integrity(
        self,
        strategy: str,
        amount: float,
        target_position: Signal,
        exec_model: AbstractExecModel,
    ) -> None:
        """
        Is the postion resulting from transaction the same as was
        required?
        """
        log.debug(
            f"Verifying transaction integrity: "
            f"target_position direction: {target_position}, "
            f"position: {np.sign(exec_model.position)}"
        )
        try:
            assert np.sign(exec_model.position) == target_position
            assert exec_model.position == abs(amount)
        except AssertionError:
            log.critical(f"Wrong position for {strategy}", exc_info=True)

    def position(self, strategy: str) -> float:
        """
        Return information about position held by strategy identified
        by `strategy`.

        If this `strategy` hasn't been created, than there is no position
        for it.  Otherwise :class:`AbstractExecModel` contains info
        about `strategy`'s position.
        """
        # Check not only position but pending position openning orders
        # THIS IS NOT READY !!!! -----> TODO
        exec_model_or_none = self.data.get(strategy)
        assert exec_model_or_none
        if exec_model_or_none.position:
            return exec_model_or_none.position
        elif orders := self._orders_by_strategy[strategy]:
            for order in orders:
                if self.orders[order.orderId].action == "OPEN":
                    return order.totalQuantity * (
                        1 if order.action.upper() == "BUY" else -1
                    )
        return 0.0

    # def verify_positions_and_orders(self):
    #     positions = self.ib.positions()
    #     orders = self.ib.openOrders()

    def locked(self, strategy: str) -> Lock:
        lock_or_none = self._locks.get(strategy)
        if lock_or_none:
            return lock_or_none
        else:
            return 0

    def register_lock(self, strategy: str, trade: ibi.Trade) -> None:
        self._locks[strategy] = 1 if trade.order.action == "BUY" else -1
        log.debug(f"Registered lock: {strategy}: {self._locks[strategy]}")

    def new_position_callbacks(self, strategy: str, trade: ibi.Trade) -> None:
        """Additional methods may be added in sub-class"""
        # When exactly should the lock be registered to prevent double-dip?
        self.register_lock(strategy, trade)

    def register_position(self, strategy: str, trade: ibi.Trade):
        self._position_dict[trade.contract].append(strategy)
        log.debug(
            f"Registered position for {trade.contract}, position: "
            f"{self._position_dict[trade.contract]}"
        )
        self.verify_positions()

    # def total_positions(self, contract: ibi.Contract) -> float:
    #     total = 0.0
    #     log.debug(f"self._position_dict[contract]={self._position_dict[contract]}")
    #     for strategy_key in self._position_dict[contract]:
    #         total += self.data[strategy_key].position
    #     return total

    def total_positions(self) -> defaultdict[int, float]:
        d: defaultdict[int, float] = defaultdict(float)
        for exec_model in self.data.values():
            if exec_model.active_contract:
                d[exec_model.active_contract.conId] += exec_model.position
        log.debug(f"Total positions: {d}")
        return d

    def position_for_contract(self, conId: int) -> float:
        return self.total_positions().get(conId) or 0.0

    def verify_positions(self) -> list[ibi.Contract]:
        """
        Compare positions actually held with broker with position
        records.  Return differences if any and log an error.
        """

        # list of contracts where differences occur
        difference: list[ibi.Contract] = []

        broker_positions = self.ib.positions()
        for position in broker_positions:
            if position.position != self.position_for_contract(position.contract.conId):
                log.error(
                    f"Position discrepancy for {position.contract.conId}, "
                    f"mine: {self.position_for_contract(position.contract.conId)}, "
                    f"theirs: {position.position}"
                )
                difference.append(position.contract)
            else:
                log.info(f"Position for {position.contract} checks out.")
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
        self._orders_by_strategy[strategy].append(trade.order)

        if action.upper() == "OPEN":
            trade.filledEvent += partial(self.new_position_callbacks, strategy)

        trade.filledEvent += partial(self.register_position, strategy)

    def register_cancel(self, trade, exec_model):
        strategy = self.orders[trade.order.orderId].strategy
        del self.orders[trade.order.orderId]
        self._orders_by_strategy[strategy].remove(trade.order)

    async def handleNewOrderEvent(self, trade: ibi.Trade) -> None:
        """
        Check if the system knows about the order that was just posted
        to the broker.

        This is an event handler (callback).  Connected (subscribed)
        to :meth:`ibi.IB.newOrderEvent` in :meth:`__init__`
        """

        await asyncio.sleep(0.1)
        existing_order_record = self.orders.get(trade.order.orderId)
        if not existing_order_record:
            log.critical(f"Unknown trade in the system {trade.order}")
        else:
            self._orders_by_strategy[existing_order_record.strategy].append(trade.order)

    def handleOrderStatusEvent(self, trade: ibi.Trade) -> None:
        if trade.order.orderId < 0:
            log.error(f"Manual trade: {trade}")
        elif trade.orderStatus.status == ibi.OrderStatus.Inactive:
            messages = ";".join([m.message for m in trade.log])
            log.error(f"Rejected order: {trade.order}, messages: {messages}")
        elif trade.isDone():
            try:
                strategy = self.orders[trade.order.orderId].strategy
                self._orders_by_strategy[strategy].remove(trade.order)
                log.debug(f"Order removed from active list: {trade.order}")
            except KeyError:
                log.error(f"Unknown trade in the system: {trade}")

    def handleErrorEvent(
        self, reqId: int, errorCode: int, errorString: str, contract: ibi.Contract
    ) -> None:
        # reqId is most likely orderId
        # order rejected is errorCode = 201
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"
