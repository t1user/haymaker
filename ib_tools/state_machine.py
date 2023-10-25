from __future__ import annotations

import asyncio
import logging
from collections import UserDict, defaultdict
from functools import partial
from typing import TYPE_CHECKING, Any, Iterator, NamedTuple, Optional
from weakref import ref

import ib_insync as ibi

from .base import Atom
from .misc import Callback, Lock, Signal, sign

if TYPE_CHECKING:
    from .execution_models import AbstractExecModel


log = logging.getLogger(__name__)


class OrderInfo(NamedTuple):
    strategy: str
    action: str
    trade: ibi.Trade
    callback: Optional[Callback]


class OrderContainer(UserDict):
    def strategy(self, strategy: str) -> Iterator[OrderInfo]:
        return (oi for oi in self.data.values() if oi.strategy == strategy)


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

        self.data: dict[str, AbstractExecModel] = {}
        self._locks: dict[str, Lock] = {}
        # self._position_dict: defaultdict[ibi.Contract, list[str]] = defaultdict(list)
        self.orders = OrderContainer()

    def onStart(self, data: dict[str, Any], *args: AbstractExecModel) -> None:
        """
        Register all strategies that are in use.
        """

        strategy = data["strategy"]
        exec_model, *_ = args
        self.data[strategy] = exec_model
        log.log(5, f"Registered execution models {list(self.data.keys())}")

    async def onData(self, data, *args) -> None:
        """
        After obtaining transaction details from execution model,
        verify if the intended effect is the same as achieved effect.
        """
        try:
            strategy = data["strategy"]
            amount = data["amount"]
            target_position = data["target_position"]
            exec_model = data["exec_model"]
            await asyncio.sleep(15)
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
            f"Transaction OK? ->{sign(exec_model.position) == target_position}<- "
            f"target_position: {target_position}, position: {sign(exec_model.position)}"
            f"->> {exec_model.strategy}"
        )
        try:
            assert sign(exec_model.position) == target_position
            # Investigate why this may be necessary:
            # assert exec_model.position == abs(amount)
        except AssertionError:
            log.critical(f"Wrong position for {strategy}", exc_info=True)

    def register_order(
        self,
        strategy: str,
        action: str,
        trade: ibi.Trade,
        callback: Optional[Callback] = None,
    ) -> None:
        """
        Register order, register lock, verify that position has been registered.

        Register order that has just been posted to the broker.  If
        it's an order openning a new position register a lock on this
        strategy (the lock may or may not be used by strategy itself,
        it doesn't matter here, locks are registered for all
        positions).  Verify that position has been registered.

        This method is called by :class:`Controller`.
        """

        self.orders[trade.order.orderId] = OrderInfo(strategy, action, trade, callback)

        if action.upper() == "OPEN":
            trade.filledEvent += partial(self.new_position_callbacks, strategy)

    def delete_order(self, orderId: int) -> None:
        del self.orders[orderId]

    def get_order(self, orderId: int) -> Optional[OrderInfo]:
        return self.orders.get(orderId)

    def position(self, strategy: str) -> float:
        """
        Return position for strategy.  Orders openning strategy are
        treated as if they were already open.
        """
        # Check not only position but pending position openning orders
        exec_model = self.data.get(strategy)
        assert exec_model
        if exec_model.position:
            return exec_model.position
        elif orders := self.orders.strategy(strategy):
            log.debug(
                f"Orders for {strategy}: {list(map(lambda x: x.trade.order, orders))}"
            )
            for order_info in orders:
                trade = order_info.trade
                if order_info.action == "OPEN" and trade.isActive():
                    return trade.order.totalQuantity * (
                        1.0 if trade.order.action.upper() == "BUY" else -1.0
                    )
        return 0.0

    def locked(self, strategy: str) -> Lock:
        return self._locks.get(strategy, 0)

    # ### TODO: Re-do this
    def register_lock(self, strategy: str, trade: ibi.Trade) -> None:
        self._locks[strategy] = 1 if trade.order.action == "BUY" else -1
        log.debug(f"Registered lock: {strategy}: {self._locks[strategy]}")

    def new_position_callbacks(self, strategy: str, trade: ibi.Trade) -> None:
        # When exactly should the lock be registered to prevent double-dip?
        self.register_lock(strategy, trade)

    # ###

    def total_positions(self) -> defaultdict[ibi.Contract, float]:
        d: defaultdict[ibi.Contract, float] = defaultdict(float)
        for exec_model in self.data.values():
            if exec_model.active_contract:
                d[exec_model.active_contract] += exec_model.position
        log.debug(f"Total positions: {d}")
        return d

    def position_for_contract(self, contract: ibi.Contract) -> float:
        return self.total_positions().get(contract) or 0.0

    def verify_positions(self) -> list[ibi.Contract]:
        """
        NOT IN USE
        Compare positions actually held with broker with position
        records.  Return differences if any and log an error.
        """

        # list of contracts where differences occur
        difference: list[ibi.Contract] = []

        broker_positions = self.ib.positions()
        for position in broker_positions:
            if position.position != self.position_for_contract(position.contract):
                log.error(
                    f"Position discrepancy for {position.contract}, "
                    f"mine: {self.position_for_contract(position.contract)}, "
                    f"theirs: {position.position}"
                )
                difference.append(position.contract)

        return difference

    def register_cancel(self, trade, exec_model):
        del self.orders[trade.order.orderId]

    def __repr__(self):
        return self.__class__.__name__ + "()"
