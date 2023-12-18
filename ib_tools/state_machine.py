from __future__ import annotations

import asyncio
import logging
from collections import UserDict, defaultdict
from dataclasses import dataclass, fields
from functools import partial
from typing import TYPE_CHECKING, Any, Iterator, Optional

import ib_insync as ibi

from ib_tools.saver import MongoSaver, SaveManager

from .base import Atom
from .misc import Lock, Signal, decode_tree, sign, tree

if TYPE_CHECKING:
    from .execution_models import AbstractExecModel


log = logging.getLogger(__name__)


@dataclass
class OrderInfo:
    strategy: str
    action: str
    trade: ibi.Trade
    params: dict

    @property
    def active(self):
        return not self.trade.isDone()

    def __iter__(self) -> Iterator[Any]:
        for f in (*(x.name for x in fields(self)), "active"):
            yield getattr(self, f)

    def encode(self) -> dict[str, Any]:
        return {
            "orderId": self.trade.order.orderId,
            **{k: tree(v) for k, v in self.__dict__.items()},
            "active": self.active,
        }

    def decode(self, data: dict[str, Any]) -> None:
        data_ = decode_tree(data)
        data_.pop("actve")
        data_.pop("orderId")
        self.__dict__.update(**data_)


class OrderContainer(UserDict):
    def __init__(self, dict=None, /, max_size: int = 10) -> None:
        self.max_size = max_size
        super().__init__(dict)

    def __setitem__(self, key, item):
        if self.max_size and (len(self.data) >= self.max_size):
            self.data.pop(min(self.data.keys()))
        super().__setitem__(key, item)

    def strategy(self, strategy: str) -> Iterator[OrderInfo]:
        """Get active orders associated with strategy."""
        # returns only active orders
        return (oi for oi in self.values() if (oi.strategy == strategy and oi.active))

    def get(self, key, default=None, active_only=False):
        """Get records for an order."""
        value = self.data.get(key, None)
        if value:
            if (not active_only) or value.active:
                return value
        return default


class ExecModelContainer(dict):
    def encode(self) -> dict[str, Any]:
        return tree({k: v.encode() for k, v in self.items()})


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

        7. /No longer valid/ Make sure order events connected to call-backs

    Collect all information needed to restore state.  This info will
    be stored to database.  If the app crashes, after restart, all
    info about state required by any object must be found here.
    """

    _instance: Optional["StateMachine"] = None

    # TODO: determined by CONFIG
    save_order = SaveManager(MongoSaver("orders", query_key="orderId"))
    save_model = SaveManager(MongoSaver("models"))

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            return cls._instance
        else:
            raise TypeError("Attempt to instantiate StateMachine more than once.")

    def __init__(self) -> None:
        super().__init__()

        self.data = ExecModelContainer()  # dict of ExecModel
        self.orders = OrderContainer()  # dict of OrderInfo

    def update_trades(self, *trades: ibi.Trade) -> list[ibi.Trade]:
        """
        Update trade object for a given order.  Used after re-start to
        bring records up to date with IB
        """
        error_trades = []
        for trade in trades:
            oi = self.orders.get(trade.order.orderId)
            if oi:
                log.debug(f"Trade will be updated: {oi}")
                new_oi = OrderInfo(oi.strategy, oi.action, trade, oi.params)
                self.orders[trade.order.orderId] = new_oi
                self.save_order(new_oi.encode())
                # TODO: What about trades that became incactive while disconnected?
            else:
                log.debug(f"Error trade: {trade}")
                error_trades.append(trade)
        return error_trades

    def review_trades(self, *trades: ibi.Trade) -> list[ibi.Trade]:
        """
        On restart review all trades on record and compare their
        status with IB.

        Args:
        -----

        *trades - list of open trades from :meth:`ibi.openTrades()`.

        Returns:
        --------

        List of trades that we have as open, while IB has them as
        done.  We have to reconcile those trades' status and report
        them as appropriate.
        """
        unresolved_trades = []
        open_trades = {trade.order.orderId: trade for trade in trades}
        for orderId, oi in self.orders.copy().items():
            if orderId not in open_trades:
                # if inactive it's already been dealt with before restart
                if oi.active:
                    # this is a trade that we have as active in self.orders
                    # but IB doesn't have it in open orders
                    # we have to figure out what happened to this trade
                    # while we were disconnected and report it as appropriate
                    unresolved_trades.append(oi)
                else:
                    # this order is no longer of interest
                    # it's inactive in our orders and inactive in IB
                    self.orders.pop(orderId)

        return unresolved_trades

    def restore_from_store(self):
        # self.data
        # self.orders
        pass

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
        super().onData(data)
        try:
            strategy = data["strategy"]
            amount = data["amount"]
            target_position = data["target_position"]
            # exec_model = data["exec_model"]
            await asyncio.sleep(15)
            self.verify_transaction_integrity(strategy, amount, target_position)
        except KeyError:
            log.exception(
                "Unable to verify transaction integrity", extra={"data": data}
            )

    def verify_transaction_integrity(
        self,
        strategy: str,
        amount: float,
        target_position: Signal,
    ) -> None:
        """
        Is the postion resulting from transaction the same as was
        required?
        """
        exec_model = self.data.get(strategy)
        if exec_model:
            log.debug(
                f"Transaction OK? ->{sign(exec_model.position) == target_position}<- "
                f"target_position: {target_position}, "
                f"position: {sign(exec_model.position)}"
                f"->> {exec_model.strategy}"
            )
            try:
                assert sign(exec_model.position) == target_position
                # Investigate why this may be necessary:
                # assert exec_model.position == abs(amount)
            except AssertionError:
                log.critical(f"Wrong position for {strategy}", exc_info=True)
        else:
            log.critical(f"Attempt to trade for unknow strategy: {strategy}")

    def register_order(
        self,
        strategy: str,
        action: str,
        trade: ibi.Trade,
        exec_model: AbstractExecModel,
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
        params = exec_model.params.get(action.lower(), {})
        order_info = OrderInfo(strategy, action, trade, params)
        self.orders[trade.order.orderId] = order_info
        log.debug(
            f"{trade.order.orderType} orderId: {trade.order.orderId} registered for: "
            f"{trade.contract.localSymbol}"
        )

        if action.upper() == "OPEN":
            trade.filledEvent += partial(
                self.new_position_callbacks, exec_model.strategy
            )

        self.save_order(order_info.encode())
        self.save_model(self.data.encode())

    def report_done_order(self, trade: ibi.Trade) -> Optional[OrderInfo]:
        order_info = self.orders.get(trade.order.orderId)
        if order_info:
            dict_for_saving = order_info.encode()
        else:
            dict_for_saving = {
                "orderId": trade.order.orderId,
                "strategy": "unknown",
                "action": "unknown",
                "trade": trade,
                "params": {},
            }
        log.debug("Reporting done order")
        self.save_order(dict_for_saving)
        self.save_model(self.data.encode())
        return order_info

    def report_new_order(self, trade: ibi.Trade) -> None:
        log.debug("Reporting new order.")
        self.save_model(self.data.encode())

    def get_order(self, orderId: int) -> Optional[OrderInfo]:
        return self.orders.get(orderId)

    def delete_order(self, orderId: int) -> None:
        del self.orders[orderId]

    def get_strategy(self, strategy: str) -> Optional[AbstractExecModel]:
        return self.data.get(strategy)

    def position(self, strategy: str) -> float:
        """
        Return position for strategy.  Orders openning strategy are
        treated as if they were already open.
        """
        # Verify that broker's position records are the same as mine

        exec_model = self.data.get(strategy)
        assert exec_model

        if exec_model.position:
            return exec_model.position

        # Check not only position but pending position openning orders
        elif orders := self.orders.strategy(strategy):
            # log.debug(
            #     f"Orders for {strategy}: {list(map(lambda x: x.trade.order, orders))}"
            # )
            for order_info in orders:
                trade = order_info.trade
                log.debug(
                    f"Order action: {order_info.action} is active: {trade.isActive()} "
                    f"amount: {trade.order.totalQuantity}, "
                    f"direction {trade.order.action.upper()}"
                )
                if order_info.action == "OPEN":  # and trade.isActive():
                    return trade.order.totalQuantity * (
                        1.0 if trade.order.action.upper() == "BUY" else -1.0
                    )
        return 0.0

    def verify_position_for_contract(self, contract):
        my_position = self.position_for_contract[contract]
        ib_position = self.ib_position_for_contract[contract]

    def locked(self, strategy: str) -> Lock:
        return self.data[strategy].lock

    def position_for_contract(self, contract: ibi.Contract) -> float:
        return self.total_positions().get(contract) or 0.0

    def ib_position_for_contract(self, contract: ibi.Contract) -> float:
        return next(
            (v.position for v in self.ib.positions() if v.contract == contract), 0
        )

        # positions = {p.contract: p.position for p in self.ib.positions()}
        # return positions.get(contract, 0.0)

    def total_positions(self) -> defaultdict[ibi.Contract, float]:
        d: defaultdict[ibi.Contract, float] = defaultdict(float)
        for exec_model in self.data.values():
            if exec_model.active_contract:
                d[exec_model.active_contract] += exec_model.position
        log.debug(f"Total positions: {d}")
        return d

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

    # ### TODO: Re-do this
    def register_lock(self, strategy: str, trade: ibi.Trade) -> None:
        self.data[strategy].lock = 1 if trade.order.action == "BUY" else -1
        log.debug(f"Registered lock: {strategy}: {self.data[strategy]}")

    def new_position_callbacks(self, strategy: str, trade: ibi.Trade) -> None:
        # When exactly should the lock be registered to prevent double-dip?
        self.register_lock(strategy, trade)

    # ###

    # def register_cancel(self, trade, exec_model):
    #     del self.orders[trade.order.orderId]

    def __repr__(self):
        return self.__class__.__name__ + "()"
