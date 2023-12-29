from __future__ import annotations

import asyncio
import logging
from collections import UserDict, defaultdict
from dataclasses import dataclass, fields
from functools import partial  # , singledispatchmethod
from typing import Any, Iterator, Optional, Union

import ib_insync as ibi

from ib_tools.saver import MongoSaver, SaveManager, async_runner

from .base import Atom
from .misc import Lock, Signal, decode_tree, sign, tree

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
        data.pop("active")
        self.__dict__.update(**data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OrderInfo:
        data.pop("active")
        return cls(**data)


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

    def decode(self, data: dict) -> None:
        try:
            log.debug(f"{self} will decode data: {len(data)} items.")
        except Exception:
            log.debug(f"{self} data for decoding: {data}")

        decoded = decode_tree(data)

        for item in decoded:
            item.pop("_id")
            orderId = item.pop("orderId")
            if existing := self.data.get(orderId):
                existing.decode(item)
            else:
                self.data[orderId] = OrderInfo.from_dict(item)
            log.debug(
                f"Order for : {item['trade'].order.orderId} "
                f"{item['trade'].contract.symbol} decoded."
            )

    def __repr__(self) -> str:
        return f"OrderContainer({self.data}, max_size={self.max_size})"


class Model(dict):
    def __getattr__(self, key):
        return self.get(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        self.__delitem__(key)


class ModelContainer(UserDict):
    def __setitem__(self, key: str, item: dict):
        if not isinstance(item, dict):
            raise ValueError(
                f"{self.__class__.__qualname__} can be used to store only dicts."
            )
        self.data[key] = Model(item)

    def __missing__(self, key):
        log.debug(f"Will reset data for {key}")
        self.data[key] = Model(
            {
                "active_contract": None,
                "position": 0.0,
                "params": {},
                "lock": 0,
                "position_id": "",
            }
        )
        return self.data[key]

    # @singledispatchmethod
    # def position(self, key: str) -> float:
    #     if model := self.get(key):
    #         return model.position
    #     else:
    #         return 0.0

    # @position.register
    # def _(self, key: ibi.Contract):
    #     c: defaultdict[ibi.Contract, float] = defaultdict(float)
    #     for model in self.data.values():
    #         if model.get("active_contract"):
    #             c[model["active_contract"]] += model["position"]
    #     return c[key]

    def encode(self) -> dict[str, Any]:
        return tree(self.data)

    def decode(self, data: dict) -> None:
        try:
            log.debug(f"{self} will decode data: {len(data.keys())-1} keys.")
        except Exception:
            log.debug(f"data for decoding: {data}")

        decoded = decode_tree(data)

        try:
            decoded.pop("_id")
        except KeyError:
            log.warning("It's weird, no '_id' in data from mongo.")

        log.debug(f"Decoded keys: {list(decoded.keys())}")
        self.data.update(**{k: Model(v) for k, v in decoded.items()})

    def __repr__(self) -> str:
        return f"ModelContainer({self.data})"


# Consider allowing for use of different savers
model_saver = MongoSaver("models")
order_saver = MongoSaver("orders", query_key="orderId")


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

    save_order = SaveManager(order_saver)
    save_model = SaveManager(model_saver)

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            return cls._instance
        else:
            raise TypeError("Attempt to instantiate StateMachine more than once.")

    def __init__(self) -> None:
        super().__init__()

        self.data = ModelContainer()  # dict of ExecModel data
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
                log.debug(
                    f"Trade will be updated - id: {oi.trade.order.orderId} "
                    f"permId: {oi.trade.order.permId}"
                )
                new_oi = OrderInfo(oi.strategy, oi.action, trade, oi.params)
                self.orders[trade.order.orderId] = new_oi
                self.save_order(new_oi.encode())
                # TODO: What about trades that became inactive while disconnected?
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

        *trades - list of open trades from :meth:`ibi.IB.openTrades()`.

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
                    unresolved_trades.append(oi.trade)
                else:
                    # this order is no longer of interest
                    # it's inactive in our orders and inactive in IB
                    self.orders.pop(orderId)
        return unresolved_trades

    def override_inactive_trades(self, *trades: ibi.Trade) -> None:
        """
        Trades that we have as active but IB doesn't know about them.
        Used for cold restarts.
        """
        for trade in trades:
            log.debug(f"Will delete trade: {trade.order.orderId}")
            oi_dict = self.orders[trade.order.orderId].encode()
            oi_dict.update({"active": False})
            self.save_order(oi_dict)
            del self.orders[trade.order.orderId]

    async def read_from_store(self):
        log.debug("Will read data from store...")
        model_data = await async_runner(model_saver.read_latest)
        order_data = await async_runner(order_saver.read, {"active": True})
        self.data.decode(model_data)
        self.orders.decode(order_data)

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
        data = self.data.get(strategy)
        if data:
            log.debug(
                f"Transaction OK? ->{sign(data.position) == target_position}<- "
                f"target_position: {target_position}, "
                f"position: {sign(data.position)}"
                f"->> {strategy}"
            )
            try:
                assert sign(data.position) == target_position
                # Investigate why this may be necessary:
                # assert exec_model.position == abs(amount)
            except AssertionError:
                log.critical(f"Wrong position for {strategy}", exc_info=True)
        else:
            log.critical(f"Attempt to trade for unknow strategy: {strategy}")

    def register_order(
        self, strategy: str, action: str, trade: ibi.Trade, data: Model
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
        params = data["params"].get(action.lower(), {})
        order_info = OrderInfo(strategy, action, trade, params)
        self.orders[trade.order.orderId] = order_info
        log.debug(
            f"{trade.order.orderType} orderId: {trade.order.orderId} "
            f"permId: {trade.order.permId} registered for: "
            f"{trade.contract.localSymbol}"
        )

        if action.upper() == "OPEN":
            trade.filledEvent += partial(self.new_position_callbacks, strategy)

        self.save_order(order_info.encode())
        self.save_model(self.data.encode())

    async def report_done_order(self, trade: ibi.Trade) -> Optional[OrderInfo]:
        log.debug(f"reporting trade: {trade.order.orderId} {trade.order.permId}")
        order_info = self.orders.get(trade.order.orderId)
        log.debug(
            f"existing record: {order_info.trade.order.orderId} "
            f"{order_info.trade.order.permId}"
        )
        if not trade.order.permId:
            log.debug("Will wait for permId for 5 secs...")
            await asyncio.sleep(5)
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
        self.save_model(self.data.encode())
        self.save_order(dict_for_saving)
        return order_info

    def report_new_order(self, trade: ibi.Trade) -> None:
        # log.debug("Reporting new order.")
        # self.save_model(self.data.encode())
        pass

    def get_order(self, orderId: int) -> Optional[OrderInfo]:
        return self.orders.get(orderId)

    def delete_order(self, orderId: int) -> None:
        del self.orders[orderId]

    def get_strategy(self, strategy: str) -> Optional[Model]:
        return self.data.get(strategy)

    def position(self, strategy: str) -> float:
        """
        Return position for strategy.  Orders openning strategy are
        treated as if they were already open.
        """
        # Verify that broker's position records are the same as mine
        data = self.data.get(strategy)
        assert data

        if data.position:
            return data.position

        # Check not only position but pending position openning orders
        elif orders := self.orders.strategy(strategy):
            for order_info in orders:
                trade = order_info.trade
                if order_info.action == "OPEN":
                    return trade.order.totalQuantity * (
                        1.0 if trade.order.action.upper() == "BUY" else -1.0
                    )
        return 0.0

    def verify_position_for_contract(self, contract) -> Union[bool, float]:
        # NOT IN USE
        my_position = self.position_for_contract(contract)
        ib_position = self.ib_position_for_contract(contract)
        return (my_position == ib_position) or (my_position - ib_position)

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
        for data in self.data.values():
            if data.active_contract:
                d[data.active_contract] += data.position
        log.debug(f"Total positions: {d}")
        return d

    def verify_positions(self) -> dict[ibi.Contract, float]:
        """
        Compare positions actually held with broker with position
        records.  Return differences if any and log an error.
        """

        # list of contracts where differences occur
        # difference: list[ibi.Contract] = []

        broker_positions_dict = {i.contract: i.position for i in self.ib.positions()}
        my_positions_dict = self.total_positions()
        log.debug(f"Broker positions: {broker_positions_dict}")
        log.debug(f"My positions: {my_positions_dict}")
        diff = {
            i: (
                (my_positions_dict.get(i) or 0.0)
                - (broker_positions_dict.get(i) or 0.0)
            )
            for i in set([*broker_positions_dict.keys(), *my_positions_dict.keys()])
        }
        return {k: v for k, v in diff.items() if v != 0}
        # for position in broker_positions:
        #     if position.position != self.position_for_contract(position.contract):
        #         log.error(
        #             f"Position discrepancy for {position.contract}, "
        #             f"mine: {self.position_for_contract(position.contract)}, "
        #             f"theirs: {position.position}"
        #         )
        #         difference.append(position.contract)

        # return difference

    # ### TODO: Re-do this
    def register_lock(self, strategy: str, trade: ibi.Trade) -> None:
        self.data[strategy].lock = 1 if trade.order.action == "BUY" else -1
        log.debug(f"Registered lock: {strategy}: {self.data[strategy]}")

    def new_position_callbacks(self, strategy: str, trade: ibi.Trade) -> None:
        # When exactly should the lock be registered to prevent double-dip?
        self.register_lock(strategy, trade)

    def __repr__(self):
        return self.__class__.__name__ + "()"
