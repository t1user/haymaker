from __future__ import annotations

import asyncio
import logging
from collections import UserDict, defaultdict
from dataclasses import dataclass, fields
from functools import partial
from typing import Any, Iterator, Optional, Union

import eventkit as ev  # type: ignore
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

    @property
    def permId(self):
        return self.trade.order.permId

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
    def __init__(self, dict=None, /, max_size: int = 50) -> None:
        self.max_size = max_size
        self.index: dict[int, int] = {}
        self.setitemEvent = ev.Event("setitemEvent")
        self.setitemEvent += self.onSetitemEvent
        super().__init__(dict)

    def __setitem__(self, key: int, item: OrderInfo) -> None:
        if self.max_size and (len(self.data) >= self.max_size):
            self.data.pop(min(self.data.keys()))
        self.setitemEvent.emit(key, item)
        super().__setitem__(key, item)

    def __getitem__(self, key):
        if key in self.data:
            return self.data[key]
        elif key in self.index:
            if self.index[key] in self.data:
                return self.data[self.index[key]]
        if hasattr(self.__class__, "__missing__"):
            return self.__class__.__missing__(self, key)
        raise KeyError(key)

    def __delitem__(self, key):
        if self.index.get(self.data[key].permId):
            del self.index[self.data[key].permId]
        super().__delitem__(key)

    async def onSetitemEvent(self, key: int, item: OrderInfo) -> None:
        while not item.permId:
            await asyncio.sleep(0.1)
        self.index[item.permId] = key

    def strategy(self, strategy: str) -> Iterator[OrderInfo]:
        """Get active orders associated with strategy."""
        # returns only active orders
        return (oi for oi in self.values() if (oi.strategy == strategy and oi.active))

    def get(self, key, default=None, active_only=False):
        """Get records for an order."""
        try:
            value = self[key]
        except KeyError:
            return default
        # value = self.data.get(key, None)
        # if value:
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


class Strategy(dict):
    def __getattr__(self, key):
        return self.get(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        self.__delitem__(key)


class StrategyContainer(UserDict):
    def __setitem__(self, key: str, item: dict):
        if not isinstance(item, dict):
            raise ValueError(
                f"{self.__class__.__qualname__} can be used to store only dicts."
            )
        self.data[key] = Strategy(item)

    def __missing__(self, key):
        log.debug(f"Will reset data for {key}")
        self.data[key] = Strategy(
            {
                "active_contract": None,
                "position": 0.0,
                "params": {},
                "lock": 0,
                "position_id": "",
            }
        )
        return self.data[key]

    def total_positions(self) -> defaultdict[ibi.Contract, float]:
        d: defaultdict[ibi.Contract, float] = defaultdict(float)
        for data in self.data.values():
            if data.active_contract:
                d[data.active_contract] += data.position
        return d

    def strategies_by_contract(self) -> defaultdict[ibi.Contract, list[str]]:
        d: defaultdict[ibi.Contract, list[str]] = defaultdict(list)
        for data in self.data.values():
            if data.active_contract:
                d[data.active_contract].append(data.strategy)
        return d

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
        self.data.update(**{k: Strategy(v) for k, v in decoded.items()})

    def __repr__(self) -> str:
        return f"StrategyContainer({self.data})"


@dataclass
class StrategyWrapper:
    # DOESN'T SEEM TO BE IN USE AND CAN'T REMEMBER WHAT IT WAS FOR
    _strategy: Strategy
    _orders: OrderContainer

    @property
    def orders(self):
        return self._orders.strategy(self._strategy)

    def __getitem__(self, item):
        return self._strategy[item]

    def __setitem__(self, key, value):
        self._strategy[key] = value

    def __getattr__(self, key):
        return self._strategy.get(key)

    def __setattr__(self, key, value):
        self._strategy[key] = value

    def __delattr__(self, key):
        self._strategy.__delitem__(key)


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

        self._data = StrategyContainer()  # dict of ExecModel data
        self._orders = OrderContainer()  # dict of OrderInfo

    def update_trade(self, trade: ibi.Trade) -> Optional[ibi.Trade]:
        """
        Update trade object for a given order.  Used after re-start to
        bring records up to date with IB
        """
        oi = self._orders.get(trade.order.orderId)
        if oi:
            log.debug(
                f"Trade will be updated - id: {oi.trade.order.orderId} "
                f"permId: {oi.trade.order.permId}"
            )
            new_oi = OrderInfo(oi.strategy, oi.action, trade, oi.params)
            self._orders[trade.order.orderId] = new_oi
            self.save_order(new_oi.encode())
            # TODO: What about trades that became inactive while disconnected?
            return None
        else:
            return trade

    def prune_order(self, orderId) -> None:
        """
        This order is no longer of interest it's inactive in our
        orders and inactive in IB.  Called by startup sync routines.
        """
        self._orders.pop(orderId)

    def override_inactive_trades(self, *trades: ibi.Trade) -> None:
        """
        Trades that we have as active but IB doesn't know about them.
        Used for cold restarts.
        """
        for trade in trades:
            log.debug(f"Will delete trade: {trade.order.orderId}")
            oi_dict = self._orders[trade.order.orderId].encode()
            oi_dict.update({"active": False})
            self.save_order(oi_dict)
            del self._orders[trade.order.orderId]

    async def read_from_store(self):
        log.debug("Will read data from store...")
        strategy_data = await async_runner(model_saver.read_latest)
        order_data = await async_runner(order_saver.read, {"active": True})
        self._data.decode(strategy_data)
        self._orders.decode(order_data)

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
        data = self._data.get(strategy)
        if data:
            log.debug(
                f"Transaction OK? ->{sign(data.position) == target_position}<- "
                f"target_position: {target_position}, "
                f"position: {sign(data.position)}"
                f"->> {strategy}"
            )
            log.debug(
                f"{data.active_contract.symbol}: "
                f"{self.verify_position_for_contract(data.active_contract)}"
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
        self, strategy: str, action: str, trade: ibi.Trade, data: Strategy
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
        self._orders[trade.order.orderId] = order_info
        log.debug(
            f"{trade.order.orderType} orderId: {trade.order.orderId} "
            f"permId: {trade.order.permId} registered for: "
            f"{trade.contract.localSymbol}"
        )

        if action.upper() == "OPEN":
            trade.filledEvent += partial(self.new_position_callbacks, strategy)

        self.save_order(order_info.encode())
        self.save_model(self._data.encode())

        # ### following is for debugging, should be deleted ####
        async def _nothing(*args):
            n = 0
            while not trade.order.permId:
                n += 1
                await asyncio.sleep(0.1)
            log.debug(
                f"register_order acquired permId after: {n*.1} seconds."
                f"{trade.order.orderId} {trade.order.permId}"
            )

        testing_event = ev.Event("testingEvent")
        testing_event += _nothing

        if not trade.order.permId:
            testing_event.emit()
        else:
            log.debug(
                f"register_order got permId without delay: "
                f"{trade.order.orderId} {trade.order.permId}"
            )

    async def save_order_status(self, trade: ibi.Trade) -> Optional[OrderInfo]:
        log.debug(f"updating trade status: {trade.order.orderId} {trade.order.permId}")
        order_info = self._orders.get(trade.order.orderId)
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
                "action": "MANUAL" if trade.order.orderId < 0 else "UNKNOWN",
                "trade": trade,
                "params": {},
            }
        log.debug(f"Dict for saving: {dict_for_saving}")
        try:
            self.save_model(self._data.encode())
            self.save_order(dict_for_saving)
        except Exception as e:
            log.exception(e)
        return order_info

    def report_new_order(self, trade: ibi.Trade) -> None:
        log.debug(f"Reporting new order: {trade.order.orderId}, {trade.order.permId}.")
        # self.save_model(self._data.encode())
        pass

    # ### These are data access and modification methods ###

    @property
    def strategy(self):
        """
        Access strategies by ``state_machine.strategy['strategy_name']``
        or ``state_machine.strategy.get('strategy_name')``
        """
        return self._data

    @property
    def order(self):
        """
        Access order by ``state_machine.order[orderId]``
        or ``state_machine.order.get(orderId)``
        """
        return self._orders

    @property
    def position(self) -> defaultdict[ibi.Contract, float]:
        """
        Access positions for given contract: ``state_machine.position[contract]``
        or ``state_machine.position.get(contract)``
        """
        return self._data.total_positions()

    @property
    def for_contract(self) -> defaultdict[ibi.Contract, list[str]]:
        """
        Access strategies for given contract:
        ``state_machine.for_contract[contract]`` or
        ``state_machine.for_contract.get(contract)``
        """
        return self._data.strategies_by_contract()

    # def get_order(self, orderId: int) -> Optional[OrderInfo]:
    #     return self._orders.get(orderId)

    def delete_order(self, orderId: int) -> None:
        del self._orders[orderId]

    # def get_strategy(self, strategy: str) -> Optional[Strategy]:
    #     return self._data.get(strategy)

    def locked(self, strategy: str) -> Lock:
        return self._data[strategy].lock

    # ### End data access and modification methods ###

    # def position(self, strategy: str) -> float:
    #     """
    #     PROBABLY NOT IN USE

    #     Return position for strategy.  Orders openning strategy are
    #     treated as if they were already open.
    #     """
    #     # Verify that broker's position records are the same as mine
    #     data = self._data.get(strategy)
    #     assert data

    #     if data.position:
    #         return data.position

    #     # Check not only position but pending position openning orders
    #     elif orders := self._orders.strategy(strategy):
    #         for order_info in orders:
    #             trade = order_info.trade
    #             if order_info.action == "OPEN":
    #                 return trade.order.totalQuantity * (
    #                     1.0 if trade.order.action.upper() == "BUY" else -1.0
    #                 )
    #     return 0.0

    def verify_position_for_contract(
        self, contract: ibi.Contract
    ) -> Union[bool, float]:
        my_position = self.position.get(contract, 0.0)
        ib_position = self.ib_position_for_contract(contract)
        return (my_position == ib_position) or (my_position - ib_position)

    def ib_position_for_contract(self, contract: ibi.Contract) -> float:
        # MOVE THIS OUT OF THIS CLASS
        return next(
            (v.position for v in self.ib.positions() if v.contract == contract), 0
        )

        # positions = {p.contract: p.position for p in self.ib.positions()}
        # return positions.get(contract, 0.0)

    def verify_positions(self) -> dict[ibi.Contract, float]:
        """
        Compare positions actually held with broker with position
        records.  Return differences if any and log an error.
        """

        # NOT IN USE

        # list of contracts where differences occur
        # difference: list[ibi.Contract] = []

        broker_positions_dict = {i.contract: i.position for i in self.ib.positions()}
        my_positions_dict = self._data.total_positions()
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
        self._data[strategy].lock = 1 if trade.order.action == "BUY" else -1
        log.debug(f"Registered lock: {strategy}: {self._data[strategy].lock}")

    def new_position_callbacks(self, strategy: str, trade: ibi.Trade) -> None:
        # When exactly should the lock be registered to prevent double-dip?
        self.register_lock(strategy, trade)

    def __repr__(self):
        return self.__class__.__name__ + "()"
