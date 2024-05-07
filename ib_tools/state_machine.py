from __future__ import annotations

import asyncio
import logging
from collections import UserDict, defaultdict
from dataclasses import dataclass, field, fields
from functools import partial
from typing import Any, Iterator, Optional

import eventkit as ev  # type: ignore
import ib_insync as ibi

from ib_tools.saver import MongoSaver, SaveManager, async_runner

from .config import CONFIG
from .misc import Lock, decode_tree, tree

log = logging.getLogger(__name__)


@dataclass
class OrderInfo:
    strategy: str
    action: str
    trade: ibi.Trade
    params: dict = field(default_factory=dict)

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
        """
        Params are being flattened in order to make searching by their
        keys easier.  However, `params` keys is kept so that object
        can be decoded into its original shape.
        """
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
    def __init__(
        self, dict=None, /, max_size: int = CONFIG["order_container_max_size"]
    ) -> None:
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
            #     self.data[orderId] = OrderInfo(
            #         item["strategy"], item["action"], item["trade"], item["params"]
            #     )
            log.debug(
                f"Order for : {item['trade'].order.orderId} "
                f"{item['trade'].contract.symbol} decoded."
            )

    def __repr__(self) -> str:
        if self.max_size:
            ms = f", max_size={self.max_size}"
        else:
            ms = ""
        return f"OrderContainer({self.data}{ms})"


class Strategy(dict):
    def __getattr__(self, key):
        return self.get(key, "UNSET")

    def __setattr__(self, key, value) -> None:
        self[key] = value

    def __delattr__(self, key) -> None:
        self.__delitem__(key)

    @property
    def active(self) -> bool:
        return self["position"] != 0


class StrategyContainer(UserDict):
    def __setitem__(self, key: str, item: dict):
        if not isinstance(item, dict):
            raise ValueError(
                f"{self.__class__.__qualname__} can be used to store only dicts."
            )
        elif key is None:
            key = "unknown"
        self.data[key] = Strategy(item)

    def __missing__(self, key):
        log.debug(f"Will reset data for {key}")
        self.data[key] = Strategy(
            {
                "strategy": str(key),
                "active_contract": None,
                "position": 0.0,
                "params": {},
                "lock": 0,
                "position_id": "",
            }
        )
        return self.data[key]

    def total_positions(self) -> defaultdict[ibi.Contract, float]:
        """
        Return a dict of positions by contract.
        """
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


# Consider allowing for use of different savers
model_saver = MongoSaver("models")
order_saver = MongoSaver("orders", query_key="orderId")


class StateMachine:
    """
    This class provides information.  It should be other classes that
    act on this information.

    Answers questions:

        1. Is a strategy in the market?

            - Which portion of the position in a contract is allocated
              to this strategy

        2. Is strategy locked?

        4. Was execution successful (i.e. record desired position
           after signal sent to execution model, then record actual
           live transactions)?

        5. Are holdings linked to strategies?

        6. Are orders linked to strategies?

    Collect all information needed to restore state.  This info is
    stored to database.  If the app crashes, after restart, all info
    about state required by any object must be found here.
    """

    _instance: Optional["StateMachine"] = None

    _save_order = SaveManager(order_saver)
    _save_model = SaveManager(model_saver)

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            return cls._instance
        else:
            raise TypeError("Attempt to instantiate StateMachine more than once.")

    def __init__(self):
        # don't make these class attributes as it messes up tests
        # (reference to dictionaries kept in between tests)
        self._data = StrategyContainer()  # dict of ExecModel data
        self._orders = OrderContainer()  # dict of OrderInfo

    def save_models(self):
        self._save_model(self._data.encode())

    def save_order(self, order: OrderInfo):
        self._save_order(order.encode())

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
            self.save_order(new_oi)
            return None
        else:
            return trade

    def prune_order(self, orderId) -> None:
        """
        This order is no longer of interest it's inactive in our
        orders and inactive in IB.  Called by startup sync routines.

        Delete trade only from local records (database is unaffected).
        """
        self._orders.pop(orderId)

    def delete_trade_record(self, trade: ibi.Trade) -> None:
        """
        Delete order related record from database.  Before deleting
        make sure it's marked as inactive so that it doesn't get
        reloaded on restart.
        """
        oi_dict = self._orders[trade.order.orderId].encode()
        oi_dict.update({"active": False})
        self._save_order(oi_dict)
        del self._orders[trade.order.orderId]

    async def read_from_store(self):
        log.debug("Will read data from store...")
        strategy_data = await async_runner(model_saver.read_latest)
        order_data = await async_runner(order_saver.read, {"active": True})
        self._data.decode(strategy_data)
        self._orders.decode(order_data)

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
            trade.filledEvent += partial(self.register_lock, strategy)
        elif action.upper() == "CLOSE":
            trade.filledEvent += partial(self.remove_lock, strategy)

        self.save_order(order_info)
        self.save_models()

    async def save_order_status(self, trade: ibi.Trade) -> Optional[OrderInfo]:
        log.debug(f"updating trade status: {trade.order.orderId} {trade.order.permId}")
        # if orderId zero it means trade objects has to be replaced
        order_info = self._orders.get(trade.order.orderId)
        if not order_info:
            order_info = OrderInfo(
                strategy="unknown",
                action="MANUAL" if trade.order.orderId < 0 else "UNKNOWN",
                trade=trade,
            )

        try:
            self.save_models()
            self.save_order(order_info)
            log.debug("SAVING MODELS SUCCESSFUL")
        except Exception as e:
            log.exception(e)
        return order_info

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

    def orders_for_strategy(self, strategy: str) -> Iterator[OrderInfo]:
        return self._orders.strategy(strategy)

    def delete_order(self, orderId: int) -> None:
        del self._orders[orderId]

    def locked(self, strategy: str) -> Lock:
        return self._data[strategy].lock

    # ### TODO: Re-do this
    # DOES THIS BELONG IN THIS CLASS?
    # or maybe position should also be set on this class
    def register_lock(self, strategy: str, trade: ibi.Trade) -> None:
        # TODO: move to controller
        self._data[strategy].lock = 1 if trade.order.action == "BUY" else -1
        # log.debug(f"Registered lock: {strategy}: {self._data[strategy].lock}")

    def remove_lock(self, strategy: str, trade: ibi.Trade) -> None:
        self._data[strategy].lock = 0

    def __repr__(self):
        return self.__class__.__name__ + "()"
