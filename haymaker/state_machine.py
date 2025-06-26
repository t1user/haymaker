from __future__ import annotations

import asyncio
import datetime as dt
import logging
from collections import UserDict, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import Any, Iterator, TypeVar

import eventkit as ev  # type: ignore
import ib_insync as ibi

from .config import CONFIG as config
from .misc import Lock, action_to_signal, decode_tree, tree
from .saver import (
    AbstractBaseSaver,
    AsyncSaveManager,
    MongoSaver,
    SyncSaveManager,
    async_runner,
)

log = logging.getLogger(__name__)

CONFIG = config.get("state_machine") or {}


ORDER_CONTAINER_MAX_SIZE = CONFIG.get("order_container_max_size", 0)
SAVE_DELAY = CONFIG.get("save_delay", 1)
STRATEGY_COLLECTION_NAME = CONFIG.get("strategy_collection_name", "strategies")
ORDER_COLLECTION_NAME = CONFIG.get("order_collection_name", "orders")
MAX_REJECTED_ORDERS = CONFIG.get("max_rejected_orders", 3)

STRATEGY_SAVER = MongoSaver(STRATEGY_COLLECTION_NAME)
ORDER_SAVER = MongoSaver(ORDER_COLLECTION_NAME, query_key="orderId")


@dataclass
class OrderInfo:
    strategy: str
    action: str
    trade: ibi.Trade
    params: dict = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return self.trade.isActive()

    @property
    def permId(self):
        return self.trade.order.permId

    @property
    def amount(self) -> float:
        return self.trade.order.totalQuantity * action_to_signal(
            self.trade.order.action
        )

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

    @classmethod
    def from_trade(cls, trade: ibi.Trade) -> OrderInfo:
        return cls(
            strategy="UNKNOWN",
            action="MANUAL" if trade.order.orderId < 0 else "UNKNOWN",
            trade=trade,
        )


T = TypeVar("T")


class OrderContainer(UserDict):
    """
    Stores `OrderInfo` objects and allows to look them up by `orderId`
    or `permId`.
    """

    def __init__(
        self,
        saver: AbstractBaseSaver,
        max_size: int = ORDER_CONTAINER_MAX_SIZE,
        save_async: bool = True,
    ) -> None:
        self.max_size = max_size
        self.index: dict[int, int] = {}  # translation of permId to orderId
        self.setitemEvent = ev.Event("setitemEvent")
        self.setitemEvent += self.onSetitemEvent
        self.saver = saver
        self._save = (
            AsyncSaveManager(self.saver) if save_async else SyncSaveManager(self.saver)
        )
        super().__init__()

    def __setitem__(self, key: int, item: OrderInfo) -> None:
        if self.max_size and (len(self.data) >= self.max_size):
            self.data.pop(min(self.data.keys()))
        self.setitemEvent.emit(key, item)
        super().__setitem__(key, item)

    def __getitem__(self, key: int) -> OrderInfo:
        if key in self.data:
            return self.data[key]
        elif key in self.index:
            if self.index[key] in self.data:
                return self.data[self.index[key]]
        if hasattr(self.__class__, "__missing__"):
            return self.__class__.__missing__(self, key)  # type: ignore
        raise KeyError(key)

    def __delitem__(self, key: int) -> None:
        if self.index.get(self.data[key].permId):
            del self.index[self.data[key].permId]
        super().__delitem__(key)

    async def onSetitemEvent(self, key: int, item: OrderInfo) -> None:
        while not item.permId:
            await asyncio.sleep(0)
        self.index[item.permId] = key

    def strategy(self, strategy: str) -> Iterator[OrderInfo]:
        """Get active orders associated with strategy."""
        return (oi for oi in self.values() if (oi.strategy == strategy and oi.active))

    def get(
        self, key: int, default: T | None = None, active_only: bool = False
    ) -> OrderInfo | T | None:
        """Get records for an order."""
        try:
            value = self[key]
        except KeyError:
            return default
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

    def save(self, oi: OrderInfo) -> None:
        """Save data to database."""
        self[oi.trade.order.orderId] = oi
        self._save(oi.encode())

    def delete(self, orderId: int) -> None:
        """
        Delete order from dict.  In db order will be marked as
        inactive so that it's not loaded again on startup.  It stays
        in db as historical order, it will no longer be in memory for.
        """
        try:
            oi_dict = self[orderId].encode()
        except KeyError:
            log.error(f"Cannot delete {orderId}, no record.")
            return
        oi_dict.update({"active": False})
        self._save(oi_dict)
        del self[orderId]

    async def read(self) -> None:
        """Read data from database and update itself."""
        order_data = await async_runner(self.saver.read, {"active": True})
        self.decode(order_data)

    def __repr__(self) -> str:
        if self.max_size:
            ms = f", max_size={self.max_size}"
        else:
            ms = ""
        return f"OrderContainer({self.data}, {ms})"


class Strategy(UserDict):
    defaults: dict[str, Any] = {
        "active_contract": None,
        "position": 0.0,
        "params": {},
        "lock": 0,
        "position_id": "",
    }

    def __init__(
        self,
        dict=None,
        strategyChangeEvent=None,
    ) -> None:
        # don't change instruction order
        if strategyChangeEvent is None:
            raise ValueError("Strategy must be initiated with `strategyChangeEvent")
        self.strategyChangeEvent = strategyChangeEvent
        self.data = {**deepcopy(self.defaults)}
        if dict is not None:
            self.update(dict)

    def __getitem__(self, key):
        return self.data.get(key, "UNSET")

    def __setitem__(self, key, item):
        self.data[key] = item
        self.data["timestamp"] = dt.datetime.now(tz=dt.timezone.utc)
        self.strategyChangeEvent.emit()

    def __delitem__(self, key):
        del self.data[key]
        self.strategyChangeEvent.emit()

    @property
    def active(self) -> bool:
        return self["position"] != 0

    def __setattr__(self, key, item):
        if key in {"data", "strategyChangeEvent"}:
            super().__setattr__(key, item)
        else:
            self.__setitem__(key, item)

    def __getattribute__(self, attr):
        if attr == "data":
            return self.__dict__.get("data", {})
        return super().__getattribute__(attr)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}({repr(self.data)})"

    __getattr__ = __getitem__
    __delattr__ = __delitem__


class StrategyContainer(UserDict):

    def __init__(
        self, saver: AbstractBaseSaver, save_delay=SAVE_DELAY, save_async: bool = True
    ) -> None:
        self._strategyChangeEvent = ev.Event("strategyChangeEvent")
        self.strategyChangeEvent = self._strategyChangeEvent.debounce(save_delay, False)
        self._strategyChangeEvent += self.strategyChangeEvent
        # will automatically save strategies to db on every change
        # (but not more often than defined in CONFIG['state_machine']['save_delay'])
        self.strategyChangeEvent += self.save
        self.saver = saver
        self._save = (
            AsyncSaveManager(self.saver) if save_async else SyncSaveManager(self.saver)
        )
        super().__init__()

    def __setitem__(self, key: str, item: dict):
        if not isinstance(item, (Strategy, dict)):
            raise ValueError(
                f"{self.__class__.__qualname__} takes only `Strategy` or `dict`."
            )
        if isinstance(item, dict):
            self.data[key] = Strategy(item, self._strategyChangeEvent)
        elif isinstance(item, Strategy):
            item.strategyChangeEvent = self._strategyChangeEvent
            self.data[key] = item
        self.data[key].strategy = key

    def __missing__(self, key):
        log.debug(f"Creating strategy: {key}")
        self.data[key] = Strategy(
            {"strategy": key},
            self._strategyChangeEvent,
        )
        return self.data[key]

    def get(self, key, default=None):
        if key in self.data:
            return self.data[key]
        elif default is None:
            return self[key]
        else:
            return default

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
        data = tree(self.data)
        # `self` is a `dict` so `data` must always be a `dict`
        data["timestamp"] = dt.datetime.now(tz=dt.timezone.utc)  # type: ignore
        return data  # type: ignore

    def decode(self, data: dict) -> None:
        if data is None:
            return
        try:
            log.debug(f"{self} will decode data: {len(data.keys())-1} keys.")
        except Exception:
            log.debug(f"data for decoding: {data}")
            raise

        decoded = decode_tree(data)

        try:
            decoded.pop("_id")
        except KeyError:
            log.warning("It's weird, no '_id' in data from mongo.")

        try:
            decoded.pop("timestamp")
        except KeyError:
            log.warning("No timestamp in data from mongo.")

        log.debug(f"Decoded keys: {list(decoded.keys())}")
        self.data.update(
            **{k: Strategy(v, self._strategyChangeEvent) for k, v in decoded.items()}
        )

    def save(self) -> None:
        """Save data to database."""
        self._save(self.encode())

    async def read(self) -> None:
        """
        Read data from database and update itself.
        """
        strategy_data = await async_runner(self.saver.read_latest)
        self.decode(strategy_data)

    def __repr__(self) -> str:
        return f"StrategyContainer({self.data})"


class StateMachine:
    """
    This class provides information, other classes act on this
    information.

    In priciple, stores data about strategies and orders.  This data
    is synched with database and restored on restart.  It is a wrapper
    over dicts storing orders and strategies with some helper methods.

    This class must be a singleton, it will raise an error if there's
    an attempt to create a second instance.
    """

    _instance: "StateMachine | None" = None

    def __new__(cls, *args, **kwargs):
        """
        Enforce singleton.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            return cls._instance
        else:
            raise TypeError("Attempt to instantiate StateMachine more than once.")

    def __init__(
        self,
        order_saver: AbstractBaseSaver | None = None,
        strategy_saver: AbstractBaseSaver | None = None,
        save_async: bool = True,
    ) -> None:
        # don't make these class attributes as it messes up tests
        # (reference to dictionaries kept in between tests)
        order_saver = order_saver or ORDER_SAVER
        strategy_saver = strategy_saver or STRATEGY_SAVER
        # dict of OrderInfo
        self._orders = OrderContainer(order_saver, save_async=save_async)
        # dict of Strategy data (same as ExecModel data)
        self._strategies = StrategyContainer(strategy_saver, save_async=save_async)
        self.rejected_orders: dict[tuple[str, str, str], int] = defaultdict(int)

    def register_rejected_order(
        self, errorCode: int, errorString: str, contract: ibi.Contract, order: ibi.Order
    ):
        self.rejected_orders[(contract.localSymbol, order.orderType, order.action)] += 1

    def verify_for_rejections(self, contract: ibi.Contract, order: ibi.Order) -> bool:
        """Return True if order approved, False otherwise."""
        if (
            count := self.rejected_orders.get(
                (contract.localSymbol, order.orderType, order.action)
            )
        ) and (count >= MAX_REJECTED_ORDERS):
            log.info(
                f"Supressing order because of multiple rejections: {order} {contract}"
            )
            return False
        else:
            return True

    def save_strategies(self, *args) -> None:
        self._strategies.save()
        log.debug("STRATEGIES SAVED")

    def save_order(self, oi: OrderInfo) -> OrderInfo:
        self._orders.save(oi)
        log.debug(f"ORDER {oi.trade.order.orderId} SAVED")
        return oi

    def clear_strategies(self):
        self._strategies.clear()

    def update_trade(self, trade: ibi.Trade) -> ibi.Trade | None:
        """
        Update trade object for a given order.  Used after re-start to
        bring records up to date with IB.  Non-restart syncs just
        confirm that the trade is already accounted for without
        over-writting the trade object (which is the same as the
        stored one anyway).
        """
        oi = self._orders.get(trade.order.orderId)
        # this is a new trade object
        if oi and not (oi.trade is trade):
            # this runs only on re-start
            log.debug(
                f"Trade will be updated - id: {oi.trade.order.orderId} "
                f"permId: {oi.trade.order.permId}"
            )
            new_oi = OrderInfo(oi.strategy, oi.action, trade, oi.params)
            self.save_order(new_oi)
            return None
        # trade object exists and is the same as in the records
        elif oi:
            return None
        # this is an unknown trade
        else:
            return trade

    def prune_order(self, orderId: int) -> None:
        """
        This order is no longer of interest it's inactive in our
        orders and inactive in IB.  Called by startup sync routines.

        Delete trade only from local records, not the db.  However, if
        will be market as `inactive` in db regardless of its current
        status.
        """
        self._orders.delete(orderId)

    async def read_from_store(self):
        log.debug("Will read data from store...")
        await asyncio.gather(self._strategies.read(), self._orders.read())

    async def save_order_status(self, trade: ibi.Trade) -> OrderInfo:
        log.debug(
            f"updating trade status: {trade.order.orderId} {trade.order.permId} "
            f"{trade.orderStatus.status}"
        )
        # if orderId is zero, trade object has to be replaced
        order_info = self._orders.get(trade.order.orderId)
        if not order_info:
            order_info = OrderInfo.from_trade(trade)
        try:
            self.save_order(order_info)
        except Exception as e:
            log.exception(e)
        return order_info

    # ### These are data access and modification methods ###

    @property
    def strategy(self) -> StrategyContainer:
        """
        Access strategies by ``state_machine.strategy['strategy_name']``
        or ``state_machine.strategy.get('strategy_name')``
        """
        return self._strategies

    def position_and_order_for_strategy(self, strategy_str: str) -> float:
        """
        Method prevents race conditions; when an order is already
        issued but not filled, object querying if there is a position
        will get information including this pending order; only active
        orders openning or closing positions are included.
        """
        strategy = self.strategy.get(strategy_str)
        if strategy:
            position = strategy.position
        else:
            position = 0.0

        orders = sum(
            [
                order.amount
                for order in self.orders_for_strategy(strategy_str)
                if order.active and (order.action in ["OPEN", "CLOSE"])
            ]
        )

        total = position + orders
        if orders:
            log.debug(
                f"Processing signals with pending orders, strategy: {strategy_str}, "
                f"position: {position}, pending orders: {orders}, total: {total}"
            )

        return total

    @property
    def order(self) -> OrderContainer:
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
        return self._strategies.total_positions()

    @property
    def for_contract(self) -> defaultdict[ibi.Contract, list[str]]:
        """
        Access strategies for given contract:
        ``state_machine.for_contract[contract]`` or
        ``state_machine.for_contract.get(contract)``
        """
        return self._strategies.strategies_by_contract()

    def strategy_for_order(self, orderId: int) -> Strategy | None:
        if oi := self.order.get(orderId):
            return self.strategy.get(oi.strategy)
        return None

    def strategy_for_trade(self, trade: ibi.Trade) -> Strategy | None:
        return self.strategy_for_order(trade.order.orderId)

    def orders_for_strategy(self, strategy: str) -> list[OrderInfo]:
        return list(self._orders.strategy(strategy))

    def delete_order(self, orderId: int) -> None:
        del self._orders[orderId]

    def locked(self, strategy: str) -> Lock:
        return self._strategies[strategy].lock

    def update_strategy_on_order(self, orderId: int, strategy: str) -> OrderInfo:
        oi = self.order.get(orderId)
        assert oi
        oi.strategy = strategy
        return self.save_order(oi)

    def __repr__(self):
        return self.__class__.__name__ + "()"
