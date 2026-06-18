from __future__ import annotations

import asyncio
import datetime as dt
import logging
from collections import UserDict, defaultdict
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import Any, TypeVar

import eventkit as ev  # type: ignore
import ib_insync as ibi

from .config import CONFIG as config
from .misc import Lock, action_to_signal, decode_tree, tree
from .saver import AbstractBaseSaver, AsyncSaveManager, MongoLatestSaver, MongoSaver

log = logging.getLogger(__name__)

CONFIG = config.get("state_machine") or {}


SAVE_DELAY = CONFIG.get("save_delay", 1)
STRATEGY_COLLECTION_NAME = CONFIG.get("strategy_collection_name", "strategies")
ORDER_COLLECTION_NAME = CONFIG.get("order_collection_name", "orders")
MAX_REJECTED_ORDERS = CONFIG.get("max_rejected_orders", 3)


class UnknownZeroOrderIdError(ValueError):
    """Raised when an unknown broker trade still has placeholder orderId 0."""


@dataclass
class OrderInfo:
    strategy: str
    action: str
    trade: ibi.Trade
    params: dict = field(default_factory=dict)
    accounted_exec_ids: list[str] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.trade.isActive()

    @property
    def permId(self):
        return self.trade.order.permId

    @property
    def order_key(self) -> int:
        """Return the local storage key for this order record, which
        is ``orderId == 0`` if non-zero or ``permId`` otherwise.
        """
        return self.trade.order.orderId or self.trade.order.permId

    @property
    def amount(self) -> float:
        return self.trade.order.totalQuantity * action_to_signal(
            self.trade.order.action
        )

    def execution_key(self, fill: ibi.Fill) -> str:
        """Return stable key identifying a broker execution fill."""
        if exec_id := fill.execution.execId:
            return exec_id

        execution = fill.execution
        time = execution.time.isoformat() if execution.time else ""
        return (
            f"{self.trade.order.permId}:{self.trade.order.orderId}:"
            f"{time}:{execution.side}:{execution.shares}:{execution.price}"
        )

    def execution_accounted(self, fill: ibi.Fill) -> bool:
        """Return True if this execution already changed local position."""
        return self.execution_key(fill) in self.accounted_exec_ids

    def mark_execution(self, fill: ibi.Fill) -> None:
        """Record that this execution has changed local position."""
        exec_key = self.execution_key(fill)
        if exec_key not in self.accounted_exec_ids:
            self.accounted_exec_ids.append(exec_key)

    def __iter__(self) -> Iterator[Any]:
        yield self.strategy
        yield self.action
        yield self.trade
        yield self.params
        yield self.active

    def encode(self) -> dict[str, Any]:
        return {
            "orderId": self.order_key,
            **{k: tree(v) for k, v in self.__dict__.items()},
            "active": self.active,
            "priority": self._priority(self.trade),
        }

    @staticmethod
    def _priority(trade: ibi.Trade) -> int:
        if not trade.log:
            return 0
        return int(max([t.time.timestamp() for t in trade.log]) * 1000)

    def decode(self, data: dict[str, Any]) -> None:
        self.__dict__.update(**self._clear_keys(data))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OrderInfo:
        clear_data = cls._clear_keys(data)
        clear_data.setdefault("accounted_exec_ids", [])
        return cls(**clear_data)

    @classmethod
    def from_trade(cls, trade: ibi.Trade) -> OrderInfo:
        if trade.order.orderId == 0:
            raise UnknownZeroOrderIdError(
                "Cannot create unknown OrderInfo for trade with orderId 0."
            )
        log.error(f"Creating unknown strategy for trade: {trade}")
        return cls(
            strategy="UNKNOWN",
            action="MANUAL" if trade.order.orderId < 0 else "UNKNOWN",
            trade=trade,
        )

    @classmethod
    def _clear_keys(cls, data: dict[str, Any]) -> dict[str, Any]:
        clear_data = dict(data)
        clear_data.pop("active", None)
        clear_data.pop("priority", None)
        valid_fields = {field.name for field in fields(cls)}
        return {key: value for key, value in clear_data.items() if key in valid_fields}


T = TypeVar("T")


class OrderContainer(UserDict):
    """
    Stores `OrderInfo` objects keyed by their local order key.
    """

    def __init__(
        self,
        saver: AbstractBaseSaver,
        save_async: bool = True,
    ) -> None:
        self.saver = (
            AsyncSaveManager(saver, name="OrderContainer") if save_async else saver
        )
        super().__init__()

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
        self[oi.order_key] = oi
        self.saver.save(oi.encode())

    def delete(self, orderId: int) -> None:
        """
        Delete order from dict.  In db order will be marked as
        inactive so that it's not loaded again on startup.  It stays
        in db as historical order, it will no longer be in memory.
        """
        try:
            oi_dict = self[orderId].encode()
        except KeyError:
            log.error(f"Cannot delete {orderId}, no record.")
            return
        oi_dict.update({"active": False})
        self.saver.save(oi_dict)
        del self[orderId]

    async def read(self) -> None:
        """Read data from database and update itself."""
        order_data = await self.saver.read({"active": True})
        self.decode(order_data)

    def __repr__(self) -> str:
        return f"OrderContainer({self.data})"


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
        # todo: think about it
        self.data["timestamp"] = dt.datetime.now(tz=dt.timezone.utc)
        self.strategyChangeEvent.emit()

    def __delitem__(self, key):
        del self.data[key]
        self.strategyChangeEvent.emit()

    @property
    def active(self) -> bool:
        return self["position"] != 0

    def register_fill(self, fill: ibi.Fill) -> None:
        """Apply a broker execution fill to this strategy's local position."""
        if fill.execution.side == "BOT":
            self.position += fill.execution.shares
        elif fill.execution.side == "SLD":
            self.position -= fill.execution.shares
        else:
            raise ValueError(f"Ambiguous fill side: {fill.execution.side}")

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
        self,
        saver: AbstractBaseSaver,
        save_delay=SAVE_DELAY,
        save_async: bool = True,
    ) -> None:
        self._strategyChangeEvent = ev.Event("strategyChangeEvent")
        self.strategyChangeEvent = self._strategyChangeEvent.debounce(save_delay, False)
        self._strategyChangeEvent += self.strategyChangeEvent
        # will automatically save strategies to db on every change
        # (but not more often than defined in CONFIG['state_machine']['save_delay'])
        self.strategyChangeEvent += self.save
        self.saver = (
            AsyncSaveManager(saver, name="StrategyContainer") if save_async else saver
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
            if data.active_contract and data.position:
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
        self.saver.save(self.encode())

    async def read(self) -> None:
        """
        Read data from database and update itself.
        """
        strategy_data = await self.saver.read()
        assert isinstance(strategy_data, dict)
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
    an attempt to create a second instance. It shouldn't be sub-classed.
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

        if order_saver is None:
            order_saver = MongoSaver(ORDER_COLLECTION_NAME, query_key="orderId")
        if strategy_saver is None:
            strategy_saver = MongoLatestSaver(STRATEGY_COLLECTION_NAME)

        # dict of OrderInfo
        self._orders = OrderContainer(order_saver, save_async=save_async)
        # dict of Strategy data (same as ExecModel data)
        self._strategies = StrategyContainer(strategy_saver, save_async=save_async)
        self.rejected_orders: dict[str, int] = defaultdict(int)
        log.debug(f"StateMachine initialized: {self}")

    def register_rejected_order(self, strategy_str: str) -> None:
        self.rejected_orders[strategy_str] += 1

    def verify_for_rejections(self, strategy_str: str) -> bool:
        """Return True if order approved, False otherwise."""
        if (count := self.rejected_orders.get(strategy_str)) and (
            count >= MAX_REJECTED_ORDERS
        ):
            log.info(
                f"Supressing order because of multiple rejections for strategy: "
                f"{strategy_str}"
            )
            return False
        else:
            return True

    def save_strategies(self, *args) -> None:
        # used by sync routines/error handlers
        self._strategies.save()
        log.debug("STRATEGIES SAVED")

    def save_order(self, oi: OrderInfo) -> OrderInfo:
        self._orders.save(oi)
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
            new_oi = OrderInfo(
                oi.strategy,
                oi.action,
                trade,
                oi.params,
                list(oi.accounted_exec_ids),
            )
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

        Delete trade only from local records, not the db.  However, it
        will be market as `inactive` in db regardless of its current
        status.
        """
        self._orders.delete(orderId)

    async def read_from_store(self):
        log.debug("Will read data from store...")
        await asyncio.gather(self._strategies.read(), self._orders.read())

    def save_order_status(self, trade: ibi.Trade) -> OrderInfo | None:

        # if orderId is zero, trade object has to be replaced
        order_info = self._orders.get(trade.order.orderId)
        if not order_info:
            try:
                order_info = OrderInfo.from_trade(trade)
            except UnknownZeroOrderIdError as e:
                log.warning(
                    f"{e} Trade skipped: {trade.order.orderId=} {trade.order.permId=}"
                )
                return None
        try:
            self.save_order(order_info)
        except Exception as e:
            log.exception(e)
        return order_info

    @contextmanager
    def guard_execution(
        self, trade: ibi.Trade, fill: ibi.Fill
    ) -> Generator[bool, None, None]:
        """Guard against applying the same broker execution more than once."""
        order_info = self.order.get(trade.order.orderId)
        if not order_info:
            log.error(
                f"Attempt to register execution for unknown order: {trade.order.orderId} "
                f"{trade.contract.localSymbol}"
            )
            yield False
        else:
            yield order_info.execution_accounted(fill)
            order_info.mark_execution(fill)
            self.save_order(order_info)

    # ### data access and modification methods ###

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

    def order_by_permId(self, perm_id: int) -> OrderInfo | None:
        """Return the order record matching a broker permanent id."""
        if not perm_id:
            return None
        for order_info in self.order.values():
            if order_info.trade.order.permId == perm_id:
                return order_info
        return None

    def strategy_for_trade(self, trade: ibi.Trade) -> Strategy | None:
        if order_info := self.order.get(trade.order.orderId):
            return self.strategy.get(order_info.strategy)
        return None

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

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"({', '.join([f'{k}={v}' for k, v in self.__dict__.items()])})"
        )
