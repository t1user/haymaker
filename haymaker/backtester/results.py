"""Backtest result values and in-memory persistence helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import ib_insync as ibi

from haymaker.saver import AbstractBaseSaver


def _finite(value: float, name: str) -> float:
    """Return ``value`` as a finite float.

    Args:
        value: Number to normalize.
        name: Field name used in validation errors.

    Returns:
        The normalized finite value.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If ``value`` is not finite.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, kw_only=True)
class BacktestFill:
    """One full simulated execution.

    Attributes:
        time: A timezone-aware simulation timestamp.
        contract: Concrete qualified IB contract.
        order_id: Broker-local simulated order identifier.
        perm_id: Stable simulated permanent identifier.
        execution_id: Stable simulated execution identifier.
        action: ``BUY`` or ``SELL``.
        order_type: Simulated IB order type.
        quantity: Positive filled quantity.
        price: Simulated execution price after slippage.
        commission: Commission charged for this execution.
        realized_pnl: Gross realized PnL produced by this execution.
    """

    time: datetime
    contract: ibi.Contract
    order_id: int
    perm_id: int
    execution_id: str
    action: str
    order_type: str
    quantity: float
    price: float
    commission: float
    realized_pnl: float

    def __post_init__(self) -> None:
        """Validate immutable execution evidence."""

        if self.time.tzinfo is None or self.time.utcoffset() is None:
            raise ValueError("time must be timezone-aware")
        if not isinstance(self.contract, ibi.Contract):
            raise TypeError("contract must be an ib_insync.Contract")
        if not self.contract.conId:
            raise ValueError("contract must have a non-zero conId")
        if self.order_id <= 0:
            raise ValueError("order_id must be positive")
        if self.perm_id <= 0:
            raise ValueError("perm_id must be positive")
        if not self.execution_id:
            raise ValueError("execution_id must not be empty")
        action = self.action.upper()
        if action not in {"BUY", "SELL"}:
            raise ValueError("action must be BUY or SELL")
        object.__setattr__(self, "action", action)
        order_type = self.order_type.upper()
        if order_type not in {"MKT", "LMT", "STP"}:
            raise ValueError("order_type must be MKT, LMT, or STP")
        object.__setattr__(self, "order_type", order_type)
        quantity = _finite(self.quantity, "quantity")
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "price", _finite(self.price, "price"))
        commission = _finite(self.commission, "commission")
        if commission < 0:
            raise ValueError("commission must not be negative")
        object.__setattr__(self, "commission", commission)
        object.__setattr__(
            self,
            "realized_pnl",
            _finite(self.realized_pnl, "realized_pnl"),
        )


@dataclass(frozen=True, kw_only=True)
class BacktestOrder:
    """Final immutable snapshot of one simulated order.

    Attributes:
        submitted_at: Replay submission time, or ``None`` for an order created
            by the initial startup hook before the first clock point.
        status: Final IB order-status string at the end of the replay.
    """

    submitted_at: datetime | None
    contract: ibi.Contract
    order_id: int
    perm_id: int
    action: str
    order_type: str
    quantity: float
    status: str
    filled: float
    remaining: float
    limit_price: float | None = None
    stop_price: float | None = None
    oca_group: str = ""

    def __post_init__(self) -> None:
        """Validate immutable order evidence."""

        if self.submitted_at is not None and (
            self.submitted_at.tzinfo is None or self.submitted_at.utcoffset() is None
        ):
            raise ValueError("submitted_at must be timezone-aware")
        if not isinstance(self.contract, ibi.Contract):
            raise TypeError("contract must be an ib_insync.Contract")
        if not self.contract.conId:
            raise ValueError("contract must have a non-zero conId")
        if self.order_id <= 0 or self.perm_id <= 0:
            raise ValueError("order_id and perm_id must be positive")
        action = self.action.upper()
        if action not in {"BUY", "SELL"}:
            raise ValueError("action must be BUY or SELL")
        object.__setattr__(self, "action", action)
        order_type = self.order_type.upper()
        if order_type not in {"MKT", "LMT", "STP"}:
            raise ValueError("order_type must be MKT, LMT, or STP")
        object.__setattr__(self, "order_type", order_type)
        if not self.status:
            raise ValueError("status must not be empty")
        for name in ("quantity", "filled", "remaining"):
            value = _finite(getattr(self, name), name)
            if value < 0:
                raise ValueError(f"{name} must not be negative")
            object.__setattr__(self, name, value)
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        for name in ("limit_price", "stop_price"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite(value, name))

    @property
    def is_working(self) -> bool:
        """Return whether the replay ended with this order still active."""

        return self.status in ibi.OrderStatus.ActiveStates


@dataclass(frozen=True, kw_only=True)
class ContractResult:
    """Final accounting projection for one concrete contract."""

    contract: ibi.Contract
    position: float
    average_price: float
    last_price: float | None
    multiplier: float
    realized_pnl: float
    unrealized_pnl: float
    commission: float
    cash_change: float = 0.0

    def __post_init__(self) -> None:
        """Validate one final contract projection."""

        if not isinstance(self.contract, ibi.Contract):
            raise TypeError("contract must be an ib_insync.Contract")
        if not self.contract.conId:
            raise ValueError("contract must have a non-zero conId")
        for name in (
            "position",
            "average_price",
            "multiplier",
            "realized_pnl",
            "unrealized_pnl",
            "commission",
            "cash_change",
        ):
            value = _finite(getattr(self, name), name)
            object.__setattr__(self, name, value)
        if self.last_price is not None:
            object.__setattr__(
                self, "last_price", _finite(self.last_price, "last_price")
            )
        if self.multiplier <= 0:
            raise ValueError("multiplier must be positive")
        if self.commission < 0:
            raise ValueError("commission must not be negative")

    @property
    def net_pnl(self) -> float:
        """Return realized and unrealized PnL after commission."""

        return self.realized_pnl + self.unrealized_pnl - self.commission


@dataclass(frozen=True, kw_only=True)
class BacktestResult:
    """Immutable final account projection for one simulation.

    ``realized_pnl`` and ``unrealized_pnl`` are gross of commission. The
    reported ``net_pnl`` and ``equity`` deduct commission exactly once.
    """

    initial_cash: float
    contracts: tuple[ContractResult, ...]
    fills: tuple[BacktestFill, ...]
    orders: tuple[BacktestOrder, ...] = ()

    def __post_init__(self) -> None:
        """Normalize immutable collection and cash values."""

        object.__setattr__(
            self, "initial_cash", _finite(self.initial_cash, "initial_cash")
        )
        object.__setattr__(self, "contracts", tuple(self.contracts))
        object.__setattr__(self, "fills", tuple(self.fills))
        object.__setattr__(self, "orders", tuple(self.orders))

    @property
    def realized_pnl(self) -> float:
        """Return gross realized PnL across all contracts."""

        return sum(item.realized_pnl for item in self.contracts)

    @property
    def unrealized_pnl(self) -> float:
        """Return mark-to-market PnL across all open positions."""

        return sum(item.unrealized_pnl for item in self.contracts)

    @property
    def commission(self) -> float:
        """Return total simulated commission."""

        return sum(item.commission for item in self.contracts)

    @property
    def net_pnl(self) -> float:
        """Return total PnL after commission."""

        return self.realized_pnl + self.unrealized_pnl - self.commission

    @property
    def equity(self) -> float:
        """Return final marked account equity."""

        return self.initial_cash + self.net_pnl

    @property
    def ending_cash(self) -> float:
        """Return cash after simulated notional, PnL, and commission flows."""

        return self.initial_cash + sum(item.cash_change for item in self.contracts)

    @property
    def working_orders(self) -> tuple[BacktestOrder, ...]:
        """Return orders still active when the replay range ended."""

        return tuple(order for order in self.orders if order.is_working)

    @property
    def positions(self) -> Mapping[int, float]:
        """Return non-zero final positions keyed by contract identifier."""

        return {
            item.contract.conId: item.position
            for item in self.contracts
            if item.position
        }


class InMemorySaver(AbstractBaseSaver):
    """Small upserting saver for a non-persistent backtest ``Book``.

    Args:
        query_key: Optional document field used as the upsert identity. When
            omitted, ``orderId`` and ``state_key`` are inferred from each
            document where possible.
    """

    def __init__(self, query_key: str | None = None) -> None:
        self.query_key = query_key
        self._documents: list[dict[str, Any]] = []

    @property
    def documents(self) -> tuple[Mapping[str, Any], ...]:
        """Return copies of all currently retained documents."""

        return tuple(dict(document) for document in self._documents)

    def save(self, data: Any, /, *args: Any) -> None:
        """Copy and upsert one mapping.

        Args:
            data: Mapping to retain.
            *args: Unused compatibility arguments accepted by saver callers.

        Raises:
            TypeError: If ``data`` is not a mapping.
        """

        del args
        if not isinstance(data, Mapping):
            raise TypeError("InMemorySaver accepts only mappings")
        document = dict(data)
        query_key = self.query_key or next(
            (name for name in ("orderId", "state_key") if name in document),
            None,
        )
        if query_key is not None and query_key in document:
            identity = document[query_key]
            for index, current in enumerate(self._documents):
                if current.get(query_key) == identity:
                    self._documents[index] = document
                    return
        self._documents.append(document)

    def save_many(self, data: Sequence[Mapping[str, Any]], /, *args: Any) -> None:
        """Retain several mappings using the same upsert policy."""

        for document in data:
            self.save(document, *args)

    def read(self, key: Any = None, /, *args: Any) -> list[dict[str, Any]]:
        """Return copied documents matching an optional query.

        Args:
            key: ``None`` or an empty mapping returns every document. A
                mapping applies exact field matching. A scalar matches the
                configured ``query_key``.
            *args: Unused compatibility arguments accepted by saver callers.

        Returns:
            Matching document copies.
        """

        del args
        if key is None or key == {}:
            selected = self._documents
        elif isinstance(key, Mapping):
            selected = [
                document
                for document in self._documents
                if all(document.get(name) == value for name, value in key.items())
            ]
        elif self.query_key is not None:
            selected = [
                document
                for document in self._documents
                if document.get(self.query_key) == key
            ]
        else:
            selected = []
        return [dict(document) for document in selected]


__all__ = [
    "BacktestFill",
    "BacktestOrder",
    "BacktestResult",
    "ContractResult",
    "InMemorySaver",
]
