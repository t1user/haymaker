"""Deterministic bar-driven broker adapter for framework backtests."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from typing import Any

import eventkit as ev  # type: ignore
import ib_insync as ibi

from .exceptions import BacktestError, UnsupportedBacktestFeatureError
from .results import BacktestFill, BacktestOrder, BacktestResult, ContractResult

_SUPPORTED_ORDER_FIELDS = frozenset(
    {
        "action",
        "auxPrice",
        "clientId",
        "lmtPrice",
        "ocaGroup",
        "ocaType",
        "orderId",
        "orderRef",
        "orderType",
        "outsideRth",
        "permId",
        "tif",
        "totalQuantity",
    }
)
_DEFAULT_ORDER = ibi.Order()


class UnsupportedBacktestOrderError(UnsupportedBacktestFeatureError):
    """Raised when an IB order cannot be simulated without hidden assumptions."""


class BacktestBrokerError(BacktestError):
    """Raised when simulated broker state or inputs are inconsistent."""


@dataclass
class _PositionLedger:
    """Mutable broker-side position and PnL state for one contract."""

    contract: ibi.Contract
    quantity: float = 0.0
    average_price: float = 0.0
    multiplier: float = 1.0
    realized_pnl: float = 0.0
    commission: float = 0.0
    cash_change: float = 0.0
    last_price: float | None = None

    def apply(self, signed_quantity: float, price: float) -> float:
        """Apply one full fill and return its gross realized PnL."""

        previous = self.quantity
        if previous == 0 or _same_sign(previous, signed_quantity):
            total = abs(previous) + abs(signed_quantity)
            self.average_price = (
                abs(previous) * self.average_price + abs(signed_quantity) * price
            ) / total
            self.quantity = previous + signed_quantity
            return 0.0

        closing = min(abs(previous), abs(signed_quantity))
        direction = 1.0 if previous > 0 else -1.0
        realized = closing * (price - self.average_price) * direction * self.multiplier
        remaining = previous + signed_quantity
        if remaining == 0:
            self.average_price = 0.0
        elif not _same_sign(previous, remaining):
            self.average_price = price
        self.quantity = remaining
        self.realized_pnl += realized
        return realized

    @property
    def unrealized_pnl(self) -> float:
        """Return gross marked PnL for the open quantity."""

        if not self.quantity or self.last_price is None:
            return 0.0
        return self.quantity * (self.last_price - self.average_price) * self.multiplier


def _same_sign(left: float, right: float) -> bool:
    """Return whether two non-zero numbers have the same sign."""

    return (left > 0 and right > 0) or (left < 0 and right < 0)


def _aware_datetime(value: date | datetime) -> datetime:
    """Normalize a replay timestamp to an aware UTC datetime."""

    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return datetime.combine(value, time.min, tzinfo=timezone.utc)


def _finite_number(value: Any, name: str) -> float:
    """Return a finite float or raise a field-specific error."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _contract_key(contract: ibi.Contract) -> int:
    """Return a qualified contract identity."""

    if not isinstance(contract, ibi.Contract):
        raise TypeError("contract must be an ib_insync.Contract")
    if not contract.conId:
        raise ValueError("backtest contracts must have a non-zero conId")
    return contract.conId


def _metadata_number(
    metadata: Mapping[str, Any], names: Sequence[str], default: float
) -> float:
    """Read the first present numeric metadata field."""

    for name in names:
        if name in metadata and metadata[name] is not None:
            value = metadata[name]
            if isinstance(value, str):
                try:
                    value = float(value)
                except ValueError as exc:
                    raise ValueError(
                        f"metadata.{name} must contain a real number"
                    ) from exc
            return _finite_number(value, f"metadata.{name}")
    return default


class SimulatedIB:
    """Duck-typed subset of :class:`ib_insync.IB` for historical replay.

    Orders are accepted immediately but are eligible only on a later actual
    bar for their concrete contract. Calls to :meth:`process_bar` therefore
    belong before publishing that bar to the strategy graph. No bar means no
    fill opportunity for that contract.

    Args:
        initial_cash: Opening account value used by :meth:`result`.
        slippage_ticks: Adverse ticks applied to every execution. A positive
            value requires ``minTick`` or ``min_tick`` in contract metadata.
        account: Simulated IB account name used in Position objects.
    """

    events = ibi.IB.events

    def __init__(
        self,
        *,
        initial_cash: float = 0.0,
        slippage_ticks: float = 0.0,
        account: str = "BACKTEST",
    ) -> None:
        ticks = _finite_number(slippage_ticks, "slippage_ticks")
        if ticks < 0:
            raise ValueError("slippage_ticks must not be negative")
        if not account:
            raise ValueError("account must not be empty")
        self.initial_cash = _finite_number(initial_cash, "initial_cash")
        self.slippage_ticks = ticks
        self.account = account
        self.connectedEvent: ev.Event
        self.disconnectedEvent: ev.Event
        self.orderModifyEvent: ev.Event
        self.newOrderEvent: ev.Event
        self.cancelOrderEvent: ev.Event
        self.orderStatusEvent: ev.Event
        self.execDetailsEvent: ev.Event
        self.commissionReportEvent: ev.Event
        self._create_events()
        self._connected = True
        self._current_time: datetime | None = None
        self._session_contracts: set[int] = set()
        self._next_order_id = 1
        self._next_perm_id = 1_000_001
        self._next_execution_id = 1
        self._trades: list[ibi.Trade] = []
        self._trades_by_order_id: dict[int, ibi.Trade] = {}
        self._fills: list[ibi.Fill] = []
        self._submitted_at: dict[int, datetime | None] = {}
        self._ledgers: dict[int, _PositionLedger] = {}
        self._metadata: dict[int, Mapping[str, Any]] = {}
        self._details: dict[int, ibi.ContractDetails] = {}
        self._historical_sources: dict[int, ibi.BarDataList] = {}
        self._tickers: dict[int, ibi.Ticker] = {}
        self._fill_records: list[BacktestFill] = []

    def _create_events(self) -> None:
        """Create the public IB event surface used by Haymaker."""

        for name in self.events:
            setattr(self, name, ev.Event(name))

    @property
    def current_time(self) -> datetime | None:
        """Return the current simulation time, if replay has started."""

        return self._current_time

    def set_current_time(self, timestamp: date | datetime) -> None:
        """Advance the broker's submission clock without processing a bar."""

        self._current_time = _aware_datetime(timestamp)

    def set_session_contracts(self, contracts: Iterable[ibi.Contract]) -> None:
        """Set Contracts with an actual stored bar at the current clock point."""

        self._session_contracts = {_contract_key(contract) for contract in contracts}

    def has_session(self, contract: ibi.Contract) -> bool:
        """Return whether ``contract`` has an actual bar at the current point."""

        return _contract_key(contract) in self._session_contracts

    def isConnected(self) -> bool:
        """Return whether the simulated broker is available."""

        return self._connected

    def connect(self, *args: Any, **kwargs: Any) -> bool:
        """Mark the in-process broker connected without external I/O."""

        del args, kwargs
        if not self._connected:
            self._connected = True
            self.connectedEvent.emit()
        return True

    def disconnect(self) -> None:
        """Disconnect and wake framework listeners."""

        if self._connected:
            self._connected = False
            self.disconnectedEvent.emit()

    def configure_contract(
        self,
        contract: ibi.Contract,
        metadata: Mapping[str, Any],
        *,
        details: ibi.ContractDetails | None = None,
    ) -> None:
        """Install datastore-derived metadata for one concrete contract."""

        key = _contract_key(contract)
        copied = dict(metadata)
        multiplier = _metadata_number(copied, ("multiplier",), 1.0)
        if multiplier <= 0:
            raise ValueError("metadata.multiplier must be positive")
        commission = _metadata_number(copied, ("commission",), 0.0)
        if commission < 0:
            raise ValueError("metadata.commission must not be negative")
        if self.slippage_ticks:
            min_tick = _metadata_number(copied, ("minTick", "min_tick"), 0.0)
            if min_tick <= 0:
                raise KeyError(
                    f"Contract {contract.localSymbol or contract.symbol} requires "
                    "positive minTick or min_tick metadata when slippage is enabled"
                )
        self._metadata[key] = copied
        self._details[key] = details or ibi.ContractDetails(
            contract=contract,
            minTick=_metadata_number(copied, ("minTick", "min_tick"), 0.0),
        )
        ledger = self._ledgers.get(key)
        if ledger is None:
            self._ledgers[key] = _PositionLedger(contract, multiplier=multiplier)
        else:
            ledger.multiplier = multiplier

    def configure_sources(self, sources: Iterable[Any]) -> None:
        """Configure contract metadata and empty streamer subscriptions.

        Each source is expected to expose ``contract`` and ``metadata`` and may
        expose ``contract_details``. This intentionally structural interface
        accepts :class:`haymaker.backtester.data.StoredSeries` without making
        the broker depend on the datastore adapter.
        """

        for source in sources:
            contract = getattr(source, "contract", None)
            metadata = getattr(source, "metadata", None)
            if not isinstance(contract, ibi.Contract) or not isinstance(
                metadata, Mapping
            ):
                raise TypeError(
                    "Backtest sources must expose Contract contract and "
                    "Mapping metadata"
                )
            details = getattr(source, "contract_details", None)
            if callable(details):
                details = details()
            if details is not None and not isinstance(details, ibi.ContractDetails):
                raise TypeError("source.contract_details must be ContractDetails")
            self.configure_contract(contract, metadata, details=details)
            bars = ibi.BarDataList()
            bars.contract = contract
            self._historical_sources[contract.conId] = bars

    def reqMktData(
        self,
        contract: ibi.Contract,
        genericTickList: str = "",
        snapshot: bool = False,
        regulatorySnapshot: bool = False,
        mktDataOptions: Sequence[ibi.TagValue] = (),
    ) -> ibi.Ticker:
        """Return one reusable in-memory ticker for a concrete contract."""

        del genericTickList, snapshot, regulatorySnapshot, mktDataOptions
        key = _contract_key(contract)
        ticker = self._tickers.get(key)
        if ticker is None:
            ticker = ibi.Ticker(contract=contract)
            self._tickers[key] = ticker
        return ticker

    async def reqHistoricalDataAsync(
        self,
        contract: ibi.Contract,
        endDateTime: date | datetime | str | None,
        durationStr: str,
        barSizeSetting: str,
        whatToShow: str,
        useRTH: bool,
        formatDate: int = 1,
        keepUpToDate: bool = False,
        chartOptions: Sequence[ibi.TagValue] = (),
        timeout: float = 60,
    ) -> ibi.BarDataList:
        """Return the configured mutable historical subscription.

        The replay engine owns history population and ``updateEvent`` emission;
        this method never queries IB or a datastore.
        """

        del timeout
        key = _contract_key(contract)
        bars = self._historical_sources.setdefault(key, ibi.BarDataList())
        bars.contract = contract
        bars.endDateTime = endDateTime
        bars.durationStr = durationStr
        bars.barSizeSetting = barSizeSetting
        bars.whatToShow = whatToShow
        bars.useRTH = useRTH
        bars.formatDate = formatDate
        bars.keepUpToDate = keepUpToDate
        bars.chartOptions = list(chartOptions)
        return bars

    def historical_subscription(self, contract: ibi.Contract) -> ibi.BarDataList:
        """Return the mutable bar list used by HistoricalDataStreamer."""

        key = _contract_key(contract)
        bars = self._historical_sources.setdefault(key, ibi.BarDataList())
        bars.contract = contract
        return bars

    async def reqContractDetailsAsync(
        self, contract: ibi.Contract
    ) -> list[ibi.ContractDetails]:
        """Return only preconfigured datastore-derived contract details."""

        if contract.conId:
            detail = self._details.get(contract.conId)
            return [detail] if detail is not None else []
        matches = [
            detail
            for detail in self._details.values()
            if detail.contract is not None
            and _contract_matches(contract, detail.contract)
        ]
        return matches

    async def qualifyContractsAsync(
        self, *contracts: ibi.Contract
    ) -> list[ibi.Contract]:
        """Resolve contracts only against configured datastore metadata."""

        qualified: list[ibi.Contract] = []
        for contract in contracts:
            details = await self.reqContractDetailsAsync(contract)
            if len(details) == 1:
                resolved = details[0].contract
                if resolved is not None:
                    qualified.append(resolved)
            elif len(details) > 1:
                raise BacktestBrokerError(f"Ambiguous contract blueprint: {contract}")
        return qualified

    def qualifyContracts(self, *contracts: ibi.Contract) -> list[ibi.Contract]:
        """Synchronous metadata-only qualification for compatibility."""

        qualified: list[ibi.Contract] = []
        for contract in contracts:
            if contract.conId and contract.conId in self._details:
                resolved = self._details[contract.conId].contract
                if resolved is not None:
                    qualified.append(resolved)
                continue
            matches = [
                detail.contract
                for detail in self._details.values()
                if detail.contract is not None
                and _contract_matches(contract, detail.contract)
            ]
            if len(matches) == 1:
                qualified.append(matches[0])
            elif len(matches) > 1:
                raise BacktestBrokerError(f"Ambiguous contract blueprint: {contract}")
        return qualified

    def placeOrder(self, contract: ibi.Contract, order: ibi.Order) -> ibi.Trade:
        """Accept or modify one supported full-fill order."""

        if not self._connected:
            raise ConnectionError("SimulatedIB is disconnected")
        key = _contract_key(contract)
        if key not in self._metadata:
            raise BacktestBrokerError(
                f"No datastore metadata configured for {contract.localSymbol or contract}"
            )
        self._validate_order(order)
        if order.orderId:
            existing = self._trades_by_order_id.get(order.orderId)
            if existing is not None:
                if existing.isDone():
                    raise BacktestBrokerError("Cannot modify a completed order")
                existing.order = order
                existing.log.append(
                    ibi.TradeLogEntry(
                        self._event_time(), existing.orderStatus.status, "Modify"
                    )
                )
                self._submitted_at[order.orderId] = self._current_time
                existing.modifyEvent.emit(existing)
                self.orderModifyEvent.emit(existing)
                return existing

        order.orderId = self._next_order_id
        self._next_order_id += 1
        order.permId = self._next_perm_id
        self._next_perm_id += 1
        order.clientId = 0
        status = ibi.OrderStatus(
            orderId=order.orderId,
            status=ibi.OrderStatus.Submitted,
            filled=0.0,
            remaining=float(order.totalQuantity),
            permId=order.permId,
            parentId=order.parentId,
        )
        trade = ibi.Trade(
            contract=contract,
            order=order,
            orderStatus=status,
            log=[ibi.TradeLogEntry(self._event_time(), status.status)],
        )
        self._trades.append(trade)
        self._trades_by_order_id[order.orderId] = trade
        self._submitted_at[order.orderId] = self._current_time
        self.newOrderEvent.emit(trade)
        return trade

    def cancelOrder(
        self, order: ibi.Order, manualCancelOrderTime: str = ""
    ) -> ibi.Trade | None:
        """Cancel one working order immediately."""

        del manualCancelOrderTime
        trade = self._trades_by_order_id.get(order.orderId)
        if trade is None or trade.isDone():
            return trade
        trade.orderStatus.status = ibi.OrderStatus.Cancelled
        trade.orderStatus.remaining = trade.remaining()
        trade.log.append(
            ibi.TradeLogEntry(self._event_time(), ibi.OrderStatus.Cancelled)
        )
        trade.cancelEvent.emit(trade)
        trade.statusEvent.emit(trade)
        self.cancelOrderEvent.emit(trade)
        self.orderStatusEvent.emit(trade)
        trade.cancelledEvent.emit(trade)
        return trade

    def trades(self) -> list[ibi.Trade]:
        """Return every submitted simulated trade."""

        return list(self._trades)

    def openTrades(self) -> list[ibi.Trade]:
        """Return currently working simulated trades."""

        return [trade for trade in self._trades if trade.isActive()]

    def fills(self) -> list[ibi.Fill]:
        """Return every simulated fill in execution order."""

        return list(self._fills)

    def positions(self, account: str = "") -> list[ibi.Position]:
        """Return non-zero broker positions as real IB objects."""

        if account and account != self.account:
            return []
        return [
            ibi.Position(
                account=self.account,
                contract=ledger.contract,
                position=ledger.quantity,
                avgCost=ledger.average_price * ledger.multiplier,
            )
            for ledger in self._ledgers.values()
            if ledger.quantity
        ]

    async def reqPositionsAsync(self) -> list[ibi.Position]:
        """Return the current in-process position snapshot."""

        return self.positions()

    async def process_bar(
        self,
        contract: ibi.Contract,
        bar: ibi.BarData,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[ibi.Trade, ...]:
        """Fill eligible orders using one actual OHLC bar.

        Args:
            contract: Concrete contract whose session produced ``bar``.
            bar: Actual stored OHLC bar. It is not synthesized.
            metadata: Optional datastore metadata update for this contract.

        Returns:
            Trades completely filled on this bar, in deterministic order.

        Notes:
            Market orders execute at the bar open before intrabar stop/limit
            ambiguity. If an OCA stop and limit both touch later in the same
            bar, the stop is evaluated first.
        """

        key = _contract_key(contract)
        timestamp = _aware_datetime(bar.date)
        self._current_time = timestamp
        if metadata is not None:
            self.configure_contract(contract, metadata, details=self._details.get(key))
        if key not in self._metadata:
            raise BacktestBrokerError(
                f"No datastore metadata configured for {contract.localSymbol or contract}"
            )
        ledger = self._ledgers[key]
        ledger.last_price = _finite_number(bar.close, "bar.close")
        candidates = [
            trade
            for trade in self.openTrades()
            if trade.contract.conId == key and self._eligible(trade, timestamp)
        ]
        first_in_group = {
            group: min(
                trade.order.orderId
                for trade in candidates
                if trade.order.ocaGroup == group
            )
            for group in {trade.order.ocaGroup for trade in candidates}
            if group
        }
        candidates.sort(
            key=lambda trade: self._execution_priority(trade, first_in_group)
        )
        executed: list[ibi.Trade] = []
        filled_oca_groups: set[str] = set()
        for trade in candidates:
            group = trade.order.ocaGroup
            if group and group in filled_oca_groups:
                continue
            base_price = self._trigger_price(trade.order, bar)
            if base_price is None:
                continue
            await self._fill_trade(trade, base_price, timestamp)
            executed.append(trade)
            if group:
                filled_oca_groups.add(group)
                self._cancel_oca_peers(trade)
        self._update_ticker(contract, bar, timestamp)
        return tuple(executed)

    def mark_to_market(self, prices: Mapping[Any, float]) -> None:
        """Apply explicit final marks keyed by contract or ``conId``."""

        for contract_or_id, price in prices.items():
            key = (
                _contract_key(contract_or_id)
                if isinstance(contract_or_id, ibi.Contract)
                else int(contract_or_id)
            )
            if key not in self._ledgers:
                raise KeyError(f"Unknown backtest contract conId={key}")
            self._ledgers[key].last_price = _finite_number(price, "mark price")

    def result(self, *, initial_cash: float | None = None) -> BacktestResult:
        """Build an immutable marked account result.

        Args:
            initial_cash: Optional result-time override of the opening account
                value configured on this broker.
        """

        contracts = tuple(
            ContractResult(
                contract=ledger.contract,
                position=ledger.quantity,
                average_price=ledger.average_price,
                last_price=ledger.last_price,
                multiplier=ledger.multiplier,
                realized_pnl=ledger.realized_pnl,
                unrealized_pnl=ledger.unrealized_pnl,
                commission=ledger.commission,
                cash_change=ledger.cash_change,
            )
            for _, ledger in sorted(self._ledgers.items())
        )
        return BacktestResult(
            initial_cash=(self.initial_cash if initial_cash is None else initial_cash),
            contracts=contracts,
            fills=tuple(self._fill_records),
            orders=tuple(self._order_snapshot(trade) for trade in self._trades),
        )

    def _order_snapshot(self, trade: ibi.Trade) -> BacktestOrder:
        """Return immutable final evidence for one submitted order."""

        order = trade.order
        order_type = order.orderType.upper()
        return BacktestOrder(
            submitted_at=self._submitted_at[order.orderId],
            contract=trade.contract,
            order_id=order.orderId,
            perm_id=order.permId,
            action=order.action,
            order_type=order_type,
            quantity=float(order.totalQuantity),
            status=trade.orderStatus.status,
            filled=float(trade.orderStatus.filled),
            remaining=float(trade.orderStatus.remaining),
            limit_price=(float(order.lmtPrice) if order_type == "LMT" else None),
            stop_price=(float(order.auxPrice) if order_type == "STP" else None),
            oca_group=order.ocaGroup,
        )

    def _event_time(self) -> datetime:
        """Return simulation time or a deterministic pre-replay epoch."""

        return self._current_time or datetime(1970, 1, 1, tzinfo=timezone.utc)

    @staticmethod
    def _validate_order(order: ibi.Order) -> None:
        """Validate the deliberately narrow execution surface."""

        if not isinstance(order, ibi.Order):
            raise TypeError("order must be an ib_insync.Order")
        if order.action.upper() not in {"BUY", "SELL"}:
            raise ValueError("order.action must be BUY or SELL")
        quantity = _finite_number(order.totalQuantity, "order.totalQuantity")
        if quantity <= 0:
            raise ValueError("order.totalQuantity must be positive")
        order_type = order.orderType.upper()
        if order_type == "TRAIL":
            raise UnsupportedBacktestOrderError(
                "TRAIL orders are not supported by the experimental backtester"
            )
        adjusted = order.adjustedOrderType.upper()
        unset_fields = (
            order.triggerPrice == ibi.util.UNSET_DOUBLE
            and order.adjustedStopPrice == ibi.util.UNSET_DOUBLE
            and order.adjustedStopLimitPrice == ibi.util.UNSET_DOUBLE
            and order.adjustedTrailingAmount == ibi.util.UNSET_DOUBLE
        )
        if adjusted or not unset_fields:
            raise UnsupportedBacktestOrderError(
                "Adjustable orders are not supported by the experimental backtester"
            )
        if order.algoStrategy or order.algoParams:
            raise UnsupportedBacktestOrderError(
                "Broker algorithms and algo parameters are not supported by the "
                "experimental backtester"
            )
        if order.conditions:
            raise UnsupportedBacktestOrderError(
                "Conditional orders are not supported by the experimental backtester"
            )
        if order.whatIf:
            raise UnsupportedBacktestOrderError(
                "What-if orders are not supported by the experimental backtester"
            )
        parent_or_transmit = (
            order.parentId
            or order.parentPermId
            or order.autoCancelParent
            or not order.transmit
        )
        if parent_or_transmit:
            raise UnsupportedBacktestOrderError(
                "Parent-child and staged transmission orders are not supported by "
                "the experimental backtester"
            )
        if (
            order.goodAfterTime
            or order.goodTillDate
            or order.activeStartTime
            or order.activeStopTime
            or order.autoCancelDate
        ):
            raise UnsupportedBacktestOrderError(
                "Scheduled order activation and expiry are not supported by the "
                "experimental backtester"
            )
        if order.tif.upper() not in {"", "GTC"}:
            raise UnsupportedBacktestOrderError(
                f"Time-in-force {order.tif!r} is not supported; use GTC or omit tif"
            )
        advanced_execution = (
            order.allOrNone
            or order.minQty != ibi.util.UNSET_INTEGER
            or bool(order.displaySize)
            or order.hidden
            or order.blockOrder
            or order.sweepToFill
            or order.cashQty != ibi.util.UNSET_DOUBLE
            or bool(order.triggerMethod)
        )
        if advanced_execution:
            raise UnsupportedBacktestOrderError(
                "Advanced quantity, display, and trigger modifiers are not "
                "supported by the experimental backtester"
            )
        if order_type not in {"MKT", "LMT", "STP"}:
            raise UnsupportedBacktestOrderError(
                f"Order type {order.orderType!r} is not supported; "
                "supported types are MKT, LMT, and STP"
            )
        if order_type == "LMT":
            _finite_order_price(order.lmtPrice, "order.lmtPrice")
        elif order_type == "STP":
            _finite_order_price(order.auxPrice, "order.auxPrice")
        unsupported_fields = sorted(
            name
            for name in set(ibi.util.dataclassNonDefaults(order))
            - _SUPPORTED_ORDER_FIELDS
            if getattr(order, name) != getattr(_DEFAULT_ORDER, name)
        )
        if unsupported_fields:
            raise UnsupportedBacktestOrderError(
                "Unsupported non-default IB order field(s): "
                + ", ".join(unsupported_fields)
            )

    def _eligible(self, trade: ibi.Trade, timestamp: datetime) -> bool:
        """Return whether an order predates this actual contract bar."""

        submitted_at = self._submitted_at[trade.order.orderId]
        return submitted_at is None or timestamp > submitted_at

    @staticmethod
    def _execution_priority(
        trade: ibi.Trade, first_in_group: Mapping[str, int]
    ) -> tuple[int, int, int, int]:
        """Model bar open before ordered intrabar OCA ambiguity."""

        order = trade.order
        order_type = order.orderType.upper()
        phase = 0 if order_type == "MKT" else 1
        oca_priority = 0 if order.ocaGroup and order_type == "STP" else 1
        return (
            phase,
            first_in_group.get(order.ocaGroup, order.orderId),
            oca_priority,
            order.orderId,
        )

    @staticmethod
    def _trigger_price(order: ibi.Order, bar: ibi.BarData) -> float | None:
        """Return a gap-aware base fill price when the order touches ``bar``."""

        open_ = _finite_number(bar.open, "bar.open")
        high = _finite_number(bar.high, "bar.high")
        low = _finite_number(bar.low, "bar.low")
        if low > high:
            raise ValueError("bar.low must not exceed bar.high")
        action = order.action.upper()
        order_type = order.orderType.upper()
        if order_type == "MKT":
            return open_
        if order_type == "LMT":
            limit = _finite_order_price(order.lmtPrice, "order.lmtPrice")
            if action == "BUY" and low <= limit:
                return min(open_, limit)
            if action == "SELL" and high >= limit:
                return max(open_, limit)
            return None
        stop = _finite_order_price(order.auxPrice, "order.auxPrice")
        if action == "BUY" and high >= stop:
            return max(open_, stop)
        if action == "SELL" and low <= stop:
            return min(open_, stop)
        return None

    async def _fill_trade(
        self, trade: ibi.Trade, base_price: float, timestamp: datetime
    ) -> None:
        """Create one full fill and emit IB-compatible callbacks."""

        order = trade.order
        metadata = self._metadata[trade.contract.conId]
        tick = _metadata_number(metadata, ("minTick", "min_tick"), 0.0)
        adverse = tick * self.slippage_ticks
        price = (
            base_price + adverse
            if order.action.upper() == "BUY"
            else base_price - adverse
        )
        if order.orderType.upper() == "LMT":
            limit = _finite_order_price(order.lmtPrice, "order.lmtPrice")
            price = (
                min(price, limit)
                if order.action.upper() == "BUY"
                else max(price, limit)
            )
        quantity = float(order.totalQuantity)
        signed_quantity = quantity if order.action.upper() == "BUY" else -quantity
        ledger = self._ledgers[trade.contract.conId]
        realized = ledger.apply(signed_quantity, price)
        commission_rate = _metadata_number(metadata, ("commission",), 0.0)
        commission = commission_rate * quantity
        ledger.commission += commission
        if trade.contract.secType == "FUT":
            ledger.cash_change += realized - commission
        else:
            ledger.cash_change -= (
                signed_quantity * price * ledger.multiplier + commission
            )

        execution_id = f"BT-{self._next_execution_id}"
        self._next_execution_id += 1
        execution = ibi.Execution(
            execId=execution_id,
            time=timestamp,
            acctNumber=self.account,
            exchange=trade.contract.exchange,
            side="BOT" if order.action.upper() == "BUY" else "SLD",
            shares=quantity,
            price=price,
            permId=order.permId,
            clientId=order.clientId,
            orderId=order.orderId,
            cumQty=quantity,
            avgPrice=price,
        )
        empty_report = ibi.CommissionReport()
        fill = ibi.Fill(
            contract=trade.contract,
            execution=execution,
            commissionReport=empty_report,
            time=timestamp,
        )
        trade.fills.append(fill)
        self._fills.append(fill)
        trade.log.append(
            ibi.TradeLogEntry(
                timestamp,
                trade.orderStatus.status,
                f"Fill {quantity}@{price}",
            )
        )
        self.execDetailsEvent.emit(trade, fill)
        trade.fillEvent.emit(trade, fill)
        await asyncio.sleep(0)

        trade.orderStatus.status = ibi.OrderStatus.Filled
        trade.orderStatus.filled = quantity
        trade.orderStatus.remaining = 0.0
        trade.orderStatus.avgFillPrice = price
        trade.orderStatus.lastFillPrice = price
        trade.log.append(ibi.TradeLogEntry(timestamp, ibi.OrderStatus.Filled))
        self.orderStatusEvent.emit(trade)
        trade.statusEvent.emit(trade)
        trade.filledEvent.emit(trade)
        await asyncio.sleep(0)

        report = ibi.CommissionReport(
            execId=execution_id,
            commission=commission,
            currency=trade.contract.currency,
            realizedPNL=realized,
        )
        fill_with_report = fill._replace(commissionReport=report)
        trade.fills[-1] = fill_with_report
        self._fills[-1] = fill_with_report
        self.commissionReportEvent.emit(trade, fill_with_report, report)
        trade.commissionReportEvent.emit(trade, fill_with_report, report)
        self._fill_records.append(
            BacktestFill(
                time=timestamp,
                contract=trade.contract,
                order_id=order.orderId,
                perm_id=order.permId,
                execution_id=execution_id,
                action=order.action,
                order_type=order.orderType,
                quantity=quantity,
                price=price,
                commission=commission,
                realized_pnl=realized,
            )
        )
        await asyncio.sleep(0)

    def _cancel_oca_peers(self, filled_trade: ibi.Trade) -> None:
        """Cancel every other working order in a filled OCA group."""

        group = filled_trade.order.ocaGroup
        if not group:
            return
        for trade in tuple(self.openTrades()):
            if trade is not filled_trade and trade.order.ocaGroup == group:
                self.cancelOrder(trade.order)

    def _update_ticker(
        self, contract: ibi.Contract, bar: ibi.BarData, timestamp: datetime
    ) -> None:
        """Update the contract ticker without publishing a strategy tick."""

        ticker = self.reqMktData(contract)
        ticker.time = timestamp
        ticker.open = bar.open
        ticker.high = bar.high
        ticker.low = bar.low
        ticker.close = bar.close
        ticker.last = bar.close


def _finite_order_price(value: Any, name: str) -> float:
    """Validate an explicitly configured IB order price."""

    result = _finite_number(value, name)
    if result == ibi.util.UNSET_DOUBLE:
        raise ValueError(f"{name} must be set")
    return result


def _contract_matches(blueprint: ibi.Contract, contract: ibi.Contract) -> bool:
    """Return whether every specified blueprint identity field matches."""

    if blueprint.conId:
        return blueprint.conId == contract.conId
    fields = ibi.util.dataclassNonDefaults(blueprint)
    if blueprint.secType == "CONTFUT":
        if contract.secType != "FUT":
            return False
        fields = {
            name: value
            for name, value in fields.items()
            if name
            in {
                "symbol",
                "multiplier",
                "exchange",
                "primaryExchange",
                "currency",
                "tradingClass",
                "secIdType",
                "secId",
                "issuerId",
            }
        }
    for name, expected in fields.items():
        if name in {"conId", "includeExpired"}:
            continue
        if expected != getattr(contract, name):
            return False
    return True


__all__ = [
    "BacktestBrokerError",
    "SimulatedIB",
    "UnsupportedBacktestOrderError",
]
