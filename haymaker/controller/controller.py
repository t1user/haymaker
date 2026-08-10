"""Broker submission, execution accounting, and reconciliation boundary."""

from __future__ import annotations

import asyncio
import datetime
import itertools
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import partial
from typing import Any, Literal, Self

import eventkit as ev  # type: ignore
import ib_insync as ibi

from haymaker import misc
from haymaker.base import Atom
from haymaker.book import OrderInfo, PositionState
from haymaker.components.messages import PositionTarget, StandardOrderRole
from haymaker.supervisor.codes import SUPERVISOR_OWNED_BROKER_CODES
from haymaker.trader import Trader

from .future_roller import FutureRoller
from .sync_brackets import MissingBracketsPolicy
from .sync_coordinator import SyncBrokenStateError, SyncCoordinator
from .terminator import Terminator

log = logging.getLogger(__name__)


class ControllerError(ValueError):
    """Raised when Controller policy configuration is invalid."""


class SyncOutcome(Enum):
    """Result of one Controller synchronization cycle."""

    OK = auto()
    FAILED = auto()
    ABORTED = auto()

    def __bool__(self) -> bool:
        return self is SyncOutcome.OK


def _broker_messages_to_ignore(
    codes: tuple[int, ...] | list[int],
) -> tuple[int, ...]:
    return tuple(sorted(set(codes) | SUPERVISOR_OWNED_BROKER_CODES))


@dataclass(eq=False)
class Controller(Atom):
    """Own broker order submission, fill accounting, and reconciliation.

    ExecutionModels call :meth:`trade` with explicit role, stable model name,
    and optional one-to-one attribution. Controller registers complete Trade
    evidence in Book immediately after broker submission, applies Fill and
    commission callbacks, writes the Book-owned blotter, and verifies accepted
    absolute targets after a delay.
    """

    trader: Trader
    cold_start: bool = True
    reset: bool = False
    zero: bool = False
    nuke: bool = False
    log_order_events: bool = False
    sync_frequency: int = 0
    health_check_frequency: int = 0
    execution_verification_delay: int = 0
    execution_verification_max_retries: int = 5
    broker_request_timeout: int = 10
    sync_max_attempts: int = 3
    sync_resync_delay: float = 1
    cancel_unknown_trades: bool = False
    missing_brackets: MissingBracketsPolicy = "ignore"
    ignore_errors: tuple[int, ...] | list[int] = field(default_factory=tuple)
    future_roll_time: tuple[int, int] | None = None
    future_roll_policies: dict[str, bool] = field(default_factory=dict)
    health_check_observables: list[list[Callable[[], bool]]] = field(
        default_factory=list
    )
    _hold: bool = field(default=True, repr=False)
    _sync_timer: ev.Timer | None = field(default=None, repr=False)
    _health_check_timer: ev.Timer | None = field(default=None, repr=False)
    _order_loggers: OrderLoggers | None = field(default=None, repr=False)
    _health_check_functions: list[Callable[[], bool]] = field(
        default_factory=list, repr=False
    )
    _health_check_triggers: list[str] = field(default_factory=list, repr=False)
    _new_position_lock: bool = False
    _trading_disabled: bool = False
    _restart_before_correction: bool = True
    _sync_abort_event: asyncio.Event | None = field(default=None, repr=False)
    _future_roll_timer: ev.Event | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        trader: Trader,
        health_check_observables: list[list[Callable[[], bool]]] | None = None,
    ) -> Self:
        """Construct a Controller from merged runtime configuration."""

        options = dict(values)
        startup = options.pop("startup", {})
        if not isinstance(startup, Mapping):
            raise TypeError("controller.startup must be a mapping")
        options.update(startup)
        roll_time = options.get("future_roll_time")
        if isinstance(roll_time, list):
            options["future_roll_time"] = tuple(roll_time)
        return cls(
            trader=trader,
            health_check_observables=health_check_observables or [],
            **options,
        )

    def __post_init__(self) -> None:
        Atom.__init__(self)
        self.ignore_errors = _broker_messages_to_ignore(self.ignore_errors)
        if self.missing_brackets not in ("ignore", "warn", "remove"):
            raise ControllerError(
                "controller.missing_brackets must be ignore, warn, or remove"
            )
        self.ib.execDetailsEvent.connect(self.onExecDetailsEvent, self._log_event_error)
        self.ib.newOrderEvent.connect(self.onNewOrderEvent, self._log_event_error)
        self.ib.orderStatusEvent.connect(self.onOrderStatusEvent, self._log_event_error)
        self.ib.orderStatusEvent.connect(self.log_order_status, self._log_event_error)
        self.ib.errorEvent.connect(self.onErrEvent, self._log_event_error)
        self.ib.commissionReportEvent.connect(
            self.onCommissionReport, self._log_event_error
        )
        if self.log_order_events:
            self._order_loggers = OrderLoggers(self.ib)
        self.set_hold()
        if missing := self.verify_have_contracts_for_positions():
            log.critical("No qualified contracts for open position: %s", missing)

    def __str__(self) -> str:
        future_roll = "off"
        if self.future_roll_time is not None:
            hour, minute = self.future_roll_time
            future_roll = f"{hour:02}:{minute:02} UTC"
        return (
            f"Controller<sync={self.sync_frequency}s, "
            f"health_check={self.health_check_frequency}s, "
            f"future_roll={future_roll}, "
            f"missing_brackets={self.missing_brackets}>"
        )

    def onStart(self, data: Any, source: Atom | None = None) -> None:
        """Ignore graph startup because Controller lifecycle is runtime-owned."""

    async def onData(
        self,
        target: PositionTarget,
        execution_model_name: str | None = None,
    ) -> None:
        """Verify one accepted target after its model has submitted work."""

        if not isinstance(target, PositionTarget):
            raise TypeError("Controller accepts only PositionTarget")
        if not execution_model_name:
            raise ValueError("execution_model_name is required")
        await asyncio.sleep(self.execution_verification_delay)
        if await self.verify_target_integrity(target, execution_model_name):
            self.verify_position_with_broker(target.contract)

    def set_health_check(self, func: Callable[[], bool]) -> None:
        self._health_check_functions.append(func)

    def set_sync_abort_event(self, event: asyncio.Event) -> None:
        self._sync_abort_event = event

    def run_health_check(self, *args: object) -> None:
        for func in itertools.chain(
            itertools.chain(*self.health_check_observables),
            self._health_check_functions,
        ):
            if not func() and func.__name__ not in self._health_check_triggers:
                log.critical("Health check failure for checker: %s", func.__name__)
                self._health_check_triggers.append(func.__name__)

    def verify_have_contracts_for_positions(self) -> list[ibi.Contract]:
        return [
            position.contract
            for position in self.ib.positions()
            if position.contract not in self.contract_registry.all_contracts
        ]

    def set_hold(self) -> None:
        self._hold = True
        log.debug("hold set")

    def release_hold(self) -> None:
        if self._hold:
            self._hold = False
            log.debug("hold released")

    def set_future_roll_policies(self, policies: Mapping[str, bool]) -> None:
        self.future_roll_policies = dict(policies)

    async def run(self) -> bool:
        """Restore Book, reconcile broker state, and arm runtime timers."""

        self._ensure_runtime_timers_started()
        self.set_hold()
        if self.nuke:
            await self.run_nuke()
        if self.cold_start:
            log.debug("Starting cold; Book state will not be loaded.")
        else:
            try:
                await self.book.read_from_store()
                self.cold_start = True
            except Exception:
                log.exception("Book state restoration failed.")
                self.disable_trading("state store read failed")
                return False
        outcome = await self.sync()
        if not outcome:
            if outcome is not SyncOutcome.ABORTED:
                log.critical("Controller startup sync failed. Trading disabled.")
            return False
        if self.zero:
            self.clear_records()
            self.zero = False
        if self.reset:
            if not await self.execute_stops_and_close_positions():
                self.disable_trading("account reset did not complete")
                return False
            self.book.clear_state()
            self.reset = False
        self._restart_before_correction = True
        return True

    def _ensure_runtime_timers_started(self) -> None:
        if self.sync_frequency and self._sync_timer is None:
            self._sync_timer = ev.Timer(self.sync_frequency)
            self._sync_timer.connect(self.sync, error=self._log_event_error)
        if self.health_check_frequency and self._health_check_timer is None:
            self._health_check_timer = ev.Timer(self.health_check_frequency)
            self._health_check_timer.connect(
                self.run_health_check, error=self._log_event_error
            )
        if self._future_roll_timer is None:
            self.schedule_future_roll()

    def roll_futures(self, *args: object) -> None:
        FutureRoller(self, self.future_roll_policies).roll()

    def schedule_future_roll(self) -> None:
        if self.future_roll_time is None:
            return
        if self._future_roll_timer is not None:
            log.warning("Future roll already scheduled; ignoring duplicate request.")
            return
        hour, minute = self.future_roll_time
        roll_time = datetime.time(hour=hour, minute=minute, tzinfo=datetime.UTC)
        self._future_roll_timer = ev.Event.timerange(
            start=roll_time, step=datetime.timedelta(days=1)  # type: ignore
        )
        self._future_roll_timer += self.roll_futures

    async def sync(self, *args: object) -> SyncOutcome:
        """Run reconciliation unless supervisor marks the connection unavailable."""

        abort_event = self._sync_abort_event
        if abort_event is None:
            return await self._sync()
        if abort_event.is_set():
            return SyncOutcome.ABORTED
        sync_task = asyncio.create_task(self._sync(), name="controller-sync")
        abort_task = asyncio.create_task(
            abort_event.wait(), name="controller-sync-abort"
        )
        try:
            done, _ = await asyncio.wait(
                (sync_task, abort_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if abort_task in done and abort_event.is_set():
                sync_task.cancel()
                await asyncio.gather(sync_task, return_exceptions=True)
                return SyncOutcome.ABORTED
            abort_task.cancel()
            await asyncio.gather(abort_task, return_exceptions=True)
            return await sync_task
        finally:
            for task in (sync_task, abort_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(sync_task, abort_task, return_exceptions=True)

    async def _sync(self) -> SyncOutcome:
        if not self.ib.isConnected():
            return SyncOutcome.FAILED
        for attempt in range(1, self.sync_max_attempts + 1):
            coordinator = SyncCoordinator(self, self._restart_before_correction)
            try:
                if await coordinator.run():
                    self._restart_before_correction = False
                    return SyncOutcome.OK
            except SyncBrokenStateError as exc:
                self.disable_trading(str(exc))
                return SyncOutcome.FAILED
            if attempt < self.sync_max_attempts:
                await asyncio.sleep(self.sync_resync_delay)
                if coordinator.request_restart:
                    self._restart_before_correction = False
                    self.ib.disconnect()
        if self._sync_abort_event is not None and self._sync_abort_event.is_set():
            return SyncOutcome.ABORTED
        self.disable_trading("sync did not converge")
        return SyncOutcome.FAILED

    def trade(
        self,
        contract: ibi.Contract,
        order: ibi.Order,
        *,
        role: str,
        execution_model_name: str,
        source_key: str | None = None,
        position_id: str | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> ibi.Trade | None:
        """Submit and immediately register one fully attributed broker order."""

        if self._trading_disabled:
            log.debug(
                "Trade suppressed while trading is disabled: %s %s %s",
                execution_model_name,
                role,
                contract.localSymbol or contract.symbol,
            )
            return None
        if role == StandardOrderRole.OPEN and self._new_position_lock:
            log.debug("New-position lock suppressed trade for %s", source_key)
            return None
        if not self.book.verify_for_rejections(execution_model_name):
            return None
        if not self.verify_market_open(contract):
            return None
        return self._submit_registered_trade(
            contract,
            order,
            role=role,
            execution_model_name=execution_model_name,
            source_key=source_key,
            position_id=position_id,
            params=params,
        )

    def _submit_registered_trade(
        self,
        contract: ibi.Contract,
        order: ibi.Order,
        *,
        role: str,
        execution_model_name: str,
        source_key: str | None = None,
        position_id: str | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> ibi.Trade:
        """Submit and register an order after caller-specific policy checks."""

        trade = self.trader.trade(contract, order)
        self.register_order(
            trade,
            role=str(role),
            execution_model_name=execution_model_name,
            source_key=source_key,
            position_id=position_id,
            params=params,
        )
        trade.filledEvent += partial(
            self.log_trade,
            reason=str(role),
            source_key=source_key or execution_model_name,
        )
        return trade

    def register_order(
        self,
        trade: ibi.Trade,
        *,
        role: str,
        execution_model_name: str,
        source_key: str | None = None,
        position_id: str | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> OrderInfo:
        """Persist complete submission evidence before later broker callbacks."""

        info = OrderInfo(
            trade=trade,
            role=role,
            submitted_at=datetime.datetime.now(datetime.timezone.utc),
            execution_model_name=execution_model_name,
            source_key=source_key,
            position_id=position_id,
            params=params or {},
        )
        self.book.save_order(info)
        log.debug(
            "%s orderId=%s permId=%s registered for %s",
            trade.order.orderType,
            trade.order.orderId,
            trade.order.permId,
            trade.contract.localSymbol or trade.contract.symbol,
        )
        return info

    def verify_market_open(self, contract: ibi.Contract) -> bool:
        details = self.contract_registry.get_details(contract)
        if details is None:
            log.warning(
                "Missing details for %s; market hours cannot be verified.",
                contract.localSymbol or contract.symbol or contract,
            )
            return True
        if details.is_open():
            return True
        log.error("Attempt to place an order while market is closed: %s", contract)
        return False

    def cancel(self, trade: ibi.Trade) -> ibi.Trade | None:
        """Cancel one live Trade through the broker gateway."""

        return self.trader.cancel(trade)

    async def onNewOrderEvent(self, trade: ibi.Trade) -> None:
        """Report a broker order that lacks immediate Book registration."""

        await asyncio.sleep(0)
        if trade.order.orderId < 0:
            return
        if self.book.order_by_id(trade.order.orderId) is None:
            log.critical(
                "Unknown broker trade: %s %s",
                trade.order,
                trade.contract.symbol,
            )

    def onOrderStatusEvent(self, trade: ibi.Trade) -> None:
        """Persist status changes and rebind current live Trade objects."""

        if self._hold:
            return
        info = self.book.order_by_id(
            trade.order.orderId
        ) or self.book.order_by_perm_id(trade.order.permId)
        if info is None:
            if not trade.order.orderId:
                log.warning(
                    "Skipping unknown order status with orderId 0, permId=%s",
                    trade.order.permId,
                )
                return
            info = self._unknown_order_info(trade)
        else:
            info.trade = trade
        self.book.save_order(info)

    def register_position(self, order_info: OrderInfo, fill: ibi.Fill) -> None:
        """Apply one execution idempotently to Book projections."""

        if isinstance(order_info.trade.contract, ibi.Bag):
            log.debug("Combo fill retained as order evidence without projection.")
            self.book.apply_fill(order_info.trade, fill)
            return
        try:
            changed = self.book.apply_fill(order_info.trade, fill)
        except (KeyError, ValueError):
            log.exception(
                "Cannot apply fill for orderId=%s", order_info.trade.order.orderId
            )
            return
        if not changed:
            log.warning(
                "Abandoned duplicate fill execId=%s orderId=%s",
                fill.execution.execId,
                order_info.trade.order.orderId,
            )

    async def onExecDetailsEvent(
        self, trade: ibi.Trade, fill: ibi.Fill
    ) -> None:
        """Match, persist, and account one broker execution callback."""

        info = (
            self.assign_manual_trade(trade)
            or self.book.order_by_id(trade.order.orderId)
            or self.match_by_permId(trade, fill)
            or self.assign_unknown_trade(trade)
        )
        if info is not None:
            info.trade = trade
            self.register_position(info, fill)

    async def onCommissionReport(
        self,
        trade: ibi.Trade,
        fill: ibi.Fill,
        report: ibi.CommissionReport,
    ) -> None:
        """Persist final commission evidence and optionally write a blotter row."""

        if self._hold or not trade.order.orderId:
            return
        await asyncio.sleep(0)
        info = self.book.order_by_id(
            trade.order.orderId
        ) or self.book.order_by_perm_id(trade.order.permId)
        if info is None:
            log.error(
                "Commission report for unknown orderId=%s", trade.order.orderId
            )
            return
        if not self.book.update_commission(trade, fill, report):
            log.warning(
                "Commission report has no normalized fill evidence: "
                "execId=%s orderId=%s",
                fill.execution.execId,
                trade.order.orderId,
            )
            info.trade = trade
            self.book.save_order(info)
        blotter = self.book.blotter
        if blotter is None:
            return
        kwargs = {
            "source_key": info.source_key,
            "position_id": info.position_id,
            "role": info.role,
            "execution_model_name": info.execution_model_name,
            "params": ibi.util.tree(dict(info.params)),
        }
        try:
            blotter.log_commission(trade, fill, report, **kwargs)
        except Exception:
            log.exception("Blotter write failed for orderId=%s", info.orderId)

    async def verify_target_integrity(
        self, target: PositionTarget, execution_model_name: str
    ) -> bool:
        """Check Book convergence when this is still the latest target.

        Returns:
            ``True`` when the target remained current and was checked, or
            ``False`` when a newer target superseded it.
        """

        retries = 0
        if not self._target_is_latest(target, execution_model_name):
            return False
        while self._active_target_orders(target, execution_model_name):
            if retries >= self.execution_verification_max_retries:
                break
            retries += 1
            await asyncio.sleep(self.execution_verification_delay)
            if not self._target_is_latest(target, execution_model_name):
                return False
        if not self._target_is_latest(target, execution_model_name):
            return False
        position = (
            self.book.position_state(target.source_key)
            if target.source_key is not None
            else None
        )
        actual = (
            position.quantity
            if position is not None
            else self.book.aggregate_quantity(target.contract)
        )
        if actual != target.target_quantity:
            log.error(
                "Target not achieved for %s: target=%s actual=%s",
                target.source_key or target.contract.localSymbol,
                target.target_quantity,
                actual,
            )
        return True

    def _active_target_orders(
        self, target: PositionTarget, execution_model_name: str
    ) -> tuple[OrderInfo, ...]:
        """Return only orders whose completion can converge this target."""

        roles = {
            StandardOrderRole.OPEN,
            StandardOrderRole.CLOSE,
            StandardOrderRole.TARGET_ADJUSTMENT,
        }
        return tuple(
            info
            for info in self.book.active_orders(
                source_key=target.source_key,
                contract=target.contract,
                execution_model_name=execution_model_name,
            )
            if info.role in roles
        )

    def _target_is_latest(
        self, target: PositionTarget, execution_model_name: str
    ) -> bool:
        """Return whether Book still identifies this exact accepted setpoint."""

        if target.source_key is not None:
            position_state = self.book.position_state(target.source_key)
            return (
                position_state is not None
                and position_state.execution_model_name == execution_model_name
                and position_state.contract is not None
                and position_state.contract.conId == target.contract.conId
                and position_state.target_quantity == target.target_quantity
                and position_state.target_created_at == target.created_at
            )
        target_state = self.book.latest_target_for_contract(target.contract)
        return (
            target_state is not None
            and target_state.execution_model_name == execution_model_name
            and target_state.target_quantity == target.target_quantity
            and target_state.target_created_at == target.created_at
        )

    def verify_position_with_broker(self, contract: ibi.Contract) -> None:
        """Compare aggregate logical Book quantity with the broker position."""

        logical = self.book.aggregate_quantity(contract)
        broker = self.trader.position_for_contract(contract)
        if logical != broker:
            log.error(
                "Wrong aggregate position for %s: logical=%s broker=%s",
                contract,
                logical,
                broker,
            )

    def match_by_permId(
        self, trade: ibi.Trade, fill: ibi.Fill
    ) -> OrderInfo | None:
        """Find and rebind an order using broker permanent id."""

        info = self.book.order_by_perm_id(trade.order.permId)
        if info is not None:
            if not trade.order.orderId:
                trade.order.orderId = info.orderId
            info.trade = trade
            self.book.save_order(info)
            log.debug(
                "Matched execId=%s by permId=%s to orderId=%s",
                fill.execution.execId,
                trade.order.permId,
                info.orderId,
            )
        return info

    def _source_for_unknown_trade(self, trade: ibi.Trade) -> str | None:
        """Attribute an unknown trade only when one logical position is clear."""

        if not trade.contract.conId:
            return None
        states = self.book.positions_for_contract(trade.contract)
        if len(states) == 1:
            return states[0].source_key
        return None

    def _unknown_order_info(
        self,
        trade: ibi.Trade,
        *,
        role: StandardOrderRole = StandardOrderRole.UNKNOWN,
    ) -> OrderInfo:
        if not trade.order.orderId:
            raise ValueError("Cannot register unknown Trade with orderId 0")
        source_key = self._source_for_unknown_trade(trade)
        state = (
            self.book.position_state(source_key) if source_key is not None else None
        )
        return OrderInfo(
            trade=trade,
            role=str(role),
            submitted_at=datetime.datetime.now(datetime.timezone.utc),
            execution_model_name=(
                state.execution_model_name
                if state is not None
                else str(role).lower()
            ),
            source_key=source_key,
            position_id=state.position_id if state is not None else None,
        )

    def assign_manual_trade(self, trade: ibi.Trade) -> OrderInfo | None:
        """Register a negative-orderId manual broker trade."""

        if trade.order.orderId >= 0:
            return None
        existing = self.book.order_by_id(trade.order.orderId)
        if existing is not None:
            return existing
        return self.book.save_order(
            self._unknown_order_info(
                trade, role=StandardOrderRole.MANUAL
            )
        )

    def assign_unknown_trade(self, trade: ibi.Trade) -> OrderInfo | None:
        """Register an unattributed non-manual broker trade."""

        if not trade.order.orderId:
            log.error(
                "Cannot persist unknown orderId=0 trade with permId=%s",
                trade.order.permId,
            )
            return None
        log.critical("Unknown broker trade: %s", trade)
        return self.book.save_order(self._unknown_order_info(trade))

    def cancel_orders_for_source(self, source_key: str) -> None:
        """Cancel every working order attributed to one source."""

        for info in self.book.active_orders(source_key=source_key):
            self.cancel(info.trade)

    def close_position_for_source(
        self, source_key: str, role: str = StandardOrderRole.LIQUIDATION
    ) -> None:
        """Submit one attributed market order to flatten a logical source."""

        state = self.book.position_state(source_key)
        if state is None or not state.quantity or state.contract is None:
            log.error("Attempt to close zero or unknown source %s", source_key)
            return
        self.trade(
            state.contract,
            ibi.MarketOrder(
                "BUY" if state.quantity < 0 else "SELL",
                abs(state.quantity),
            ),
            role=role,
            execution_model_name=state.execution_model_name,
            source_key=source_key,
            position_id=state.position_id,
        )

    def disable_trading(self, reason: str) -> None:
        if not self._trading_disabled:
            self._trading_disabled = True
            log.critical("Trading disabled: %s", reason)

    def lock_new_positions(self) -> None:
        log.error("Emergency lock for new positions.")
        self._new_position_lock = True

    def log_order_status(self, trade: ibi.Trade) -> None:
        if self._hold:
            return
        if trade.order.orderId < 0:
            log.warning("Manual trade status update: %s", trade.orderStatus)
        elif trade.isDone():
            log.debug(
                "%s order %s done %s",
                trade.contract.symbol,
                trade.order.orderId,
                trade.orderStatus.status,
            )
        else:
            log.info(
                "%s order %s status %s",
                trade.contract.symbol,
                trade.order.orderId,
                trade.orderStatus.status,
            )

    @staticmethod
    def log_trade(
        trade: ibi.Trade, reason: str = "", source_key: str = ""
    ) -> None:
        log.info(
            "%s trade filled: %s %s %s@%s --> %s orderId=%s permId=%s",
            reason,
            trade.contract.localSymbol,
            trade.order.action,
            trade.filled(),
            misc.trade_fill_price(trade),
            source_key,
            trade.order.orderId,
            trade.order.permId,
        )

    def onErrEvent(
        self,
        reqId: int,
        errorCode: int,
        errorString: str,
        contract: ibi.Contract,
    ) -> None:
        """Log broker messages and count genuine order rejections."""

        info = self.book.order_by_id(reqId)
        model_name = info.execution_model_name if info is not None else ""
        role = info.role if info is not None else ""
        order = info.trade.order if info is not None else ""
        context = f"{contract=}, {model_name} | {role} | {order}"
        if errorCode == 201:
            log.critical("ORDER REJECTED: %s errorCode=%s, %s", errorString, errorCode, context)
            self.book.register_rejected_order(model_name)
        elif errorCode == 202 and "YOUR ORDER IS NOT ACCEPTED" in errorString:
            log.error("ORDER NOT ACCEPTED: %s, %s", errorString, context)
        elif errorCode in self.ignore_errors:
            return
        elif errorCode == 202:
            log.debug("Broker message %s: %s %s", errorCode, errorString, context)
        elif errorCode in (165, 321, 322, 323):
            log.debug("Broker message %s: %s %s", errorCode, errorString, context)
        elif errorCode == 10141 or errorCode < 400:
            log.error("Broker message %s: %s %s", errorCode, errorString, context)
        else:
            log.debug("Broker message %s: %s %s", errorCode, errorString, context)

    async def execute_stops_and_close_positions(self) -> bool:
        """Run an explicit account reset and report whether it completed."""

        return await Terminator(self).run()

    def clear_records(self) -> None:
        self.book.clear_state()

    async def close_positions(self) -> None:
        for position in self.ib.positions():
            await self.ib.qualifyContractsAsync(position.contract)
            states = self.book.positions_for_contract(position.contract)
            state = states[0] if len(states) == 1 else None
            self._submit_registered_trade(
                position.contract,
                ibi.MarketOrder(
                    "BUY" if position.position < 0 else "SELL",
                    abs(position.position),
                ),
                role=StandardOrderRole.LIQUIDATION,
                execution_model_name=(
                    state.execution_model_name
                    if state is not None
                    else self.book.affinity_for_contract(position.contract)
                    or "nuke_liquidation"
                ),
                source_key=state.source_key if state is not None else None,
                position_id=state.position_id if state is not None else None,
            )

    async def run_nuke(self) -> None:
        self.ib.reqGlobalCancel()
        await self.close_positions()
        self.disable_trading("self nuke requested")
        log.critical("Emergency account liquidation requested.")


class OrderLoggers:
    """Optional detailed broker order-event loggers."""

    def __init__(self, ib: ibi.IB) -> None:
        self.ib = ib
        ib.cancelOrderEvent += self.log_cancel
        ib.orderModifyEvent += self.log_modification

    @staticmethod
    def log_cancel(trade: ibi.Trade) -> None:
        log.info(
            "%s order %s %s cancelled",
            trade.order.orderType,
            trade.order.action,
            trade.remaining(),
        )

    @staticmethod
    def log_modification(trade: ibi.Trade) -> None:
        log.debug("Order modified: %s", trade.order)

    def __repr__(self) -> str:
        return f"OrderLoggers({self.ib})"
