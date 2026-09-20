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
from haymaker.book import OrderInfo
from haymaker.components.messages import PositionTarget, StandardOrderRole
from haymaker.supervisor.codes import SUPERVISOR_OWNED_BROKER_CODES
from haymaker.trader import Trader

from .future_roller import FutureRoller
from .reset import EmergencyReset, Reset
from .sync_brackets import MissingBracketsPolicy
from .sync_coordinator import SyncBrokenStateError, SyncCoordinator

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

    ``position_mismatch_policy`` defaults to ``"fail"``: unexplained broker
    position differences disable trading without rewriting Book positions.
    ``"correct"`` opts into inferred one-to-one position/target corrections.
    """

    trader: Trader
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
    position_mismatch_policy: Literal["correct", "fail"] = "fail"
    cancel_unknown_trades: bool = False
    missing_brackets: MissingBracketsPolicy = "ignore"
    ignore_errors: tuple[int, ...] | list[int] = field(default_factory=tuple)
    future_roll_time: tuple[int, int] | None = None
    future_roll_policies: dict[str, bool] = field(default_factory=dict)
    health_check_observables: list[list[Callable[[], bool]]] = field(
        default_factory=list
    )
    _broker_ready: bool = field(default=False, init=False, repr=False)
    _broker_account: str | None = field(default=None, init=False, repr=False)
    _sync_timer: ev.Timer | None = field(default=None, repr=False)
    _health_check_timer: ev.Timer | None = field(default=None, repr=False)
    _order_loggers: OrderLoggers | None = field(default=None, repr=False)
    _health_check_functions: list[Callable[[], bool]] = field(
        default_factory=list, repr=False
    )
    _health_check_triggers: dict[int, Callable[[], bool]] = field(
        default_factory=dict, repr=False
    )
    _new_position_lock: bool = False
    _trading_disabled: bool = False
    _trading_disabled_event: asyncio.Event = field(
        default_factory=asyncio.Event, init=False, repr=False
    )
    _sync_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock, init=False, repr=False
    )
    _restart_before_correction: bool = True
    _sync_abort_event: asyncio.Event | None = field(default=None, repr=False)
    _future_roll_timer: ev.Event | None = field(default=None, init=False, repr=False)
    _protection_recovery: dict[str, Callable[[], None]] = field(
        default_factory=dict, init=False, repr=False
    )
    future_roller: FutureRoller = field(init=False, repr=False)

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
        self.future_roller = FutureRoller(self)
        self.ib.disconnectedEvent.connect(self.suspend_broker_work)
        self.ignore_errors = _broker_messages_to_ignore(self.ignore_errors)
        if self.position_mismatch_policy not in ("correct", "fail"):
            raise ControllerError(
                "controller.position_mismatch_policy must be correct or fail"
            )
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
            f"position_mismatch_policy={self.position_mismatch_policy}, "
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
            contracts = (
                (target.contract,)
                if target.source_key is None
                else self._episode_contracts(target)
            )
            for contract in contracts or (target.contract,):
                self.verify_position_with_broker(contract)

    def _episode_contracts(self, target: PositionTarget) -> tuple[ibi.Contract, ...]:
        """Verify held execution identity, not a CLOSE message's destination."""
        state = (
            self.book.positions.for_source(target.source_key)
            if target.source_key is not None
            else None
        )
        return (state.contract,) if state is not None and state.contract else ()

    def set_health_check(self, func: Callable[[], bool]) -> None:
        """Register a health check; report each failure episode independently."""
        if not callable(func):
            raise TypeError("health check must be callable")
        self._health_check_functions.append(func)

    def set_sync_abort_event(self, event: asyncio.Event) -> None:
        self._sync_abort_event = event

    def run_health_check(self, *args: object) -> None:
        """Isolate checker failures and reset suppression after a successful check."""
        for func in itertools.chain(
            itertools.chain(*self.health_check_observables),
            self._health_check_functions,
        ):
            identity = id(func)
            label = getattr(func, "__name__", type(func).__name__)
            try:
                healthy = bool(func())
            except Exception:
                if identity not in self._health_check_triggers:
                    log.exception("Health checker raised: %s", label)
                healthy = False
            if healthy:
                self._health_check_triggers.pop(identity, None)
            elif identity not in self._health_check_triggers:
                log.critical("Health check failure for checker: %s", label)
                self._health_check_triggers[identity] = func

    def verify_have_contracts_for_positions(self) -> list[ibi.Contract]:
        return [
            position.contract
            for position in self.ib.positions()
            if position.contract not in self.contract_registry.all_contracts
        ]

    def suspend_broker_work(self) -> None:
        """Defer roll discovery until reconciliation completes again."""
        self._broker_ready = False

    def verify_broker_account(self, positions: tuple[ibi.Position, ...]) -> None:
        """Enforce the process's single account across reconnects and order history."""
        accounts = set(self.ib.managedAccounts())
        accounts.update(position.account for position in positions)
        accounts.update(trade.order.account for trade in self.ib.openTrades())
        accounts.update(info.trade.order.account for info in self.book.orders.query())
        if self._broker_account is not None:
            accounts.add(self._broker_account)
        accounts.discard("")
        if len(accounts) > 1:
            raise SyncBrokenStateError(
                f"Expected one account/subaccount, received {sorted(accounts)}"
            )
        if accounts:
            self._broker_account = accounts.pop()

    @property
    def broker_ready(self) -> bool:
        """Whether scheduled broker work can use reconciled connection state."""
        return (
            self._broker_ready
            and self.ib.isConnected()
            and not self._trading_disabled
            and not self._sync_lock.locked()
            and (self._sync_abort_event is None or not self._sync_abort_event.is_set())
        )

    def set_future_roll_policies(self, policies: Mapping[str, bool]) -> None:
        """Install the one-to-one source policies collected during composition."""

        self.future_roll_policies = dict(policies)
        self.future_roller.set_policies(self.future_roll_policies)

    def register_protection_recovery(
        self, source_key: str, callback: Callable[[], None]
    ) -> None:
        """Register one model-owned initial-protection recovery hook per source.

        Framework plumbing calls these only after broker evidence and positions
        are reconciled, before missing-bracket remediation. Hooks must be
        repeat-safe and must not perform ordinary target convergence.
        """
        previous = self._protection_recovery.get(source_key)
        if previous is not None and previous != callback:
            raise ValueError(
                f"Protection recovery already registered for {source_key!r}"
            )
        self._protection_recovery[source_key] = callback

    def recover_protection(self) -> None:
        """Give models a repair opportunity before missing-bracket remediation."""
        if self.reset or self.zero or self.nuke or self._trading_disabled:
            return
        for source_key, callback in self._protection_recovery.items():
            try:
                callback()
            except Exception as exc:
                raise SyncBrokenStateError(
                    f"Initial bracket recovery failed for {source_key!r}: {exc}"
                ) from exc

    async def run(self) -> SyncOutcome:
        """Reconcile broker state and arm runtime timers."""

        self._ensure_runtime_timers_started()
        self.suspend_broker_work()
        if self.nuke:
            self.execute_emergency_reset()
        outcome = await self.sync()
        if not outcome:
            if outcome is SyncOutcome.ABORTED:
                log.debug("Controller startup sync aborted: connection unavailable.")
            else:
                log.critical("Controller startup sync failed. Trading disabled.")
            return outcome
        if self.zero:
            self.clear_records()
            self.zero = False
        if self.reset:
            if not await self.execute_reset():
                self.disable_trading("account reset did not complete")
                return SyncOutcome.FAILED
            self.book.clear_state()
            self.reset = False
        self._restart_before_correction = True
        return SyncOutcome.OK

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
        """Run one scheduled futures-roll discovery pass."""

        self.future_roller.roll()

    def request_position_sync(self) -> None:
        """Schedule reconciliation after roll discovery sees unsettled positions."""
        asyncio.get_running_loop().create_task(
            self._sync_before_roll(), name="roll-position-sync"
        )

    async def _sync_before_roll(self) -> None:
        if await self.sync() is SyncOutcome.OK:
            self.future_roller.roll()

    def schedule_future_roll(self) -> None:
        """Install the single app-lifetime daily UTC roll callback."""

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
        # One broker position request and one safety decision at a time. A
        # later timer/reconnect must not resume recovery after a terminal fault.
        async with self._sync_lock:
            if self._trading_disabled:
                return SyncOutcome.FAILED
            return await self._sync_attempts()

    async def _sync_attempts(self) -> SyncOutcome:
        """Retry transient broker state and latch terminal reconciliation faults."""
        self._broker_ready = False
        if not self.ib.isConnected():
            return SyncOutcome.FAILED
        for attempt in range(1, self.sync_max_attempts + 1):
            log.debug("Sync attempt %s/%s", attempt, self.sync_max_attempts)
            coordinator = SyncCoordinator(self, self._restart_before_correction)
            try:
                if await coordinator.run():
                    self._restart_before_correction = False
                    self._broker_ready = True
                    return SyncOutcome.OK
            except SyncBrokenStateError as exc:
                self.disable_trading(str(exc))
                return SyncOutcome.FAILED
            if self._sync_abort_event is not None and self._sync_abort_event.is_set():
                return SyncOutcome.ABORTED
            if coordinator.request_restart:
                return self._request_sync_restart()
            if attempt < self.sync_max_attempts:
                await asyncio.sleep(self.sync_resync_delay)
        if self._sync_abort_event is not None and self._sync_abort_event.is_set():
            return SyncOutcome.ABORTED
        self.disable_trading("sync did not converge")
        return SyncOutcome.FAILED

    def _request_sync_restart(self) -> SyncOutcome:
        """Request fresh broker state through the owning supervisor."""

        request_restart = self.request_restart
        if request_restart is None:
            self.disable_trading("supervisor restart callback unavailable")
            return SyncOutcome.FAILED

        accepted = request_restart(
            "controller reconciliation requires fresh broker state"
        )
        if accepted is False:
            self.disable_trading("supervisor restart request rejected")
            return SyncOutcome.FAILED

        self._restart_before_correction = False
        return SyncOutcome.ABORTED

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

        self._validate_order_attribution(
            role=str(role),
            source_key=source_key,
        )
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

        # Validate the same record schema used after submission before the
        # broker can accept an order. Only the broker Trade is replaced later.
        metadata = OrderInfo(
            trade=ibi.Trade(contract=contract, order=order),
            role=role,
            submitted_at=datetime.datetime.now(datetime.timezone.utc),
            execution_model_name=execution_model_name,
            source_key=source_key,
            position_id=position_id,
            params={} if params is None else params,
        )
        self._validate_order_attribution(
            role=metadata.role, source_key=metadata.source_key
        )
        if (
            order.account
            and self._broker_account
            and order.account != self._broker_account
        ):
            raise ValueError("Order account differs from the reconciled account")
        self.book.check_writable()
        trade = self.trader.trade(contract, order)
        self.register_order(
            trade,
            role=metadata.role,
            execution_model_name=metadata.execution_model_name,
            source_key=metadata.source_key,
            position_id=metadata.position_id,
            params=metadata.params,
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
            params={} if params is None else params,
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

    @staticmethod
    def _validate_order_attribution(*, role: str, source_key: str | None) -> None:
        """Reserve direct adjustment attribution for concrete account targets."""
        if role == StandardOrderRole.TARGET_ADJUSTMENT and source_key is not None:
            raise ValueError("TARGET_ADJUSTMENT must not have source_key")

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
        if self.book.orders.by_id(trade.order.orderId) is None:
            log.critical(
                "Unknown broker trade: %s %s",
                trade.order,
                trade.contract.symbol,
            )

    def onOrderStatusEvent(self, trade: ibi.Trade) -> None:
        """Persist status changes and rebind current live Trade objects."""
        if self.book.rebind_trade(trade) is not None:
            if not trade.order.orderId:
                log.warning(
                    "Skipping unknown order status with orderId 0, permId=%s",
                    trade.order.permId,
                )
                return
            log.debug(
                "Unattributed order status awaits reconciliation: %s",
                trade.order.orderId,
            )

    def register_position(self, trade: ibi.Trade, fill: ibi.Fill) -> None:
        """Apply one execution idempotently to Book projections."""

        if isinstance(trade.contract, ibi.Bag):
            log.debug("Combo fill retained as order evidence without projection.")
            self.book.apply_fill(trade, fill)
            return
        try:
            changed = self.book.apply_fill(trade, fill)
        except (KeyError, ValueError):
            log.exception("Cannot apply fill for orderId=%s", trade.order.orderId)
            return
        if not changed:
            log.debug(
                "Ignored duplicate fill execId=%s orderId=%s",
                fill.execution.execId,
                trade.order.orderId,
            )

    async def onExecDetailsEvent(self, trade: ibi.Trade, fill: ibi.Fill) -> None:
        """Match, persist, and account one broker execution callback."""

        info = (
            self.assign_manual_trade(trade)
            or self.book.orders.by_id(trade.order.orderId)
            or self.book.orders.by_perm_id(trade.order.permId)
            or self.assign_unknown_trade(trade)
        )
        if info is not None:
            self.register_position(trade, fill)

    async def onCommissionReport(
        self,
        trade: ibi.Trade,
        fill: ibi.Fill,
        report: ibi.CommissionReport,
    ) -> None:
        """Persist final commission evidence and optionally write a blotter row."""

        if not trade.order.orderId and not trade.order.permId:
            return
        await asyncio.sleep(0)
        info = self.book.orders.by_id(
            trade.order.orderId
        ) or self.book.orders.by_perm_id(trade.order.permId)
        if info is None:
            log.error("Commission report for unknown orderId=%s", trade.order.orderId)
            return
        if not self.book.update_commission(trade, fill, report):
            log.warning(
                "Commission report has no normalized fill evidence: "
                "execId=%s orderId=%s",
                fill.execution.execId,
                trade.order.orderId,
            )
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
        while self._active_target_orders(target, execution_model_name) or (
            self.book.rolls.for_source(target.source_key)
            if target.source_key is not None
            else self.book.rolls.for_contract(target.contract)
        ):
            if retries >= self.execution_verification_max_retries:
                break
            retries += 1
            await asyncio.sleep(self.execution_verification_delay)
            if not self._target_is_latest(target, execution_model_name):
                return False
        if not self._target_is_latest(target, execution_model_name):
            return False
        position = (
            self.book.positions.for_source(target.source_key)
            if target.source_key is not None
            else None
        )
        actual = (
            position.quantity
            if position is not None
            else (self.book.positions.quantity(target.contract))
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
        if target.source_key is None:
            orders = self.book.orders.active(
                contract=target.contract,
                execution_model_name=execution_model_name,
            )
        else:
            orders = self.book.orders.active(
                source_key=target.source_key,
                execution_model_name=execution_model_name,
            )
        return tuple(info for info in orders if info.role in roles)

    def _target_is_latest(
        self, target: PositionTarget, execution_model_name: str
    ) -> bool:
        """Return whether Book still identifies this exact accepted setpoint."""

        if target.source_key is not None:
            position_state = self.book.positions.for_source(target.source_key)
            return (
                position_state is not None
                and position_state.execution_model_name == execution_model_name
                and position_state.target_contract is not None
                and position_state.target_contract.conId == target.contract.conId
                and position_state.target_quantity == target.target_quantity
                and position_state.target_created_at == target.created_at
            )
        target_state = self.book.targets.for_contract(target.contract)
        return (
            target_state is not None
            and target_state.execution_model_name == execution_model_name
            and target_state.contract.conId == target.contract.conId
            and target_state.target_quantity == target.target_quantity
            and target_state.target_created_at == target.created_at
        )

    def verify_position_with_broker(self, contract: ibi.Contract) -> None:
        """Compare aggregate logical Book quantity with the broker position."""

        logical = self.book.positions.quantity(contract)
        broker = self.trader.position_for_contract(contract)
        if logical != broker:
            log.error(
                "Wrong aggregate position for %s: logical=%s broker=%s",
                contract,
                logical,
                broker,
            )

    def _source_for_unknown_trade(self, trade: ibi.Trade) -> str | None:
        """Attribute an unknown trade only when one logical position is clear."""

        if not trade.contract.conId:
            return None
        states = self.book.positions.source_states_for_contract(trade.contract)
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
            self.book.positions.for_source(source_key)
            if source_key is not None
            else None
        )
        return OrderInfo(
            trade=trade,
            role=str(role),
            submitted_at=datetime.datetime.now(datetime.timezone.utc),
            execution_model_name=(
                state.execution_model_name if state is not None else str(role).lower()
            ),
            source_key=source_key,
            position_id=state.position_id if state is not None else None,
        )

    def assign_manual_trade(self, trade: ibi.Trade) -> OrderInfo | None:
        """Register a negative-orderId manual broker trade."""

        if trade.order.orderId >= 0:
            return None
        existing = self.book.orders.by_id(trade.order.orderId)
        if existing is not None:
            return existing
        return self.book.save_order(
            self._unknown_order_info(trade, role=StandardOrderRole.MANUAL)
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

        for info in self.book.orders.active(source_key=source_key):
            self.cancel(info.trade)

    def close_position_for_source(
        self, source_key: str, role: str = StandardOrderRole.LIQUIDATION
    ) -> None:
        """Submit one attributed market order to flatten a logical source."""

        state = self.book.positions.for_source(source_key)
        if state is None or not state.quantity or state.contract is None:
            log.error("Attempt to close zero or unknown source %s", source_key)
            return
        exits = tuple(
            info.trade
            for info in self.book.orders.active(source_key=source_key)
            if info.role in {StandardOrderRole.STOP_LOSS, StandardOrderRole.TAKE_PROFIT}
        )
        action = "BUY" if state.quantity < 0 else "SELL"
        for trade in exits:
            if (
                trade.contract != state.contract
                or trade.order.action != action
                or trade.remaining() != abs(state.quantity)
                or not trade.order.ocaGroup
                or trade.order.ocaType not in {1, 2, 3}
            ):
                self.cancel(trade)
                if not trade.isDone():
                    raise SyncBrokenStateError(
                        "Unsafe protective exit cancellation unconfirmed for "
                        f"{source_key!r}"
                    )
        groups = {
            (trade.order.ocaGroup, trade.order.ocaType)
            for trade in exits
            if trade.isActive()
        }
        if len(groups) > 1:
            raise SyncBrokenStateError(
                f"Conflicting protective OCA groups for {source_key!r}"
            )
        options = {}
        if groups:
            group, oca_type = groups.pop()
            options = {"ocaGroup": group, "ocaType": oca_type}
        self.trade(
            state.contract,
            ibi.MarketOrder(
                action,
                abs(state.quantity),
                **options,
            ),
            role=role,
            execution_model_name=state.execution_model_name,
            source_key=source_key,
            position_id=state.position_id,
        )

    def disable_trading(self, reason: str) -> None:
        """Block submissions and notify live runtime to end strategy work.

        Args:
            reason: Actionable explanation recorded on the first failure.
        """
        if not self._trading_disabled:
            self._trading_disabled = True
            log.critical("Trading disabled: %s", reason)
        self._trading_disabled_event.set()

    async def wait_for_trading_disabled(self) -> None:
        """Wait for the process-lifetime safety latch that ends live strategy work.

        This remains set across reconnects. Only a new process may resume
        trading after the underlying accounting or safety problem is repaired.
        """
        if not self._trading_disabled:
            await self._trading_disabled_event.wait()

    def lock_new_positions(self) -> None:
        log.error("Emergency lock for new positions.")
        self._new_position_lock = True

    def log_order_status(self, trade: ibi.Trade) -> None:
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
    def log_trade(trade: ibi.Trade, reason: str = "", source_key: str = "") -> None:
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

        info = self.book.orders.by_id(reqId)
        model_name = info.execution_model_name if info is not None else ""
        role = info.role if info is not None else ""
        order = info.trade.order if info is not None else ""
        context = f"{contract=}, {model_name} | {role} | {order}"
        if errorCode == 201:
            log.critical(
                "ORDER REJECTED: %s errorCode=%s, %s", errorString, errorCode, context
            )
            self.book.orders.register_rejection(model_name)
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

    async def execute_reset(self) -> bool:
        """Execute a reset and return whether broker completion is confirmed.

        A reset closes all open positions and cancels pending orders.
        :meth:`run` clears Book state only after this method succeeds.
        """

        return await Reset(self).run()

    def execute_emergency_reset(self) -> None:
        """Request an emergency reset, disabling trading before broker operations.

        This bypasses normal submission policies but retains persistence checks.
        Completion is not verified and Book state is not cleared. Broker and
        persistence failures propagate with trading left disabled.
        """
        EmergencyReset(self).run()

    def clear_records(self) -> None:
        self.book.clear_state()


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
