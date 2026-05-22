"""Controller sync orchestration for broker and local state.

Sync starts by checking the broker connection and validating that
``ib.positions()`` agrees with ``await ib.reqPositionsAsync()``.  If either
check fails, trading is disabled before any recovery or correction action is
attempted.

After broker validation, each step reads current state directly from
``controller.ib`` or ``controller.sm`` instead of using stored broker/local
snapshots.  The ordered flow is:

1. Relink broker ``ibi.Trade`` objects to local order records and back-report
   fills for orders that completed while the process was disconnected.
2. Compare local aggregate strategy positions with broker positions and
   correct local position records when the existing recovery rules allow it.
3. Skip correction trades when unresolved unknown broker orders remain active.
4. Delegate bracket-record and broker stop-loss protection handling to
   :mod:`haymaker.controller.sync_brackets`.

Broker-affecting remedies wait for ``controller.sync_resync_delay`` before
later broker queries, but sync no longer restarts from a captured snapshot.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import ib_insync as ibi

from haymaker import misc
from haymaker.state_machine import OrderInfo, StateMachine

from .sync_brackets import BracketSyncResult, BracketSyncer

if TYPE_CHECKING:
    from .controller import Controller

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrderFindings:
    """Read-only order differences between local and broker state."""

    unknown_broker_trades: tuple[ibi.Trade, ...] = ()
    broker_trade_updates: tuple[tuple[OrderInfo, ibi.Trade], ...] = ()
    inactive_local_orders: tuple[OrderInfo, ...] = ()
    missing_active_orders: tuple[OrderInfo, ...] = ()

    @property
    def has_findings(self) -> bool:
        """Return True when any order discrepancy was found."""
        return any(
            (
                self.unknown_broker_trades,
                self.broker_trade_updates,
                self.inactive_local_orders,
                self.missing_active_orders,
            )
        )


@dataclass
class OrderRecoveryResult:
    """Actions taken while applying order reconciliation findings."""

    unknown_broker_trades: list[ibi.Trade] = field(default_factory=list)
    cancelled_unknown_trades: list[ibi.Trade] = field(default_factory=list)
    updated_orders: list[OrderInfo] = field(default_factory=list)
    pruned_orders: list[OrderInfo] = field(default_factory=list)
    done_trades: list[ibi.Trade] = field(default_factory=list)
    faulty_orders: list[OrderInfo] = field(default_factory=list)

    @property
    def changed_local(self) -> bool:
        """Return True when order sync changed local records or emitted events."""
        return any(
            (
                self.updated_orders,
                self.pruned_orders,
                self.done_trades,
                self.faulty_orders,
            )
        )

    @property
    def changed_broker(self) -> bool:
        """Return True when order sync sent broker-affecting requests."""
        return bool(self.cancelled_unknown_trades)

    @property
    def has_unresolved_unknown_orders(self) -> bool:
        """Return True when unknown broker orders were left active."""
        return bool(self.unknown_broker_trades and not self.cancelled_unknown_trades)


@dataclass(frozen=True)
class PositionFindings:
    """Read-only position differences between local and broker state."""

    errors: Mapping[ibi.Contract, float] = field(default_factory=dict)


@dataclass
class PositionRecoveryResult:
    """Actions taken while correcting local position records."""

    corrected_contracts: list[ibi.Contract] = field(default_factory=list)

    @property
    def changed_local(self) -> bool:
        """Return True when position sync changed local records."""
        return bool(self.corrected_contracts)


class BrokerStateError(Exception):
    """Raised when broker state cannot be queried safely."""

    def __init__(self, reason: str) -> None:
        """Initialize the exception with a human-readable failure reason."""
        super().__init__(reason)
        self.reason = reason


async def verify_broker_position_source(ib: ibi.IB, timeout: float) -> None:
    """Verify that synchronous and requested broker positions agree."""
    positions = tuple(ib.positions())
    try:
        requested_positions = tuple(
            await asyncio.wait_for(ib.reqPositionsAsync(), timeout)
        )
    except asyncio.TimeoutError as exc:
        reason = f"broker position request timed out after {timeout}s"
        raise BrokerStateError(reason) from exc
    except Exception as exc:
        reason = f"broker position request failed: {exc!r}"
        raise BrokerStateError(reason) from exc

    positions_dict = {
        position.contract.localSymbol: position.position for position in positions
    }
    requested_positions_dict = {
        position.contract.localSymbol: position.position
        for position in requested_positions
        if position.position
    }
    if positions_dict != requested_positions_dict:
        reason = (
            "broker position sources disagree: "
            f"positions={positions_dict} req_positions={requested_positions_dict}"
        )
        raise BrokerStateError(reason)

    log.debug(f"broker positions: {positions_dict}")


def broker_positions_by_contract(ib: ibi.IB) -> dict[ibi.Contract, float]:
    """Return current broker positions keyed by contract."""
    return {position.contract: position.position for position in ib.positions()}


def broker_trades_by_order_id_or_perm_id(ib: ibi.IB) -> dict[int, ibi.Trade]:
    """Return current session trades keyed by order id, falling back to perm id."""
    return {trade.order.orderId or trade.order.permId: trade for trade in ib.trades()}


def broker_fills_by_order_id(ib: ibi.IB) -> defaultdict[int, list[ibi.Fill]]:
    """Return current broker fills grouped by order id."""
    fills: defaultdict[int, list[ibi.Fill]] = defaultdict(list)
    for fill in ib.fills():
        fills[fill.execution.orderId].append(fill)
    return fills


def broker_position_for_contract(ib: ibi.IB, contract: ibi.Contract) -> float:
    """Return current broker position for ``contract``."""
    return broker_positions_by_contract(ib).get(contract, 0.0)


def local_positions_by_contract(sm: StateMachine) -> dict[ibi.Contract, float]:
    """Return current local positions keyed by contract."""
    return dict(sm.strategy.total_positions())


def compare_order_state(sm: StateMachine, ib: ibi.IB) -> OrderFindings:
    """Compare local order records with current broker order state."""
    unknown_broker_trades = []
    broker_trade_updates = []
    inactive_local_orders = []
    missing_active_orders = []

    open_trades = ib.openTrades()
    for trade in open_trades:
        order_info = sm.order.get(trade.order.orderId)
        if order_info is None:
            unknown_broker_trades.append(trade)
        elif order_info.trade is not trade:
            broker_trade_updates.append((order_info, trade))

    broker_open_order_ids = {trade.order.orderId for trade in open_trades}
    for order_id, order_info in sm.order.items():
        if order_id in broker_open_order_ids:
            continue
        if order_info.active:
            missing_active_orders.append(order_info)
        else:
            inactive_local_orders.append(order_info)

    findings = OrderFindings(
        unknown_broker_trades=tuple(unknown_broker_trades),
        broker_trade_updates=tuple(broker_trade_updates),
        inactive_local_orders=tuple(inactive_local_orders),
        missing_active_orders=tuple(missing_active_orders),
    )
    report_order_findings(findings)
    return findings


def report_order_findings(findings: OrderFindings) -> None:
    """Log a summary of order reconciliation findings."""
    if findings.has_findings:
        log.debug(
            "Trades on sync -> "
            f"unknown: {len(findings.unknown_broker_trades)}, "
            f"updates: {len(findings.broker_trade_updates)}, "
            f"inactive: {len(findings.inactive_local_orders)}, "
            f"unmatched: {len(findings.missing_active_orders)}"
        )
    else:
        log.debug("Orders sync OK.")

    for unknown_trade in findings.unknown_broker_trades:
        log.critical(f"Unknown trade in the system: {unknown_trade}.")


def compare_position_state(sm: StateMachine, ib: ibi.IB) -> PositionFindings:
    """Compare local strategy positions with current broker positions."""
    broker_positions = broker_positions_by_contract(ib)
    local_positions = local_positions_by_contract(sm)
    diff = {
        contract: (
            (local_positions.get(contract) or 0.0)
            - (broker_positions.get(contract) or 0.0)
        )
        for contract in set([*broker_positions.keys(), *local_positions.keys()])
    }
    errors = {contract: value for contract, value in diff.items() if value}
    if errors:
        log.error(f"errors: { {k.symbol: v for k, v in errors.items()} }")
        log.critical(f"Failed to match positions to broker: {errors}")
    else:
        log.debug("Positions matched to broker OK.")
    return PositionFindings(errors=errors)


class OrderSyncApplier:
    """Apply order reconciliation findings to local records."""

    def __init__(
        self,
        ib: ibi.IB,
        sm: StateMachine,
        controller: Controller,
        cancel_unknown_trades: bool,
    ) -> None:
        self.ib = ib
        self.sm = sm
        self.controller = controller
        self.cancel_unknown_trades = cancel_unknown_trades

    async def apply(self, findings: OrderFindings) -> OrderRecoveryResult:
        result = OrderRecoveryResult(
            unknown_broker_trades=list(findings.unknown_broker_trades)
        )

        self.update_broker_trade_records(findings, result)
        self.prune_inactive_local_orders(findings, result)
        self.recover_missing_active_orders(findings, result)
        self.handle_unknown_broker_orders(result)

        try:
            for done_trade in result.done_trades:
                self.report_done_trade(done_trade)
        except Exception as exc:
            log.exception(f"Error with done trade: {exc}")

        await asyncio.sleep(0)
        self.clear_unmatched_orders(result)
        await asyncio.sleep(0)
        return result

    def update_broker_trade_records(
        self, findings: OrderFindings, result: OrderRecoveryResult
    ) -> None:
        for order_info, broker_trade in findings.broker_trade_updates:
            log.debug(
                f"Trade will be updated - id: {order_info.trade.order.orderId} "
                f"permId: {order_info.trade.order.permId}"
            )
            new_order_info = OrderInfo(
                order_info.strategy,
                order_info.action,
                broker_trade,
                order_info.params,
                list(order_info.accounted_exec_ids),
            )
            self.sm.save_order(new_order_info)
            result.updated_orders.append(new_order_info)

    def prune_inactive_local_orders(
        self, findings: OrderFindings, result: OrderRecoveryResult
    ) -> None:
        for order_info in findings.inactive_local_orders:
            order_id = order_info.trade.order.orderId
            log.debug(f"Will prune order: {order_id}")
            self.sm.prune_order(order_id)
            result.pruned_orders.append(order_info)

    def recover_missing_active_orders(
        self,
        findings: OrderFindings,
        result: OrderRecoveryResult,
    ) -> None:
        broker_trades = broker_trades_by_order_id_or_perm_id(self.ib)
        fills_by_order_id = broker_fills_by_order_id(self.ib)

        if findings.missing_active_orders:
            inactive_trades = [
                (i.trade.order.orderId, i.trade.order.permId)
                for i in findings.missing_active_orders
            ]
            log.debug(f"inactive trades: {inactive_trades}")
            log.debug(f"ib_known_trades: {list(broker_trades)}")

        for order_info in findings.missing_active_orders:
            old_trade = order_info.trade
            if new_trade := broker_trades.get(old_trade.order.permId):
                log.warning(
                    f"Will change orderId: {new_trade.order.orderId} "
                    f"to: {old_trade.order.orderId}"
                )
                new_trade.order.orderId = old_trade.order.orderId
                self.sm.update_trade(new_trade)
                result.done_trades.append(new_trade)
            elif fills := fills_by_order_id.get(old_trade.order.orderId):
                self.reconstruct_trade_from_fills(old_trade, fills)
                self.sm.update_trade(old_trade)
                result.done_trades.append(old_trade)
            else:
                result.faulty_orders.append(order_info)

        if result.done_trades:
            log.debug(
                "done: "
                f"{[(t.order.orderId, t.order.permId) for t in result.done_trades]}"
            )

    def reconstruct_trade_from_fills(
        self, trade: ibi.Trade, fills: list[ibi.Fill]
    ) -> None:
        trade.fills = fills
        filled = sum(fill.execution.shares for fill in fills)
        remaining = trade.order.totalQuantity - filled
        trade.log.append(
            ibi.objects.TradeLogEntry(
                time=fills[-1].execution.time,
                status="Filled" if remaining == 0 else "Submitted",
                message="composed by sync",
            )
        )
        trade.orderStatus = ibi.OrderStatus(
            orderId=trade.order.orderId,
            status=(
                ibi.OrderStatus.Filled if remaining == 0 else ibi.OrderStatus.Submitted
            ),
            filled=filled,
            remaining=remaining,
        )

    def handle_unknown_broker_orders(self, result: OrderRecoveryResult) -> None:
        trades = result.unknown_broker_trades
        if not trades:
            return

        log.critical(f"Unknown broker orders during sync: {trades}.")
        if not self.cancel_unknown_trades:
            log.critical(
                "Unknown broker orders left active because "
                "cancel_unknown_trades is False."
            )
            return

        for trade in trades:
            log.debug(f"Cancelling unknown broker order: {trade.order.orderId}")
            self.controller.cancel(trade)
            result.cancelled_unknown_trades.append(trade)

    def report_done_trade(self, trade: ibi.Trade) -> None:
        log.debug(
            f"Back-reporting trade: {trade.contract.symbol} "
            f"{trade.order.action} {misc.trade_fill_price(trade)} "
            f"order id: {trade.order.orderId} {trade.order.permId} "
            f"active?: {trade.isActive()}"
        )
        self.ib.orderStatusEvent.emit(trade)
        for fill in trade.fills:
            self.ib.execDetailsEvent.emit(trade, fill)
        if trade.orderStatus.status == "Filled":
            self.ib.commissionReportEvent.emit(
                trade, trade.fills[-1], trade.fills[-1].commissionReport
            )

    def clear_unmatched_orders(self, result: OrderRecoveryResult) -> None:
        for order_info in result.faulty_orders:
            trade = order_info.trade
            log.error(
                f"Will delete record for trade that IB doesn't known about: "
                f"{trade.order.orderId}"
            )
            self.sm.prune_order(trade.order.orderId)


def apply_position_recovery(
    sm: StateMachine,
    ib: ibi.IB,
    findings: PositionFindings,
    order_recovery: OrderRecoveryResult,
) -> PositionRecoveryResult:
    """Apply position reconciliation findings to local records."""
    result = PositionRecoveryResult()
    if not findings.errors:
        return result

    log.error("Will attempt to fix position records")
    strategy_faults = {
        order_info.strategy for order_info in order_recovery.faulty_orders
    }
    for contract, diff in findings.errors.items():
        strategies = sm.for_contract.get(contract)
        log.debug(f"Strategies for contract {contract.localSymbol}: {strategies}")
        if strategies and len(strategies) == 1:
            sm.strategy[strategies[0]].position -= diff
            log.error(
                f"Corrected position records for strategy "
                f"{strategies[0]} by {-diff}"
            )
            sm.save_strategies()
            result.corrected_contracts.append(contract)
        elif strategies and broker_position_for_contract(ib, contract) == 0:
            for strategy in strategies:
                sm.strategy[strategy].position = 0
            sm.save_strategies()
            log.error(
                f"Position records zeroed for {strategies} "
                f"to reflect zero position for {contract.symbol}."
            )
            result.corrected_contracts.append(contract)
        elif strategies:
            for strategy in strategies:
                if strategy in strategy_faults:
                    sm.strategy[strategy].position = 0
                    log.error(
                        f"Position records zeroed for {strategy} "
                        f"to reflect faulty trade previously removed."
                    )
                    result.corrected_contracts.append(contract)
        else:
            log.critical(
                f"Cannot fix position records for {contract.localSymbol}, "
                f"{strategies=}."
            )
    return result


class SyncCoordinator:
    """Coordinate broker/local sync phases and trading-disable decisions.

    The coordinator is the only object in the sync path that disables trading.
    Helper functions and bracket handling return findings or result objects;
    this class decides whether those results are terminal for the running
    system.
    """

    def __init__(self, controller: Controller) -> None:
        """Initialize the coordinator for one controller sync run."""
        self.controller = controller

    async def run(self) -> bool:
        """Run one sync pass against current broker and local state.

        Returns:
            ``True`` when sync completed without a terminal safety failure,
            otherwise ``False`` after disabling trading.
        """
        log.debug("--- Sync ---")

        if not self.verify_broker_connection():
            return False
        if not await self.verify_broker_state():
            return False

        order_recovery = await self.apply_order_recovery()
        await self.wait_after_broker_change(
            "order recovery", order_recovery.changed_broker
        )

        position_findings = self.compare_positions()
        self.apply_position_recovery(position_findings, order_recovery)
        if not self.verify_position_sync():
            return False
        if self.has_unresolved_unknown_orders(order_recovery):
            return self.complete()

        bracket_result = self.apply_bracket_sync()
        if bracket_result.blocked_reason:
            self.block_trading(bracket_result.blocked_reason)
            return False
        if bracket_result.terminal_action:
            return self.complete()

        await self.wait_after_broker_change(
            "bracket recovery", bracket_result.changed_broker
        )
        return self.complete()

    def verify_broker_connection(self) -> bool:
        """Return False and disable trading if broker is disconnected."""
        if self.controller.ib.isConnected():
            return True

        log.debug("No connection. Abandoning sync.")
        self.block_trading("broker not connected")
        return False

    async def verify_broker_state(self) -> bool:
        """Return False and disable trading if broker state is not trustworthy."""
        try:
            await verify_broker_position_source(
                self.controller.ib,
                self.controller.broker_request_timeout,
            )
            return True
        except BrokerStateError as exc:
            self.block_trading(exc.reason)
            return False

    async def apply_order_recovery(self) -> OrderRecoveryResult:
        """Apply order recovery and relink broker trades to local records."""
        self.controller.release_hold()
        findings = compare_order_state(self.controller.sm, self.controller.ib)
        return await OrderSyncApplier(
            self.controller.ib,
            self.controller.sm,
            self.controller,
            self.controller.cancel_unknown_trades,
        ).apply(findings)

    def has_unresolved_unknown_orders(
        self, order_recovery: OrderRecoveryResult
    ) -> bool:
        """Return True when unknown broker orders remain active."""
        if not order_recovery.has_unresolved_unknown_orders:
            return False

        log.error(
            "Unknown broker orders remain active; "
            "skipping position-protection correction trades."
        )
        return True

    def compare_positions(self) -> PositionFindings:
        """Compare local strategy positions with broker positions."""
        return compare_position_state(self.controller.sm, self.controller.ib)

    def apply_position_recovery(
        self,
        position_findings: PositionFindings,
        order_recovery: OrderRecoveryResult,
    ) -> PositionRecoveryResult:
        """Correct local position records to match broker positions."""
        return apply_position_recovery(
            self.controller.sm,
            self.controller.ib,
            position_findings,
            order_recovery,
        )

    def verify_position_sync(self) -> bool:
        """Verify local positions match current broker positions."""
        recheck = compare_position_state(self.controller.sm, self.controller.ib)
        if not recheck.errors:
            return True

        self.block_trading("local state does not match broker state")
        return False

    def apply_bracket_sync(self) -> BracketSyncResult:
        """Check bracket policy after order and position reconciliation.

        Bracket handling runs only after broker state has been validated and
        local position records have been reconciled.  The bracket module owns
        the policy details for ``controller.missing_brackets``; the coordinator
        only interprets the returned result.
        """
        return BracketSyncer(
            self.controller,
            self.controller.missing_brackets,
        ).run()

    async def wait_after_broker_change(self, reason: str, changed_broker: bool) -> None:
        """Wait after broker-affecting remedies before later broker queries."""
        if not changed_broker:
            return

        log.error(f"Sync {reason} changed broker state; waiting for broker update.")
        await asyncio.sleep(self.controller.sync_resync_delay)

    def block_trading(self, reason: str) -> None:
        """Disable future trading from one sync-owned decision point."""
        self.controller.disable_trading(reason)

    def complete(self) -> bool:
        """Log successful sync completion."""
        log.debug("--- Sync completed ---")
        return True
