from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import ib_insync as ibi

from haymaker import misc
from haymaker.state_machine import OrderInfo, Strategy, StateMachine

if TYPE_CHECKING:
    from .controller import Controller

log = logging.getLogger(__name__)

MissingBracketsPolicy = Literal["ignore", "warn", "remove"]


@dataclass(frozen=True)
class BrokerProtectionIssue:
    """Broker position that lacks enough opposite-side stop-loss protection."""

    contract: ibi.Contract
    broker_position: float

    @property
    def close_action(self) -> str:
        """Return the order action needed to flatten the broker position."""
        return "BUY" if self.broker_position < 0 else "SELL"

    @property
    def close_quantity(self) -> float:
        """Return the absolute quantity needed to flatten the broker position."""
        return abs(self.broker_position)


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


@dataclass(frozen=True)
class BracketIssue:
    """Missing bracket information for one strategy."""

    strategy: Strategy
    existing_orders: tuple[OrderInfo, ...]
    expected_bracket_count: int


@dataclass(frozen=True)
class BracketFindings:
    """Read-only order/position bracket findings."""

    obsolete_orders: tuple[tuple[str, OrderInfo], ...] = ()
    missing_brackets: tuple[BracketIssue, ...] = ()

    @property
    def has_findings(self) -> bool:
        """Return True when any bracket discrepancy was found."""
        return bool(self.obsolete_orders or self.missing_brackets)


@dataclass(frozen=True)
class BrokerProtectionFindings:
    """Broker positions that are not protected by stop-loss orders."""

    exposed_positions: tuple[BrokerProtectionIssue, ...] = ()

    @property
    def has_findings(self) -> bool:
        """Return True when any broker position lacks stop-loss protection."""
        return bool(self.exposed_positions)


@dataclass
class BracketSyncResult:
    """Actions taken by bracket synchronization."""

    cancelled_obsolete_orders: list[OrderInfo] = field(default_factory=list)
    closed_positions: list[BrokerProtectionIssue] = field(default_factory=list)
    blocked_reason: str | None = None

    @property
    def changed_broker(self) -> bool:
        """Return True when bracket sync sent broker-affecting requests."""
        return bool(self.cancelled_obsolete_orders or self.closed_positions)

    @property
    def terminal_action(self) -> bool:
        """Return True when sync should stop after a broker action."""
        return bool(self.closed_positions)


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


def broker_open_trades_for_contract(
    ib: ibi.IB, contract: ibi.Contract
) -> list[ibi.Trade]:
    """Return current active broker trades for ``contract``."""
    return [trade for trade in ib.openTrades() if trade.contract == contract]


def broker_position_is_protected(
    ib: ibi.IB, contract: ibi.Contract, broker_position: float
) -> bool:
    """Return True if broker currently has enough opposite-side stop protection."""
    protective_action = "SELL" if broker_position > 0 else "BUY"
    protective_order_types = {
        "STP",
        "STP LMT",
        "TRAIL",
        "TRAIL LIMIT",
        "TRAILLIT",
        "TRAIL LIT",
        "TRAILLMT",
        "TRAIL LMT",
    }
    protective_quantity = sum(
        trade.order.totalQuantity
        for trade in broker_open_trades_for_contract(ib, contract)
        if (
            trade.isActive()
            and trade.order.action == protective_action
            and trade.order.orderType in protective_order_types
        )
    )
    return protective_quantity >= abs(broker_position)


def local_positions_by_contract(sm: StateMachine) -> dict[ibi.Contract, float]:
    """Return current local positions keyed by contract."""
    return dict(sm.strategy.total_positions())


def local_strategies_by_contract(sm: StateMachine) -> dict[ibi.Contract, list[str]]:
    """Return current strategy names grouped by active contract."""
    return dict(sm.strategy.strategies_by_contract())


def active_orders_for_strategy(
    sm: StateMachine, strategy_name: str
) -> tuple[OrderInfo, ...]:
    """Return current active order records for a strategy."""
    return tuple(
        order_info
        for order_info in sm.order.values()
        if order_info.active and order_info.strategy == strategy_name
    )


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


def compare_bracket_records(sm: StateMachine) -> BracketFindings:
    """Compare local positions with local bracket/order records."""
    obsolete_orders: list[tuple[str, OrderInfo]] = []
    missing_brackets: list[BracketIssue] = []

    for strategy_str, strategy in sm.strategy.items():
        order_infos = active_orders_for_strategy(sm, strategy.strategy)
        if strategy.position == 0:
            obsolete_orders.extend(
                (strategy_str, order_info)
                for order_info in order_infos
                if order_info.action != "OPEN" and order_info.active
            )
        else:
            params = strategy.get("params")
            if params:
                brackets = [
                    bracket
                    for bracket_name in ("stop-loss", "take-profit")
                    if (bracket := params.get(bracket_name))
                ]
                existing_orders = tuple(
                    order_info
                    for order_info in order_infos
                    if order_info.action in ("STOP-LOSS", "TAKE-PROFIT")
                    and order_info.active
                )
                if len(brackets) != len(existing_orders):
                    missing_brackets.append(
                        BracketIssue(
                            strategy=strategy,
                            existing_orders=existing_orders,
                            expected_bracket_count=len(brackets),
                        )
                    )

    return BracketFindings(
        obsolete_orders=tuple(obsolete_orders),
        missing_brackets=tuple(missing_brackets),
    )


def compare_broker_protection(ib: ibi.IB) -> BrokerProtectionFindings:
    """Find broker positions that lack active stop-loss protection."""
    exposed_positions = [
        BrokerProtectionIssue(
            contract=contract,
            broker_position=position,
        )
        for contract, position in broker_positions_by_contract(ib).items()
        if position and not broker_position_is_protected(ib, contract, position)
    ]
    return BrokerProtectionFindings(exposed_positions=tuple(exposed_positions))


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


class BracketSyncer:
    """Check bracket records and broker stop-loss protection."""

    def __init__(
        self,
        controller: Controller,
        missing_brackets: str,
    ) -> None:
        self.controller = controller
        self.missing_brackets = missing_brackets

    def run(self) -> BracketSyncResult:
        result = BracketSyncResult()
        if self.missing_brackets == "ignore":
            return result

        bracket_findings = compare_bracket_records(self.controller.sm)
        protection_findings = compare_broker_protection(self.controller.ib)
        self.report_bracket_record_findings(bracket_findings)
        self.report_broker_protection_findings(protection_findings)

        if self.missing_brackets == "remove":
            self.cancel_obsolete_orders(bracket_findings, result)
            self.close_exposed_positions(protection_findings, result)

        return result

    def report_bracket_record_findings(self, findings: BracketFindings) -> None:
        for strategy_str, order_info in findings.obsolete_orders:
            order_id = order_info.trade.order.orderId
            log.error(
                f"Obsolete order for " f"{strategy_str}: {order_info.action, order_id}"
            )

        for issue in findings.missing_brackets:
            strategy = issue.strategy
            log.error(
                f"Bracket error for {strategy.strategy}, "
                f"position: {strategy.position} "
                f"we have: {len(issue.existing_orders)} orders, "
                f"we should have: {issue.expected_bracket_count} orders."
            )

    def report_broker_protection_findings(
        self, findings: BrokerProtectionFindings
    ) -> None:
        for issue in findings.exposed_positions:
            log.error(
                f"Broker position lacks stop-loss protection: "
                f"{issue.contract.localSymbol} {issue.broker_position}"
            )

    def cancel_obsolete_orders(
        self, findings: BracketFindings, result: BracketSyncResult
    ) -> None:
        for strategy_str, order_info in findings.obsolete_orders:
            order_id = order_info.trade.order.orderId
            log.error(
                f"Cancelling obsolete order: "
                f"{strategy_str, order_info.action, order_id}"
            )
            self.controller.trader.cancel(order_info.trade)
            result.cancelled_obsolete_orders.append(order_info)

    def close_exposed_positions(
        self, findings: BrokerProtectionFindings, result: BracketSyncResult
    ) -> None:
        for issue in findings.exposed_positions:
            strategy = self.strategy_for_contract(issue)
            if strategy is None:
                result.blocked_reason = (
                    "cannot attribute unprotected broker position to one strategy"
                )
                continue
            self.controller.lock_new_positions()
            log.error(
                f"Closing broker position without stop-loss protection: "
                f"{strategy} {issue.close_action} {issue.close_quantity} "
                f"{issue.contract.localSymbol}"
            )
            trade = self.controller.trade(
                strategy,
                issue.contract,
                ibi.MarketOrder(issue.close_action, issue.close_quantity),
                "MISSING BRACKET EMERGENCY CLOSE",
                self.controller.sm.strategy[strategy],
            )
            if trade:
                result.closed_positions.append(issue)
            else:
                result.blocked_reason = "failed to submit missing-bracket close order"

    def strategy_for_contract(self, issue: BrokerProtectionIssue) -> str | None:
        strategies = (
            local_strategies_by_contract(self.controller.sm).get(issue.contract) or []
        )
        if len(strategies) == 1:
            return strategies[0]

        log.critical(
            f"Cannot close unprotected broker position for "
            f"{issue.contract.localSymbol}: {strategies=}."
        )
        return None


class SyncCoordinator:
    """Coordinate broker/local sync phases and trading-disable decisions."""

    def __init__(self, controller: Controller) -> None:
        """Initialize the coordinator for one controller sync run."""
        self.controller = controller

    async def run(self) -> bool:
        """Run one sync pass against current broker and local state."""
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
        """Check local bracket records and broker stop-loss protection."""
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
