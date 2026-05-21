from __future__ import annotations

import asyncio
import datetime as dt
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
class BrokerSnapshot:
    """Broker state captured once for a sync cycle."""

    positions: tuple[ibi.Position, ...]
    open_trades: tuple[ibi.Trade, ...]
    trades: tuple[ibi.Trade, ...]
    fills: tuple[ibi.Fill, ...]
    captured_at: dt.datetime

    @property
    def positions_by_contract(self) -> dict[ibi.Contract, float]:
        """Return broker positions keyed by contract."""
        return {position.contract: position.position for position in self.positions}

    @property
    def open_trades_by_order_id(self) -> dict[int, ibi.Trade]:
        """Return active broker trades keyed by order id."""
        return {trade.order.orderId: trade for trade in self.open_trades}

    @property
    def trades_by_order_id_or_perm_id(self) -> dict[int, ibi.Trade]:
        """Return session trades keyed by order id, falling back to perm id."""
        return {
            trade.order.orderId or trade.order.permId: trade for trade in self.trades
        }

    @property
    def fills_by_order_id(self) -> defaultdict[int, list[ibi.Fill]]:
        """Return broker fills grouped by order id."""
        fills: defaultdict[int, list[ibi.Fill]] = defaultdict(list)
        for fill in self.fills:
            fills[fill.execution.orderId].append(fill)
        return fills

    def position_for_contract(self, contract: ibi.Contract) -> float:
        """Return broker position for ``contract`` from this snapshot."""
        return self.positions_by_contract.get(contract, 0.0)

    def open_trades_for_contract(self, contract: ibi.Contract) -> list[ibi.Trade]:
        """Return active broker trades for ``contract`` from this snapshot."""
        return [trade for trade in self.open_trades if trade.contract == contract]

    def position_is_protected(
        self, contract: ibi.Contract, broker_position: float
    ) -> bool:
        """Return True if snapshot contains enough opposite-side stop protection."""
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
            for trade in self.open_trades_for_contract(contract)
            if (
                trade.isActive()
                and trade.order.action == protective_action
                and trade.order.orderType in protective_order_types
            )
        )
        return protective_quantity >= abs(broker_position)


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
class LocalSnapshot:
    """State-machine data captured once for a sync phase."""

    orders_by_id: Mapping[int, OrderInfo]
    strategies: Mapping[str, Strategy]
    positions_by_contract: Mapping[ibi.Contract, float]
    strategies_by_contract: Mapping[ibi.Contract, list[str]]
    orders_by_strategy: Mapping[str, tuple[OrderInfo, ...]]

    def orders_for_strategy(self, strategy: str) -> tuple[OrderInfo, ...]:
        """Return active order records for ``strategy``."""
        return self.orders_by_strategy.get(strategy, ())


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


class BrokerSnapshotError(Exception):
    """Raised when broker state cannot be captured safely."""

    def __init__(self, reason: str) -> None:
        """Initialize the exception with a human-readable failure reason."""
        super().__init__(reason)
        self.reason = reason


async def capture_broker_snapshot(ib: ibi.IB, timeout: float) -> BrokerSnapshot:
    """Capture broker state once and verify the position source."""
    positions = tuple(ib.positions())
    try:
        requested_positions = tuple(
            await asyncio.wait_for(ib.reqPositionsAsync(), timeout)
        )
    except asyncio.TimeoutError as exc:
        reason = f"broker position request timed out after {timeout}s"
        raise BrokerSnapshotError(reason) from exc
    except Exception as exc:
        reason = f"broker position request failed: {exc!r}"
        raise BrokerSnapshotError(reason) from exc

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
        raise BrokerSnapshotError(reason)

    log.debug(f"broker positions: {positions_dict}")
    return BrokerSnapshot(
        positions=positions,
        open_trades=tuple(ib.openTrades()),
        trades=tuple(ib.trades()),
        fills=tuple(ib.fills()),
        captured_at=dt.datetime.now(tz=dt.timezone.utc),
    )


def capture_local_snapshot(sm: StateMachine) -> LocalSnapshot:
    """Capture state-machine records for one sync phase."""
    orders_by_id = dict(sm.order.items())
    orders_by_strategy: defaultdict[str, list[OrderInfo]] = defaultdict(list)
    for order_info in orders_by_id.values():
        if order_info.active:
            orders_by_strategy[order_info.strategy].append(order_info)

    return LocalSnapshot(
        orders_by_id=orders_by_id,
        strategies=dict(sm.strategy.items()),
        positions_by_contract=dict(sm.strategy.total_positions()),
        strategies_by_contract=dict(sm.strategy.strategies_by_contract()),
        orders_by_strategy={
            strategy: tuple(order_infos)
            for strategy, order_infos in orders_by_strategy.items()
        },
    )


def compare_order_state(local: LocalSnapshot, broker: BrokerSnapshot) -> OrderFindings:
    """Compare local order records with a broker snapshot."""
    unknown_broker_trades = []
    broker_trade_updates = []
    inactive_local_orders = []
    missing_active_orders = []

    for trade in broker.open_trades:
        order_info = local.orders_by_id.get(trade.order.orderId)
        if order_info is None:
            unknown_broker_trades.append(trade)
        elif order_info.trade is not trade:
            broker_trade_updates.append((order_info, trade))

    broker_open_order_ids = set(broker.open_trades_by_order_id)
    for order_id, order_info in local.orders_by_id.items():
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


def compare_position_state(
    local: LocalSnapshot, broker: BrokerSnapshot
) -> PositionFindings:
    """Compare local strategy positions with a broker snapshot."""
    broker_positions = broker.positions_by_contract
    local_positions = local.positions_by_contract
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


def compare_bracket_records(local: LocalSnapshot) -> BracketFindings:
    """Compare local positions with local bracket/order records."""
    obsolete_orders: list[tuple[str, OrderInfo]] = []
    missing_brackets: list[BracketIssue] = []

    for strategy_str, strategy in local.strategies.items():
        order_infos = local.orders_for_strategy(strategy.strategy)
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


def compare_broker_protection(broker: BrokerSnapshot) -> BrokerProtectionFindings:
    """Find broker positions that lack active stop-loss protection."""
    exposed_positions = [
        BrokerProtectionIssue(
            contract=contract,
            broker_position=position,
        )
        for contract, position in broker.positions_by_contract.items()
        if position and not broker.position_is_protected(contract, position)
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

    async def apply(
        self, findings: OrderFindings, broker: BrokerSnapshot
    ) -> OrderRecoveryResult:
        result = OrderRecoveryResult(
            unknown_broker_trades=list(findings.unknown_broker_trades)
        )

        self.update_broker_trade_records(findings, result)
        self.prune_inactive_local_orders(findings, result)
        self.recover_missing_active_orders(findings, broker, result)
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
        broker: BrokerSnapshot,
        result: OrderRecoveryResult,
    ) -> None:
        broker_trades = broker.trades_by_order_id_or_perm_id
        fills_by_order_id = broker.fills_by_order_id

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
    broker: BrokerSnapshot,
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
        elif strategies and broker.position_for_contract(contract) == 0:
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
        local: LocalSnapshot,
        broker: BrokerSnapshot,
        missing_brackets: str,
    ) -> None:
        self.controller = controller
        self.local = local
        self.broker = broker
        self.missing_brackets = missing_brackets

    def run(self) -> BracketSyncResult:
        result = BracketSyncResult()
        if self.missing_brackets == "ignore":
            return result

        bracket_findings = compare_bracket_records(self.local)
        protection_findings = compare_broker_protection(self.broker)
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
        strategies = self.local.strategies_by_contract.get(issue.contract) or []
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
        self.attempt = 0
        self.broker: BrokerSnapshot | None = None
        self.local: LocalSnapshot | None = None
        self.order_findings: OrderFindings | None = None
        self.order_recovery = OrderRecoveryResult()
        self.position_findings = PositionFindings()
        self.position_recovery = PositionRecoveryResult()
        self.bracket_result = BracketSyncResult()

    async def run(self) -> bool:
        """Run sync until it completes, aborts, or fails to converge."""
        log.debug("--- Sync ---")

        for attempt in range(1, self.controller.sync_max_attempts + 1):
            self.reset_attempt(attempt)

            if not self.verify_broker_connection():
                return False
            if not await self.capture_broker_state():
                return False

            self.capture_local_state()
            self.compare_orders()
            await self.apply_order_recovery()
            if await self.should_resync("order recovery"):
                continue

            self.compare_positions()
            self.apply_position_recovery()
            if await self.should_resync("position recovery"):
                continue

            if not self.verify_position_sync():
                return False
            if self.has_unresolved_unknown_orders():
                return self.complete()

            self.apply_bracket_sync()
            if self.bracket_result.blocked_reason:
                self.block_trading(self.bracket_result.blocked_reason)
                return False
            if self.bracket_result.terminal_action:
                return self.complete()
            if await self.should_resync("bracket recovery"):
                continue

            return self.complete()

        return self.fail_to_converge()

    def reset_attempt(self, attempt: int) -> None:
        """Reset per-attempt state before acquiring fresh snapshots."""
        self.attempt = attempt
        self.broker = None
        self.local = None
        self.order_findings = None
        self.order_recovery = OrderRecoveryResult()
        self.position_findings = PositionFindings()
        self.position_recovery = PositionRecoveryResult()
        self.bracket_result = BracketSyncResult()
        log.debug(f"Sync attempt {attempt}/{self.controller.sync_max_attempts}")

    def verify_broker_connection(self) -> bool:
        """Return False and disable trading if broker is disconnected."""
        if self.controller.ib.isConnected():
            return True

        log.debug("No connection. Abandoning sync.")
        self.block_trading("broker not connected")
        return False

    async def capture_broker_state(self) -> bool:
        """Capture and validate broker state for this attempt."""
        try:
            self.broker = await capture_broker_snapshot(
                self.controller.ib,
                self.controller.broker_request_timeout,
            )
            return True
        except BrokerSnapshotError as exc:
            self.block_trading(exc.reason)
            return False

    def capture_local_state(self) -> None:
        """Capture local state-machine records for this attempt."""
        self.local = capture_local_snapshot(self.controller.sm)

    def compare_orders(self) -> None:
        """Compare local order records with broker open/session state."""
        assert self.local is not None
        assert self.broker is not None
        self.order_findings = compare_order_state(self.local, self.broker)

    async def apply_order_recovery(self) -> None:
        """Apply order recovery and relink broker trades to local records."""
        assert self.order_findings is not None
        assert self.broker is not None
        self.controller.release_hold()
        self.order_recovery = await OrderSyncApplier(
            self.controller.ib,
            self.controller.sm,
            self.controller,
            self.controller.cancel_unknown_trades,
        ).apply(self.order_findings, self.broker)

    def has_unresolved_unknown_orders(self) -> bool:
        """Return True when unknown broker orders remain active."""
        if not self.order_recovery.has_unresolved_unknown_orders:
            return False

        log.error(
            "Unknown broker orders remain active; "
            "skipping position-protection correction trades."
        )
        return True

    def compare_positions(self) -> None:
        """Compare local strategy positions with broker positions."""
        assert self.local is not None
        assert self.broker is not None
        self.position_findings = compare_position_state(self.local, self.broker)

    def apply_position_recovery(self) -> None:
        """Correct local position records to match broker positions."""
        assert self.broker is not None
        self.position_recovery = apply_position_recovery(
            self.controller.sm,
            self.broker,
            self.position_findings,
            self.order_recovery,
        )

    def verify_position_sync(self) -> bool:
        """Verify local positions still match the current broker snapshot."""
        assert self.broker is not None
        final_local = capture_local_snapshot(self.controller.sm)
        recheck = compare_position_state(final_local, self.broker)
        if not recheck.errors:
            self.local = final_local
            return True

        self.block_trading("local state does not match broker state")
        return False

    def apply_bracket_sync(self) -> None:
        """Check local bracket records and broker stop-loss protection."""
        assert self.local is not None
        assert self.broker is not None
        self.bracket_result = BracketSyncer(
            self.controller,
            self.local,
            self.broker,
            self.controller.missing_brackets,
        ).run()

    async def should_resync(self, reason: str) -> bool:
        """Return True when recovery changed state and snapshots should refresh."""
        if not self.changed_local and not self.changed_broker:
            return False

        log.error(f"Sync {reason} changed state; will resync with fresh snapshots.")
        await asyncio.sleep(self.controller.sync_resync_delay)
        return True

    @property
    def changed_local(self) -> bool:
        """Return True when this attempt changed local state."""
        return self.order_recovery.changed_local or self.position_recovery.changed_local

    @property
    def changed_broker(self) -> bool:
        """Return True when this attempt changed broker state."""
        return self.order_recovery.changed_broker or self.bracket_result.changed_broker

    def block_trading(self, reason: str) -> None:
        """Disable future trading from one sync-owned decision point."""
        self.controller.disable_trading(reason)

    def fail_to_converge(self) -> bool:
        """Block trading when repeated recovery prevents a stable sync."""
        self.block_trading("sync did not converge")
        return False

    def complete(self) -> bool:
        """Log successful sync completion."""
        log.debug("--- Sync completed ---")
        return True
