from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import ib_insync as ibi

from haymaker import misc
from haymaker.state_machine import OrderInfo, StateMachine

from .sync_reconciliation import BracketReconciler, BrokerProtectionReconciler
from .sync_types import (
    BracketFindings,
    BracketSyncResult,
    BrokerProtectionFindings,
    BrokerProtectionIssue,
    BrokerSnapshot,
    LocalSnapshot,
    OrderFindings,
    OrderRecoveryResult,
    PositionFindings,
    PositionRecoveryResult,
)

if TYPE_CHECKING:
    from .controller import Controller

log = logging.getLogger(__name__)


class OrderSyncApplier:
    """Apply order reconciliation findings to local records and broker actions."""

    def __init__(
        self,
        ib: ibi.IB,
        sm: StateMachine,
        controller: Controller,
        cancel_unknown_trades: bool,
    ) -> None:
        """Initialize the applier with sync dependencies."""
        self.ib = ib
        self.sm = sm
        self.controller = controller
        self.cancel_unknown_trades = cancel_unknown_trades

    async def apply(
        self, findings: OrderFindings, broker: BrokerSnapshot
    ) -> OrderRecoveryResult:
        """Apply order findings and return a summary of recovery actions."""
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
        """Replace local trade objects with active broker trade objects."""
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
        """Prune local inactive orders that are not active at the broker."""
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
        """Resolve local active orders that are missing from broker open orders."""
        broker_trades = broker.trades_by_order_id_or_perm_id
        fills_by_order_id = broker.fills_by_order_id

        if findings.missing_active_orders:
            inactive_trades = [
                (i.trade.order.orderId, i.trade.order.permId)
                for i in findings.missing_active_orders
            ]
            log.debug(f"inactive trades: " f"{inactive_trades}")
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
        """Update ``trade`` from broker fills when no completed trade is available."""
        trade.fills = fills
        filled = sum(fill.execution.shares for fill in fills)
        remaining = trade.order.totalQuantity - filled
        trade.log.append(
            ibi.objects.TradeLogEntry(
                time=fills[-1].execution.time,
                status="Filled" if remaining == 0 else "Submitted",
                message="composed by sync_actions",
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
        """Cancel or report broker orders that do not exist in local records."""
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
        """Emit synthetic broker events for a done trade found during sync."""
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
        """Prune local active orders that broker state could not resolve."""
        for order_info in result.faulty_orders:
            trade = order_info.trade
            log.error(
                f"Will delete record for trade that IB doesn't known about: "
                f"{trade.order.orderId}"
            )
            self.sm.prune_order(trade.order.orderId)


class PositionRecordApplier:
    """Apply position reconciliation findings to local records."""

    def __init__(self, sm: StateMachine, broker: BrokerSnapshot) -> None:
        """Initialize the applier with local state and broker snapshot."""
        self.sm = sm
        self.broker = broker

    def apply(
        self, findings: PositionFindings, order_recovery: OrderRecoveryResult
    ) -> PositionRecoveryResult:
        """Correct local position records based on position findings."""
        result = PositionRecoveryResult()
        if not findings.errors:
            return result

        log.error("Will attempt to fix position records")
        strategy_faults = {
            order_info.strategy for order_info in order_recovery.faulty_orders
        }
        for contract, diff in findings.errors.items():
            strategies = self.sm.for_contract.get(contract)
            log.debug(f"Strategies for contract {contract.localSymbol}: {strategies}")
            if strategies and len(strategies) == 1:
                self.sm.strategy[strategies[0]].position -= diff
                log.error(
                    f"Corrected position records for strategy "
                    f"{strategies[0]} by {-diff}"
                )
                self.sm.save_strategies()
                result.corrected_contracts.append(contract)
            elif strategies and self.broker.position_for_contract(contract) == 0:
                for strategy in strategies:
                    self.sm.strategy[strategy].position = 0
                self.sm.save_strategies()
                log.error(
                    f"Position records zeroed for {strategies} "
                    f"to reflect zero position for {contract.symbol}."
                )
                result.corrected_contracts.append(contract)
            elif strategies:
                for strategy in strategies:
                    if strategy in strategy_faults:
                        self.sm.strategy[strategy].position = 0
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
        """Initialize bracket sync with local records and broker snapshot."""
        self.controller = controller
        self.local = local
        self.broker = broker
        self.missing_brackets = missing_brackets

    def run(self) -> BracketSyncResult:
        """Apply configured bracket checks and return broker action summary."""
        result = BracketSyncResult()
        if self.missing_brackets == "ignore":
            return result

        bracket_findings = BracketReconciler().compare(self.local)
        protection_findings = BrokerProtectionReconciler().compare(self.broker)
        self.report_bracket_record_findings(bracket_findings)
        self.report_broker_protection_findings(protection_findings)

        if self.missing_brackets == "remove":
            self.cancel_obsolete_orders(bracket_findings, result)
            self.close_exposed_positions(protection_findings, result)

        return result

    def report_bracket_record_findings(self, findings: BracketFindings) -> None:
        """Log local bracket record inconsistencies."""
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
        """Log broker positions that lack stop-loss protection."""
        for issue in findings.exposed_positions:
            log.error(
                f"Broker position lacks stop-loss protection: "
                f"{issue.contract.localSymbol} {issue.broker_position}"
            )

    def cancel_obsolete_orders(
        self, findings: BracketFindings, result: BracketSyncResult
    ) -> None:
        """Cancel active local bracket/closing orders for flat strategies."""
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
        """Close unprotected broker positions when they can be attributed safely."""
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
        """Return the single local strategy for a broker position contract."""
        strategies = self.local.strategies_by_contract.get(issue.contract) or []
        if len(strategies) == 1:
            return strategies[0]

        log.critical(
            f"Cannot close unprotected broker position for "
            f"{issue.contract.localSymbol}: {strategies=}."
        )
        return None
