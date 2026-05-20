from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import TYPE_CHECKING

import ib_insync as ibi

from haymaker import misc
from haymaker.state_machine import OrderInfo, StateMachine, Strategy

from .sync_types import (
    BracketFindings,
    BrokerSnapshot,
    OrderFindings,
    OrderRecoveryResult,
    PositionFindings,
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
        self.handle_unknown_broker_orders(result.unknown_broker_trades)

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

    def handle_unknown_broker_orders(self, trades: list[ibi.Trade]) -> None:
        """Cancel or report broker orders that do not exist in local records."""
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
    ) -> None:
        """Correct local position records based on position findings."""
        if not findings.errors:
            return

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
            elif strategies and self.broker.position_for_contract(contract) == 0:
                for strategy in strategies:
                    self.sm.strategy[strategy].position = 0
                self.sm.save_strategies()
                log.error(
                    f"Position records zeroed for {strategies} "
                    f"to reflect zero position for {contract.symbol}."
                )
            elif strategies:
                for strategy in strategies:
                    if strategy in strategy_faults:
                        self.sm.strategy[strategy].position = 0
                        log.error(
                            f"Position records zeroed for {strategy} "
                            f"to reflect faulty trade previously removed."
                        )
            else:
                log.critical(
                    f"Cannot fix position records for {contract.localSymbol}, "
                    f"{strategies=}."
                )


class EmergencyCloseGuard:
    """Authorize emergency closes from local state and a broker snapshot."""

    def __init__(
        self,
        sm: StateMachine,
        broker: BrokerSnapshot,
        trading_disabled: bool,
    ) -> None:
        """Initialize the guard with local records and captured broker state."""
        self.sm = sm
        self.broker = broker
        self.trading_disabled = trading_disabled

    def can_close(self, strategy: Strategy) -> bool:
        """Return True if broker state confirms an unprotected position."""
        if self.trading_disabled:
            log.debug(
                f"Emergency close suppressed because trading is disabled: "
                f"{strategy.strategy}"
            )
            return False

        contract = strategy.active_contract
        broker_position = self.broker.position_for_contract(contract)
        if not broker_position:
            log.error(
                f"Emergency close suppressed for {strategy.strategy}: "
                f"broker reports no position for {contract.localSymbol}."
            )
            return False

        strategies = self.sm.for_contract.get(contract) or []
        if len(strategies) != 1:
            log.error(
                f"Emergency close suppressed for {strategy.strategy}: "
                f"contract {contract.localSymbol} is shared by strategies "
                f"{strategies}."
            )
            return False

        if self.broker.position_is_protected(contract, broker_position):
            log.error(
                f"Emergency close suppressed for {strategy.strategy}: "
                f"broker position for {contract.localSymbol} appears protected."
            )
            return False

        return True


class BracketActionExecutor:
    """Apply bracket findings after broker/local recovery has completed."""

    def __init__(
        self,
        controller: Controller,
        broker: BrokerSnapshot,
        cancel_stray_orders: bool,
        handle_missing_brackets: str,
    ) -> None:
        """Initialize the executor with controller policy and broker snapshot."""
        self.controller = controller
        self.broker = broker
        self.cancel_stray_orders = cancel_stray_orders
        self.handle_missing_brackets = handle_missing_brackets
        self.emergency_close_guard = EmergencyCloseGuard(
            controller.sm,
            broker,
            controller._trading_disabled,
        )

    def apply(self, findings: BracketFindings) -> None:
        """Apply bracket findings according to controller policy."""
        self.handle_obsolete_orders(findings)
        if self.handle_missing_brackets not in ["remove", "warn"]:
            return
        self.handle_missing_bracket_issues(findings)

    def handle_obsolete_orders(self, findings: BracketFindings) -> None:
        """Cancel or report active closing orders for flat strategies."""
        for strategy_str, order_info in findings.obsolete_orders:
            order_id = order_info.trade.order.orderId
            if self.cancel_stray_orders:
                log.error(
                    f"Cancelling obsolete order: "
                    f"{strategy_str, order_info.action, order_id}"
                )
                self.controller.trader.cancel(order_info.trade)
            else:
                log.critical(
                    f"Obsolete order for "
                    f"{strategy_str}: {order_info.action, order_id}"
                )

    def handle_missing_bracket_issues(self, findings: BracketFindings) -> None:
        """Warn or close positions for strategies with missing brackets."""
        for issue in findings.missing_brackets:
            strategy = issue.strategy
            log.error(
                f"Bracket error for {strategy.strategy}, "
                f"position: {strategy.position} "
                f"we have: {len(issue.existing_orders)} orders, "
                f"we should have: {issue.expected_bracket_count} orders."
            )

        if self.handle_missing_brackets != "remove":
            return

        non_cancelling_positions: defaultdict[ibi.Contract, float] = defaultdict(float)
        for issue in findings.missing_brackets:
            strategy = issue.strategy
            non_cancelling_positions[strategy.active_contract] += strategy.position
            self.controller.lock_new_positions()

        for issue in findings.missing_brackets:
            strategy = issue.strategy
            if not non_cancelling_positions.get(strategy.active_contract):
                continue
            if not self.emergency_close_guard.can_close(strategy):
                continue
            log.error(
                f"Closing positions for strategy with missing bracket: "
                f"{strategy.strategy}"
            )
            non_cancelling_positions[strategy.active_contract] -= strategy.position
            self.controller.close_positions_for_strategy(
                strategy.strategy, "MISSING BRACKET EMERGENCY CLOSE"
            )
