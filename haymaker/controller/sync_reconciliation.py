from __future__ import annotations

import logging

from haymaker.state_machine import OrderInfo, Strategy

from .sync_types import (
    BracketFindings,
    BracketIssue,
    BrokerProtectionFindings,
    BrokerProtectionIssue,
    BrokerSnapshot,
    LocalSnapshot,
    OrderFindings,
    PositionFindings,
)

log = logging.getLogger(__name__)


class OrderReconciler:
    """Compare local order records with a broker snapshot."""

    def compare(self, local: LocalSnapshot, broker: BrokerSnapshot) -> OrderFindings:
        """Return read-only order findings without mutating local state."""
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
        self.report(findings)
        return findings

    def report(self, findings: OrderFindings) -> None:
        """Log a summary of order findings."""
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


class PositionReconciler:
    """Compare local strategy positions with a broker snapshot."""

    def compare(self, local: LocalSnapshot, broker: BrokerSnapshot) -> PositionFindings:
        """Return read-only position findings without mutating local state."""
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


class BracketReconciler:
    """Compare local positions with local bracket/order records."""

    def compare(self, local: LocalSnapshot) -> BracketFindings:
        """Return read-only bracket findings without mutating local state."""
        obsolete_orders: list[tuple[str, OrderInfo]] = []
        missing_brackets: list[BracketIssue] = []

        for strategy_str, strategy in local.strategies.items():
            order_infos = local.orders_for_strategy(strategy.strategy)
            if strategy.position == 0:
                obsolete_orders.extend(
                    (strategy_str, order_info)
                    for order_info in self.find_obsolete_orders(order_infos)
                )
            elif issue := self.find_missing_brackets(strategy, order_infos):
                missing_brackets.append(issue)

        return BracketFindings(
            obsolete_orders=tuple(obsolete_orders),
            missing_brackets=tuple(missing_brackets),
        )

    def find_obsolete_orders(
        self, order_infos: tuple[OrderInfo, ...]
    ) -> list[OrderInfo]:
        """Find active closing orders for a strategy that has no position."""
        return [
            order_info
            for order_info in order_infos
            if order_info.action != "OPEN" and order_info.active
        ]

    def find_missing_brackets(
        self, strategy: Strategy, order_infos: tuple[OrderInfo, ...]
    ) -> BracketIssue | None:
        """Return missing bracket issue for ``strategy``, if one exists."""
        params = strategy.get("params")
        if not params:
            return None

        brackets = [
            bracket
            for bracket_name in ("stop-loss", "take-profit")
            if (bracket := params.get(bracket_name))
        ]
        existing_orders = tuple(
            order_info
            for order_info in order_infos
            if order_info.action in ("STOP-LOSS", "TAKE-PROFIT") and order_info.active
        )
        if len(brackets) == len(existing_orders):
            return None

        return BracketIssue(
            strategy=strategy,
            existing_orders=existing_orders,
            expected_bracket_count=len(brackets),
        )


class BrokerProtectionReconciler:
    """Find broker positions that lack stop-loss protection."""

    def compare(self, broker: BrokerSnapshot) -> BrokerProtectionFindings:
        """Return broker positions not covered by active stop-loss orders."""
        exposed_positions = [
            BrokerProtectionIssue(
                contract=contract,
                broker_position=position,
            )
            for contract, position in broker.positions_by_contract.items()
            if position and not broker.position_is_protected(contract, position)
        ]
        return BrokerProtectionFindings(exposed_positions=tuple(exposed_positions))
