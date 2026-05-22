"""Bracket and broker-protection checks used during controller sync.

The bracket policy is configured globally with
``controller.missing_brackets``.  It is intentionally not strategy-specific:
the running system either expects bracket protection or it does not.

Policy values:
    ``ignore``:
        Skip local bracket-record checks and broker stop-loss protection
        checks.
    ``warn``:
        Log local bracket-record mismatches and broker positions that do not
        have enough opposite-side stop-loss protection.  Do not send broker
        orders or cancellations.
    ``remove``:
        Perform the same checks as ``warn``, cancel obsolete local bracket
        orders, and submit emergency close orders for broker positions that
        can be attributed to exactly one local strategy and lack stop-loss
        protection.

The broker protection decision is based on direct broker state queried during
sync.  Local state is used only to identify obsolete local bracket orders and
to attribute an exposed broker contract to one strategy before submitting an
emergency close.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import ib_insync as ibi

from haymaker.state_machine import OrderInfo, StateMachine, Strategy

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


def active_orders_for_strategy(
    sm: StateMachine, strategy_name: str
) -> tuple[OrderInfo, ...]:
    """Return current active order records for a strategy."""
    return tuple(
        order_info
        for order_info in sm.order.values()
        if order_info.active and order_info.strategy == strategy_name
    )


def broker_positions_by_contract(ib: ibi.IB) -> dict[ibi.Contract, float]:
    """Return current broker positions keyed by contract."""
    return {position.contract: position.position for position in ib.positions()}


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


def local_strategies_by_contract(sm: StateMachine) -> dict[ibi.Contract, list[str]]:
    """Return current strategy names grouped by active contract."""
    return dict(sm.strategy.strategies_by_contract())


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


class BracketSyncer:
    """Check bracket records and broker stop-loss protection.

    The syncer applies ``missing_brackets`` as a global system policy.  It
    never creates replacement brackets.  In ``remove`` mode it only removes
    stale bracket orders and flattens broker exposure that lacks stop-loss
    protection, because continuing to trade an unprotected broker position is
    more dangerous than leaving local records untouched.
    """

    def __init__(
        self,
        controller: Controller,
        missing_brackets: MissingBracketsPolicy,
    ) -> None:
        """Initialize bracket synchronization for one controller sync run."""
        self.controller = controller
        self.missing_brackets = missing_brackets

    def run(self) -> BracketSyncResult:
        """Apply configured bracket/protection checks and remedies.

        Returns:
            A result describing broker-affecting actions and any condition
            that should block the broader sync from continuing normally.
        """
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
        """Log local bracket record mismatches."""
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
        """Cancel active bracket orders for strategies without a position."""
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
        """Submit emergency closes for attributable unprotected broker positions.

        A close is submitted only when exactly one local strategy maps to the
        exposed broker contract.  Ambiguous attribution is reported through
        ``result.blocked_reason`` so the coordinator can disable trading
        rather than guess which strategy should receive the closing trade.
        """
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
        """Return the single local strategy owning an exposed broker contract."""
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
