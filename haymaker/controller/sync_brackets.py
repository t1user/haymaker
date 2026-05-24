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
        orders, and close local strategy positions when local records say a
        bracket should exist but no active local bracket order is present.

The broker protection decision is based on direct broker state queried during
sync.  Local state is used to identify obsolete local bracket orders and local
strategy positions whose recorded bracket orders are missing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Self

import ib_insync as ibi
from haymaker.state_machine import OrderInfo, StateMachine, Strategy

if TYPE_CHECKING:
    from .controller import Controller

log = logging.getLogger(__name__)

MissingBracketsPolicy = Literal["ignore", "warn", "remove"]


class BracketSyncError(Exception):
    """Raised when bracket synchronization leaves state that must stop trading."""


@dataclass(frozen=True)
class BracketIssue:
    """Missing bracket information for one strategy."""

    strategy: Strategy
    existing_orders: tuple[OrderInfo, ...]
    expected_bracket_count: int


@dataclass(frozen=True)
class ProtectionIssue:
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


@dataclass
class BracketSync:
    """Collect local bracket-record issues and broker stop-protection issues."""

    controller: Controller
    missing_brackets: list[BracketIssue] = field(default_factory=list)
    obsolete_brackets: list[tuple[str, OrderInfo]] = field(default_factory=list)
    exposed_positions: list[ProtectionIssue] = field(default_factory=list)

    _sm: StateMachine = field(init=False, repr=False)
    _ib: ibi.IB = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Read current local and broker state for bracket checks."""
        self._sm = self.controller.sm
        self._ib = self.controller.ib

        self.compare_bracket_records()
        self.check_stop_protection()

    @property
    def has_issues(self) -> bool:
        """Return True when any local bracket or broker protection issue exists."""
        return any(
            (self.missing_brackets, self.obsolete_brackets, self.exposed_positions)
        )

    @property
    def local_records_issue(self) -> bool:
        """Return True when local strategy and bracket-order records disagree."""
        return any((self.missing_brackets, self.obsolete_brackets))

    def compare_bracket_records(self) -> None:
        """Compare local positions with local bracket/order records."""

        for strategy_str, strategy in self._sm.strategy.items():
            order_infos = self._sm.orders_for_strategy(strategy_str)
            # Find stop/take-profit/close orders for a strategy that
            # has no position
            if strategy.position == 0:
                self.obsolete_brackets.extend(
                    self._find_obsolete_brackets(strategy_str, order_infos)
                )
            elif bracket_issue := self._find_missing_brackets(strategy, order_infos):
                self.missing_brackets.append(bracket_issue)

    def _find_obsolete_brackets(
        self, strategy_str: str, order_infos: list[OrderInfo]
    ) -> list[tuple[str, OrderInfo]]:
        """Find stop/take-profit/close orders for a strategy that has no position."""
        return [
            (strategy_str, order_info)
            for order_info in order_infos
            # there is no position and we're not trying to open a new one
            if order_info.action != "OPEN" and order_info.active
        ]

    def _find_missing_brackets(
        self, strategy: Strategy, order_infos: list[OrderInfo]
    ) -> BracketIssue | None:
        """Find missing local bracket orders for a strategy with a position."""
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
                return BracketIssue(
                    strategy=strategy,
                    existing_orders=existing_orders,
                    expected_bracket_count=len(brackets),
                )

    def check_stop_protection(self) -> None:
        """Find broker positions that lack active stop-loss protection."""
        self.exposed_positions = [
            ProtectionIssue(contract=contract, broker_position=position)
            for contract, position in self._positions_by_contract().items()
            if position and not self._position_is_protected(contract, position)
        ]

    def _positions_by_contract(self) -> dict[ibi.Contract, float]:
        """Return current broker positions keyed by contract."""
        return {
            position.contract: position.position for position in self._ib.positions()
        }

    def _trades_for_contract(self, contract: ibi.Contract) -> list[ibi.Trade]:
        """Return current active broker trades for ``contract``."""
        return [trade for trade in self._ib.openTrades() if trade.contract == contract]

    def _position_is_protected(
        self, contract: ibi.Contract, broker_position: float
    ) -> bool:
        """Return True if broker currently has enough opposite-side stop protection."""
        protective_action = "SELL" if broker_position > 0 else "BUY"
        protective_order_types = {"STP", "STP LMT", "TRAIL"}
        protective_quantity = sum(
            trade.order.totalQuantity
            for trade in self._trades_for_contract(contract)
            if (
                trade.isActive()
                and trade.order.action == protective_action
                and trade.order.orderType in protective_order_types
            )
        )
        return protective_quantity >= abs(broker_position)


class BracketSyncAction(ABC):
    """Apply the configured bracket policy to the current sync pass."""

    def __init__(self, controller: Controller) -> None:
        """Collect bracket state and apply the concrete policy."""
        self.bracket_sync = BracketSync(controller)
        self.controller = controller

        self.sync()

    @abstractmethod
    def sync(self) -> None: ...

    @classmethod
    def from_policy(cls, policy: MissingBracketsPolicy, controller: Controller) -> Self:
        """Create and run the action class for ``policy``."""
        try:
            return {
                "ignore": IgnoreBracketSyncAction,
                "warn": WarnBracketSyncAction,
                "remove": RemoveBracketSyncAction,
            }[policy](controller)
        except KeyError:
            raise ValueError(
                f"MissingBracketsPolicy must be one of "
                f"'ignore', 'warn', 'remove', not {policy}"
            )

    def close_exposed_positions(self) -> None:
        """
        Submit emergency closes for unprotected positions.
        CURRENTLY NOT IN USE.
        Might be used in future revisions.
        """
        for issue in self.bracket_sync.exposed_positions:
            log.error(
                f"Closing position without stop-loss protection: "
                f"{issue.contract.localSymbol or issue.contract.symbol or issue.contract} "
                f"{issue.close_action} {issue.close_quantity} "
            )
            self.controller.trader.trade(
                issue.contract,
                ibi.MarketOrder(issue.close_action, issue.close_quantity),
            )

    def close_positions_without_brackets(self) -> None:
        """
        Close local strategy positions whose recorded brackets are missing.

        This handles local records saying a strategy should have a bracket,
        while active local bracket order records are not present.
        """
        for issue in self.bracket_sync.missing_brackets:
            strategy = issue.strategy
            log.error(
                f"Closing positions for strategy with missing bracket: "
                f"{strategy.strategy}"
            )
            self.controller.close_positions_for_strategy(
                strategy.strategy, "MISSING BRACKET EMERGENCY CLOSE"
            )

    def cancel_obsolete_brackets(self) -> None:
        """Cancel active bracket orders for strategies without a position."""
        for strategy_str, order_info in self.bracket_sync.obsolete_brackets:
            order_id = order_info.trade.order.orderId
            log.error(
                f"Cancelling obsolete order: "
                f"{strategy_str, order_info.action, order_id}"
            )
            self.controller.trader.cancel(order_info.trade)
            # result.cancelled_obsolete_orders.append(order_info)

    def report_bracket_record_findings(self) -> None:
        """Log local bracket record mismatches."""
        for strategy_str, order_info in self.bracket_sync.obsolete_brackets:
            order_id = order_info.trade.order.orderId
            log.error(
                f"Obsolete order for " f"{strategy_str}: {order_info.action, order_id}"
            )

        for issue in self.bracket_sync.missing_brackets:
            strategy = issue.strategy
            log.error(
                f"Bracket error for {strategy.strategy}, "
                f"position: {strategy.position} "
                f"we have: {len(issue.existing_orders)} orders, "
                f"we should have: {issue.expected_bracket_count} orders."
            )

    def report_broker_protection_findings(self) -> None:
        """Log broker positions that lack stop-loss protection."""
        for issue in self.bracket_sync.exposed_positions:
            log.error(
                f"Existing position lacks stop-loss protection: "
                f"{issue.contract.localSymbol} {issue.broker_position}"
            )


class IgnoreBracketSyncAction(BracketSyncAction):
    """Skip all bracket and broker-protection checks."""

    def __init__(self, *args) -> None:
        """Do not collect or act on bracket state for the ``ignore`` policy."""
        pass

    def sync(self) -> None:
        """No-op for the ``ignore`` policy."""
        pass


class WarnBracketSyncAction(BracketSyncAction):
    """Log bracket and broker-protection findings without changing state."""

    def sync(self) -> None:
        """Report findings without sending cancellation or close orders."""
        self.report_bracket_record_findings()
        self.report_broker_protection_findings()


class RemoveBracketSyncAction(BracketSyncAction):
    """Apply local bracket remedies and stop trading on remaining local issues."""

    def sync(self) -> None:
        """Close missing-bracket strategy positions and cancel obsolete brackets."""
        self.close_positions_without_brackets()
        self.cancel_obsolete_brackets()
        self.report_broker_protection_findings()
        if self.bracket_sync.local_records_issue:
            raise BracketSyncError()
