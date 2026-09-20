"""Critical stop-loss and optional take-profit checks for Controller sync."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import ib_insync as ibi

from haymaker.book import OrderInfo, PositionState
from haymaker.components.messages import StandardOrderRole

if TYPE_CHECKING:
    from .controller import Controller

log = logging.getLogger(__name__)

MissingBracketsPolicy = Literal["ignore", "warn", "remove"]


class BracketSyncError(Exception):
    """Raised when configured bracket remediation changed local state."""


@dataclass(frozen=True)
class BracketIssue:
    """Describe a one-to-one position missing its protective stop."""

    state: PositionState
    existing_orders: tuple[OrderInfo, ...]


@dataclass(frozen=True)
class ProtectionIssue:
    """Describe a broker position lacking enough stop protection."""

    contract: ibi.Contract
    broker_position: float


@dataclass
class BracketSync:
    """Collect critical stop-loss attribution and broker-protection issues.

    A configured stop-loss is required for an established bracket-managed
    position. Take-profit orders are optional execution conveniences: their
    absence is not a synchronization issue.
    """

    controller: Controller
    missing_brackets: list[BracketIssue] = field(default_factory=list)
    obsolete_brackets: list[tuple[str, OrderInfo]] = field(default_factory=list)
    exposed_positions: list[ProtectionIssue] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.compare_bracket_records()
        self.check_stop_protection()

    @property
    def local_records_issue(self) -> bool:
        return bool(self.missing_brackets or self.obsolete_brackets)

    def compare_bracket_records(self) -> None:
        """Require attributed stops without requiring optional take-profits."""

        for source_key, state in self.controller.book.positions.source_states().items():
            orders = self.controller.book.orders.active(source_key=source_key)
            entry_active = any(info.role == StandardOrderRole.OPEN for info in orders)
            exit_active = any(info.role == StandardOrderRole.CLOSE for info in orders)
            brackets = tuple(
                info
                for info in orders
                if info.role
                in {
                    StandardOrderRole.STOP_LOSS,
                    StandardOrderRole.TAKE_PROFIT,
                }
            )
            if not state.quantity and not exit_active:
                self.obsolete_brackets.extend((source_key, info) for info in brackets)
            elif (
                state.quantity
                and state.bracket_inputs
                and not entry_active
                and not exit_active
                and self.controller.book.rolls.for_source(source_key) is None
                and not self._episode_is_protected(state, brackets)
            ):
                self.missing_brackets.append(
                    BracketIssue(state=state, existing_orders=brackets)
                )

    @staticmethod
    def _episode_is_protected(
        state: PositionState, brackets: tuple[OrderInfo, ...]
    ) -> bool:
        """Require a stop covering this episode's concrete exposure."""
        return any(
            info.role == StandardOrderRole.STOP_LOSS
            and info.position_id == state.position_id
            and info.trade.contract == state.contract
            and info.trade.order.action == ("SELL" if state.quantity > 0 else "BUY")
            and info.trade.order.orderType in {"STP", "STP LMT", "TRAIL", "FIX PEGGED"}
            and info.trade.remaining() >= abs(state.quantity)
            for info in brackets
        )

    def check_stop_protection(self) -> None:
        """Find broker positions without enough opposite-side stop quantity."""

        self.exposed_positions = [
            ProtectionIssue(position.contract, position.position)
            for position in self.controller.ib.positions()
            if position.position
            and not self._position_is_protected(position.contract, position.position)
        ]

    def _position_is_protected(self, contract: ibi.Contract, quantity: float) -> bool:
        action = "SELL" if quantity > 0 else "BUY"
        types = {"STP", "STP LMT", "TRAIL", "FIX PEGGED"}
        protected = sum(
            trade.remaining()
            for trade in self.controller.ib.openTrades()
            if trade.contract == contract
            and trade.isActive()
            and trade.order.action == action
            and trade.order.orderType in types
        )
        return protected >= abs(quantity)


class BracketSyncAction(ABC):
    """Apply one configured missing-bracket policy."""

    def __init__(self, controller: Controller) -> None:
        self.controller = controller
        self.bracket_sync = BracketSync(controller)
        self.sync()

    @abstractmethod
    def sync(self) -> None:
        """Apply the concrete policy."""

    @classmethod
    def from_policy(
        cls, policy: MissingBracketsPolicy, controller: Controller
    ) -> BracketSyncAction:
        actions: dict[str, type[BracketSyncAction]] = {
            "ignore": IgnoreBracketSyncAction,
            "warn": WarnBracketSyncAction,
            "remove": RemoveBracketSyncAction,
        }
        try:
            action = actions[policy]
        except KeyError as exc:
            raise ValueError(f"Unknown missing-brackets policy: {policy!r}") from exc
        return action(controller)

    def report(self) -> None:
        for source_key, info in self.bracket_sync.obsolete_brackets:
            log.error(
                "Obsolete bracket for %s: %s orderId=%s",
                source_key,
                info.role,
                info.orderId,
            )
        for issue in self.bracket_sync.missing_brackets:
            log.error(
                "Missing stop for %s position=%s",
                issue.state.source_key,
                issue.state.quantity,
            )
        for protection in self.bracket_sync.exposed_positions:
            log.error(
                "Broker position lacks stop protection: %s %s",
                protection.contract.localSymbol,
                protection.broker_position,
            )


class IgnoreBracketSyncAction(BracketSyncAction):
    """Skip all bracket checks."""

    def __init__(self, controller: Controller) -> None:
        self.controller = controller

    def sync(self) -> None:
        pass


class WarnBracketSyncAction(BracketSyncAction):
    """Report bracket findings without changing broker or Book state."""

    def sync(self) -> None:
        self.report()


class RemoveBracketSyncAction(BracketSyncAction):
    """Close missing-bracket sources and cancel obsolete brackets."""

    def sync(self) -> None:
        self.report()
        for issue in self.bracket_sync.missing_brackets:
            self.controller.close_position_for_source(
                issue.state.source_key,
                role="MISSING_BRACKET_EMERGENCY_CLOSE",
            )
        for _, info in self.bracket_sync.obsolete_brackets:
            self.controller.cancel(info.trade)
        if self.bracket_sync.local_records_issue:
            raise BracketSyncError()
