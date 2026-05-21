from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from .sync_actions import BracketSyncer, OrderSyncApplier, PositionRecordApplier
from .sync_reconciliation import OrderReconciler, PositionReconciler
from .sync_snapshots import (
    BrokerSnapshotError,
    capture_broker_snapshot,
    capture_local_snapshot,
)
from .sync_types import (
    BracketSyncResult,
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
        self.order_findings = OrderReconciler().compare(self.local, self.broker)

    async def apply_order_recovery(self) -> None:
        """Apply order recovery and relink broker trades to local records."""
        assert self.order_findings is not None
        assert self.broker is not None
        # IB events will be handled so matched trades can be sent to blotter.
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
        self.position_findings = PositionReconciler().compare(self.local, self.broker)

    def apply_position_recovery(self) -> None:
        """Correct local position records to match broker positions."""
        assert self.broker is not None
        self.position_recovery = PositionRecordApplier(
            self.controller.sm,
            self.broker,
        ).apply(self.position_findings, self.order_recovery)

    def verify_position_sync(self) -> bool:
        """Verify local positions still match the current broker snapshot."""
        assert self.broker is not None
        final_local = capture_local_snapshot(self.controller.sm)
        recheck = PositionReconciler().compare(final_local, self.broker)
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
