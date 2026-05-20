from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .sync_actions import (
    BracketActionExecutor,
    OrderSyncApplier,
    PositionRecordApplier,
)
from .sync_reconciliation import BracketReconciler, OrderReconciler, PositionReconciler
from .sync_snapshots import (
    BrokerSnapshotError,
    capture_broker_snapshot,
    capture_local_snapshot,
    verify_broker_connected,
)
from .sync_types import (
    BrokerSnapshot,
    OrderRecoveryResult,
    PositionFindings,
    SyncResult,
)

if TYPE_CHECKING:
    from .controller import Controller

log = logging.getLogger(__name__)


class SyncCoordinator:
    """Coordinate broker/local sync phases and trading-disable decisions."""

    def __init__(self, controller: Controller) -> None:
        """Initialize the coordinator for one controller sync run."""
        self.controller = controller

    async def run(self) -> SyncResult:
        """Run one complete sync cycle."""
        log.debug("--- Sync ---")

        connection_result = verify_broker_connected(self.controller.ib)
        if not connection_result.ok:
            self.controller.disable_trading(connection_result.reason)
            return connection_result

        broker_result = await self.capture_valid_broker_state()
        if isinstance(broker_result, SyncResult):
            return broker_result
        broker = broker_result

        local = capture_local_snapshot(self.controller.sm)
        order_findings = OrderReconciler().compare(local, broker)

        # IB events will be handled so that matched trades can be sent to blotter.
        self.controller.release_hold()
        order_recovery = await OrderSyncApplier(
            self.controller.ib,
            self.controller.sm,
            self.controller,
            self.controller.cancel_unknown_trades,
        ).apply(order_findings, broker)

        local_after_orders = capture_local_snapshot(self.controller.sm)
        position_findings = PositionReconciler().compare(local_after_orders, broker)
        PositionRecordApplier(self.controller.sm, broker).apply(
            position_findings, order_recovery
        )

        final_local = capture_local_snapshot(self.controller.sm)
        recheck = PositionReconciler().compare(final_local, broker)
        if recheck.errors:
            reason = "local state does not match broker state"
            self.controller.disable_trading(reason)
            return SyncResult(False, reason)

        if self.recovery_happened(order_recovery, position_findings):
            log.error(
                "Order or position recovery happened during sync; "
                "skipping correction trades for this sync cycle."
            )
            log.debug("--- Sync completed ---")
            return SyncResult(True, "recovery completed")

        bracket_findings = BracketReconciler().compare(final_local)
        BracketActionExecutor(
            self.controller,
            broker,
            self.controller.cancel_stray_orders,
            self.controller.handle_missing_brackets,
        ).apply(bracket_findings)

        log.debug("--- Sync completed ---")
        return SyncResult(True)

    async def capture_valid_broker_state(self) -> BrokerSnapshot | SyncResult:
        """Return a broker snapshot or a failed sync result."""
        try:
            return await capture_broker_snapshot(
                self.controller.ib,
                self.controller.broker_request_timeout,
            )
        except BrokerSnapshotError as exc:
            self.controller.disable_trading(exc.result.reason)
            return exc.result

    def recovery_happened(
        self,
        order_recovery: OrderRecoveryResult,
        position_findings: PositionFindings,
    ) -> bool:
        """Return True when sync performed or found recovery work."""
        return order_recovery.recovery_happened or bool(position_findings.errors)
