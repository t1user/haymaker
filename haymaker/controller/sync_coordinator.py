"""Single-pass controller sync checks for broker and local state.

Sync starts by comparing cached ``ib.positions()`` with a fresh
``await ib.reqPositionsAsync()`` response. A transient disagreement returns
``False`` for a local retry, while an unavailable request asks the owning
supervisor for fresh broker state. The successfully requested positions are
the sole broker snapshot consumed by the rest of that pass.

After broker validation, each step reads current Book and order state while
retaining that one broker-position snapshot. The ordered flow is:

1. Relink broker ``ibi.Trade`` objects to local order records and back-report
   fills for orders that completed while the process was disconnected.
2. Compare local aggregate logical Book positions with the fresh broker
   snapshot, deferring Contracts with active OPEN/CLOSE work.
3. Correct local position records when the existing recovery rules allow it,
   aligning their persisted targets to the authoritative broker quantity.
4. Skip correction trades when unresolved unknown broker orders remain active.
5. Delegate bracket-record and broker stop-loss protection handling to
   :mod:`haymaker.controller.sync_brackets`.

The coordinator does not disable trading and does not retry.  Any recovery
action returns ``False`` so :meth:`Controller.sync` can start a fresh pass from
current broker/local state.  A caller can request a reconnect-before-correction
mode so known-fill housekeeping runs first, but unresolved order or position
errors only set ``request_restart`` instead of mutating broker/local state.
Non-retryable unsafe state raises ``SyncBrokenStateError``.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum, auto
from typing import TYPE_CHECKING

import ib_insync as ibi

from haymaker import misc
from haymaker.book import OrderInfo, PositionState
from haymaker.components.messages import StandardOrderRole

from .sync_brackets import BracketSyncAction, BracketSyncError
from .sync_routines import OrderSync, PositionSync

if TYPE_CHECKING:
    from .controller import Controller

log = logging.getLogger(__name__)


class SyncBrokenStateError(Exception):
    """Raised when sync detects unsafe state that must stop trading."""


class PositionsOutOfSync(Exception):
    """Raised when local positions cannot be reconciled to broker positions."""


class BrokerPositionStatus(Enum):
    """Classify freshness verification of broker position sources."""

    MATCH = auto()
    SNAPSHOT_DISAGREEMENT = auto()
    REQUEST_UNAVAILABLE = auto()


@dataclass(frozen=True)
class BrokerPositionSnapshot:
    """Return explicit broker verification status and the fresh positions."""

    status: BrokerPositionStatus
    positions: tuple[ibi.Position, ...] = ()


class SyncCoordinator:
    """Run one broker/local sync pass and report the outcome.

    ``run()`` returns ``True`` only when the current pass completed cleanly.
    It returns ``False`` after any recovery action so the caller can retry from
    fresh broker/local reads.  Retryable broker connection and broker-state
    verification failures also return ``False``.  Terminal safety failures
    raise ``SyncBrokenStateError``; :class:`Controller` owns the decision to
    disable trading.  Known completed trades are back-reported before position
    comparison even when corrective actions are gated behind a reconnect.
    """

    def __init__(
        self, controller: Controller, restart_before_correction: bool = False
    ) -> None:
        """Initialize the coordinator for one controller sync run.

        Args:
            controller: Controller whose broker and local state should be
                reconciled.
            restart_before_correction: When ``True``, unresolved order or
                position mismatches set ``request_restart`` and end the pass
                before cancelling unknown orders, pruning local order records,
                or changing logical positions.  Trade-object refreshes and
                known completed-fill back-reporting still run first.
        """
        self.controller = controller
        self.request_restart = False
        self._faulty_trades: list[OrderInfo] = []
        self._restart_before_correction = restart_before_correction

    async def run(self) -> bool:
        """Run one sync pass against current broker and local state.

        Returns:
            ``True`` when sync completed without recovery actions or terminal
            safety failures.  ``False`` means the controller should retry the
            sync from fresh broker/local reads.  If ``request_restart`` is set
            after ``False``, the caller should refresh the broker connection
            before the next pass.

        Raises:
            SyncBrokenStateError: Raised for unsafe state that should stop
                trading immediately.
        """

        position_snapshot = await verify_broker_position_source(
            self.controller.ib,
            self.controller.broker_request_timeout,
        )
        if position_snapshot.status is not BrokerPositionStatus.MATCH:
            self.request_restart = (
                position_snapshot.status is BrokerPositionStatus.REQUEST_UNAVAILABLE
            )
            return False

        order_sync = OrderSync(self.controller.ib, self.controller.book)

        self.controller.release_hold()
        if order_sync.done:
            self.handle_done_trades(order_sync.done)
            await asyncio.sleep(0)

        position_sync = PositionSync(
            position_snapshot.positions,
            self.controller.book,
        )
        position_errors = self._defer_active_position_errors(position_sync.errors)

        if (order_sync.is_error or position_errors) and self._restart_before_correction:
            self.request_restart = True
            return False

        # these are corrective actions that will be taken only if they
        # persist after a restart

        # order fixes
        if order_sync.errors:
            self.handle_error_trades(order_sync.errors)
            await asyncio.sleep(0)
        if order_sync.unknown:
            if self.handle_unknown_trades(order_sync.unknown):
                return False
            return True
        if order_sync.done or order_sync.errors:
            return False

        # position fixes
        if position_errors:
            try:
                self.handle_error_positions(
                    position_errors,
                    position_sync.broker_positions,
                )
            except PositionsOutOfSync as exc:
                raise SyncBrokenStateError(
                    "local state does not match broker state"
                ) from exc
            return False

        try:
            BracketSyncAction.from_policy(
                self.controller.missing_brackets,
                self.controller,
            )
        except BracketSyncError as exc:
            raise SyncBrokenStateError("bracket sync failed") from exc
        return True

    def handle_unknown_trades(self, trades: list[ibi.Trade]) -> bool:
        """Cancel unknown broker trades when configured and report if broker changed."""
        log.critical(f"Unknown broker orders during sync: {trades}.")
        if not self.controller.cancel_unknown_trades:
            log.critical(
                "Unknown broker orders left active because "
                "cancel_unknown_trades is False."
            )
            return False

        for trade in trades:
            log.debug(f"Cancelling unknown broker order: {trade.order.orderId}")
            self.controller.cancel(trade)
        return True

    def handle_done_trades(self, trades: list[ibi.Trade]) -> None:
        """
        Events artificially emitted here will trigger registering
        position and saving trade to blotter.
        """
        for trade in trades:
            log.debug(
                f"Back-reporting trade: {trade.contract.symbol} "
                f"{trade.order.action} {misc.trade_fill_price(trade)} "
                f"order id: {trade.order.orderId} {trade.order.permId} "
                f"active?: {trade.isActive()}"
            )
            self.controller.ib.orderStatusEvent.emit(trade)
            for fill in trade.fills:
                self.controller.ib.execDetailsEvent.emit(trade, fill)
            if trade.orderStatus.status == "Filled":
                self.controller.ib.commissionReportEvent.emit(
                    trade, trade.fills[-1], trade.fills[-1].commissionReport
                )

    def handle_error_trades(self, trades: list[ibi.Trade]) -> None:
        """
        Local trades unknown to broker. Local state is corrected, but
        record of trades is kept for further investigation.
        """
        for trade in trades:
            order_id = trade.order.orderId
            info = self.controller.book.order_by_id(order_id)
            if info is not None:
                self._faulty_trades.append(info)
            self.controller.book.prune_order(order_id)
            log.warning(
                "Pruned stale local order %s; order was absent at broker.",
                order_id,
            )

    def _defer_active_position_errors(
        self,
        errors: dict[ibi.Contract, float],
    ) -> dict[ibi.Contract, float]:
        """Defer position correction while one-to-one work can still fill."""

        actionable: dict[ibi.Contract, float] = {}
        for contract, difference in errors.items():
            adjustments = tuple(
                info
                for info in self.controller.book.active_orders(contract=contract)
                if info.role
                in {
                    StandardOrderRole.OPEN,
                    StandardOrderRole.CLOSE,
                }
            )
            if adjustments:
                log.info(
                    "Deferring position reconciliation for %s while "
                    "OPEN/CLOSE order(s) remain active: %s",
                    contract.localSymbol or contract.symbol,
                    [info.orderId for info in adjustments],
                )
            else:
                actionable[contract] = difference
        return actionable

    @staticmethod
    def _corrected_position_state(
        state: PositionState,
        quantity: float,
        corrected_at: datetime,
    ) -> PositionState:
        """Align one recovered episode and target to broker authority."""

        flat = quantity == 0
        return replace(
            state,
            quantity=quantity,
            target_quantity=quantity,
            target_created_at=corrected_at,
            position_id=None if flat else state.position_id,
            bracket_inputs={} if flat else state.bracket_inputs,
            updated_at=corrected_at,
        )

    def handle_error_positions(
        self,
        errors: dict[ibi.Contract, float],
        broker_positions: dict[ibi.Contract, float],
    ) -> None:
        """Correct recoverable Book positions and supersede stale targets."""

        log.error("Will attempt to fix position records")
        for contract, diff in errors.items():
            states = self.controller.book.positions_for_contract(contract)
            corrected_at = datetime.now(timezone.utc)
            log.debug(
                "Sources for contract %s: %s",
                contract.localSymbol,
                [state.source_key for state in states],
            )
            if len(states) == 1:
                state = states[0]
                self.controller.book.update_position(
                    self._corrected_position_state(
                        state,
                        state.quantity - diff,
                        corrected_at,
                    )
                )
                log.error(
                    "Corrected position records for source %s by %s",
                    state.source_key,
                    -diff,
                )

            elif states and broker_positions.get(contract, 0.0) == 0:
                for state in states:
                    self.controller.book.update_position(
                        self._corrected_position_state(
                            state,
                            0.0,
                            corrected_at,
                        )
                    )
                log.error(
                    "Position records zeroed for %s to reflect broker flat.",
                    [state.source_key for state in states],
                )
            elif states:
                source_faults = [
                    order_info.source_key
                    for order_info in self._faulty_trades
                    if order_info.source_key is not None
                ]
                for state in states:
                    if state.source_key in source_faults:
                        self.controller.book.update_position(
                            self._corrected_position_state(
                                state,
                                0.0,
                                corrected_at,
                            )
                        )
                        log.error(
                            "Position records zeroed for %s after faulty trade.",
                            state.source_key,
                        )

            else:
                # too risky to make assumptions about strategy (what about sl?)
                log.critical(
                    f"Cannot fix position records for {contract.localSymbol}, "
                    f"{states=}."
                )
                raise PositionsOutOfSync
            self._faulty_trades.clear()


async def verify_broker_position_source(
    ib: ibi.IB,
    timeout: float,
) -> BrokerPositionSnapshot:
    """Verify cached broker positions and return one fresh snapshot.

    Cached/fresh disagreement can result from ordinary event propagation and
    therefore requests only a local sync retry. A timeout or failed request
    means broker position state is unavailable and requires supervisor-owned
    recovery.

    Args:
        ib: Connected IB client.
        timeout: Maximum seconds to wait for the fresh position request.

    Returns:
        Explicit verification status and the requested positions when the
        broker request completed.
    """
    cached_positions = tuple(ib.positions())
    try:
        requested_positions = tuple(
            await asyncio.wait_for(ib.reqPositionsAsync(), timeout)
        )
    except asyncio.TimeoutError:
        log.warning("Broker position request timed out after %ss", timeout)
        return BrokerPositionSnapshot(BrokerPositionStatus.REQUEST_UNAVAILABLE)
    except Exception as exc:
        log.warning("Broker position request failed: %r", exc)
        return BrokerPositionSnapshot(BrokerPositionStatus.REQUEST_UNAVAILABLE)

    cached_quantities = {
        position.contract.localSymbol: position.position
        for position in cached_positions
        if position.position
    }
    requested_quantities = {
        position.contract.localSymbol: position.position
        for position in requested_positions
        if position.position
    }
    if cached_quantities != requested_quantities:
        log.debug(
            "Broker position sources disagree: positions=%s req_positions=%s",
            cached_quantities,
            requested_quantities,
        )
        return BrokerPositionSnapshot(
            BrokerPositionStatus.SNAPSHOT_DISAGREEMENT,
            requested_positions,
        )

    log.debug("Broker positions: %s", requested_quantities)
    return BrokerPositionSnapshot(
        BrokerPositionStatus.MATCH,
        requested_positions,
    )
