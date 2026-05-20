from __future__ import annotations

import asyncio
import datetime as dt
import logging
from collections import defaultdict

import ib_insync as ibi

from haymaker.state_machine import OrderInfo, StateMachine

from .sync_types import BrokerSnapshot, LocalSnapshot, SyncResult

log = logging.getLogger(__name__)


class BrokerSnapshotError(Exception):
    """Raised when broker state cannot be captured safely."""

    def __init__(self, result: SyncResult) -> None:
        """Initialize the exception with the failed sync result."""
        super().__init__(result.reason)
        self.result = result


def verify_broker_connected(ib: ibi.IB) -> SyncResult:
    """Return a failed sync result when the broker is disconnected."""
    if not ib.isConnected():
        reason = "broker not connected"
        log.debug("No connection. Abandoning sync.")
        return SyncResult(False, reason)
    return SyncResult(True)


async def capture_broker_snapshot(ib: ibi.IB, timeout: float) -> BrokerSnapshot:
    """Capture broker state once and verify the position source."""
    positions = tuple(ib.positions())
    try:
        requested_positions = tuple(
            await asyncio.wait_for(ib.reqPositionsAsync(), timeout)
        )
    except asyncio.TimeoutError as exc:
        reason = f"broker position request timed out after {timeout}s"
        raise BrokerSnapshotError(SyncResult(False, reason)) from exc
    except Exception as exc:
        reason = f"broker position request failed: {exc!r}"
        raise BrokerSnapshotError(SyncResult(False, reason)) from exc

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
        raise BrokerSnapshotError(SyncResult(False, reason))

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
