import asyncio
import ib_insync as ibi
from .sync_types import SyncResult
import logging

log = logging.getLogger(__name__)


def verify_broker_connected(ib: ibi.IB) -> SyncResult:
    if not ib.isConnected():
        reason = "broker not connected"
        log.debug("No connection. Abandoning sync.")
        return SyncResult(False, reason)
    else:
        return SyncResult(True)


async def verify_broker_position_source(ib: ibi.IB, timeout: int) -> SyncResult:
    positions = {p.contract.localSymbol: p.position for p in ib.positions()}
    try:
        req_positions_response = await asyncio.wait_for(ib.reqPositionsAsync(), timeout)
    except asyncio.TimeoutError:
        reason = "broker position request timed out " f"after {timeout}s"

        return SyncResult(False, reason)
    except Exception as e:
        reason = f"broker position request failed: {e!r}"
        return SyncResult(False, reason)

    req_positions = {
        p.contract.localSymbol: p.position for p in req_positions_response if p.position
    }
    if positions != req_positions:
        reason = f"broker position sources disagree: {positions=} {req_positions=}"
        return SyncResult(False, reason)

    log.debug(f"{positions=}")
    return SyncResult(True)
