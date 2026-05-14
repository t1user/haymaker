"""IB account data functions based on the existing account notebook."""

from __future__ import annotations

import sys
import time
from typing import Any, NamedTuple

import pandas as pd

from dashboard.data.common import ensure_event_loop


ACCOUNT_TAGS = {
    ("NetLiquidation", "USD"),
    ("CashBalance", "BASE"),
    ("TotalCashBalance", "BASE"),
    ("EquityWithLoanValue", "USD"),
    ("FullMaintMarginReq", "USD"),
    ("FullAvailableFunds", "USD"),
    ("FuturesPNL", "BASE"),
    ("MaintMarginReq", "USD"),
    ("RealizedPnL", "BASE"),
    ("UnrealizedPnL", "BASE"),
}


class ContractTuple(NamedTuple):
    symbol: str
    conId: int
    position: float


class IBConnectionError(RuntimeError):
    """Raised when every attempted IB client id fails."""


def _ib_insync() -> Any:
    ensure_event_loop()
    import ib_insync as ibi

    return ibi


def _is_retryable_client_id_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        isinstance(exc, TimeoutError)
        or "client id" in message
        or "clientid" in message
        or "already in use" in message
    )


def connect_ib(
    *,
    host: str,
    port: int,
    client_id: int,
    timeout: float = 2,
    max_client_id_attempts: int = 10,
) -> Any:
    ibi = _ib_insync()
    errors: list[str] = []

    for offset in range(max_client_id_attempts):
        attempted_client_id = client_id + offset
        ib = ibi.IB()
        try:
            return ib.connect(
                host=host,
                port=port,
                clientId=attempted_client_id,
                timeout=timeout,
            )
        except Exception as exc:
            try:
                ib.disconnect()
            except Exception:
                pass
            errors.append(f"{attempted_client_id}: {type(exc).__name__} {exc}")
            if not _is_retryable_client_id_error(exc):
                break

    tried = f"{client_id}-{client_id + max_client_id_attempts - 1}"
    raise IBConnectionError(
        f"Could not connect using client IDs {tried}. Last errors: "
        + "; ".join(errors[-3:])
    )


def client_id(ib: Any) -> int | None:
    client = getattr(ib, "client", None)
    return getattr(client, "clientId", None)


def account_values_frame(ib: Any) -> pd.DataFrame:
    rows = []
    for value in ib.accountValues():
        if (value.tag, value.currency) not in ACCOUNT_TAGS:
            continue
        rows.append(
            {
                "tag": value.tag,
                "currency": value.currency,
                "value": round(float(value.value) / 1000, 2),
            }
        )
    return pd.DataFrame(rows).set_index("tag") if rows else pd.DataFrame()


def _sleep_ib(ib: Any, seconds: float) -> None:
    sleep = getattr(ib, "sleep", None)
    if callable(sleep):
        sleep(seconds)
    else:
        time.sleep(seconds)


def _pnl_value(value: Any) -> float | None:
    if value is None or value == sys.float_info.max:
        return None
    return value


def _collect_pnl_rows(
    ib: Any, account: str, contracts: list[ContractTuple]
) -> pd.DataFrame:
    rows = {}
    for contract in contracts:
        pnl_items = ib.pnlSingle(account, "", contract.conId)
        if not pnl_items:
            continue
        pnl = pnl_items[0]
        rows[contract.symbol] = {
            "dailyPnL": _pnl_value(pnl.dailyPnL),
            "unrealizedPnL": _pnl_value(pnl.unrealizedPnL),
            "realizedPnL": _pnl_value(pnl.realizedPnL),
            "position": pnl.position,
        }
    return pd.DataFrame.from_dict(rows, orient="index").round(2)


def _has_pnl_values(frame: pd.DataFrame) -> bool:
    if frame.empty:
        return False
    pnl_columns = ["dailyPnL", "unrealizedPnL", "realizedPnL"]
    existing = [column for column in pnl_columns if column in frame]
    if not existing:
        return False
    return bool(frame[existing].notna().any().any())


def open_position_pnl_frame(
    ib: Any,
    account: str,
    *,
    force_refresh: bool = False,
    attempts: int = 5,
    poll_interval: float = 0.35,
) -> pd.DataFrame:
    contracts = [
        ContractTuple(p.contract.localSymbol, p.contract.conId, p.position)
        for p in ib.positions()
    ]
    if not contracts:
        return pd.DataFrame()

    for contract in contracts:
        if force_refresh:
            try:
                ib.cancelPnLSingle(account, "", contract.conId)
            except Exception:
                pass
        try:
            ib.reqPnLSingle(account, "", contract.conId)
        except AssertionError:
            pass

    if force_refresh:
        _sleep_ib(ib, poll_interval)

    frame = pd.DataFrame()
    for attempt in range(attempts):
        frame = _collect_pnl_rows(ib, account, contracts)
        if _has_pnl_values(frame):
            return frame
        if attempt < attempts - 1:
            _sleep_ib(ib, poll_interval)

    return frame
