"""Display formatting helpers for dashboard tables."""

from __future__ import annotations

import pandas as pd


MONETARY_COLUMNS = {
    "avg_pnl",
    "avgFillPrice",
    "auxPrice",
    "close_price",
    "commission",
    "commission_roll",
    "commissions",
    "dailyPnL",
    "fills",
    "gross_cashflow",
    "lmtPrice",
    "open_price",
    "pnl",
    "price",
    "price_roll",
    "realizedPNL",
    "realizedPnL",
    "unrealizedPnL",
    "value",
}


def round_money(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a display copy with money/price columns rounded to two decimals."""

    if frame.empty:
        return frame

    output = frame.copy()
    for column in MONETARY_COLUMNS.intersection(output.columns):
        values = pd.to_numeric(output[column], errors="coerce")
        if values.notna().any():
            output[column] = values.round(2)
    return output
