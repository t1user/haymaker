"""Blotter queries and notebook-equivalent position PnL calculations."""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping
from typing import Any

import pandas as pd
import pymongo  # type: ignore

from dashboard.data.common import (
    as_utc_datetime,
    clean_object_id,
    find_nested_value,
    first_present,
    flatten_unique,
    integer,
    numeric,
)

SIDE_SIGN = {"BUY": 1.0, "SELL": -1.0}
ROLL_ACTION = "FUTURE-ROLL"


def _date_match_stage(start: dt.datetime, end: dt.datetime) -> list[dict[str, Any]]:
    return [
        {
            "$addFields": {
                "_fill_dt": {
                    "$convert": {
                        "input": "$last_fill_time",
                        "to": "date",
                        "onError": None,
                        "onNull": None,
                    }
                }
            }
        },
        {"$match": {"_fill_dt": {"$gte": start, "$lt": end}}},
    ]


def fetch_blotter_docs(
    collection: pymongo.collection.Collection,
    *,
    start: dt.datetime | None = None,
    end: dt.datetime | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if start is not None and end is not None:
        pipeline = [
            *_date_match_stage(start, end),
            {"$sort": {"_fill_dt": -1, "_id": -1}},
        ]
        if limit is not None:
            pipeline.append({"$limit": limit})
        try:
            return list(collection.aggregate(pipeline, allowDiskUse=True))
        except pymongo.errors.PyMongoError:
            pass

    cursor = collection.find().sort("_id", pymongo.DESCENDING)
    if limit is not None:
        cursor = cursor.limit(limit)
    docs = list(cursor)
    if start is None or end is None:
        return docs

    frame = pd.DataFrame(docs)
    if frame.empty or "last_fill_time" not in frame:
        return docs
    fill_time = as_utc_datetime(frame["last_fill_time"])
    mask = (fill_time >= pd.Timestamp(start)) & (fill_time < pd.Timestamp(end))
    return [doc for doc, keep in zip(docs, mask, strict=False) if keep]


def fetch_position_docs(
    collection: pymongo.collection.Collection, position_id: str
) -> list[dict[str, Any]]:
    if not position_id:
        return []
    return list(
        collection.find({"position_id": position_id}).sort("_id", pymongo.ASCENDING)
    )


def _extract_multiplier(doc: Mapping[str, Any]) -> tuple[float, str]:
    direct = numeric(doc.get("multiplier"))
    if not math.isnan(direct):
        return direct, "blotter.multiplier"

    nested = numeric(find_nested_value(doc, "multiplier"))
    if not math.isnan(nested):
        return nested, "nested contract"

    return math.nan, ""


def _extract_contract_id(doc: Mapping[str, Any]) -> int | None:
    return integer(
        first_present(doc.get("conId"), find_nested_value(doc, "conId"))
    )


def transactions_frame(
    docs: list[dict[str, Any]],
    *,
    multiplier_map: Mapping[int, float] | None = None,
) -> pd.DataFrame:
    multiplier_map = multiplier_map or {}
    rows: list[dict[str, Any]] = []

    for doc in docs:
        con_id = _extract_contract_id(doc)
        multiplier, source = _extract_multiplier(doc)
        if math.isnan(multiplier) and con_id is not None and con_id in multiplier_map:
            multiplier = multiplier_map[con_id]
            source = "orders conId map"

        side = str(doc.get("side", "")).upper()
        side_sign = SIDE_SIGN.get(side, math.nan)
        amount = numeric(doc.get("amount"))
        price = numeric(doc.get("price"))
        commission = numeric(doc.get("commission"))
        if math.isnan(commission):
            commission = 0.0

        gross_cashflow = -(price * side_sign * amount * multiplier)
        pnl = gross_cashflow - commission

        rows.append(
            {
                "_id": clean_object_id(doc.get("_id")),
                "last_fill_time": doc.get("last_fill_time"),
                "position_id": doc.get("position_id"),
                "order_id": doc.get("order_id"),
                "perm_id": doc.get("perm_id"),
                "contract": doc.get("contract")
                or find_nested_value(doc, "localSymbol"),
                "symbol": doc.get("symbol") or find_nested_value(doc, "symbol"),
                "localSymbol": find_nested_value(doc, "localSymbol"),
                "conId": con_id,
                "action": doc.get("action"),
                "strategy": doc.get("strategy"),
                "side": side,
                "side_sign": side_sign,
                "amount": amount,
                "price": price,
                "commission": commission,
                "realizedPNL": numeric(doc.get("realizedPNL")),
                "multiplier": multiplier,
                "multiplier_source": source,
                "gross_cashflow": gross_cashflow,
                "pnl": pnl,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["last_fill_time"] = as_utc_datetime(frame["last_fill_time"])
    return frame.sort_values(["last_fill_time", "_id"], ascending=[False, False])


def positions_frame(transactions: pd.DataFrame) -> pd.DataFrame:
    """Return closed positions using the same open/close/roll model as the notebook."""

    if transactions.empty or "position_id" not in transactions:
        return pd.DataFrame()

    opens = transactions[transactions["action"] == "OPEN"].copy()
    closes = transactions[
        (transactions["action"] != "OPEN")
        & (transactions["action"] != ROLL_ACTION)
    ].copy()
    rolls = transactions[transactions["action"] == ROLL_ACTION].copy()

    if opens.empty or closes.empty:
        return pd.DataFrame()

    roll_summary = (
        rolls.groupby("position_id")
        .agg(
            price_roll=("price", "sum"),
            commission_roll=("commission", "sum"),
            roll_count=("action", "count"),
        )
        if not rolls.empty
        else pd.DataFrame(
            columns=["price_roll", "commission_roll", "roll_count"]
        )
    )

    merged = pd.merge(
        opens,
        closes,
        on="position_id",
        suffixes=("_open", "_close"),
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame()

    merged = merged.merge(
        roll_summary,
        left_on="position_id",
        right_index=True,
        how="left",
    )
    merged[["price_roll", "commission_roll", "roll_count"]] = merged[
        ["price_roll", "commission_roll", "roll_count"]
    ].fillna(0)

    multiplier = merged["multiplier_open"].where(
        merged["multiplier_open"].notna(), merged["multiplier_close"]
    )
    multiplier_source = merged["multiplier_source_open"].where(
        merged["multiplier_source_open"].notna(),
        merged["multiplier_source_close"],
    )

    pnl_points = -(
        merged["price_close"] * merged["side_sign_close"]
        + merged["price_open"] * merged["side_sign_open"]
        + merged["price_roll"] * merged["side_sign_open"]
    )
    commissions = (
        merged["commission_open"]
        + merged["commission_close"]
        + merged["commission_roll"]
    )

    output = pd.DataFrame(
        {
            "strategy": merged["strategy_open"],
            "pnl": pnl_points * multiplier * merged["amount_open"] - commissions,
            "open_price": merged["price_open"],
            "close_price": merged["price_close"],
            "contract": merged["contract_open"].where(
                merged["contract_open"].notna(), merged["localSymbol_open"]
            ),
            "side": merged["side_open"],
            "open_time": merged["last_fill_time_open"],
            "close_time": merged["last_fill_time_close"],
            "amount": merged["amount_open"],
            "close_action": merged["action_close"],
            "symbol": merged["symbol_open"],
            "position_id": merged["position_id"],
            "pnl_points": pnl_points,
            "commissions": commissions,
            "multiplier": multiplier,
            "multiplier_source": multiplier_source,
            "duration_hours": (
                merged["last_fill_time_close"] - merged["last_fill_time_open"]
            )
            / pd.Timedelta(hours=1),
            "rolls": merged["roll_count"].astype(int),
            "price_roll": merged["price_roll"],
            "commission_roll": merged["commission_roll"],
            "open_order_id": merged["order_id_open"],
            "close_order_id": merged["order_id_close"],
        }
    )
    return output.sort_values("close_time", ascending=False)


def audit_positions_frame(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty or "position_id" not in transactions:
        return pd.DataFrame()

    grouped_rows: list[dict[str, Any]] = []
    grouped = transactions.dropna(subset=["position_id"]).groupby(
        "position_id", sort=False
    )

    for position_id, group in grouped:
        chronological = group.sort_values("last_fill_time")
        non_roll = chronological[chronological["action"] != ROLL_ACTION]
        qty_balance = (non_roll["side_sign"] * non_roll["amount"]).sum()
        open_rows = chronological[chronological["action"] == "OPEN"]
        exit_rows = chronological[
            (chronological["action"] != "OPEN")
            & (chronological["action"] != ROLL_ACTION)
        ]
        missing_multiplier = chronological["multiplier"].isna().any()
        if missing_multiplier:
            pnl = math.nan
            gross = math.nan
        else:
            pnl = chronological["pnl"].sum(min_count=1)
            gross = chronological["gross_cashflow"].sum(min_count=1)
        commission = chronological["commission"].sum(min_count=1)

        open_time = chronological["last_fill_time"].min()
        close_time = (
            exit_rows["last_fill_time"].max() if not exit_rows.empty else pd.NaT
        )
        duration = (
            (close_time - open_time) / pd.Timedelta(hours=1)
            if pd.notna(open_time) and pd.notna(close_time)
            else math.nan
        )

        grouped_rows.append(
            {
                "position_id": position_id,
                "symbol": chronological["symbol"].dropna().iloc[0]
                if chronological["symbol"].notna().any()
                else None,
                "localSymbol": flatten_unique(chronological["localSymbol"]),
                "strategy": chronological["strategy"].dropna().iloc[0]
                if chronological["strategy"].notna().any()
                else None,
                "open_time": open_time,
                "close_time": close_time,
                "status": "OPEN" if abs(qty_balance) > 1e-9 else "CLOSED",
                "side": open_rows["side"].iloc[0] if not open_rows.empty else None,
                "quantity_balance": qty_balance,
                "transactions": len(chronological),
                "rolls": int((chronological["action"] == ROLL_ACTION).sum()),
                "actions": flatten_unique(chronological["action"]),
                "gross_cashflow": gross,
                "commission": commission,
                "pnl": pnl,
                "duration_hours": duration,
                "open_rows": len(open_rows),
                "exit_rows": len(exit_rows),
                "missing_multiplier": missing_multiplier,
            }
        )

    output = pd.DataFrame(grouped_rows)
    if output.empty:
        return output
    return output.sort_values(["close_time", "open_time"], ascending=[False, False])


def filter_positions_by_close_time(
    positions: pd.DataFrame,
    *,
    start: dt.datetime | None,
    end: dt.datetime | None,
) -> pd.DataFrame:
    if positions.empty or start is None or end is None:
        return positions
    close_time = positions["close_time"]
    return positions[
        (close_time >= pd.Timestamp(start)) & (close_time < pd.Timestamp(end))
    ]


def payout_ratio(pnl: pd.Series) -> float:
    winners = pnl[pnl > 0]
    losers = pnl[pnl < 0]
    if winners.empty or losers.empty:
        return math.nan
    return -winners.mean() / losers.mean()


def win_rate(pnl: pd.Series) -> float:
    if pnl.empty:
        return math.nan
    return (pnl > 0).sum() / pnl.count()


def strategy_stats(positions: pd.DataFrame) -> pd.DataFrame:
    if positions.empty:
        return pd.DataFrame()
    grouped = positions.groupby("strategy", dropna=False)
    output = grouped.agg(
        positions=("position_id", "count"),
        pnl=("pnl", "sum"),
        avg_pnl=("pnl", "mean"),
        commissions=("commissions", "sum"),
        wins=("pnl", lambda x: (x > 0).sum()),
        win_rate=("pnl", win_rate),
        payout_ratio=("pnl", payout_ratio),
    )
    return output.sort_values("pnl", ascending=False)


def pnl_by_contract(positions: pd.DataFrame) -> pd.DataFrame:
    if positions.empty:
        return pd.DataFrame()
    return (
        positions.groupby("symbol", dropna=False)
        .agg(
            positions=("position_id", "count"),
            pnl=("pnl", "sum"),
            avg_pnl=("pnl", "mean"),
        )
        .sort_values("pnl", ascending=False)
    )


def pnl_by_month(positions: pd.DataFrame) -> pd.DataFrame:
    if positions.empty:
        return pd.DataFrame()
    monthly = positions.dropna(subset=["close_time"]).copy()
    monthly["month"] = (
        monthly["close_time"].dt.tz_convert(None).dt.to_period("M").astype(str)
    )
    return monthly.groupby("month").agg(
        positions=("position_id", "count"), pnl=("pnl", "sum")
    )


def issues_frame(
    transactions: pd.DataFrame, audit_positions: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not transactions.empty:
        no_position = transactions[
            transactions["position_id"].isna() | (transactions["position_id"] == "")
        ]
        for _, row in no_position.iterrows():
            rows.append(
                {
                    "severity": "warning",
                    "issue": "missing position_id",
                    "position_id": None,
                    "transaction_id": row.get("order_id"),
                    "detail": (
                        f"{row.get('action')} "
                        f"{row.get('symbol')} "
                        f"{row.get('side')}"
                    ),
                }
            )

        missing_multiplier = transactions[transactions["multiplier"].isna()]
        for _, row in missing_multiplier.iterrows():
            rows.append(
                {
                    "severity": "error",
                    "issue": "missing multiplier",
                    "position_id": row.get("position_id"),
                    "transaction_id": row.get("order_id"),
                    "detail": f"conId={row.get('conId')} symbol={row.get('symbol')}",
                }
            )

    if not audit_positions.empty:
        bad_open_count = audit_positions[audit_positions["open_rows"] != 1]
        for _, row in bad_open_count.iterrows():
            rows.append(
                {
                    "severity": "warning",
                    "issue": "open row count is not 1",
                    "position_id": row.get("position_id"),
                    "transaction_id": None,
                    "detail": f"open_rows={row.get('open_rows')}",
                }
            )

        no_close = audit_positions[audit_positions["exit_rows"] == 0]
        for _, row in no_close.iterrows():
            rows.append(
                {
                    "severity": "info",
                    "issue": "open or unmatched position",
                    "position_id": row.get("position_id"),
                    "transaction_id": None,
                    "detail": "No close transaction found; excluded from closed PnL.",
                }
            )

        multi_close = audit_positions[audit_positions["exit_rows"] > 1]
        for _, row in multi_close.iterrows():
            rows.append(
                {
                    "severity": "warning",
                    "issue": "multiple close rows",
                    "position_id": row.get("position_id"),
                    "transaction_id": None,
                    "detail": f"exit_rows={row.get('exit_rows')}",
                }
            )

    return pd.DataFrame(rows)


def strip_mongo_ids(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for doc in docs:
        cleaned = dict(doc)
        cleaned.pop("_id", None)
        output.append(cleaned)
    return output


def position_display_frame(positions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "strategy",
        "pnl",
        "open_price",
        "close_price",
        "contract",
        "side",
        "open_time",
        "close_time",
        "amount",
        "close_action",
        "symbol",
        "position_id",
        "pnl_points",
        "commissions",
        "multiplier",
        "duration_hours",
        "rolls",
    ]
    return positions[[column for column in columns if column in positions]]


def transaction_display_frame(transactions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "last_fill_time",
        "symbol",
        "side",
        "amount",
        "order_id",
        "perm_id",
        "realizedPNL",
        "action",
        "price",
        "commission",
        "strategy",
        "position_id",
        "contract",
        "multiplier",
        "multiplier_source",
    ]
    return transactions[[column for column in columns if column in transactions]]


def rolls_frame(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame()
    rolls = transactions[transactions["action"] == ROLL_ACTION].copy()
    if rolls.empty:
        return pd.DataFrame()

    open_sides = (
        transactions[transactions["action"] == "OPEN"]
        .dropna(subset=["position_id"])
        .drop_duplicates("position_id")
        .set_index("position_id")["side"]
    )
    rolls["open_side"] = rolls["position_id"].map(open_sides)
    rolls["side_matches_open"] = rolls["side"] == rolls["open_side"]
    def _effect(value: float) -> str:
        if value > 0:
            return "increases pnl"
        if value < 0:
            return "decreases pnl"
        return "flat"

    rolls["pnl_effect"] = rolls["gross_cashflow"].apply(_effect)
    columns = [
        "last_fill_time",
        "position_id",
        "symbol",
        "localSymbol",
        "open_side",
        "side",
        "side_matches_open",
        "price",
        "amount",
        "multiplier",
        "gross_cashflow",
        "commission",
        "pnl_effect",
    ]
    return rolls[columns].sort_values("last_fill_time", ascending=False)
