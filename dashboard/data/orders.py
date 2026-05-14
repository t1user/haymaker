"""Order and model-state queries for the dashboard."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import pandas as pd
import pymongo  # type: ignore

from dashboard.data.common import (
    clean_object_id,
    ensure_event_loop,
    find_nested_value,
    numeric,
)

ensure_event_loop()

from haymaker.misc import decode_tree, weighted_average


def _decode_docs(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        decoded = decode_tree(docs)
    except Exception:
        return docs
    return decoded if isinstance(decoded, list) else [decoded]


def fetch_recent_order_docs(
    collection: pymongo.collection.Collection, limit: int
) -> list[dict[str, Any]]:
    return list(collection.find().sort("_id", pymongo.DESCENDING).limit(limit))


def fetch_active_order_docs(
    collection: pymongo.collection.Collection,
) -> list[dict[str, Any]]:
    return list(collection.find({"active": True}).sort("_id", pymongo.DESCENDING))


def fetch_order_docs(
    collection: pymongo.collection.Collection, order_id: int
) -> list[dict[str, Any]]:
    return list(collection.find({"orderId": order_id}).sort("_id", pymongo.ASCENDING))


def _fill_price(fills: Any) -> float:
    try:
        values = [(fill.execution.price, fill.execution.shares) for fill in fills]
        return weighted_average(*values)
    except Exception:
        return math.nan


def _last_log_time(log: Any) -> Any:
    try:
        return log[-1].time if log else None
    except Exception:
        return None


def orders_frame(docs: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for raw, row in zip(docs, _decode_docs(docs), strict=False):
        trade = row.get("trade") if isinstance(row, dict) else None
        order = getattr(trade, "order", None)
        status = getattr(trade, "orderStatus", None)
        contract = getattr(trade, "contract", None)
        fills = getattr(trade, "fills", None)
        log = getattr(trade, "log", None)

        rows.append(
            {
                "_id": clean_object_id(raw.get("_id")),
                "orderId": row.get("orderId") or getattr(order, "orderId", None),
                "permId": getattr(order, "permId", None),
                "utc_time": _last_log_time(log),
                "contract": getattr(contract, "localSymbol", None)
                or getattr(contract, "symbol", None),
                "symbol": getattr(contract, "symbol", None),
                "s_action": row.get("action"),
                "action": getattr(order, "action", None),
                "active": row.get("active"),
                "strategy": row.get("strategy"),
                "totalQuantity": getattr(order, "totalQuantity", None),
                "orderType": getattr(order, "orderType", None),
                "auxPrice": getattr(order, "auxPrice", None),
                "lmtPrice": getattr(order, "lmtPrice", None),
                "filled": getattr(status, "filled", None),
                "avgFillPrice": getattr(status, "avgFillPrice", None),
                "status": getattr(status, "status", None),
                "fills": _fill_price(fills),
                "position_id": row.get("params", {}).get("position_id")
                if isinstance(row.get("params"), dict)
                else None,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["utc_time"] = pd.to_datetime(frame["utc_time"], errors="coerce", utc=True)
    return frame.sort_values(["orderId", "_id"], ascending=[False, False])


def contract_multiplier_map(
    collection: pymongo.collection.Collection,
) -> dict[int, float]:
    mapping: dict[int, float] = {}
    cursor = collection.find({}, {"trade": 1, "params": 1, "contract": 1})
    for doc in cursor:
        con_id = find_nested_value(doc, "conId")
        multiplier = numeric(find_nested_value(doc, "multiplier"))
        try:
            con_id_int = int(con_id)
        except (TypeError, ValueError):
            continue
        if not math.isnan(multiplier):
            mapping[con_id_int] = multiplier
    return mapping


def latest_model_doc(
    collection: pymongo.collection.Collection,
) -> dict[str, Any] | None:
    return collection.find_one(sort=[("_id", pymongo.DESCENDING)])


def models_frame(
    model_doc: dict[str, Any] | None, active_orders: pd.DataFrame
) -> pd.DataFrame:
    if not model_doc:
        return pd.DataFrame()

    doc = dict(model_doc)
    doc.pop("_id", None)
    timestamp = doc.pop("timestamp", None)
    order_map: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
    stops: dict[str, float] = defaultdict(float)

    if not active_orders.empty:
        for row in active_orders.itertuples(index=False):
            strategy = getattr(row, "strategy")
            item = (
                getattr(row, "s_action"),
                getattr(row, "orderType"),
                getattr(row, "action"),
                getattr(row, "totalQuantity"),
                getattr(row, "orderId"),
            )
            order_map[strategy].append(item)
            if getattr(row, "s_action") == "STOP-LOSS":
                side = 1 if getattr(row, "action") == "BUY" else -1
                stops[strategy] += side * numeric(getattr(row, "totalQuantity"))

    rows: list[dict[str, Any]] = []
    for strategy, state in doc.items():
        if not isinstance(state, dict):
            continue
        contract = state.get("active_contract")
        try:
            decoded_contract = decode_tree(contract)
            contract_label = getattr(decoded_contract, "localSymbol", decoded_contract)
        except Exception:
            contract_label = contract

        stop_qty = stops.get(strategy, 0.0)
        position = numeric(state.get("position"))
        rows.append(
            {
                "strategy": strategy,
                "timestamp": timestamp,
                "position": position,
                "lock": state.get("lock"),
                "contract": contract_label,
                "orders": repr(order_map.get(strategy, [])),
                "stops": stop_qty,
                "OK?": "-" if position + stop_qty == 0 else False,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("contract")
