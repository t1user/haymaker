#!/usr/bin/env python3
"""Convert legacy Haymaker Mongo records into the direct-cutover Book schema.

The command is dry-run by default. It requires distinct explicit source and
target database names and never modifies the source database.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import ib_insync as ibi
from pymongo import MongoClient  # type: ignore

from haymaker.book import FillRecord
from haymaker.misc import decode_tree, tree

MIGRATION_VERSION = "components-book-v2"

KNOWN_ROLES = {
    "OPEN": "OPEN",
    "CLOSE": "CLOSE",
    "TARGET-ADJUSTMENT": "TARGET_ADJUSTMENT",
    "TARGET_ADJUSTMENT": "TARGET_ADJUSTMENT",
    "STOP": "STOP_LOSS",
    "STOP-LOSS": "STOP_LOSS",
    "STOP_LOSS": "STOP_LOSS",
    "TAKE-PROFIT": "TAKE_PROFIT",
    "TAKE_PROFIT": "TAKE_PROFIT",
    "FUTURE-ROLL": "ROLL",
    "ROLL": "ROLL",
    "RESET": "LIQUIDATION",
    "LIQUIDATION": "LIQUIDATION",
    "MANUAL": "MANUAL",
    "UNKNOWN": "UNKNOWN",
}


def order_role(action: str) -> str:
    """Map known legacy actions while retaining custom action strings."""

    normalized = str(action).upper()
    return KNOWN_ROLES.get(normalized, str(action))


def provenance(
    *,
    source_database: str,
    source_collection: str,
    source_id: Any,
) -> dict[str, Any]:
    """Return deterministic rerun provenance for one source document."""

    source_id_string = str(source_id)
    return {
        "migration_version": MIGRATION_VERSION,
        "source_database": source_database,
        "source_collection": source_collection,
        "source_id": source_id_string,
        "migration_key": (
            f"{MIGRATION_VERSION}:{source_database}:"
            f"{source_collection}:{source_id_string}"
        ),
    }


def _submitted_at(trade: ibi.Trade) -> datetime:
    """Return the earliest honest Trade log time.

    Raises:
        ValueError: If the legacy Trade contains no submission chronology.
    """

    if not trade.log:
        raise ValueError(
            f"orderId={trade.order.orderId} has no honest submission timestamp"
        )
    timestamp = min(entry.time for entry in trade.log)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp


def _legacy_timestamp(*values: Any, name: str) -> datetime:
    """Return the first available legacy timestamp normalized to UTC."""

    for value in values:
        if isinstance(value, datetime):
            if value.tzinfo is None or value.utcoffset() is None:
                return value.replace(tzinfo=timezone.utc)
            return value
    raise ValueError(f"legacy {name} timestamp is required")


def _legacy_bracket_inputs(params: Mapping[str, Any]) -> dict[str, Any]:
    """Recover volatility inputs recorded by legacy bracket creation."""

    inputs: dict[str, Any] = {}
    for label in ("stop-loss", "take-profit"):
        memo = params.get(label)
        if not isinstance(memo, Mapping):
            continue
        name = memo.get("vol_field_name")
        if isinstance(name, str) and name and "vol_field_value" in memo:
            inputs[name] = memo["vol_field_value"]

    opening = params.get("open")
    if isinstance(opening, Mapping):
        for name in ("atr", "sl_points", "min_tick"):
            if name in opening:
                inputs.setdefault(name, opening[name])

    for name in ("atr", "sl_points", "min_tick"):
        if name in params:
            inputs.setdefault(name, params[name])
    return inputs


def convert_order(
    document: Mapping[str, Any], *, source_database: str
) -> dict[str, Any]:
    """Convert one legacy order document without modifying the input."""

    trade = decode_tree(document["trade"])
    if not isinstance(trade, ibi.Trade):
        raise TypeError("legacy order trade did not decode to ib_insync.Trade")
    if not trade.order.orderId:
        raise ValueError("legacy order with orderId 0 cannot use natural identity")
    source_key = str(document.get("strategy") or "UNKNOWN")
    action = str(document.get("action") or "UNKNOWN")
    role = order_role(action)
    target_key = (
        f"legacy:{source_key}:{trade.contract.conId}"
        if role == "TARGET_ADJUSTMENT"
        else None
    )
    fills = [FillRecord.from_fill(trade, fill).encode() for fill in trade.fills]
    source_id = document.get("_id", trade.order.orderId)
    migration = provenance(
        source_database=source_database,
        source_collection="orders",
        source_id=source_id,
    )
    params = dict(document.get("params") or {})
    params.setdefault("legacy_action", action)
    return {
        "orderId": trade.order.orderId,
        "clientId": trade.order.clientId,
        "permId": trade.order.permId,
        "trade": tree(trade),
        "role": role,
        "submitted_at": _submitted_at(trade),
        "execution_model_name": f"legacy:{source_key}",
        "target_key": target_key,
        "source_key": None if target_key is not None else source_key,
        "position_id": None if target_key is not None else params.get("position_id"),
        "params": tree(params),
        "fills": fills,
        "applied_fill_keys": [fill["deduplication_key"] for fill in fills],
        "active": bool(document.get("active", trade.isActive())),
        "priority": document.get("priority", 0),
        **migration,
    }


def convert_latest_strategy_snapshot(
    document: Mapping[str, Any], *, source_database: str
) -> list[dict[str, Any]]:
    """Convert useful latest strategy state and skip historical snapshots."""

    converted: list[dict[str, Any]] = []
    snapshot_id = document.get("_id", "latest")
    for source_key, raw_state in document.items():
        if source_key in {"_id", "timestamp"}:
            continue
        state = decode_tree(raw_state)
        if not isinstance(state, Mapping):
            continue
        contract = state.get("active_contract")
        quantity = float(state.get("position", 0.0))
        lock = state.get("lock")
        blocked_direction = int(lock) if not quantity and lock in (-1, 1) else None
        params = dict(state.get("params") or {})
        bracket_inputs = _legacy_bracket_inputs(params)
        migration = provenance(
            source_database=source_database,
            source_collection="strategies",
            source_id=f"{snapshot_id}:{source_key}",
        )
        snapshot_time = _legacy_timestamp(
            state.get("timestamp"),
            document.get("timestamp"),
            name=f"strategy {source_key!r}",
        )
        converted.append(
            {
                "state_key": f"position:{source_key}",
                "state_type": "position",
                "source_key": source_key,
                "execution_model_name": f"legacy:{source_key}",
                "contract": tree(contract),
                "quantity": quantity,
                "target_quantity": quantity,
                "target_created_at": snapshot_time,
                "position_id": state.get("position_id") or None,
                "blocked_direction": blocked_direction,
                "bracket_inputs": tree(bracket_inputs),
                "updated_at": snapshot_time,
                **migration,
            }
        )
    return converted


def convert_blotter(
    document: Mapping[str, Any], *, source_database: str
) -> dict[str, Any]:
    """Copy one blotter row while normalizing logical attribution."""

    result = dict(document)
    source_id = result.pop("_id", result.get("order_id", "unknown"))
    source_key = result.pop("strategy", None) or result.get("source_key")
    action = result.pop("action", None) or result.get("role", "UNKNOWN")
    result.update(
        {
            "source_key": source_key,
            "role": order_role(str(action)),
            **provenance(
                source_database=source_database,
                source_collection="blotter",
                source_id=source_id,
            ),
        }
    )
    return result


def _latest_snapshot(collection: Any) -> Mapping[str, Any] | None:
    """Read only the latest legacy whole-system strategy snapshot."""

    return collection.find_one({}, sort=[("timestamp", -1)])


def _ensure_target_compatible(database: Any, *, source_database: str) -> None:
    """Refuse a non-empty target that was not created by this converter."""

    for collection_name in ("orders", "state", "blotter"):
        collection = database[collection_name]
        count = collection.estimated_document_count()
        compatible = collection.count_documents(
            {
                "migration_version": MIGRATION_VERSION,
                "source_database": source_database,
            }
        )
        if count != compatible:
            raise RuntimeError(
                f"Target collection {collection_name!r} is incompatible or "
                "contains foreign/non-migration data"
            )


def _upsert_documents(
    collection: Any,
    documents: Iterable[Mapping[str, Any]],
    *,
    identity_field: str,
) -> int:
    """Idempotently upsert converted documents by deterministic provenance."""

    count = 0
    for document in documents:
        collection.replace_one(
            {identity_field: document[identity_field]},
            dict(document),
            upsert=True,
        )
        count += 1
    return count


def validation_report(
    orders: Sequence[Mapping[str, Any]],
    states: Sequence[Mapping[str, Any]],
    blotter: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return reconciliation totals and representative grouping counts."""

    commissions = sum(float(row.get("commission") or 0) for row in blotter)
    realized_pnl = sum(float(row.get("realizedPNL") or 0) for row in blotter)
    grouping = Counter(
        (
            str(row.get("source_key")),
            str(row.get("position_id")),
        )
        for row in blotter
    )
    order_ids = {order.get("orderId") for order in orders}
    blotter_ids = {row.get("order_id") for row in blotter}
    episode_groups = [
        f"{source_key}|{position_id}"
        for source_key, position_id in grouping
        if source_key != "None" and position_id != "None"
    ]
    return {
        "counts": {
            "orders": len(orders),
            "state": len(states),
            "blotter": len(blotter),
            "active_orders": sum(bool(order.get("active")) for order in orders),
            "active_positions": sum(bool(state.get("quantity")) for state in states),
        },
        "totals": {
            "commission": commissions,
            "realized_pnl": realized_pnl,
        },
        "source_position_groups": {
            f"{source_key}|{position_id}": count
            for (source_key, position_id), count in grouping.items()
        },
        "unmatched_blotter_order_ids": sorted(
            order_id for order_id in blotter_ids - order_ids if order_id is not None
        ),
        "broker_identifiers": {
            "orders_missing_order_id": sum(
                not order.get("orderId") for order in orders
            ),
            "orders_missing_perm_id": sum(not order.get("permId") for order in orders),
            "fills_missing_exec_id": sum(
                not fill.get("execution", {}).get("Execution", {}).get("execId")
                for order in orders
                for fill in order.get("fills", ())
            ),
        },
        "representative_position_episodes": sorted(episode_groups)[:10],
    }


def migrate(
    client: MongoClient,
    *,
    source_database: str,
    target_database: str,
    apply: bool = False,
) -> dict[str, Any]:
    """Convert records and optionally write a fresh target database."""

    if not source_database or not target_database:
        raise ValueError("source and target database names are required")
    if source_database == target_database:
        raise ValueError("source and target databases must be different")
    source = client[source_database]
    target = client[target_database]
    _ensure_target_compatible(
        target,
        source_database=source_database,
    )

    orders = [
        convert_order(document, source_database=source_database)
        for document in source["orders"].find({})
    ]
    snapshot = _latest_snapshot(source["strategies"])
    states = (
        convert_latest_strategy_snapshot(snapshot, source_database=source_database)
        if snapshot is not None
        else []
    )
    blotter = [
        convert_blotter(document, source_database=source_database)
        for document in source["blotter"].find({})
    ]
    report = validation_report(orders, states, blotter)
    report["mode"] = "apply" if apply else "dry-run"
    report["source_database"] = source_database
    report["target_database"] = target_database
    report["source_counts"] = {
        "orders": source["orders"].estimated_document_count(),
        "strategy_snapshots": source["strategies"].estimated_document_count(),
        "blotter": source["blotter"].estimated_document_count(),
    }
    if apply:
        target["orders"].create_index("orderId", unique=True)
        target["orders"].create_index("migration_key", unique=True)
        target["state"].create_index("state_key", unique=True)
        target["state"].create_index("migration_key", unique=True)
        target["blotter"].create_index("migration_key", unique=True)
        _upsert_documents(
            target["orders"],
            orders,
            identity_field="orderId",
        )
        _upsert_documents(
            target["state"],
            states,
            identity_field="state_key",
        )
        _upsert_documents(
            target["blotter"],
            blotter,
            identity_field="migration_key",
        )
        report["target_counts"] = {
            collection_name: target[collection_name].estimated_document_count()
            for collection_name in ("orders", "state", "blotter")
        }
    else:
        report["target_counts"] = {
            collection_name: target[collection_name].estimated_document_count()
            for collection_name in ("orders", "state", "blotter")
        }
    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line options without defaulting database identities."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    parser.add_argument("--source-db", required=True)
    parser.add_argument("--target-db", required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write the target; omitted means dry-run",
    )
    return parser.parse_args()


def main() -> None:
    """Run conversion and print its validation report as JSON."""

    args = parse_args()
    client = MongoClient(args.mongo_uri)
    report = migrate(
        client,
        source_database=args.source_db,
        target_database=args.target_db,
        apply=args.apply,
    )
    print(json.dumps(report, default=str, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
