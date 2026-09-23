#!/usr/bin/env python3
"""Convert legacy Haymaker Mongo records into the direct-cutover Book schema.

The command is dry-run by default. It requires distinct explicit source and
target database names and never modifies the source database.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

import ib_insync as ibi
from bson import json_util  # type: ignore
from pymongo import MongoClient  # type: ignore

from haymaker.book import (
    Book,
    ContractPosition,
    FillRecord,
    OrderInfo,
    PositionState,
    TargetState,
    RollState,
)
from haymaker.misc import decode_tree, tree
from haymaker.saver import AbstractBaseSaver

MIGRATION_VERSION = "components-book-v5-settled-cutover"
COLLECTION_IDENTITIES = {
    "orders": "orderId",
    "state": "state_key",
    "blotter": "migration_key",
}

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
        # Custom legs choose their own field (for example DualThrust's range).
        # Keep the original opening inputs; an installed bracket memo wins.
        for name, value in opening.items():
            inputs.setdefault(name, value)

    for name in ("atr", "sl_points", "min_tick"):
        if name in params:
            inputs.setdefault(name, params[name])
    return inputs


def convert_order(
    document: Mapping[str, Any], *, source_database: str
) -> dict[str, Any]:
    """Convert legacy or component orders without discarding explicit evidence."""

    if "execution_model_name" in document and "role" in document:
        result = dict(document)
        source_id = result.pop("_id", result.get("orderId"))
        result.pop("target_key", None)
        # Explicit fills, commission updates, timestamps and IB identifiers are
        # already authoritative. Do not rebuild them from diagnostic Trade data.
        OrderInfo.decode(result)
        result.update(
            provenance(
                source_database=source_database,
                source_collection="orders",
                source_id=source_id,
            )
        )
        return result

    trade = decode_tree(document["trade"])
    if not isinstance(trade, ibi.Trade):
        raise TypeError("legacy order trade did not decode to ib_insync.Trade")
    if not trade.order.orderId:
        raise ValueError(
            "Legacy order with orderId 0 cannot use Book's natural identity; "
            "reconcile its real broker orderId by permId before conversion"
        )
    source_key = str(document.get("strategy") or "UNKNOWN")
    action = str(document.get("action") or "UNKNOWN")
    role = order_role(action)
    direct = role == "TARGET_ADJUSTMENT"
    fills = [FillRecord.from_fill(trade, fill).encode() for fill in trade.fills]
    source_id = document.get("_id", trade.order.orderId)
    migration = provenance(
        source_database=source_database,
        source_collection="orders",
        source_id=source_id,
    )
    params = dict(document.get("params") or {})
    params.setdefault("legacy_action", action)
    if role == "ROLL":
        for old_name, new_name in (("old", "old_contract"), ("new", "new_contract")):
            if old_name in params:
                params.setdefault(new_name, params[old_name])
    if "active" in document and bool(document["active"]) != trade.isActive():
        raise ValueError(
            f"orderId={trade.order.orderId} active flag disagrees with Trade status; "
            "reconcile the source instead of inventing a broker status"
        )
    result = {
        "orderId": trade.order.orderId,
        "clientId": trade.order.clientId,
        "permId": trade.order.permId,
        "trade": tree(trade),
        "role": role,
        "submitted_at": _submitted_at(trade),
        "execution_model_name": f"legacy:{source_key}",
        "source_key": None if direct else source_key,
        "position_id": None if direct else params.get("position_id"),
        "params": tree(params),
        "fills": fills,
        "applied_fill_keys": [fill["deduplication_key"] for fill in fills],
        "active": bool(document.get("active", trade.isActive())),
        "priority": document.get("priority", 0),
        **migration,
    }
    if "accounted_exec_ids" in document:
        result["legacy_accounted_exec_ids"] = list(document["accounted_exec_ids"])
    OrderInfo.decode(result)
    return result


def convert_latest_strategy_snapshot(
    document: Mapping[str, Any],
    *,
    source_database: str,
    orders: Sequence[Mapping[str, Any]] = (),
    source_collection: str = "strategies",
) -> list[dict[str, Any]]:
    """Convert useful latest strategy state and skip historical snapshots."""

    converted: list[dict[str, Any]] = []
    snapshot_id = document.get("_id", "latest")
    for source_key, raw_state in document.items():
        if source_key in {"_id", "timestamp"}:
            continue
        state = decode_tree(raw_state)
        if not isinstance(state, Mapping):
            raise ValueError(f"Invalid legacy strategy snapshot for {source_key!r}")
        contract = state.get("active_contract")
        quantity = float(state.get("position", 0.0))
        lock = state.get("lock")
        blocked_direction = int(lock) if not quantity and lock in (-1, 1) else None
        params = dict(state.get("params") or {})
        bracket_inputs = _legacy_bracket_inputs(params)
        migration = provenance(
            source_database=source_database,
            source_collection=source_collection,
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
                "target_contract": tree(contract),
                "target_bracket_inputs": tree(bracket_inputs),
                "position_id": state.get("position_id") or None,
                "blocked_direction": blocked_direction,
                "bracket_inputs": tree(bracket_inputs),
                "updated_at": snapshot_time,
                "applied_fill_keys": _position_fill_checkpoint(
                    source_key,
                    orders,
                    snapshot_time=_legacy_timestamp(
                        document.get("timestamp"), name="whole-system snapshot"
                    ),
                ),
                **migration,
            }
        )
    return converted


def _position_fill_checkpoint(
    source_key: str,
    orders: Sequence[Mapping[str, Any]],
    *,
    snapshot_time: datetime | None = None,
) -> list[str]:
    """Treat converted state as the operator-selected accounting baseline."""
    keys: set[str] = set()
    for order in orders:
        if order.get("source_key") != source_key:
            continue
        is_bag = isinstance(decode_tree(order["trade"]).contract, ibi.Bag)
        accounted = order.get("legacy_accounted_exec_ids")
        if accounted is not None and not is_bag:
            # Rebinding a legacy Trade can lose older Fill objects while its
            # accounting keys survive. Late recovery must not count them twice.
            if any(not isinstance(key, str) or not key for key in accounted):
                raise ValueError(f"Source {source_key!r} has invalid legacy fill keys")
            keys.update(accounted)
        for fill in order.get("fills", ()):
            record = FillRecord.decode(fill)
            if snapshot_time is not None and record.time > snapshot_time:
                raise ValueError(
                    f"Source {source_key!r} has a fill newer than its snapshot; "
                    "reconcile and flush the old runtime before migration"
                )
            if is_bag:
                continue
            if accounted is not None and record.deduplication_key not in accounted:
                raise ValueError(
                    f"Source {source_key!r} has an unaccounted legacy fill "
                    f"{record.deduplication_key!r}; reconcile before migration"
                )
            keys.add(record.deduplication_key)
    return sorted(keys)


def _exposure(
    orders: Sequence[Mapping[str, Any]], cutoff: datetime | None = None
) -> dict[int, ibi.Contract]:
    """Locate non-flat concrete exposure using normalized fills, including BAG legs."""
    quantities: dict[int, float] = defaultdict(float)
    contracts: dict[int, ibi.Contract] = {}
    seen: set[str] = set()
    for document in orders:
        info = OrderInfo.decode(
            {k: v for k, v in document.items() if k != "target_key"}
        )
        has_legs = any(not isinstance(fill.contract, ibi.Bag) for fill in info.fills)
        for fill in info.fills:
            if fill.deduplication_key in seen or (cutoff and fill.time <= cutoff):
                continue
            seen.add(fill.deduplication_key)
            quantity = fill.execution.shares * (
                1 if fill.execution.side in {"BOT", "BUY"} else -1
            )
            legs = [(fill.contract, quantity)]
            if isinstance(fill.contract, ibi.Bag):
                if has_legs:
                    continue
                old, new = info.params.get("old_contract"), info.params.get(
                    "new_contract"
                )
                if not isinstance(old, ibi.Contract) or not isinstance(
                    new, ibi.Contract
                ):
                    raise ValueError(
                        "Cannot reconcile BAG evidence without explicit roll endpoints"
                    )
                legs = [(old, -quantity), (new, quantity)]
            for contract, delta in legs:
                contracts[contract.conId] = contract
                quantities[contract.conId] += delta
    return {
        key: contracts[key]
        for key, quantity in quantities.items()
        if abs(quantity) > 1e-12
    }


def convert_component_states(
    documents: Sequence[Mapping[str, Any]],
    orders: Sequence[Mapping[str, Any]],
    *,
    source_database: str,
) -> list[dict[str, Any]]:
    """Convert old keyed/component state only when concrete ownership is unambiguous.

    Pending roll work must finish under its original implementation. Old source
    state without separate held/target fields is reconciled to episode evidence.
    A keyed direct target spanning other held/working Contracts cannot be split
    automatically; the operator must resolve that allocation before conversion.
    """
    converted: list[dict[str, Any]] = []
    identities: set[str] = set()
    for document in documents:
        result = dict(document)
        source_id = result.pop("_id", result.get("state_key"))
        kind = result.get("state_type")
        if kind == "balance":
            # Rebuild derived totals from converted episode/order/roll evidence
            # at Book startup; copying them would duplicate active-state totals.
            ContractPosition.decode(result)
            continue
        if kind == "position":
            result.setdefault(
                "applied_fill_keys",
                _position_fill_checkpoint(str(result["source_key"]), orders),
            )
            if "target_contract" not in result:
                result["target_contract"] = result.get("contract")
                result["target_bracket_inputs"] = result.get("bracket_inputs", {})
                episode_orders = [
                    order
                    for order in orders
                    if order.get("source_key") == result["source_key"]
                    and order.get("position_id") == result.get("position_id")
                ]
                held = _exposure(episode_orders)
                pending = {
                    decode_tree(order["trade"])
                    .contract.conId: decode_tree(order["trade"])
                    .contract
                    for order in episode_orders
                    if order.get("active") and order["role"] == "OPEN"
                }
                candidates = held if result.get("quantity") else pending
                if len(candidates) > 1 or (result.get("quantity") and not candidates):
                    raise ValueError(
                        f"Cannot establish held Contract for source {result['source_key']!r}"
                    )
                if candidates:
                    result["contract"] = tree(next(iter(candidates.values())))
                    openings = [
                        order for order in episode_orders if order["role"] == "OPEN"
                    ]
                    if openings:
                        opening = min(
                            openings,
                            key=lambda order: decode_tree(order["submitted_at"]),
                        )
                        params = decode_tree(opening.get("params", {}))
                        required = decode_tree(result.get("bracket_inputs", {}))
                        if any(name not in params for name in required):
                            raise ValueError(
                                "Original episode bracket inputs are missing from OPEN evidence"
                            )
                        result["bracket_inputs"] = tree(
                            {name: params[name] for name in required}
                        )
                elif not result.get("quantity"):
                    result["contract"] = None
            PositionState.decode(result)
        elif kind == "target":
            contract = decode_tree(result["contract"])
            key = result.pop("target_key", None)
            related = [
                order
                for order in orders
                if order.get("source_key") is None
                and (
                    order.get("target_key") == key
                    if key is not None
                    else decode_tree(order["trade"]).contract.conId == contract.conId
                )
            ]
            cutoff = decode_tree(result.get("fill_evidence_start_at"))
            members = set(_exposure(related, cutoff))
            members.update(
                decode_tree(order["trade"]).contract.conId
                for order in related
                if order.get("active")
            )
            if members - {contract.conId}:
                raise ValueError(
                    f"Ambiguous concrete allocation for target {key!r}: conIds={sorted(members)}"
                )
            result["conId"] = contract.conId
            result["state_key"] = f"target:{contract.conId}"
            TargetState.decode(result)
        elif kind == "roll":
            if result.get("stage") != "COMPLETE":
                raise ValueError(
                    "Finish pending roll work before changing the roll schema"
                )
            result["participants"] = [
                {
                    key: value
                    for key, value in participant.items()
                    if key != "target_key"
                }
                for participant in result["participants"]
            ]
            if result.get("target_transfers"):
                raise ValueError(
                    "Review stored roll transfers before component schema conversion"
                )
            RollState.decode(result)
        elif kind != "portfolio":
            raise ValueError(f"Unknown component state_type: {kind!r}")
        identity = result["state_key"]
        if identity in identities:
            raise ValueError(
                f"Multiple old states map to {identity!r}; allocation must be resolved explicitly"
            )
        identities.add(identity)
        result.update(
            provenance(
                source_database=source_database,
                source_collection="state",
                source_id=source_id,
            )
        )
        converted.append(result)
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


def _ensure_target_compatible(
    database: Any, *, source_database: str, source_fingerprint: str
) -> None:
    """Refuse a non-empty target that was not created by this converter."""

    foreign = set(database.list_collection_names()) - COLLECTION_IDENTITIES.keys()
    if foreign:
        raise RuntimeError(
            f"Target database contains foreign collections: {sorted(foreign)}"
        )
    for collection_name in ("orders", "state", "blotter"):
        collection = database[collection_name]
        count = collection.count_documents({})
        compatible = collection.count_documents(
            {
                "migration_version": MIGRATION_VERSION,
                "source_database": source_database,
                "source_fingerprint": source_fingerprint,
            }
        )
        if count != compatible:
            raise RuntimeError(
                f"Target collection {collection_name!r} is incompatible or "
                "contains foreign, changed-source, or runtime-written data"
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


def _fingerprint(value: Any) -> str:
    """Hash BSON values canonically, including timestamps and source identities."""
    encoded = json_util.dumps(
        value, sort_keys=True, json_options=json_util.CANONICAL_JSON_OPTIONS
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def _read_source(database: Any, strategy_collection: str) -> dict[str, Any]:
    """Capture accounting evidence without instantiating runtime database services."""
    return {
        "orders": list(database["orders"].find({})),
        "state": list(database["state"].find({})),
        "blotter": list(database["blotter"].find({})),
        "strategy_collection": strategy_collection,
        "snapshot": _latest_snapshot(database[strategy_collection]),
        "strategy_snapshots": database[strategy_collection].count_documents({}),
    }


def _source_fingerprint(source: Mapping[str, Any]) -> str:
    """Make source identity independent of Mongo cursor ordering."""
    return _fingerprint(
        {
            **source,
            **{
                name: sorted(_fingerprint(document) for document in source[name])
                for name in COLLECTION_IDENTITIES
            },
        }
    )


def _require_unique(documents: Sequence[Mapping[str, Any]], field: str) -> None:
    """Reject missing or duplicate natural identities before any target writes."""
    identities = [document.get(field) for document in documents]
    if any(identity is None for identity in identities):
        raise ValueError(f"Missing {field} in converted records")
    duplicates = [
        identity for identity, count in Counter(identities).items() if count > 1
    ]
    if duplicates:
        raise ValueError(f"Duplicate {field} in converted records: {duplicates}")


class _ValidationSaver(AbstractBaseSaver):
    """Keep Book restoration and projection repairs entirely in memory."""

    def __init__(self, documents: Sequence[Mapping[str, Any]], identity: str) -> None:
        self.documents = {
            document[identity]: deepcopy(dict(document)) for document in documents
        }
        self.identity = identity

    def save(self, data: Any, /, *args: Any) -> None:
        self.documents[data[self.identity]] = deepcopy(data)

    def read(self, key: Any = None, /, *args: Any) -> list[dict[str, Any]]:
        return deepcopy(
            [
                document
                for document in self.documents.values()
                if all(
                    document.get(name) == value for name, value in (key or {}).items()
                )
            ]
        )


def _validate_restoration(
    orders: Sequence[Mapping[str, Any]],
    states: Sequence[Mapping[str, Any]],
    *,
    legacy: bool,
) -> dict[str, Any]:
    """Restore real Book accounting using isolated savers and inspect identities."""
    book = Book(
        order_saver=_ValidationSaver(orders, "orderId"),
        state_saver=_ValidationSaver(states, "state_key"),
        save_async=False,
        restore=True,
    )
    perm_ids: set[int] = set()
    fill_keys: set[str] = set()
    accounts: set[str] = set()
    for info in book.orders.query():
        if info.permId and info.permId in perm_ids:
            raise ValueError(f"Ambiguous broker permId={info.permId}")
        perm_ids.add(info.permId)
        if info.trade.order.account:
            accounts.add(info.trade.order.account)
        for record in info.fills:
            if record.deduplication_key in fill_keys:
                raise ValueError(
                    f"Duplicate fill identity {record.deduplication_key!r}"
                )
            fill_keys.add(record.deduplication_key)
            if record.execution.acctNumber:
                accounts.add(record.execution.acctNumber)
    if len(accounts) > 1:
        raise ValueError("Source contains more than one account/subaccount")
    if legacy:
        for document in states:
            expected = PositionState.decode(document)
            restored = book.positions.for_source(expected.source_key)
            if restored is None or restored.encode() != expected.encode():
                raise ValueError(
                    f"Book restoration changes legacy source {expected.source_key!r}"
                )
        _validate_legacy_protection(book)
    return {
        "validated": True,
        "accounts": sorted(accounts),
        "positions_by_con_id": {
            str(contract.conId): quantity
            for contract, quantity in book.positions.by_contract().items()
        },
        "sources": {
            state.source_key: {
                "quantity": state.quantity,
                "target_quantity": state.target_quantity,
                "position_id": state.position_id,
                "blocked_direction": state.blocked_direction,
                "execution_model_name": state.execution_model_name,
            }
            for state in book.positions.source_states().values()
        },
    }


def _validate_legacy_protection(book: Book) -> None:
    """Require settled episodes with matching live protection before cutover."""
    for state in book.positions.source_states().values():
        if state.quantity and (
            state.contract is None or not state.contract.conId or not state.position_id
        ):
            raise ValueError(
                f"Held source {state.source_key!r} lacks a concrete Contract or episode"
            )
        stops: list[OrderInfo] = []
        protection = book.orders.active(source_key=state.source_key)
        for info in protection:
            remaining = info.trade.order.totalQuantity - sum(
                record.execution.shares for record in info.fills
            )
            if (
                not state.quantity
                or not info.permId
                or info.position_id != state.position_id
                or info.trade.contract != state.contract
                or info.trade.order.action != ("SELL" if state.quantity > 0 else "BUY")
                or remaining != abs(state.quantity)
            ):
                raise ValueError(
                    f"Protective orderId={info.orderId} disagrees with source {state.source_key!r}"
                )
            if info.role == "STOP_LOSS":
                stops.append(info)
        if state.quantity and len(stops) != 1:
            raise ValueError(
                f"Held source {state.source_key!r} requires exactly one active stop before cutover"
            )
        if len(protection) > 1:
            groups = {info.trade.order.ocaGroup for info in protection}
            if len(groups) != 1 or "" in groups:
                raise ValueError(
                    f"Source {state.source_key!r} has inconsistent protection OCA groups"
                )


def _validate_settled_orders(
    orders: Sequence[Mapping[str, Any]],
    states: Sequence[Mapping[str, Any]],
    *,
    legacy: bool,
) -> None:
    """Reject working trading intent and incomplete legacy attribution."""
    sources = {
        state.get("source_key")
        for state in states
        if state.get("state_type") == "position"
    }
    for document in orders:
        info = OrderInfo.decode(document)
        if bool(document["active"]) != info.active:
            raise ValueError(
                f"orderId={info.orderId} active flag disagrees with Trade status"
            )
        if info.active and info.role not in {"STOP_LOSS", "TAKE_PROFIT"}:
            raise ValueError(
                f"Finish pending {info.role} orderId={info.orderId} under the old runtime before migration"
            )
        if (
            legacy
            and info.source_key not in sources
            and (info.fills or info.active or document.get("legacy_accounted_exec_ids"))
        ):
            raise ValueError(
                f"Missing legacy snapshot source for orderId={info.orderId}: {info.source_key!r}"
            )


def _verify_written_target(database: Any, expected: Mapping[str, Any]) -> None:
    """Read back every target record and refuse missing, extra or altered data."""
    for collection in COLLECTION_IDENTITIES:
        actual = [
            {key: value for key, value in document.items() if key != "_id"}
            for document in database[collection].find({})
        ]
        if sorted(map(_fingerprint, actual)) != sorted(
            map(_fingerprint, expected[collection])
        ):
            raise RuntimeError(
                f"Target {collection!r} does not match the complete conversion plan; do not launch"
            )


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
    fill_records = {
        fill["deduplication_key"]: FillRecord.decode(fill)
        for order in orders
        for fill in order.get("fills", ())
    }
    reports = [
        record.commission_report
        for record in fill_records.values()
        if record.commission_report is not None
    ]
    order_groups = Counter(
        (str(order.get("source_key")), str(order.get("position_id")))
        for order in orders
    )
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
        "fill_totals": {
            "unique_fills": len(fill_records),
            "commission": sum(report.commission for report in reports),
            "realized_pnl": sum(report.realizedPNL for report in reports),
        },
        "order_source_position_groups": {
            f"{source}|{position}": count
            for (source, position), count in order_groups.items()
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
    strategy_collection: str = "strategies",
    source_stopped: bool = False,
) -> dict[str, Any]:
    """Copy settled accounting into a separate database, preserving the source.

    Args:
        client: Mongo client configured with timezone-aware decoding.
        source_database: Existing database, read only throughout conversion.
        target_database: Fresh database, or an unfinished identical conversion.
        apply: Write only after source and in-memory restoration checks pass.
        strategy_collection: Legacy whole-system snapshot collection.
        source_stopped: Operator confirms the old runtime was stopped after
            reconciliation and its final persistence flush. Required to apply.

    Returns:
        Evidence counts, restoration results, source fingerprint and write status.

    Raises:
        ValueError: If accounting or cutover preconditions are ambiguous.
        RuntimeError: If the source changes or target compatibility/verification fails.
    """

    if not source_database or not target_database:
        raise ValueError("source and target database names are required")
    if source_database == target_database:
        raise ValueError("source and target databases must be different")
    if not strategy_collection or strategy_collection in COLLECTION_IDENTITIES:
        raise ValueError(
            "strategy_collection must be a distinct legacy snapshot collection"
        )
    if apply and not source_stopped:
        raise ValueError(
            "--apply requires --source-stopped after reconciliation and final persistence flush"
        )
    source = client[source_database]
    target = client[target_database]
    original = _read_source(source, strategy_collection)
    source_fingerprint = _source_fingerprint(original)
    _ensure_target_compatible(
        target,
        source_database=source_database,
        source_fingerprint=source_fingerprint,
    )

    original_orders = original["orders"]
    orders = [
        convert_order(document, source_database=source_database)
        for document in original_orders
    ]
    snapshot = original["snapshot"]
    component_states = original["state"]
    if snapshot is None and not component_states:
        raise ValueError(
            f"No snapshot in {strategy_collection!r} and no component state; "
            "select the real legacy collection with --strategy-collection"
        )
    states = (
        convert_latest_strategy_snapshot(
            snapshot,
            source_database=source_database,
            orders=orders,
            source_collection=strategy_collection,
        )
        if snapshot is not None
        else []
    )
    if component_states:
        if snapshot is not None:
            raise ValueError(
                "Source mixes strategy snapshots and component state; choose one authoritative source"
            )
        states = convert_component_states(
            component_states, original_orders, source_database=source_database
        )
    blotter = [
        convert_blotter(document, source_database=source_database)
        for document in original["blotter"]
    ]
    converted = {"orders": orders, "state": states, "blotter": blotter}
    for collection, identity in COLLECTION_IDENTITIES.items():
        _require_unique(converted[collection], identity)
        _require_unique(converted[collection], "migration_key")
        for document in converted[collection]:
            document["source_fingerprint"] = source_fingerprint
    legacy = snapshot is not None
    _validate_settled_orders(orders, states, legacy=legacy)
    restoration = _validate_restoration(orders, states, legacy=legacy)
    if (
        _source_fingerprint(_read_source(source, strategy_collection))
        != source_fingerprint
    ):
        raise RuntimeError(
            "Source changed during conversion; stop and reconcile the old runtime, then retry"
        )
    report = validation_report(orders, states, blotter)
    report["mode"] = "apply" if apply else "dry-run"
    report["source_database"] = source_database
    report["target_database"] = target_database
    report["strategy_collection"] = strategy_collection
    report["source_fingerprint"] = source_fingerprint
    report["source_stopped_confirmed"] = source_stopped
    report["book_restoration"] = restoration
    report["arctic_libraries"] = "untouched; new runtime starts separate audit history"
    report["source_counts"] = {
        "orders": len(original_orders),
        "strategy_snapshots": original["strategy_snapshots"],
        "state": len(component_states),
        "blotter": len(original["blotter"]),
    }
    if apply:
        target["orders"].create_index("orderId", unique=True)
        target["orders"].create_index("migration_key", unique=True, sparse=True)
        target["state"].create_index("state_key", unique=True)
        target["state"].create_index("migration_key", unique=True, sparse=True)
        target["blotter"].create_index("migration_key", unique=True, sparse=True)
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
        _verify_written_target(target, converted)
        report["target_verified"] = True
    report["target_counts"] = {
        collection_name: target[collection_name].count_documents({})
        for collection_name in COLLECTION_IDENTITIES
    }
    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line options without defaulting database identities."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    parser.add_argument("--source-db", required=True)
    parser.add_argument("--target-db", required=True)
    parser.add_argument(
        "--strategy-collection",
        default="strategies",
        help="legacy snapshot collection (use the old profile's strategy_collection_name)",
    )
    parser.add_argument(
        "--source-stopped",
        action="store_true",
        help="confirm the old runtime stopped after reconciliation and final persistence flush",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write the target; omitted means dry-run",
    )
    return parser.parse_args()


def main() -> None:
    """Run conversion and print its validation report as JSON."""

    args = parse_args()
    with MongoClient(args.mongo_uri, tz_aware=True) as client:
        report = migrate(
            client,
            source_database=args.source_db,
            target_database=args.target_db,
            apply=args.apply,
            strategy_collection=args.strategy_collection,
            source_stopped=args.source_stopped,
        )
    print(json.dumps(report, default=str, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
