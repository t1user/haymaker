from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from runpy import run_path

import ib_insync as ibi
import pytest

from haymaker.book import Book, PositionState
from haymaker.misc import decode_tree, tree

_MIGRATION = run_path(
    str(Path(__file__).parents[1] / "scripts" / "migrate_components_state.py")
)
MIGRATION_VERSION = _MIGRATION["MIGRATION_VERSION"]
convert_blotter = _MIGRATION["convert_blotter"]
convert_latest_strategy_snapshot = _MIGRATION["convert_latest_strategy_snapshot"]
convert_order = _MIGRATION["convert_order"]
convert_component_states = _MIGRATION["convert_component_states"]
migrate = _MIGRATION["migrate"]
order_role = _MIGRATION["order_role"]
validation_report = _MIGRATION["validation_report"]


def legacy_trade():
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    contract = ibi.Future(conId=1, symbol="ES", exchange="CME")
    order = ibi.Order(
        orderId=7,
        clientId=2,
        permId=99,
        action="BUY",
        totalQuantity=1,
    )
    trade = ibi.Trade(
        contract=contract,
        order=order,
        orderStatus=ibi.OrderStatus(
            orderId=7,
            status=ibi.OrderStatus.Filled,
            filled=1,
            remaining=0,
        ),
        fills=[
            ibi.Fill(
                contract=contract,
                execution=ibi.Execution(
                    execId="exec-1",
                    orderId=7,
                    permId=99,
                    side="BOT",
                    shares=1,
                    price=100,
                    time=timestamp,
                ),
                commissionReport=ibi.CommissionReport(
                    execId="exec-1",
                    commission=1,
                    realizedPNL=2,
                ),
                time=timestamp,
            )
        ],
        log=[
            ibi.TradeLogEntry(
                time=timestamp,
                status=ibi.OrderStatus.Submitted,
                message="submitted",
            )
        ],
    )
    return trade


@pytest.mark.parametrize("checkpoint", [None, [], ["exec-1"]])
def test_conversion_preserves_or_establishes_source_fill_checkpoint(
    checkpoint, order_saver, state_saver
):
    """Only old uncheckpointed snapshots establish an operator-reviewed baseline."""
    trade = legacy_trade()
    order = convert_order(
        {"strategy": "alpha", "action": "OPEN", "trade": tree(trade)},
        source_database="old",
    )
    state = PositionState(
        source_key="alpha",
        execution_model_name="legacy:alpha",
        contract=trade.contract,
        quantity=0,
    ).encode()
    if checkpoint is None:
        state.pop("applied_fill_keys")
        with pytest.raises(ValueError, match="fill checkpoint"):
            PositionState.decode(state)
    else:
        state["applied_fill_keys"] = checkpoint
    converted = convert_component_states([state], [order], source_database="old")[0]
    assert converted["applied_fill_keys"] == (
        ["exec-1"] if checkpoint is None else checkpoint
    )
    order_saver.save(order)
    state_saver.save(converted)
    recovered = Book(
        order_saver=order_saver,
        state_saver=state_saver,
        save_async=False,
        restore=True,
    )
    assert recovered.positions.for_source("alpha").quantity == (
        1 if checkpoint == [] else 0
    )


@pytest.mark.parametrize(
    ("legacy", "expected"),
    [
        ("STOP-LOSS", "STOP_LOSS"),
        ("TAKE-PROFIT", "TAKE_PROFIT"),
        ("FUTURE-ROLL", "ROLL"),
        ("CUSTOM_ALGO", "CUSTOM_ALGO"),
    ],
)
def test_role_mapping_preserves_custom_values(legacy, expected):
    assert order_role(legacy) == expected


def test_order_conversion_preserves_trade_fill_and_identifiers():
    trade = legacy_trade()
    converted = convert_order(
        {
            "_id": "legacy-order",
            "strategy": "alpha",
            "action": "OPEN",
            "trade": tree(trade),
            "params": {"position_id": "episode"},
            "active": False,
        },
        source_database="legacy",
    )

    assert converted["orderId"] == 7
    assert converted["clientId"] == 2
    assert converted["permId"] == 99
    assert converted["source_key"] == "alpha"
    assert converted["position_id"] == "episode"
    assert converted["params"]["legacy_action"] == "OPEN"
    assert converted["fills"][0]["deduplication_key"] == "exec-1"
    assert converted["migration_version"] == MIGRATION_VERSION


def test_direct_adjustment_conversion_supplies_current_target_identity():
    trade = legacy_trade()

    converted = convert_order(
        {
            "strategy": "alpha",
            "action": "TARGET_ADJUSTMENT",
            "trade": tree(trade),
        },
        source_database="legacy",
    )

    assert "target_key" not in converted
    assert decode_tree(converted["trade"]).contract.conId == trade.contract.conId
    assert converted["source_key"] is None
    assert converted["position_id"] is None


def test_order_conversion_refuses_invented_submission_timestamp():
    trade = legacy_trade()
    trade.log = []

    with pytest.raises(ValueError, match="honest submission"):
        convert_order(
            {"strategy": "alpha", "action": "OPEN", "trade": tree(trade)},
            source_database="legacy",
        )


def test_latest_strategy_conversion_preserves_episode_and_lock():
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    states = convert_latest_strategy_snapshot(
        {
            "_id": "snapshot",
            "timestamp": timestamp,
            "alpha": tree(
                {
                    "active_contract": ibi.Future(conId=1, symbol="ES", exchange="CME"),
                    "position": 0,
                    "lock": -1,
                    "position_id": "episode",
                    "params": {
                        "open": {"atr": 5},
                        "unrelated": "ignored",
                    },
                    "timestamp": timestamp,
                }
            ),
        },
        source_database="legacy",
    )

    assert states[0]["state_type"] == "position"
    assert states[0]["source_key"] == "alpha"
    assert states[0]["position_id"] == "episode"
    assert states[0]["blocked_direction"] == -1
    assert states[0]["bracket_inputs"] == {"atr": 5}


def test_component_conversion_rebuilds_derived_balances_in_book():
    """A saved account total must not become a second position in conversion."""
    balance = {
        "state_key": "balance:1",
        "state_type": "balance",
        "conId": 1,
        "contract": tree(legacy_trade().contract),
        "quantity": 1,
        "updated_at": datetime.now(timezone.utc),
    }
    assert convert_component_states([balance], [], source_database="source") == []
    with pytest.raises(ValueError, match="identity"):
        convert_component_states(
            [{**balance, "conId": 2}], [], source_database="source"
        )


def test_latest_strategy_conversion_recovers_custom_bracket_field():
    """Legacy bracket memos retain the configured volatility field."""

    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    states = convert_latest_strategy_snapshot(
        {
            "_id": "snapshot",
            "timestamp": timestamp,
            "alpha": tree(
                {
                    "active_contract": ibi.Future(conId=1, symbol="ES", exchange="CME"),
                    "position": 1,
                    "params": {
                        "open": {"custom_volatility": 7},
                        "stop-loss": {
                            "vol_field_name": "custom_volatility",
                            "vol_field_value": 7,
                            "sl_points": 14,
                        },
                    },
                    "timestamp": timestamp,
                }
            ),
        },
        source_database="legacy",
    )

    assert states[0]["bracket_inputs"] == {"custom_volatility": 7}


def test_blotter_conversion_and_validation_totals():
    converted = convert_blotter(
        {
            "_id": "row",
            "strategy": "alpha",
            "action": "CLOSE",
            "position_id": "episode",
            "order_id": 7,
            "commission": 1.5,
            "realizedPNL": 10,
        },
        source_database="legacy",
    )
    report = validation_report(
        [{"orderId": 7, "active": False}],
        [{"quantity": 0}],
        [converted],
    )

    assert converted["source_key"] == "alpha"
    assert converted["role"] == "CLOSE"
    assert report["totals"] == {"commission": 1.5, "realized_pnl": 10}
    assert report["unmatched_blotter_order_ids"] == []


class FakeCollection:
    def __init__(self, documents=None):
        self.documents = [dict(document) for document in documents or ()]

    def find(self, query):
        return [
            document
            for document in self.documents
            if all(document.get(key) == value for key, value in query.items())
        ]

    def find_one(self, query, sort=None):
        matches = self.find(query)
        if sort:
            name, direction = sort[0]
            matches.sort(
                key=lambda document: document.get(name),
                reverse=direction < 0,
            )
        return matches[0] if matches else None

    def estimated_document_count(self):
        return len(self.documents)

    def count_documents(self, query):
        return len(self.find(query))

    def create_index(self, *args, **kwargs):
        return None

    def replace_one(self, query, document, upsert=False):
        for index, existing in enumerate(self.documents):
            if all(existing.get(key) == value for key, value in query.items()):
                self.documents[index] = dict(document)
                return
        if upsert:
            self.documents.append(dict(document))


class FakeDatabase:
    def __init__(self):
        self.collections = {}

    def __getitem__(self, name):
        return self.collections.setdefault(name, FakeCollection())

    def __setitem__(self, name, collection):
        self.collections[name] = collection

    def list_collection_names(self):
        return list(self.collections)


class FakeClient:
    def __init__(self):
        self.databases = {}

    def __getitem__(self, name):
        return self.databases.setdefault(name, FakeDatabase())


def migration_client():
    client = FakeClient()
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    client["legacy"]["orders"] = FakeCollection(
        [
            {
                "_id": "legacy-order",
                "strategy": "alpha",
                "action": "OPEN",
                "trade": tree(legacy_trade()),
                "params": {"position_id": "episode"},
            }
        ]
    )
    stop = legacy_trade()
    stop.order.orderId = 8
    stop.order.permId = 100
    stop.order.action = "SELL"
    stop.order.orderType = "STP"
    stop.order.auxPrice = 95
    stop.order.ocaGroup = "episode-brackets"
    stop.order.ocaType = 2
    stop.orderStatus = ibi.OrderStatus(orderId=8, status="Submitted", remaining=1)
    stop.fills = []
    client["legacy"]["orders"].documents.append(
        {
            "_id": "stop",
            "strategy": "alpha",
            "action": "STOP-LOSS",
            "trade": tree(stop),
            "params": {
                "position_id": "episode",
                "vol_field_name": "atr",
                "vol_field_value": 5,
            },
        }
    )
    client["legacy"]["strategies"] = FakeCollection(
        [
            {
                "_id": "snapshot",
                "timestamp": timestamp,
                "alpha": tree(
                    {
                        "active_contract": ibi.Future(
                            conId=1, symbol="ES", exchange="CME"
                        ),
                        "position": 1,
                        "position_id": "episode",
                        "timestamp": timestamp,
                    }
                ),
            }
        ]
    )
    client["legacy"]["blotter"] = FakeCollection(
        [
            {
                "_id": "row",
                "strategy": "alpha",
                "action": "OPEN",
                "position_id": "episode",
                "order_id": 7,
                "commission": 1,
                "realizedPNL": 2,
            }
        ]
    )
    return client


def test_migration_defaults_to_non_mutating_dry_run():
    client = migration_client()

    report = migrate(
        client,
        source_database="legacy",
        target_database="fresh",
    )

    assert report["mode"] == "dry-run"
    assert report["source_counts"] == {
        "orders": 2,
        "strategy_snapshots": 1,
        "state": 0,
        "blotter": 1,
    }
    assert report["target_counts"] == {
        "orders": 0,
        "state": 0,
        "blotter": 0,
    }


def test_migration_apply_is_idempotent_and_reports_target_counts():
    client = migration_client()

    first = migrate(
        client,
        source_database="legacy",
        target_database="fresh",
        apply=True,
        source_stopped=True,
    )
    second = migrate(
        client,
        source_database="legacy",
        target_database="fresh",
        apply=True,
        source_stopped=True,
    )

    assert first["target_counts"] == {
        "orders": 2,
        "state": 1,
        "blotter": 1,
    }
    assert second["target_counts"] == first["target_counts"]


def test_migration_refuses_mixed_or_foreign_target_database():
    client = migration_client()
    client["fresh"]["orders"] = FakeCollection([{"unrelated": True}])

    with pytest.raises(RuntimeError, match="incompatible"):
        migrate(
            client,
            source_database="legacy",
            target_database="fresh",
        )


def component_order(*, source_key=None, target_key=None):
    """Build old component evidence with explicit fills absent from Trade.fills."""
    from haymaker.book import FillRecord, OrderInfo

    trade = legacy_trade()
    record = FillRecord.from_fill(trade, trade.fills[0])
    trade.fills.clear()
    document = OrderInfo(
        trade=trade,
        role="OPEN" if source_key else "TARGET_ADJUSTMENT",
        source_key=source_key,
        position_id="episode" if source_key else None,
        execution_model_name="model",
        submitted_at=trade.log[0].time,
        params={"atr": 5},
        fills=(record,),
    ).encode()
    if target_key is not None:
        document["target_key"] = target_key
    return document


def keyed_target(con_id=1, key="old-key"):
    """Represent the old logical-key direct state."""
    from haymaker.book import TargetState

    document = TargetState(
        contract=ibi.Future("ES", conId=con_id, exchange="CME"),
        execution_model_name="model",
        target_quantity=1,
        target_created_at=datetime.now(timezone.utc),
    ).encode()
    document["target_key"] = key
    document["state_key"] = f"target:{key}"
    return document


def test_component_order_keeps_authoritative_fills_and_commissions():
    """Trade is diagnostic; conversion must preserve its separate Fill evidence."""
    original = component_order(target_key="old-key")
    converted = convert_order(original, source_database="old")
    assert "target_key" not in converted
    assert original["target_key"] == "old-key"
    assert converted["fills"] == original["fills"]
    assert converted["submitted_at"] == original["submitted_at"]
    assert converted["execution_model_name"] == "model"
    report = validation_report([converted], [], [])
    assert report["fill_totals"] == {
        "unique_fills": 1,
        "commission": 1,
        "realized_pnl": 2,
    }


def test_component_state_separates_held_and_pending_contract_and_inputs():
    """The overwritten old target Contract cannot be mistaken for the holding."""
    from haymaker.book import PositionState

    incoming = ibi.Future("ES", conId=2, exchange="CME")
    state = PositionState(
        source_key="alpha",
        execution_model_name="model",
        contract=incoming,
        quantity=1,
        target_quantity=-1,
        target_created_at=datetime.now(timezone.utc),
        position_id="episode",
        bracket_inputs={"atr": 9},
    ).encode()
    del state["target_contract"]
    del state["target_bracket_inputs"]
    result = convert_component_states(
        [state], [component_order(source_key="alpha")], source_database="old"
    )[0]
    restored = PositionState.decode(result)
    assert restored.contract.conId == 1
    assert restored.target_contract == incoming
    assert restored.bracket_inputs == {"atr": 5}
    assert restored.target_bracket_inputs == {"atr": 9}


def test_keyed_direct_state_converts_only_unambiguous_concrete_ownership():
    """Do not invent allocation between an old holding and a different target."""
    orders = [component_order(target_key="old-key")]
    converted = convert_component_states(
        [keyed_target()], orders, source_database="old"
    )[0]
    assert converted["state_key"] == "target:1"
    assert "target_key" not in converted
    with pytest.raises(ValueError, match="Ambiguous concrete allocation"):
        convert_component_states(
            [keyed_target(con_id=2)], orders, source_database="old"
        )
    with pytest.raises(ValueError, match="Multiple old states"):
        convert_component_states(
            [keyed_target(), keyed_target(key="other")], orders, source_database="old"
        )


def test_component_conversion_apply_is_idempotent_and_source_is_unchanged():
    """A fresh target can be safely retried with deterministic provenance."""
    from copy import deepcopy

    client = FakeClient()
    orders = [component_order(target_key="old-key")]
    states = [keyed_target()]
    original = deepcopy((orders, states))
    client["old"]["orders"] = FakeCollection(orders)
    client["old"]["state"] = FakeCollection(states)
    first = migrate(
        client,
        source_database="old",
        target_database="fresh",
        apply=True,
        source_stopped=True,
    )
    second = migrate(
        client,
        source_database="old",
        target_database="fresh",
        apply=True,
        source_stopped=True,
    )
    assert (
        first["target_counts"]
        == second["target_counts"]
        == {"orders": 1, "state": 1, "blotter": 0}
    )
    assert (orders, states) == original


def test_pending_roll_schema_change_is_refused_before_writes():
    """A converter does not reinterpret in-flight broker sequencing."""
    client = FakeClient()
    client["old"]["state"] = FakeCollection(
        [{"state_type": "roll", "stage": "ROLL_ORDER_ACTIVE"}]
    )
    with pytest.raises(ValueError, match="Finish pending roll"):
        migrate(
            client,
            source_database="old",
            target_database="fresh",
            apply=True,
            source_stopped=True,
        )
    assert client["fresh"]["orders"].documents == []
    assert client["fresh"]["state"].documents == []


def test_models_collection_copy_preserves_source_and_restores_existing_protection(
    atom_runtime_factory, order_saver, state_saver, monkeypatch
):
    """A protected position survives conversion and processes its next real fill."""
    from haymaker.components import BracketExecutionModel, TrailingStop
    from haymaker.controller import Controller

    client = migration_client()
    client["legacy"]["models"] = client["legacy"]["strategies"]
    del client["legacy"].collections["strategies"]
    snapshot = client["legacy"]["models"].documents[0]
    snapshot["alpha"]["params"] = {"open": {"range": 17, "position_id": "episode"}}
    snapshot["locked"] = {"position": 0, "lock": -1, "timestamp": snapshot["timestamp"]}
    original = deepcopy(
        {name: col.documents for name, col in client["legacy"].collections.items()}
    )
    report = migrate(
        client,
        source_database="legacy",
        target_database="fresh",
        strategy_collection="models",
        source_stopped=True,
        apply=True,
    )
    assert report["target_verified"] is True
    assert report["book_restoration"]["positions_by_con_id"] == {"1": 1}
    assert {
        name: col.documents
        for name, col in client["legacy"].collections.items()
        if col.documents
    } == original
    for document in client["fresh"]["orders"].documents:
        order_saver.save(document)
    for document in client["fresh"]["state"].documents:
        state_saver.save(document)
        assert document["source_collection"] == "models"
    book = Book(
        order_saver=order_saver, state_saver=state_saver, restore=True, save_async=False
    )
    runtime = atom_runtime_factory(book_=book)
    runtime.bind_controller(Controller(trader=runtime.trader))

    def forbidden(*args, **kwargs):
        raise AssertionError("Recovery must preserve the existing broker orders")

    monkeypatch.setattr(runtime.ib, "placeOrder", forbidden)
    monkeypatch.setattr(runtime.ib, "cancelOrder", forbidden)
    model = BracketExecutionModel(
        "alpha", name="legacy:alpha", stop=TrailingStop(1, vol_field="range")
    )
    model.recover()
    assert book.positions.for_source("alpha").bracket_inputs["range"] == 17
    assert book.positions.blocked_direction("locked") == -1
    assert not book.apply_fill(book.orders.by_id(7).trade, legacy_trade().fills[0])
    stop = book.orders.by_id(8).trade
    stop.orderStatus.status = ibi.OrderStatus.Filled
    stop.orderStatus.filled = 1
    stop.orderStatus.remaining = 0
    fill = ibi.Fill(
        stop.contract,
        ibi.Execution(
            execId="after-cutover",
            orderId=8,
            permId=100,
            side="SLD",
            shares=1,
            price=95,
            time=snapshot["timestamp"] + timedelta(seconds=10),
        ),
        ibi.CommissionReport(),
        snapshot["timestamp"] + timedelta(seconds=10),
    )
    stop.fills.append(fill)
    assert book.apply_fill(stop, fill)
    assert book.positions.quantity(stop.contract) == 0
    assert book.positions.blocked_direction("alpha") == 1
    assert not book.apply_fill(stop, fill)
    recovered = Book(
        order_saver=order_saver, state_saver=state_saver, restore=True, save_async=False
    )
    assert recovered.positions.quantity(stop.contract) == 0
    assert recovered.positions.blocked_direction("alpha") == 1


def test_missing_models_selection_refuses_before_writes():
    client = migration_client()
    client["legacy"]["models"] = client["legacy"]["strategies"]
    del client["legacy"].collections["strategies"]
    with pytest.raises(ValueError, match="strategy-collection"):
        migrate(
            client,
            source_database="legacy",
            target_database="fresh",
            source_stopped=True,
            apply=True,
        )
    assert client["fresh"]["orders"].documents == []


def test_legacy_accounted_keys_survive_missing_trade_fill_history(
    order_saver, state_saver
):
    """Rebound legacy Trades can omit fills that the snapshot already includes."""
    client = migration_client()
    client["legacy"]["orders"].documents[0]["accounted_exec_ids"] = [
        "exec-1",
        "older-fill",
    ]
    migrate(
        client,
        source_database="legacy",
        target_database="fresh",
        source_stopped=True,
        apply=True,
    )
    for document in client["fresh"]["orders"].documents:
        order_saver.save(document)
    for document in client["fresh"]["state"].documents:
        state_saver.save(document)
    book = Book(
        order_saver=order_saver, state_saver=state_saver, restore=True, save_async=False
    )
    old_fill = deepcopy(legacy_trade().fills[0])
    old_fill.execution.execId = "older-fill"
    old_fill.commissionReport.execId = "older-fill"
    assert book.apply_fill(book.orders.by_id(7).trade, old_fill)
    assert len(book.orders.by_id(7).fills) == 2
    assert book.positions.quantity(old_fill.contract) == 1
    restored = Book(
        order_saver=order_saver, state_saver=state_saver, restore=True, save_async=False
    )
    assert restored.positions.quantity(old_fill.contract) == 1


def test_apply_requires_explicit_stopped_source_and_distinct_databases():
    client = migration_client()
    with pytest.raises(ValueError, match="source-stopped"):
        migrate(client, source_database="legacy", target_database="fresh", apply=True)
    with pytest.raises(ValueError, match="must be different"):
        migrate(
            client,
            source_database="legacy",
            target_database="legacy",
            source_stopped=True,
            apply=True,
        )
    assert client["fresh"]["orders"].documents == []


@pytest.mark.parametrize(
    "role", ["OPEN", "CLOSE", "FUTURE-ROLL", "MANUAL", "TARGET_ADJUSTMENT"]
)
def test_legacy_pending_work_is_refused_before_writes(role):
    client = migration_client()
    client["legacy"]["orders"].documents[1]["action"] = role
    with pytest.raises(ValueError, match="Finish pending"):
        migrate(
            client,
            source_database="legacy",
            target_database="fresh",
            source_stopped=True,
            apply=True,
        )
    assert client["fresh"]["orders"].documents == []


@pytest.mark.parametrize("stale", ["newer_fill", "unaccounted_fill"])
def test_inconsistent_snapshot_cannot_hide_a_fill(stale):
    client = migration_client()
    if stale == "newer_fill":
        client["legacy"]["strategies"].documents[0]["timestamp"] -= timedelta(seconds=1)
    else:
        client["legacy"]["orders"].documents[0]["accounted_exec_ids"] = []
    with pytest.raises(ValueError, match="newer than|unaccounted legacy fill"):
        migrate(
            client,
            source_database="legacy",
            target_database="fresh",
            source_stopped=True,
            apply=True,
        )
    assert client["fresh"]["orders"].documents == []


def test_corrected_snapshot_remains_authoritative_over_historical_fills():
    client = migration_client()
    client["legacy"]["orders"].documents[0]["accounted_exec_ids"] = ["exec-1"]
    snapshot = client["legacy"]["strategies"].documents[0]
    snapshot["alpha"]["position"] = 0
    snapshot["alpha"]["lock"] = 1
    stop = decode_tree(client["legacy"]["orders"].documents[1]["trade"])
    stop.orderStatus.status = ibi.OrderStatus.Cancelled
    client["legacy"]["orders"].documents[1]["trade"] = tree(stop)
    report = migrate(client, source_database="legacy", target_database="fresh")
    state = report["book_restoration"]["sources"]["alpha"]
    assert state["quantity"] == 0
    assert state["target_quantity"] == 0
    assert state["blocked_direction"] == 1


@pytest.mark.parametrize("collision", ["orderId", "permId", "fill"])
def test_duplicate_identities_do_not_overwrite_evidence(collision):
    client = migration_client()
    duplicate = deepcopy(client["legacy"]["orders"].documents[0])
    duplicate["_id"] = "another-order"
    trade = decode_tree(duplicate["trade"])
    if collision != "orderId":
        trade.order.orderId = 90
    if collision != "permId":
        trade.order.permId = 900
    if collision != "fill":
        trade.fills[0].execution.execId = "another-fill"
    duplicate["trade"] = tree(trade)
    client["legacy"]["orders"].documents.append(duplicate)
    with pytest.raises(ValueError, match="Duplicate|Ambiguous broker"):
        migrate(
            client,
            source_database="legacy",
            target_database="fresh",
            source_stopped=True,
            apply=True,
        )
    assert client["fresh"]["orders"].documents == []


@pytest.mark.parametrize("problem", ["absent", "quantity", "episode", "active_flag"])
def test_inconsistent_protection_is_refused(problem):
    client = migration_client()
    document = client["legacy"]["orders"].documents[1]
    trade = decode_tree(document["trade"])
    if problem == "absent":
        trade.orderStatus.status = ibi.OrderStatus.Cancelled
    elif problem == "quantity":
        trade.order.totalQuantity = 2
    elif problem == "episode":
        document["params"]["position_id"] = "wrong-episode"
    else:
        document["active"] = False
    document["trade"] = tree(trade)
    with pytest.raises(ValueError, match="active stop|disagrees"):
        migrate(
            client,
            source_database="legacy",
            target_database="fresh",
            source_stopped=True,
            apply=True,
        )
    assert client["fresh"]["orders"].documents == []


@pytest.mark.parametrize("evidence", ["fills", "accounting_keys"])
def test_missing_source_cannot_resurrect_historical_positions(evidence):
    client = migration_client()
    if evidence == "accounting_keys":
        document = client["legacy"]["orders"].documents[0]
        trade = decode_tree(document["trade"])
        trade.fills = []
        document["trade"] = tree(trade)
        document["accounted_exec_ids"] = ["exec-1"]
        client["legacy"]["orders"].documents.pop()
    del client["legacy"]["strategies"].documents[0]["alpha"]
    with pytest.raises(ValueError, match="Missing legacy snapshot source"):
        migrate(
            client,
            source_database="legacy",
            target_database="fresh",
            source_stopped=True,
            apply=True,
        )


def test_source_changes_during_conversion_are_refused(monkeypatch):
    client = migration_client()
    collection = client["legacy"]["orders"]
    original_find = collection.find
    calls = 0

    def changing_find(query):
        nonlocal calls
        calls += 1
        if calls == 2:
            collection.documents[0]["priority"] = 100
        return original_find(query)

    monkeypatch.setattr(collection, "find", changing_find)
    with pytest.raises(RuntimeError, match="Source changed"):
        migrate(
            client,
            source_database="legacy",
            target_database="fresh",
            source_stopped=True,
            apply=True,
        )
    assert client["fresh"]["orders"].documents == []


def test_changed_source_cannot_overwrite_previous_conversion():
    client = migration_client()
    migrate(
        client,
        source_database="legacy",
        target_database="fresh",
        source_stopped=True,
        apply=True,
    )
    before = deepcopy(client["fresh"]["orders"].documents)
    client["legacy"]["orders"].documents[0]["priority"] = 100
    with pytest.raises(RuntimeError, match="changed-source"):
        migrate(
            client,
            source_database="legacy",
            target_database="fresh",
            source_stopped=True,
            apply=True,
        )
    assert client["fresh"]["orders"].documents == before


def test_foreign_target_collection_is_refused():
    client = migration_client()
    client["fresh"]["foreign"] = FakeCollection([{"value": 1}])
    with pytest.raises(RuntimeError, match="foreign collections"):
        migrate(client, source_database="legacy", target_database="fresh")


def test_target_readback_catches_missing_writes(monkeypatch):
    client = migration_client()
    monkeypatch.setattr(
        client["fresh"]["orders"], "replace_one", lambda *args, **kwargs: None
    )
    with pytest.raises(RuntimeError, match="complete conversion plan"):
        migrate(
            client,
            source_database="legacy",
            target_database="fresh",
            source_stopped=True,
            apply=True,
        )


def test_interrupted_target_can_resume_identical_source(monkeypatch):
    client = migration_client()
    original = client["fresh"]["state"].replace_one

    def fail(*args, **kwargs):
        raise RuntimeError("interrupted write")

    monkeypatch.setattr(client["fresh"]["state"], "replace_one", fail)
    with pytest.raises(RuntimeError, match="interrupted write"):
        migrate(
            client,
            source_database="legacy",
            target_database="fresh",
            source_stopped=True,
            apply=True,
        )
    assert len(client["fresh"]["orders"].documents) == 2
    monkeypatch.setattr(client["fresh"]["state"], "replace_one", original)
    report = migrate(
        client,
        source_database="legacy",
        target_database="fresh",
        source_stopped=True,
        apply=True,
    )
    assert report["target_verified"]
    assert report["target_counts"] == {"orders": 2, "state": 1, "blotter": 1}


def test_migration_indexes_allow_subsequent_runtime_writes():
    """Sparse provenance indexes must allow multiple ordinary runtime records."""
    import mongomock
    from haymaker.saver import MongoSaver

    fixture = migration_client()
    client = mongomock.MongoClient(tz_aware=True)
    for name, collection in fixture["legacy"].collections.items():
        client["legacy"][name].insert_many(deepcopy(collection.documents))
    source_before = {
        name: list(client["legacy"][name].find())
        for name in client["legacy"].list_collection_names()
    }
    migrate(
        client,
        source_database="legacy",
        target_database="fresh",
        source_stopped=True,
        apply=True,
    )
    book = Book(
        order_saver=MongoSaver(
            "orders",
            client=client,
            database="fresh",
            query_key="orderId",
            tz_aware=True,
        ),
        state_saver=MongoSaver(
            "state",
            client=client,
            database="fresh",
            query_key="state_key",
            tz_aware=True,
        ),
        restore=True,
        save_async=False,
    )
    for source in ("new-a", "new-b"):
        book.update_position(
            PositionState(source_key=source, execution_model_name="new")
        )
    for order_id in (20, 21):
        client["fresh"]["orders"].insert_one({"orderId": order_id})
        client["fresh"]["blotter"].insert_one({"order_id": order_id})
    assert client["fresh"]["state"].count_documents({}) == 4
    assert {
        name: list(client["legacy"][name].find())
        for name in client["legacy"].list_collection_names()
    } == source_before
    with pytest.raises(RuntimeError, match="runtime-written"):
        migrate(client, source_database="legacy", target_database="fresh")
