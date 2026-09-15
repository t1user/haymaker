from datetime import datetime, timezone
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
        "orders": 1,
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
    )
    second = migrate(
        client,
        source_database="legacy",
        target_database="fresh",
        apply=True,
    )

    assert first["target_counts"] == {
        "orders": 1,
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
    first = migrate(client, source_database="old", target_database="fresh", apply=True)
    second = migrate(client, source_database="old", target_database="fresh", apply=True)
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
        migrate(client, source_database="old", target_database="fresh", apply=True)
    assert client["fresh"]["orders"].documents == []
    assert client["fresh"]["state"].documents == []
