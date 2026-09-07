from dataclasses import FrozenInstanceError, is_dataclass
from datetime import datetime, timezone
from types import SimpleNamespace

import ib_insync as ibi
import numpy as np
import pandas as pd
import pytest

from haymaker import misc
from haymaker.async_wrappers import QueueShutdownPolicy
from haymaker.components import (
    PandasSignalModel,
    Signal,
    SignalCalculation,
    SignalPair,
    SignalModel,
    SignalType,
)
from haymaker.datastore import QueuedSignalFramePersistence
from haymaker.enums import ActiveNext


class Model(PandasSignalModel):
    def df(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        result["signal"] = result["close"].apply(lambda value: 1 if value > 10 else 0)
        result["atr"] = 2.5
        return result


def test_signal_contract_hook_can_pass_blueprint_to_direct_portfolio(atom_runtime):
    """User overrides only Contract selection; the framework builds the envelope."""

    class BlueprintModel(Model):
        def select_signal_contract(self) -> ibi.Contract:
            """Leave concrete Contract selection to the direct Portfolio."""
            return self.contract_blueprint

    blueprint = ibi.Stock("AAPL", "SMART", "USD")
    model = BlueprintModel("alpha", blueprint, SignalType.STATE)
    qualified = ibi.Stock("AAPL", "SMART", "USD", conId=123)
    atom_runtime.contract_registry.reset_data(
        [[ibi.ContractDetails(contract=qualified)]]
    )
    frame = pd.DataFrame({"close": [12]})
    signal = model.create_signal(frame)
    assert model.contract == qualified
    assert signal.contract == blueprint
    assert signal.contract.conId == 0
    assert signal.source_key == "alpha"
    assert signal.value == 1
    assert signal.metadata["atr"] == 2.5


class EntryExitModel(PandasSignalModel):
    def df(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        result["in"] = result["close"].apply(lambda value: 1 if value > 10 else 0)
        result["out"] = result["close"].apply(lambda value: -1 if value < 10 else 0)
        result["atr"] = 2.5
        return result


class FakeAuditSink:
    shutdown_policy = QueueShutdownPolicy.DRAIN

    def __init__(self):
        self.calls = []

    def enqueue_write(self, symbol, data, meta=None):
        self.calls.append(("write", symbol, data.copy(), meta))

    def enqueue_append(self, symbol, data, meta=None, upsert=True):
        self.calls.append(("append", symbol, data.copy(), meta))

    def enqueue_write_metadata(self, symbol, meta):
        self.calls.append(("metadata", symbol, meta))


class RecordingPersistence:
    """Record synchronous persistence calls and return a configured reference."""

    def __init__(self, reference: str | None = "calculation-reference") -> None:
        """Initialize the reference and empty call history."""

        self.reference = reference
        self.calls: list[tuple[pd.DataFrame, str, ibi.Contract, datetime]] = []

    def save(
        self,
        frame: pd.DataFrame,
        *,
        source_key: str,
        active_contract: ibi.Contract,
        run_started_at: datetime,
    ) -> str | None:
        """Record one non-blocking persistence request."""

        self.calls.append((frame, source_key, active_contract, run_started_at))
        return self.reference


def queued_persistence(
    sink: FakeAuditSink | None = None,
) -> QueuedSignalFramePersistence:
    """Return default Signal persistence around a test sink."""

    return QueuedSignalFramePersistence(sink or FakeAuditSink())


def model(**kwargs):
    return Model(
        "alpha",
        ibi.Future(conId=1, symbol="ES", exchange="CME", localSymbol="ESM6"),
        SignalType.STATE,
        **kwargs,
    )


def frame(last=11):
    return pd.DataFrame(
        {"close": [9, last]},
        index=pd.DatetimeIndex(
            [
                datetime(2026, 1, 1, tzinfo=timezone.utc),
                datetime(2026, 1, 2, tzinfo=timezone.utc),
            ]
        ),
    )


def install_selector(
    monkeypatch: pytest.MonkeyPatch,
    atom_runtime,
    *,
    active: ibi.Contract,
    next_contract: ibi.Contract | None = None,
) -> None:
    """Install a resolved ACTIVE/NEXT selector on the test runtime."""

    selector = SimpleNamespace(
        active_contract=active,
        next_contract=next_contract or active,
    )
    monkeypatch.setattr(
        atom_runtime.contract_registry,
        "get_selector",
        lambda blueprint: selector,
    )


def test_signal_models_are_identity_based_dataclasses_without_roll_policy(
    atom_runtime,
):
    first = model()
    second = model()

    assert is_dataclass(first)
    assert first != second
    assert atom_runtime.future_roll_policies == {}


def test_signal_model_rejects_non_string_source_key(atom_runtime):
    with pytest.raises(TypeError, match="source_key must be a string"):
        Model(
            1,
            ibi.Future(conId=1, symbol="ES", exchange="CME"),
            SignalType.STATE,
        )


def test_pandas_model_emits_structured_signal(atom_runtime):
    output = []
    subject = model()
    subject.dataEvent += output.append

    subject.onData(frame())

    assert output == [
        Signal(
            source_key="alpha",
            contract=subject.contract,
            value=1,
            signal_type=SignalType.STATE,
            as_of=datetime(2026, 1, 2, tzinfo=timezone.utc),
            created_at=output[0].created_at,
            metadata={"close": 11, "atr": 2.5},
        )
    ]


def test_pandas_model_accepts_numpy_integer_signal(atom_runtime):
    class IntegerModel(PandasSignalModel):
        def df(self, data):
            return pd.DataFrame(
                {"signal": np.array([0, 1], dtype=np.int64)},
                index=data.index,
            )

    result = IntegerModel(
        "alpha",
        ibi.Future(conId=1, symbol="ES", exchange="CME"),
        SignalType.STATE,
    ).create_signal(frame())

    assert result.value == 1.0


def test_signal_calculation_is_frozen_and_metadata_is_optional():
    calculation = SignalCalculation(value=1)

    assert calculation.metadata == {}
    assert calculation.as_of is None
    with pytest.raises(FrozenInstanceError):
        calculation.value = 2


def test_signal_model_owns_signal_envelope_and_both_timestamps(atom_runtime):
    observed_at = datetime(2026, 1, 1, tzinfo=timezone.utc)

    class CalculationModel(SignalModel):
        def calculate_signal(self, data):
            return SignalCalculation(
                value=data,
                metadata={"atr": 2},
                as_of=observed_at,
            )

    subject = CalculationModel(
        "alpha",
        ibi.Future(conId=1, symbol="ES", exchange="CME"),
        SignalType.STATE,
    )
    created_after = datetime.now(timezone.utc)

    result = subject.create_signal(1)

    assert result.source_key == "alpha"
    assert result.contract == subject.contract
    assert result.value == 1.0
    assert result.signal_type is SignalType.STATE
    assert result.as_of is observed_at
    assert result.created_at >= created_after
    assert result.metadata == {"atr": 2}


def test_signal_model_requires_signal_calculation(atom_runtime):
    class WrongCalculationModel(SignalModel):
        def calculate_signal(self, data):
            return data

    subject = WrongCalculationModel(
        "alpha",
        ibi.Future(conId=1, symbol="ES", exchange="CME"),
        SignalType.STATE,
    )

    with pytest.raises(TypeError, match="must return SignalCalculation"):
        subject.onData(1)


def test_signal_model_allows_model_specific_value_validation(atom_runtime):
    class BinaryModel(SignalModel):
        def calculate_signal(self, data):
            return SignalCalculation(value=data)

        def validate_signal_value(self, value):
            if value not in (-1.0, 0.0, 1.0):
                raise ValueError("value must be binary")

    subject = BinaryModel(
        "alpha",
        ibi.Future(conId=1, symbol="ES", exchange="CME"),
        SignalType.STATE,
    )

    with pytest.raises(ValueError, match="must be binary"):
        subject.onData(2)


def test_pandas_model_treats_naive_observation_index_as_utc(atom_runtime):
    data = frame()
    data.index = data.index.tz_localize(None)

    result = model().create_signal(data)

    assert result.as_of.tzinfo is timezone.utc


def test_pandas_model_uses_custom_scalar_signal_field(atom_runtime):
    class AlternateModel(Model):
        def df(self, data):
            result = super().df(data)
            result["direction"] = -1
            return result

    subject = AlternateModel(
        "alpha",
        ibi.Future(conId=1, symbol="ES", exchange="CME"),
        SignalType.STATE,
        signal_fields="direction",
    )

    result = subject.create_signal(frame())

    assert result.value == -1
    assert "direction" not in result.metadata
    assert result.metadata["signal"] == 1


def test_pandas_model_builds_pair_and_excludes_both_fields_from_metadata(
    atom_runtime,
):
    subject = EntryExitModel(
        "alpha",
        ibi.Future(conId=1, symbol="ES", exchange="CME"),
        SignalType.EVENT,
        signal_fields=("in", "out"),
    )

    result = subject.create_signal(frame())

    assert result.value == SignalPair(entry=1, exit=0)
    assert result.signal_type is SignalType.EVENT
    assert result.metadata == {"close": 11, "atr": 2.5}


@pytest.mark.parametrize(
    ("signal_fields", "exception", "message"),
    [
        (("entry",), ValueError, "exactly two"),
        (("entry", "exit", "other"), ValueError, "exactly two"),
        (("entry", 1), TypeError, "must be strings"),
        (["entry", "exit"], TypeError, "field name or a two-field tuple"),
    ],
)
def test_pandas_model_rejects_invalid_signal_fields(
    atom_runtime, signal_fields, exception, message
):
    with pytest.raises(exception, match=message):
        model(signal_fields=signal_fields)


@pytest.mark.parametrize(
    "signal_fields",
    ("", ("entry", "entry"), ("entry", "")),
)
def test_pandas_model_accepts_string_field_names(
    atom_runtime,
    signal_fields: str | tuple[str, str],
) -> None:
    """Accept any pandas column names once the field shape is valid."""

    subject = model(signal_fields=signal_fields)

    assert subject.signal_fields == signal_fields


@pytest.mark.parametrize(
    ("metadata_fields", "expected"),
    [
        (None, {"close": 11, "atr": 2.5}),
        ((), {}),
        (("atr",), {"atr": 2.5}),
    ],
)
def test_pandas_model_selects_metadata_fields(
    atom_runtime,
    metadata_fields,
    expected,
) -> None:
    subject = model(metadata_fields=metadata_fields)

    result = subject.create_signal(frame())

    assert result.metadata == expected


@pytest.mark.parametrize("metadata_fields", ["atr", ("atr", 1)])
def test_pandas_model_rejects_invalid_metadata_fields(
    atom_runtime,
    metadata_fields,
) -> None:
    with pytest.raises(TypeError, match="metadata_fields"):
        model(metadata_fields=metadata_fields)


def test_pandas_model_reports_all_missing_pair_fields_before_persistence(
    atom_runtime,
):
    class MissingPair(PandasSignalModel):
        def df(self, data):
            return data

    sink = FakeAuditSink()
    subject = MissingPair(
        "alpha",
        ibi.Future(conId=1, symbol="ES", exchange="CME"),
        SignalType.STATE,
        signal_fields=("in", "out"),
        persistence=queued_persistence(sink),
    )

    with pytest.raises(KeyError, match="'in'.*'out'"):
        subject.onData(frame())

    assert sink.calls == []


def test_custom_row_conversion_must_return_signal_calculation(atom_runtime):
    class InvalidRowModel(Model):
        def row_to_calculation(self, row):
            return {"value": row["signal"]}

    subject = InvalidRowModel(
        "alpha",
        ibi.Future(conId=1, symbol="ES", exchange="CME", localSymbol="ESM6"),
        SignalType.STATE,
    )

    with pytest.raises(TypeError, match="must return SignalCalculation"):
        subject.create_signal(frame())


def test_custom_row_conversion_supplies_only_calculated_fields(atom_runtime):
    observed_at = datetime(2025, 12, 31, tzinfo=timezone.utc)

    class CustomRowModel(Model):
        def row_to_calculation(self, row):
            return SignalCalculation(
                value=row["signal"],
                metadata={"custom": row["atr"]},
                as_of=observed_at,
            )

    subject = CustomRowModel(
        "alpha",
        ibi.Future(conId=1, symbol="ES", exchange="CME", localSymbol="ESM6"),
        SignalType.STATE,
    )

    result = subject.create_signal(frame())

    assert result.source_key == "alpha"
    assert result.contract == subject.contract
    assert result.signal_type is SignalType.STATE
    assert result.value == 1.0
    assert result.metadata == {"custom": 2.5}
    assert result.as_of is observed_at


def test_invalid_custom_calculation_creates_no_persistence_generation(atom_runtime):
    class InvalidCalculationModel(Model):
        def row_to_calculation(self, row):
            return SignalCalculation(value=float("nan"))

    sink = FakeAuditSink()
    subject = InvalidCalculationModel(
        "alpha",
        ibi.Future(conId=1, symbol="ES", exchange="CME", localSymbol="ESM6"),
        SignalType.STATE,
        persistence=queued_persistence(sink),
    )

    with pytest.raises(ValueError, match="finite"):
        subject.onData(frame())

    assert sink.calls == []


def test_model_specific_validation_precedes_persistence(atom_runtime):
    class RejectingModel(Model):
        def validate_signal_value(self, value):
            raise ValueError("rejected calculated value")

    sink = FakeAuditSink()
    subject = RejectingModel(
        "alpha",
        ibi.Future(conId=1, symbol="ES", exchange="CME"),
        SignalType.STATE,
        persistence=queued_persistence(sink),
    )

    with pytest.raises(ValueError, match="rejected calculated value"):
        subject.onData(frame())

    assert sink.calls == []


def test_missing_explicit_metadata_creates_no_persistence_generation(atom_runtime):
    sink = FakeAuditSink()
    subject = model(
        metadata_fields=("missing",),
        persistence=queued_persistence(sink),
    )

    with pytest.raises(KeyError, match="metadata.*'missing'"):
        subject.onData(frame())

    assert sink.calls == []


def test_successful_persistence_writes_full_frame_then_only_new_rows(
    atom_runtime,
    monkeypatch,
):
    active = ibi.Future(
        conId=1,
        symbol="ES",
        exchange="CME",
        localSymbol="ESM6",
    )
    install_selector(monkeypatch, atom_runtime, active=active)
    sink = FakeAuditSink()
    output = []
    subject = model(persistence=queued_persistence(sink))
    subject.dataEvent += output.append
    subject.onData(frame())
    subject.onData(frame())
    extended = pd.concat(
        [
            frame(),
            pd.DataFrame(
                {"close": [12]},
                index=[datetime(2026, 1, 3, tzinfo=timezone.utc)],
            ),
        ]
    )

    subject.onData(extended)
    result = output[-1]

    assert [call[0] for call in sink.calls] == ["write", "append"]
    assert len(sink.calls[0][2]) == 2
    assert list(sink.calls[1][2].index) == [datetime(2026, 1, 3, tzinfo=timezone.utc)]
    assert result.metadata["audit_symbol"] == sink.calls[0][1]
    assert sink.calls[0][3]["source_key"] == "alpha"
    assert sink.calls[0][3]["active_contract"] == misc.tree(active)


def test_active_contract_change_starts_new_complete_generation(
    atom_runtime,
    monkeypatch,
):
    """ACTIVE rotation should not append adjusted history to the prior run."""

    first = ibi.Future(
        conId=1,
        symbol="ES",
        exchange="CME",
        localSymbol="ESM6",
    )
    second = ibi.Future(
        conId=2,
        symbol="ES",
        exchange="CME",
        localSymbol="ESU6",
    )
    selector = SimpleNamespace(active_contract=first, next_contract=first)
    monkeypatch.setattr(
        atom_runtime.contract_registry,
        "get_selector",
        lambda blueprint: selector,
    )
    sink = FakeAuditSink()
    subject = model(persistence=queued_persistence(sink))

    subject.onData(frame())
    selector.active_contract = second
    selector.next_contract = second
    subject.onData(frame())

    assert [call[0] for call in sink.calls] == ["write", "write"]
    assert [len(call[2]) for call in sink.calls] == [2, 2]
    assert "ESM6" in sink.calls[0][1]
    assert "ESU6" in sink.calls[1][1]


def test_signal_uses_selected_contract_but_saved_data_uses_active(
    atom_runtime,
    monkeypatch,
):
    active = ibi.Future(
        conId=1,
        symbol="ES",
        exchange="CME",
        localSymbol="ESM6",
    )
    next_contract = ibi.Future(
        conId=2,
        symbol="ES",
        exchange="CME",
        localSymbol="ESU6",
    )
    install_selector(
        monkeypatch,
        atom_runtime,
        active=active,
        next_contract=next_contract,
    )

    class NextModel(Model):
        which_contract = ActiveNext.NEXT

    sink = FakeAuditSink()
    subject = NextModel(
        "alpha",
        ibi.Future(symbol="ES", exchange="CME"),
        SignalType.STATE,
        persistence=queued_persistence(sink),
    )
    output = []
    subject.dataEvent += output.append

    subject.onData(frame())
    result = output[0]

    assert result.contract == next_contract
    assert "ESM6" in sink.calls[0][1]
    assert sink.calls[0][3]["active_contract"] == misc.tree(active)


def test_create_signal_has_no_persistence_side_effect(atom_runtime):
    """Direct calculation must not unexpectedly write audit data."""

    persistence = RecordingPersistence()
    subject = model(persistence=persistence)

    result = subject.create_signal(frame())

    assert persistence.calls == []
    assert "audit_symbol" not in result.metadata


def test_custom_persistence_is_queued_before_signal_emission(
    atom_runtime,
    monkeypatch,
):
    """Queue acceptance should provide metadata before downstream callbacks."""

    active = ibi.Future(
        conId=1,
        symbol="ES",
        exchange="CME",
        localSymbol="ESM6",
    )
    install_selector(monkeypatch, atom_runtime, active=active)
    order = []

    class OrderedPersistence(RecordingPersistence):
        def save(self, *args, **kwargs):
            """Record ordering before delegating the persistence request."""

            order.append("persistence")
            return super().save(*args, **kwargs)

    subject = model(persistence=OrderedPersistence())
    output = []

    def receive(signal):
        """Record downstream Signal delivery order."""

        order.append("signal")
        output.append(signal)

    subject.dataEvent += receive

    subject.onData(frame())

    assert order == ["persistence", "signal"]
    assert output[0].metadata["audit_symbol"] == "calculation-reference"


def test_true_persistence_resolves_runtime_default_once(
    atom_runtime,
    monkeypatch,
):
    """The runtime factory should create one policy for each model."""

    active = ibi.Future(
        conId=1,
        symbol="ES",
        exchange="CME",
        localSymbol="ESM6",
    )
    install_selector(monkeypatch, atom_runtime, active=active)
    persistence = RecordingPersistence()
    calls = []

    def factory():
        """Return the model-specific test policy."""

        calls.append("factory")
        return persistence

    atom_runtime.signal_persistence_factory = factory
    subject = model(persistence=True)
    subject.onData(frame())
    subject.onData(frame())

    assert subject.persistence is persistence
    assert calls == ["factory"]
    assert len(persistence.calls) == 2


def test_false_persistence_does_not_consult_runtime_default(atom_runtime):
    """Disabled persistence should require no runtime storage configuration."""

    calls = []

    def factory():
        """Record an unexpected request for default persistence."""

        calls.append("factory")
        return RecordingPersistence()

    atom_runtime.signal_persistence_factory = factory
    subject = model(persistence=False)
    output = []
    subject.dataEvent += output.append

    subject.onData(frame())

    assert calls == []
    assert len(output) == 1
    assert "audit_symbol" not in output[0].metadata


@pytest.mark.parametrize("persistence", [None, 1, "enabled"])
def test_pandas_model_rejects_invalid_persistence(atom_runtime, persistence):
    """Only explicit booleans and structural persistence objects are valid."""

    with pytest.raises(TypeError, match="persistence must be"):
        model(persistence=persistence)


def test_persistence_failure_does_not_suppress_signal(
    atom_runtime,
    monkeypatch,
    caplog,
):
    """An optional audit failure must not interrupt the trading pipeline."""

    active = ibi.Future(
        conId=1,
        symbol="ES",
        exchange="CME",
        localSymbol="ESM6",
    )
    install_selector(monkeypatch, atom_runtime, active=active)

    class FailingPersistence(RecordingPersistence):
        def save(self, *args, **kwargs):
            """Simulate a queue that can no longer accept work."""

            raise RuntimeError("queue is closed")

    subject = model(persistence=FailingPersistence())
    output = []
    subject.dataEvent += output.append

    subject.onData(frame())

    assert len(output) == 1
    assert "audit_symbol" not in output[0].metadata
    assert "could not queue calculated dataframe persistence" in caplog.text


def test_failed_calculation_creates_no_persistence_generation(atom_runtime):
    class Failing(Model):
        def df(self, data):
            raise RuntimeError("calculation failed")

    sink = FakeAuditSink()
    subject = Failing(
        "alpha",
        ibi.Future(conId=1, symbol="ES", exchange="CME"),
        SignalType.STATE,
        persistence=queued_persistence(sink),
    )

    with pytest.raises(RuntimeError):
        subject.onData(frame())

    assert sink.calls == []


def test_invalid_calculated_row_creates_no_persistence_generation(atom_runtime):
    class MissingSignal(Model):
        def df(self, data):
            return data

    sink = FakeAuditSink()
    subject = MissingSignal(
        "alpha",
        ibi.Future(conId=1, symbol="ES", exchange="CME"),
        SignalType.STATE,
        persistence=queued_persistence(sink),
    )

    with pytest.raises(KeyError, match="signal"):
        subject.onData(frame())

    assert sink.calls == []


def test_queued_persistence_requires_drain_policy():
    """The standard audit implementation must preserve accepted shutdown work."""

    sink = FakeAuditSink()
    sink.shutdown_policy = QueueShutdownPolicy.DISCARD

    with pytest.raises(ValueError, match="DRAIN"):
        QueuedSignalFramePersistence(sink)
