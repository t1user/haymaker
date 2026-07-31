from dataclasses import is_dataclass
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
    SignalPair,
    SignalModel,
    SignalType,
)
from haymaker.enums import ActiveNext


class Model(PandasSignalModel):
    def df(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        result["signal"] = result["close"].apply(lambda value: 1 if value > 10 else 0)
        result["atr"] = 2.5
        return result


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


def test_signal_model_rejects_changed_signal_identity(atom_runtime):
    class WrongSourceModel(SignalModel):
        def create_signal(self, data):
            return Signal(
                source_key="other",
                contract=self.contract,
                value=data,
                signal_type=self.signal_type,
            )

    subject = WrongSourceModel(
        "alpha",
        ibi.Future(conId=1, symbol="ES", exchange="CME"),
        SignalType.STATE,
    )

    with pytest.raises(ValueError, match=r"create_signal\(\) changed source_key"):
        subject.onData(1)


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
        ("", ValueError, "must not be empty"),
        (("entry",), ValueError, "exactly two"),
        (("entry", "exit", "other"), ValueError, "exactly two"),
        (("entry", "entry"), ValueError, "distinct"),
        (("entry", ""), ValueError, "must not be empty"),
        (("entry", 1), TypeError, "must be strings"),
        (["entry", "exit"], TypeError, "field name or a two-field tuple"),
    ],
)
def test_pandas_model_rejects_invalid_signal_fields(
    atom_runtime, signal_fields, exception, message
):
    with pytest.raises(exception, match=message):
        model(signal_fields=signal_fields)


def test_pandas_model_reports_all_missing_pair_fields_before_audit(
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
        audit_sink=sink,
    )

    with pytest.raises(KeyError, match="'in'.*'out'"):
        subject.create_signal(frame())

    assert sink.calls == []


def test_custom_row_hook_must_return_signal(atom_runtime):
    subject = model(row_to_signal=lambda row, contract: {"value": row["signal"]})

    with pytest.raises(TypeError, match="must return Signal"):
        subject.create_signal(frame())


def test_custom_row_hook_cannot_change_owned_signal_identity(atom_runtime):
    sink = FakeAuditSink()
    subject = model(
        audit_sink=sink,
        row_to_signal=lambda row, contract: Signal(
            source_key="other",
            contract=contract,
            value=row["signal"],
            signal_type=SignalType.STATE,
        ),
    )

    with pytest.raises(ValueError, match="source_key"):
        subject.create_signal(frame())

    assert sink.calls == []


def test_successful_audit_writes_full_frame_then_only_new_rows(
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
    subject = model(audit_sink=sink)
    subject.create_signal(frame())
    extended = pd.concat(
        [
            frame(),
            pd.DataFrame(
                {"close": [12]},
                index=[datetime(2026, 1, 3, tzinfo=timezone.utc)],
            ),
        ]
    )

    result = subject.create_signal(extended)

    assert [call[0] for call in sink.calls] == ["write", "append"]
    assert len(sink.calls[0][2]) == 2
    assert list(sink.calls[1][2].index) == [datetime(2026, 1, 3, tzinfo=timezone.utc)]
    assert result.metadata["audit_symbol"] == sink.calls[0][1]
    assert sink.calls[0][3]["source_key"] == "alpha"
    assert sink.calls[0][3]["active_contract"] == misc.tree(active)


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
        audit_sink=sink,
    )

    result = subject.create_signal(frame())

    assert result.contract == next_contract
    assert "ESM6" in sink.calls[0][1]
    assert sink.calls[0][3]["active_contract"] == misc.tree(active)


def test_saving_requires_initialized_contract_selector(atom_runtime):
    subject = model(audit_sink=FakeAuditSink())

    with pytest.raises(RuntimeError, match="Contract selector"):
        subject.create_signal(frame())


def test_failed_calculation_creates_no_audit_generation(atom_runtime):
    class Failing(Model):
        def df(self, data):
            raise RuntimeError("calculation failed")

    sink = FakeAuditSink()
    subject = Failing(
        "alpha",
        ibi.Future(conId=1, symbol="ES", exchange="CME"),
        SignalType.STATE,
        audit_sink=sink,
    )

    with pytest.raises(RuntimeError):
        subject.create_signal(frame())

    assert sink.calls == []


def test_invalid_calculated_row_creates_no_audit_generation(atom_runtime):
    class MissingSignal(Model):
        def df(self, data):
            return data

    sink = FakeAuditSink()
    subject = MissingSignal(
        "alpha",
        ibi.Future(conId=1, symbol="ES", exchange="CME"),
        SignalType.STATE,
        audit_sink=sink,
    )

    with pytest.raises(KeyError, match="signal"):
        subject.create_signal(frame())

    assert sink.calls == []


def test_audit_sink_requires_drain_policy(atom_runtime):
    sink = FakeAuditSink()
    sink.shutdown_policy = QueueShutdownPolicy.DISCARD

    with pytest.raises(ValueError, match="DRAIN"):
        model(audit_sink=sink)
