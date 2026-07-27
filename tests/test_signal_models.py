from datetime import datetime, timezone

import ib_insync as ibi
import pandas as pd
import pytest

from haymaker.async_wrappers import QueueShutdownPolicy
from haymaker.components import (
    PandasSignalModel,
    Signal,
    SignalPair,
    SignalType,
    read_signal_audit,
)


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


def test_successful_audit_writes_full_frame_then_only_new_rows(atom_runtime):
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


def test_signal_model_registers_source_roll_policy(atom_runtime):
    model(auto_roll_futures=False)

    assert atom_runtime.future_roll_policies["alpha"] is False


def test_conflicting_source_roll_policy_raises(atom_runtime):
    model(auto_roll_futures=True)

    with pytest.raises(ValueError, match="Conflicting"):
        model(auto_roll_futures=False)


@pytest.mark.asyncio
async def test_audit_lookup_uses_metadata_identity_and_normalizes_legacy_utc():
    data = frame()

    class Store:
        async def keys(self):
            return ["alpha_ES_older", "alpha_ES_wrong", "beta_ES_newer"]

        async def read_metadata(self, symbol):
            return {
                "alpha_ES_older": {
                    "source_key": "alpha",
                    "run_started_at": "2026-01-01T00:00:00",
                },
                "alpha_ES_wrong": {
                    "source_key": "other",
                    "run_started_at": "2026-01-02T00:00:00+00:00",
                },
                "beta_ES_newer": {
                    "source_key": "beta",
                    "run_started_at": "2026-01-03T00:00:00+00:00",
                },
            }[symbol]

        async def read(self, symbol, start_date=None, end_date=None):
            assert symbol == "alpha_ES_older"
            assert end_date == datetime(2026, 1, 2, tzinfo=timezone.utc)
            return data

    result = await read_signal_audit(
        Store(),
        source_key="alpha",
        created_at=datetime(2026, 1, 4, tzinfo=timezone.utc),
        as_of=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )

    assert result is data
