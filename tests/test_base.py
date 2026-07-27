import logging

import ib_insync as ibi
import pytest

from haymaker.base import Atom, ContractRollData, MissingContractError, Pipe
from haymaker.enums import ActiveNext


class Transform(Atom):
    def __init__(self, name):
        self.name = name
        super().__init__()

    def onData(self, data, *args):
        self.dataEvent.emit((data, self.name))


def test_atom_has_explicit_events():
    atom = Atom()

    assert atom.startEvent.name() == "startEvent"
    assert atom.dataEvent.name() == "dataEvent"
    assert atom.feedbackEvent.name() == "feedbackEvent"


def test_base_on_data_requires_explicit_implementation():
    with pytest.raises(NotImplementedError):
        Atom().onData(object())


def test_base_on_start_preserves_arbitrary_mutable_data():
    atom = Atom()
    received = []
    atom.startEvent += lambda data, source: received.append((data, source))
    payload = []

    atom.onStart(payload)

    assert received == [(payload, atom)]


def test_base_on_feedback_passes_arbitrary_payload():
    atom = Atom()
    received = []
    atom.feedbackEvent += received.append
    payload = object()

    atom.onFeedback(payload)

    assert received == [payload]


def test_atom_has_book_not_sm(atom_runtime):
    atom = Atom()

    assert atom.book is atom_runtime.book
    assert not hasattr(atom, "sm")


def test_atom_has_no_automatic_strategy_or_startup():
    atom = Atom()

    assert not hasattr(atom, "strategy")
    assert not hasattr(atom, "startup")
    assert not hasattr(atom, "strategy_data")
    assert not hasattr(atom, "data")


def test_connect_wires_start_data_and_reverse_feedback():
    source = Transform("source")
    target = Transform("target")
    starts = []
    data = []
    feedback = []
    target.startEvent += lambda payload, sender: starts.append((payload, sender))
    target.dataEvent += data.append
    source.feedbackEvent += feedback.append
    source.connect(target)

    source.onStart({})
    source.onData("input")
    target.onFeedback("result")

    assert starts[0][0] == {}
    assert data == [(("input", "source"), "target")]
    assert feedback == ["result"]


def test_connect_validates_all_targets_before_modifying_connections():
    source = Transform("source")
    accepted = Transform("accepted")

    class Reject(Transform):
        def validate_source(self, source):
            raise TypeError("incompatible")

    rejected = Reject("rejected")

    with pytest.raises(TypeError, match="incompatible"):
        source.connect(accepted, rejected)

    assert len(source.startEvent) == 0
    assert len(source.dataEvent) == 0


def test_fan_out_shares_one_object_reference():
    source = Atom()
    seen = []

    class Sink(Atom):
        def onData(self, data, *args):
            seen.append(data)

    source.connect(Sink(), Sink())
    payload = {"value": 1}

    source.dataEvent.emit(payload)

    assert seen[0] is payload
    assert seen[1] is payload


def test_duplicate_connection_is_replaced_not_multiplied():
    source = Atom()
    target = Transform("target")
    source.connect(target)
    source.connect(target)
    source.connect(target)

    assert len(source.startEvent) == 1
    assert len(source.dataEvent) == 1


def test_disconnect_removes_reverse_feedback():
    source = Atom()
    target = Transform("target")
    source.connect(target)

    source.disconnect(target)

    assert len(source.startEvent) == 0
    assert len(source.dataEvent) == 0
    assert len(target.feedbackEvent) == 0


def test_pipe_connects_members_in_order():
    first = Transform("first")
    second = Transform("second")
    third = Transform("third")
    output = []
    pipe = Pipe(first, second, third)
    pipe.dataEvent += output.append

    pipe.onData("input")

    assert output == [((("input", "first"), "second"), "third")]
    assert pipe.first is first
    assert pipe.last is third
    assert len(pipe) == 3


def test_pipe_requires_atom_members():
    with pytest.raises(ValueError, match="at least one"):
        Pipe()
    with pytest.raises(TypeError, match="Atom"):
        Pipe(object())


def test_pipe_delegates_source_validation_to_first_member():
    class SignalOnly(Transform):
        def validate_source(self, source):
            if getattr(source, "output_type", None) != "signal":
                raise TypeError("signal required")

    source = Transform("source")
    pipe = Pipe(SignalOnly("first"), Transform("last"))

    with pytest.raises(TypeError, match="signal required"):
        source.connect(pipe)

    assert len(source.dataEvent) == 0


def test_pipe_connects_downstream_from_last_member():
    pipe = Pipe(Transform("first"), Transform("second"))
    sink = Transform("sink")
    output = []
    sink.dataEvent += output.append

    pipe.connect(sink)
    pipe.onData("input")

    assert output == [((("input", "first"), "second"), "sink")]


def test_contract_descriptor_registers_blueprint(atom_runtime):
    atom = Atom()
    future = ibi.Future("ES", exchange="CME")

    atom.contract = future

    assert future in atom_runtime.contract_registry.blueprints
    assert atom.contract == future


def test_contract_descriptor_rejects_wrong_type(atom_runtime):
    with pytest.raises(TypeError):
        Atom().contract = "ES"


def test_missing_qualified_contract_raises_domain_error(
    atom_runtime, monkeypatch
):
    atom = Atom()
    future = ibi.Future("ES", exchange="CME")
    atom.contract = future
    monkeypatch.setattr(
        atom_runtime.contract_registry,
        "get_contract",
        lambda *args: (_ for _ in ()).throw(KeyError("missing")),
    )

    with pytest.raises(MissingContractError):
        _ = atom.contract


def test_contract_change_emits_and_records_roll(atom_runtime):
    atom = Atom()
    old = ibi.Future(conId=1, symbol="ES", exchange="CME")
    new = ibi.Future(conId=2, symbol="ES", exchange="CME")
    atom._contract_memo = old
    atom_runtime.contract_registry.get_contract = lambda *args: new
    atom._contract_blueprint = old

    atom._process_contract_change()

    assert atom._roll_contract_data == ContractRollData(old, new)


def test_request_restart_delegates_to_runtime(atom_runtime):
    assert Atom().request_restart("stale")
    assert atom_runtime.restart_requests == ["stale"]


def test_request_restart_absent_without_runtime(monkeypatch):
    monkeypatch.delattr(Atom, "runtime", raising=False)

    assert Atom().request_restart is None


def test_repr_excludes_default_active_role():
    atom = Transform("alpha")

    assert repr(atom) == "Transform(name=alpha)"
    atom.which_contract = ActiveNext.NEXT
    assert "which_contract=NEXT" in repr(atom)


def test_event_callback_failures_are_logged(caplog):
    source = Atom()

    class Broken(Atom):
        def onData(self, data, *args):
            raise RuntimeError("broken callback")

    source.connect(Broken())

    with caplog.at_level(logging.ERROR):
        source.dataEvent.emit(object())

    assert "broken callback" in caplog.text
