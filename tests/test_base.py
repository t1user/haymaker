import logging

import ib_insync as ibi
import pytest

from haymaker.base import Atom, Details, DetailsContainer, Pipe
from haymaker.state_machine import Strategy


class NewAtom(Atom):
    onStart_string = ""
    onData_string = ""
    onFeedback_string = ""
    onData_checksum = 0
    onStart_checksum = 0
    onFeedback_checksum = 0

    def __init__(self, name):
        self.name = name
        super().__init__()

    def onStart(self, data, *args):
        self.onStart_string += data
        self.onStart_checksum += 1
        self.startEvent.emit(f"{data}_{self.name}")
        return data

    def onData(self, data, *args):
        self.onData_string += data
        self.onData_checksum += 1
        self.dataEvent.emit(f"{data}_{self.name}")
        return data

    def onFeedback(self, data, *args):
        self.onFeedback_string += data
        self.onFeedback_checksum += 1
        self.feedbackEvent.emit(f"{data}_{self.name}")
        return data


class TestAtom:
    @pytest.fixture
    def atom1(self):
        return NewAtom("atom1")

    @pytest.fixture
    def atom2(self):
        return NewAtom("atom2")

    def test_events_exist(self, atom1):
        assert hasattr(atom1, "startEvent")
        assert hasattr(atom1, "dataEvent")

    def test_connect(self, atom1, atom2):
        atom3 = NewAtom("atom3")
        atom3.connect(atom1, atom2)
        assert len(atom3.startEvent) == 2
        assert len(atom3.dataEvent) == 2

    def test_connect_startEvent(self, atom1, atom2):
        atom1.connect(atom2)
        atom1.startEvent.emit("start_test_string")
        assert atom2.onStart_string == "start_test_string"

    def test_connect_dataEvent(self, atom1, atom2):
        atom1.connect(atom2)
        atom1.dataEvent.emit("data_test_string")
        assert atom2.onData_string == "data_test_string"

    def test_connect_feedbackEvent(self, atom1, atom2):
        atom1.connect(atom2)
        atom2.feedbackEvent.emit("data_test_string")
        assert atom1.onFeedback_string == "data_test_string"

    def test_disconnect(self, atom1, atom2):
        atom1.connect(atom2)
        atom1.disconnect(atom2)
        assert len(atom1.startEvent) == 0
        assert len(atom1.dataEvent) == 0
        assert len(atom2.feedbackEvent) == 0

    def test_disconnect_1(self, atom1, atom2):
        atom1.connect(atom2)
        atom1.disconnect(atom2)
        atom1.startEvent.emit("test_data_string")
        assert "test_data_string" not in atom2.onStart_string

    def test_disconnect_2(self, atom1, atom2):
        atom1.connect(atom2)
        atom1.disconnect(atom2)
        atom2.feedbackEvent.emit("test_data_string")
        assert "test_data_string" not in atom1.onFeedback_string

    def test_clear(self, atom1, atom2):
        atom3 = NewAtom("atom3")
        atom3.connect(atom1, atom2)
        atom3.clear()
        assert len(atom3.startEvent) == 0
        assert len(atom3.dataEvent) == 0
        assert len(atom1.feedbackEvent) == 0

    def test_iadd(self, atom1, atom2):
        atom1 += atom2
        atom1.startEvent.emit("test_string")
        atom1.dataEvent.emit("test_string")
        atom2.feedbackEvent.emit("new_test_string")
        assert atom2.onStart_string == "test_string"
        assert atom2.onData_string == "test_string"
        assert atom1.onFeedback_string == "new_test_string"

    def test_isub(self, atom1, atom2):
        atom1.connect(atom2)
        atom1 -= atom2
        assert len(atom1.startEvent) == 0
        assert len(atom1.dataEvent) == 0
        assert len(atom2.feedbackEvent) == 0

    def test_union(self, atom1, atom2):
        atom3 = NewAtom("atom3")
        atom3.union(atom1, atom2)
        atom3.startEvent.emit("test_string")
        assert atom1.onStart_string == "test_string"
        assert atom2.onStart_string == "test_string"

    def test_union_1(self, atom1, atom2):
        atom3 = NewAtom("atom3")
        atom3.union(atom1, atom2)
        atom2.feedbackEvent.emit("test_string")
        assert atom3.onFeedback_string == "test_string"

    def test_unequality(self, atom1):
        ato = NewAtom("atom1")
        assert ato != atom1

    def test_repr(self, atom1):
        assert repr(atom1) == "NewAtom(name=atom1)"

    def test_no_duplicate_connections(self, atom1, atom2):
        atom1.connect(atom2)
        atom1.connect(atom2)
        atom1.connect(atom2)
        atom1.startEvent.emit("test_string")
        atom1.dataEvent.emit("test_string")
        atom2.feedbackEvent.emit("bla")
        assert len(atom1.startEvent) == 1
        assert len(atom1.dataEvent) == 1
        assert len(atom2.feedbackEvent) == 1

    def test_no_duplicate_connections_1(self, atom1, atom2):
        atom1.connect(atom2)
        atom1.connect(atom2)
        atom1.connect(atom2)
        atom1.connect(atom2)
        atom1.startEvent.emit("test_string")
        atom1.dataEvent.emit("test_string")
        atom2.feedbackEvent.emit("bla")
        assert atom2.onStart_checksum == 1
        assert atom2.onData_checksum == 1
        assert atom1.onFeedback_checksum == 1


class TestPipe:
    @pytest.fixture
    def atoms(self):
        return NewAtom("x"), NewAtom("y"), NewAtom("z")

    @pytest.fixture
    def pipe_(self, atoms):
        x, y, z = atoms
        pipe = x.pipe(y, z)
        return pipe

    @pytest.fixture
    def pass_through_pipe(self, atoms, pipe_):
        x, y, z = atoms
        pipe = pipe_
        start = NewAtom("start")
        end = NewAtom("end")
        start.connect(pipe)
        pipe.connect(end)
        start.startEvent.emit("StartEvent")
        start.dataEvent.emit("DataEvent")
        end.feedbackEvent.emit("FeedbackEvent")
        return start, end, pipe

    def test_pipe_type(self, pipe_):
        assert isinstance(pipe_, Pipe)

    def test_pipe_members(self, atoms, pipe_):
        x, y, z = atoms
        pipe = pipe_
        assert pipe[0] == x
        assert pipe[1] == y
        assert pipe[2] == z

    def test_pipe_lenght(self, pipe_):
        assert len(pipe_) == 3

    def test_inside_atoms_start_event(self, pass_through_pipe):
        _, _, pipe = pass_through_pipe
        assert pipe[0].onStart_string == "StartEvent"
        assert pipe[1].onStart_string == "StartEvent_x"
        assert pipe[2].onStart_string == "StartEvent_x_y"

    def test_inside_atoms_data_event(self, pass_through_pipe):
        _, _, pipe = pass_through_pipe
        assert pipe[0].onData_string == "DataEvent"
        assert pipe[1].onData_string == "DataEvent_x"
        assert pipe[2].onData_string == "DataEvent_x_y"

    def test_inside_atoms_feedback_event(self, pass_through_pipe):
        _, _, pipe = pass_through_pipe
        assert pipe[2].onFeedback_string == "FeedbackEvent"
        assert pipe[1].onFeedback_string == "FeedbackEvent_z"
        assert pipe[0].onFeedback_string == "FeedbackEvent_z_y"

    def test_pass_through_startEvent(self, pass_through_pipe):
        start, end, pipe = pass_through_pipe
        assert end.onStart_string == "StartEvent_x_y_z"

    def test_pass_through_dataEvent(self, pass_through_pipe):
        start, end, pipe = pass_through_pipe
        assert end.onData_string == "DataEvent_x_y_z"

    def test_pass_through_feedbackEvent(self, pass_through_pipe):
        start, end, pipe = pass_through_pipe
        assert start.onFeedback_string == "FeedbackEvent_z_y_x"

    def test_connect_multiple_objects(self, atoms, pipe_):
        start = NewAtom("start")
        end1 = NewAtom("end1")
        end2 = NewAtom("end2")
        start.connect(pipe_)
        pipe_.connect(end1, end2)
        start.dataEvent.emit("test_string")
        assert end1.onData_string == "test_string_x_y_z"
        assert end1.onData_string == "test_string_x_y_z"

    def test_connect_multiple_objects_1(self, atoms, pipe_):
        start = NewAtom("start")
        end1 = NewAtom("end1")
        end2 = NewAtom("end2")
        start.connect(pipe_)
        pipe_.connect(end1, end2)
        start.dataEvent.emit("test_string")
        assert end1.onData_checksum == 1
        assert end1.onData_checksum == 1

    def test_connect_multiple_objects_feedback(self, atoms, pipe_):
        start = NewAtom("start")
        end1 = NewAtom("end1")
        end2 = NewAtom("end2")
        start.connect(pipe_)
        pipe_.connect(end1, end2)
        end1.feedbackEvent.emit("test_string")
        assert start.onFeedback_string == "test_string_z_y_x"

    def test_connect_multiple_objects_feedback_1(self, atoms, pipe_):
        start = NewAtom("start")
        end1 = NewAtom("end1")
        end2 = NewAtom("end2")
        start.connect(pipe_)
        pipe_.connect(end1, end2)
        end1.feedbackEvent.emit("test_string")
        end2.feedbackEvent.emit("bla")
        assert start.onFeedback_string == "test_string_z_y_xbla_z_y_x"

    def test_connect_multiple_objects_feedback_2(self, atoms, pipe_):
        start = NewAtom("start")
        end1 = NewAtom("end1")
        end2 = NewAtom("end2")
        start.connect(pipe_)
        pipe_.connect(end1, end2)
        end1.feedbackEvent.emit("test_string")
        end2.feedbackEvent.emit("bla")
        assert start.onFeedback_checksum == 2

    def test_disconnect(self, atoms, pipe_):
        start = NewAtom("start")
        end1 = NewAtom("end1")
        end2 = NewAtom("end2")
        end3 = NewAtom("end3")
        start.connect(pipe_)
        pipe_.connect(end1, end2, end3)
        pipe_.disconnect(end1, end2)
        start.dataEvent.emit("test_string")
        # this one is still connected
        assert end3.onData_string == "test_string_x_y_z"
        # those ones should be disconnected
        assert end1.onData_string == ""
        assert end2.onData_string == ""

    def test_disconnect_feedback(self, atoms, pipe_):
        start = NewAtom("start")
        end1 = NewAtom("end1")
        end2 = NewAtom("end2")
        end3 = NewAtom("end3")
        start.connect(pipe_)
        pipe_.connect(end1, end2, end3)
        pipe_.disconnect(end1, end2)
        end3.feedbackEvent.emit("test_string")
        end1.feedbackEvent.emit("bla")
        end2.feedbackEvent.emit("bla")
        # end1 is still connected, but end2 and end3 aren't
        # if they were 'bla' would be captured
        assert start.onFeedback_string == "test_string_z_y_x"

    def test_pass_through_startEvent_checksum(self, pass_through_pipe):
        """
        Each object in the chain should have been acted upon, except for the first.
        """
        start, end, pipe = pass_through_pipe
        assert start.onStart_checksum == 0
        assert end.onStart_checksum == 1
        assert pipe[0].onStart_checksum == 1
        assert pipe[1].onStart_checksum == 1
        assert pipe[2].onStart_checksum == 1

    def test_pass_through_onDataEvent_checksum(self, pass_through_pipe):
        """
        Each object in the chain should have been acted upon, except for the first.
        """
        start, end, pipe = pass_through_pipe
        assert start.onData_checksum == 0
        assert end.onData_checksum == 1
        assert pipe[0].onData_checksum == 1
        assert pipe[1].onData_checksum == 1
        assert pipe[2].onData_checksum == 1

    def test_pass_through_onFeedback_checksum(self, pass_through_pipe):
        """
        Each object in the chain should have been acted upon, except for the first.
        """
        start, end, pipe = pass_through_pipe
        assert end.onFeedback_checksum == 0
        assert start.onFeedback_checksum == 1
        assert pipe[0].onFeedback_checksum == 1
        assert pipe[1].onFeedback_checksum == 1
        assert pipe[2].onFeedback_checksum == 1

    def test_one_atom_in_multiple_pipes(self):
        a = NewAtom("a")
        b = NewAtom("b")
        c = NewAtom("c")
        d = NewAtom("d")
        e = NewAtom("e")
        Pipe(a, b, c)
        Pipe(a, d, e)
        a.dataEvent.emit("test_message")
        a.startEvent.emit("another_test_message")
        assert c.onData_string == "test_message_b"
        assert e.onData_string == "test_message_d"
        assert c.onStart_string == "another_test_message_b"
        assert e.onStart_string == "another_test_message_d"

    def test_one_atom_in_multiple_pipes_feedback(self):
        a = NewAtom("a")
        b = NewAtom("b")
        c = NewAtom("c")
        d = NewAtom("d")
        e = NewAtom("e")
        Pipe(c, b, a)
        Pipe(e, d, a)
        a.feedbackEvent.emit("test_message")
        assert c.onFeedback_string == "test_message_b"
        assert e.onFeedback_string == "test_message_d"


class TestUnionPipe:
    @pytest.fixture
    def atoms(self):
        return NewAtom("start"), NewAtom("end")

    @pytest.fixture
    def pipe1(self):
        x, y, z = NewAtom("x"), NewAtom("y"), NewAtom("z")
        pipe = Pipe(x, y, z)
        return pipe

    @pytest.fixture
    def pipe2(self):
        a, b, c = NewAtom("a"), NewAtom("b"), NewAtom("c")
        pipe = Pipe(a, b, c)
        return pipe

    def test_union_pipe_onStart(self, pipe1, pipe2, atoms):
        start, end = atoms
        p1 = pipe1
        p2 = pipe2
        start.union(p1, p2)
        start.startEvent.emit("test_string")
        assert p1[-1].onStart_string == "test_string_x_y"
        assert p2[-1].onStart_string == "test_string_a_b"

    def test_union_pipe_onData(self, pipe1, pipe2, atoms):
        start, end = atoms
        p1 = pipe1
        p2 = pipe2
        start.union(p1, p2)
        start.dataEvent.emit("test_string")
        assert p1[-1].onData_string == "test_string_x_y"
        assert p2[-1].onData_string == "test_string_a_b"

    def test_union_pipe_onFeedback_1(self, pipe1, pipe2, atoms):
        start, end = atoms
        p1 = pipe1
        p2 = pipe2
        start.union(p1, p2)
        p1.feedbackEvent.emit("test_string")
        assert start.onFeedback_string == "test_string"

    def test_union_pipe_onFeedback_2(self, pipe1, pipe2, atoms):
        start, end = atoms
        p1 = pipe1
        p2 = pipe2
        start.union(p1, p2)
        p2.feedbackEvent.emit("test_string")
        assert start.onFeedback_string == "test_string"

    def test_union_pipe_output(self, pipe1, pipe2, atoms):
        start, end = atoms
        p1 = pipe1
        p2 = pipe2
        start.union(p1, p2)
        p1.connect(end)
        start.startEvent.emit("test_string")
        assert end.onStart_string == "test_string_x_y_z"

    def test_union_pipe_output_feedback(self, pipe1, pipe2, atoms):
        start, end = atoms
        p1 = pipe1
        p2 = pipe2
        start.union(p1, p2)
        p1.connect(end)
        end.feedbackEvent.emit("test_string")
        assert start.onFeedback_string == "test_string_z_y_x"


class AtomWithContract(Atom):
    def __init__(self, contract):
        self.contract = contract


class TestContract:
    @pytest.fixture
    def contract(self):
        return ibi.Contract("ES", "CME")

    @pytest.fixture
    def atom(self):
        return AtomWithContract(ibi.Future("NQ", "CME"))

    def test_can_assign_and_get_contract(self, atom):
        assert isinstance(atom.contract, ibi.Future)

    def test_same_contract_returned_as_assigned(self, contract):
        c = contract
        a = AtomWithContract(c)
        assert a.contract is c

    def test_can_assign_and_retrieve_list_of_contracts(self):
        c1 = ibi.Contract("ES", "CME")
        c2 = ibi.Contract("MES", "CME")
        atom = AtomWithContract([c1, c2])
        assert isinstance(atom.contract, list)
        assert atom.contract == [c1, c2]

    def test_can_assign_and_retrieve_same_list_of_contracts(self):
        c1 = ibi.Contract("ES", "CME")
        c2 = ibi.Contract("MES", "CME")
        list_of_contracts = [c1, c2]
        atom = AtomWithContract(list_of_contracts)
        assert atom.contract == list_of_contracts


class TestContractList:
    @pytest.fixture
    def contract(self):
        return ibi.Contract("YM")

    @pytest.fixture
    def list_of_contracts(self, contract):
        return [contract, ibi.Contract("NQ")]

    @pytest.fixture
    def atom_with_contract(self, contract):
        class NewAtomWithContract(Atom):
            def __init__(self, contract):
                self.contract = contract

        a = NewAtomWithContract(contract)
        yield a
        a.contracts.clear()

    @pytest.fixture
    def atom_with_list_of_contracts(self, list_of_contracts):
        class AtomWithListOfContracts(Atom):
            def __init__(self, contract):
                self.contract = contract

        a = AtomWithListOfContracts(list_of_contracts)
        yield a
        a.contracts.clear()

    def test_newly_added_contract_in_the_list(self):
        class NewNewAtom(Atom):
            def __init__(self, x):
                self.contract = x

        NewNewAtom("a")
        assert "a" in Atom.contracts

    def test_newly_added_contract_in_Atom_list(self, atom_with_contract):
        atom_with_contract.contract = "b"
        assert "b" in atom_with_contract.contracts

    def test_contract_list_on_instance_contains_contracts(
        self, atom_with_contract, contract
    ):
        a = atom_with_contract
        assert a.contracts == [contract]

    def test_contract_list_on_class_contains_contracts(
        self, atom_with_contract, contract
    ):
        assert Atom.contracts == [contract]

    def test_ContractList_new_instance_contains_contracts(
        self, atom_with_contract, contract
    ):
        atom_with_contract
        alt_list = Atom.contracts
        assert alt_list == [contract]

    def test_contract_list_works_with_lists(
        self, atom_with_list_of_contracts, list_of_contracts
    ):
        a = atom_with_list_of_contracts
        assert a.contracts == list_of_contracts


class Test_keep_adding_contracts:
    def atom(self, contract):
        class A(Atom):
            def __init__(self, contract):
                self.contract = contract

        return A(contract)

    def test_single(self):
        self.atom("a")
        assert "a" in Atom.contracts

    def test_list(self):
        self.atom(["x", "y", "z"])
        assert "x" in Atom.contracts
        assert "y" in Atom.contracts
        assert "z" in Atom.contracts
        assert "z" in Atom.contracts


def test_onData_sets_attribute_if_dict_passed():
    class Source(Atom):
        def run(self):
            self.startEvent.emit({"strategy": "xxx"})

    class Output(Atom):
        pass

    source = Source()
    out = Output()
    source += out
    source.run()

    assert out.strategy == "xxx"


def test_onData_emits_self_if_dict_passed():
    class Source(Atom):
        def run(self):
            self.startEvent.emit({"strategy": "xxx"})

    class NewAtom(Atom):
        pass

    class Output(Atom):
        data = None
        args = None

        def onStart(self, data, *args):
            self.data = data
            self.args = args

    source = Source()
    inner = NewAtom()
    out = Output()
    Pipe(source, inner, out)
    source.run()
    assert isinstance(out.args[0], NewAtom)


def test_onData_ignores_attribute_if_no_dict_passed():
    class Source(Atom):
        def run(self):
            self.startEvent.emit(("strategy", "xxx"))

    class Output(Atom):
        pass

    source = Source()
    out = Output()
    source += out
    source.run()

    with pytest.raises(AttributeError):
        out.strategy


def test_onData_sets_attribute_downstream_if_dict_passed():
    class Source(Atom):
        def run(self):
            self.startEvent.emit({"strategy": "xxx"})

    class NewAtom(Atom):
        pass

    source = Source()
    inner1 = NewAtom()
    inner2 = NewAtom()
    out = NewAtom()
    Pipe(source, inner1, inner2, out)
    source.run()

    assert out.strategy == inner1.strategy == inner2.strategy == "xxx"


def test_event_error_logged(caplog):
    class CustomException(Exception):
        pass

    class ErrorRaisingAtom(NewAtom):
        def onData(*args):
            raise CustomException("CustomError")

    a = NewAtom("a")
    b = ErrorRaisingAtom("b")
    a.connect(b)
    a.dataEvent.emit("xxx")
    assert "Event error dataEvent: CustomError" in caplog.messages


def test_event_error_logged_with_correct_logger(caplog):
    class CustomException(Exception):
        pass

    class ErrorRaisingAtom(NewAtom):
        def onData(*args):
            raise CustomException("CustomError")

    a = NewAtom("a")
    b = ErrorRaisingAtom("b")
    a.connect(b)
    a.dataEvent.emit("xxx")
    assert caplog.record_tuples == [
        ("strategy.NewAtom", logging.ERROR, "Event error dataEvent: CustomError")
    ]


class TestAtomDetails:
    @pytest.fixture
    def mock_atom(self, details):
        class MockAtom(Atom):
            def __init__(self):
                self.contract = details.contract
                super().__init__()

        Atom.contract_details[details.contract] = details
        return MockAtom()

    def test_details_set_properly(self, mock_atom, details):
        assert isinstance(mock_atom.details, Details)

    def test_trading_hours_processed(self, mock_atom, details):
        assert isinstance(mock_atom.details.trading_hours, list)

    def test_trading_hours_is_open(self, mock_atom, details):
        assert isinstance(mock_atom.details.is_open(), bool)

    def test_if_no_contract_set_all_details_returned(self, mock_atom):
        class NewMockAtom(Atom):
            pass

        atom = NewMockAtom()
        assert isinstance(atom.details, DetailsContainer)
        # mock atom has been created, so there are details for one contract
        # even though I have no contract set on my `atom`
        assert len(atom.details.keys()) == 1

    def test_missing_details_log(self, caplog, mock_atom, details):
        caplog.set_level(logging.DEBUG)

        del Atom.contract_details[details.contract]

        mock_atom.details
        assert f"Missing contract details for: {details.contract}" in caplog.messages


class Test_data_property:
    # data property test depend on StateMachine being properly set as attribute of Atom
    # and StateMachine singleton being destroyed betewen tests
    # for this `atom` fixture should be used

    def test_data_property_without_strategy(self, Atom):
        class A(Atom):
            pass

        a = A()
        assert isinstance(a.data, Strategy)

    def test_data_property_with_strategy_first_access(self, Atom):
        """If we're using non-existing strategy, one should be created."""

        class A(Atom):
            def __init__(self, strategy):
                self.strategy = strategy

        a = A("xxx")
        print(a.data)

        assert a.data.strategy == "xxx"

    def test_data_property_with_strategy_access_correct_essential_keys_in_data(
        self, Atom
    ):
        """Newly created strategy must have certain keys by default."""

        class A(Atom):
            def __init__(self, strategy):
                self.strategy = strategy

        a = A("xxx")

        assert {"position", "lock", "strategy", "active_contract"}.issubset(
            set(a.data.keys())
        )

    def test_data_property_with_strategy_access_correct_position(self, Atom):
        class A(Atom):
            def __init__(self, strategy):
                self.strategy = strategy

        a = A("xxx")
        b = A("xxx")

        a.data.position += 1
        assert b.data.position == 1

    def test_data_property_multiple_strategies_access_correct_position(self, Atom):
        class A(Atom):
            pass

        a = A()
        b = A()
        c = A()
        d = A()

        a.strategy = "xxx"
        b.strategy = "xxx"
        c.strategy = "yyy"
        d.strategy = "yyy"

        a.data.position += 1
        b.data.position += 1

        assert b.data.position == 2
        assert c.data.position == 0

    def test_data_property_multiple_strategies_access_correct_position_1(self, Atom):
        class A(Atom):
            pass

        a = A()
        b = A()
        c = A()
        d = A()

        a.strategy = "xxx"
        b.strategy = "xxx"
        c.strategy = "yyy"
        d.strategy = "yyy"

        a.data.position += 1
        b.data.position += 1
        c.data.position += 1
        assert a.data.position == 2
        assert d.data.position == 1
