import logging
from unittest.mock import ANY, Mock

import ib_insync as ibi
import pytest

from haymaker.base import ActiveNext, Atom, Pipe
from haymaker.contract_registry import ContractRegistry
from haymaker.details_processor import Details
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

    def test_events_exist(self, atom1: NewAtom):
        assert hasattr(atom1, "startEvent")
        assert hasattr(atom1, "dataEvent")

    def test_connect(self, atom1: NewAtom, atom2: NewAtom):
        atom3 = NewAtom("atom3")
        atom3.connect(atom1, atom2)
        assert len(atom3.startEvent) == 2
        assert len(atom3.dataEvent) == 2

    def test_connect_startEvent(self, atom1: NewAtom, atom2: NewAtom):
        atom1.connect(atom2)
        atom1.startEvent.emit("start_test_string")
        assert atom2.onStart_string == "start_test_string"

    def test_connect_dataEvent(self, atom1: NewAtom, atom2: NewAtom):
        atom1.connect(atom2)
        atom1.dataEvent.emit("data_test_string")
        assert atom2.onData_string == "data_test_string"

    def test_connect_feedbackEvent(self, atom1: NewAtom, atom2: NewAtom):
        atom1.connect(atom2)
        atom2.feedbackEvent.emit("data_test_string")
        assert atom1.onFeedback_string == "data_test_string"

    def test_disconnect(self, atom1: NewAtom, atom2: NewAtom):
        atom1.connect(atom2)
        atom1.disconnect(atom2)
        assert len(atom1.startEvent) == 0
        assert len(atom1.dataEvent) == 0
        assert len(atom2.feedbackEvent) == 0

    def test_disconnect_1(self, atom1: NewAtom, atom2: NewAtom):
        atom1.connect(atom2)
        atom1.disconnect(atom2)
        atom1.startEvent.emit("test_data_string")
        assert "test_data_string" not in atom2.onStart_string

    def test_disconnect_2(self, atom1: NewAtom, atom2: NewAtom):
        atom1.connect(atom2)
        atom1.disconnect(atom2)
        atom2.feedbackEvent.emit("test_data_string")
        assert "test_data_string" not in atom1.onFeedback_string

    def test_clear(self, atom1: NewAtom, atom2: NewAtom):
        atom3 = NewAtom("atom3")
        atom3.connect(atom1, atom2)
        atom3.clear()
        assert len(atom3.startEvent) == 0
        assert len(atom3.dataEvent) == 0
        assert len(atom1.feedbackEvent) == 0

    def test_iadd(self, atom1: NewAtom, atom2: NewAtom):
        atom1 += atom2
        atom1.startEvent.emit("test_string")
        atom1.dataEvent.emit("test_string")
        atom2.feedbackEvent.emit("new_test_string")
        assert atom2.onStart_string == "test_string"
        assert atom2.onData_string == "test_string"
        assert atom1.onFeedback_string == "new_test_string"

    def test_isub(self, atom1: NewAtom, atom2: NewAtom):
        atom1.connect(atom2)
        atom1 -= atom2
        assert len(atom1.startEvent) == 0
        assert len(atom1.dataEvent) == 0
        assert len(atom2.feedbackEvent) == 0

    def test_union(self, atom1: NewAtom, atom2: NewAtom):
        atom3 = NewAtom("atom3")
        atom3.union(atom1, atom2)
        atom3.startEvent.emit("test_string")
        assert atom1.onStart_string == "test_string"
        assert atom2.onStart_string == "test_string"

    def test_union_1(self, atom1: NewAtom, atom2: NewAtom):
        atom3 = NewAtom("atom3")
        atom3.union(atom1, atom2)
        atom2.feedbackEvent.emit("test_string")
        assert atom3.onFeedback_string == "test_string"

    def test_unequality(self, atom1: NewAtom):
        ato = NewAtom("atom1")
        assert ato != atom1

    def test_repr(self, atom1: NewAtom):
        assert repr(atom1) == "NewAtom(name=atom1)"

    def test_repr_next_contract(self, atom1: NewAtom):
        """
        Repr should only show non-default attributes, `which_contract`
        has default value of `ActiveNext.ACTIVE`.
        """

        atom1.which_contract = ActiveNext.NEXT
        assert repr(atom1) == "NewAtom(name=atom1, which_contract=NEXT)"

    def test_no_duplicate_connections(self, atom1: NewAtom, atom2: NewAtom):
        atom1.connect(atom2)
        atom1.connect(atom2)
        atom1.connect(atom2)
        atom1.startEvent.emit("test_string")
        atom1.dataEvent.emit("test_string")
        atom2.feedbackEvent.emit("bla")
        assert len(atom1.startEvent) == 1
        assert len(atom1.dataEvent) == 1
        assert len(atom2.feedbackEvent) == 1

    def test_no_duplicate_connections_1(self, atom1: NewAtom, atom2: NewAtom):
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
    def pipe_(self, atoms: tuple[NewAtom, NewAtom, NewAtom]):
        x, y, z = atoms
        pipe = x.pipe(y, z)
        return pipe

    @pytest.fixture
    def pass_through_pipe(self, pipe_: Pipe):
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

    def test_pipe_members(self, atoms: tuple[NewAtom, NewAtom, NewAtom], pipe_: Pipe):
        x, y, z = atoms
        pipe = pipe_
        assert pipe[0] == x
        assert pipe[1] == y
        assert pipe[2] == z

    def test_pipe_lenght(self, pipe_: Pipe):
        assert len(pipe_) == 3

    def test_inside_atoms_start_event(
        self, pass_through_pipe: tuple[NewAtom, NewAtom, Pipe]
    ):
        _, _, pipe = pass_through_pipe
        assert pipe[0].onStart_string == "StartEvent"  # type: ignore
        assert pipe[1].onStart_string == "StartEvent_x"  # type: ignore
        assert pipe[2].onStart_string == "StartEvent_x_y"  # type: ignore

    def test_inside_atoms_data_event(
        self, pass_through_pipe: tuple[NewAtom, NewAtom, Pipe]
    ):
        _, _, pipe = pass_through_pipe
        assert pipe[0].onData_string == "DataEvent"  # type: ignore
        assert pipe[1].onData_string == "DataEvent_x"  # type: ignore
        assert pipe[2].onData_string == "DataEvent_x_y"  # type: ignore

    def test_inside_atoms_feedback_event(
        self, pass_through_pipe: tuple[NewAtom, NewAtom, Pipe]
    ):
        _, _, pipe = pass_through_pipe
        assert pipe[2].onFeedback_string == "FeedbackEvent"  # type: ignore
        assert pipe[1].onFeedback_string == "FeedbackEvent_z"  # type: ignore
        assert pipe[0].onFeedback_string == "FeedbackEvent_z_y"  # type: ignore

    def test_pass_through_startEvent(
        self, pass_through_pipe: tuple[NewAtom, NewAtom, Pipe]
    ):
        start, end, pipe = pass_through_pipe
        assert end.onStart_string == "StartEvent_x_y_z"

    def test_pass_through_dataEvent(
        self, pass_through_pipe: tuple[NewAtom, NewAtom, Pipe]
    ):
        start, end, pipe = pass_through_pipe
        assert end.onData_string == "DataEvent_x_y_z"

    def test_pass_through_feedbackEvent(
        self, pass_through_pipe: tuple[NewAtom, NewAtom, Pipe]
    ):
        start, end, pipe = pass_through_pipe
        assert start.onFeedback_string == "FeedbackEvent_z_y_x"

    def test_connect_multiple_objects(
        self, atoms: tuple[NewAtom, NewAtom, NewAtom], pipe_: Pipe
    ):
        start = NewAtom("start")
        end1 = NewAtom("end1")
        end2 = NewAtom("end2")
        start.connect(pipe_)
        pipe_.connect(end1, end2)
        start.dataEvent.emit("test_string")
        assert end1.onData_string == "test_string_x_y_z"
        assert end1.onData_string == "test_string_x_y_z"

    def test_connect_multiple_objects_1(
        self, atoms: tuple[NewAtom, NewAtom, NewAtom], pipe_
    ):
        start = NewAtom("start")
        end1 = NewAtom("end1")
        end2 = NewAtom("end2")
        start.connect(pipe_)
        pipe_.connect(end1, end2)
        start.dataEvent.emit("test_string")
        assert end1.onData_checksum == 1
        assert end1.onData_checksum == 1

    def test_connect_multiple_objects_feedback(
        self, atoms: tuple[NewAtom, NewAtom, NewAtom], pipe_: Pipe
    ):
        start = NewAtom("start")
        end1 = NewAtom("end1")
        end2 = NewAtom("end2")
        start.connect(pipe_)
        pipe_.connect(end1, end2)
        end1.feedbackEvent.emit("test_string")
        assert start.onFeedback_string == "test_string_z_y_x"

    def test_connect_multiple_objects_feedback_1(
        self, atoms: tuple[NewAtom, NewAtom, NewAtom], pipe_: Pipe
    ):
        start = NewAtom("start")
        end1 = NewAtom("end1")
        end2 = NewAtom("end2")
        start.connect(pipe_)
        pipe_.connect(end1, end2)
        end1.feedbackEvent.emit("test_string")
        end2.feedbackEvent.emit("bla")
        assert start.onFeedback_string == "test_string_z_y_xbla_z_y_x"

    def test_connect_multiple_objects_feedback_2(
        self, atoms: tuple[NewAtom, NewAtom, NewAtom], pipe_: Pipe
    ):
        start = NewAtom("start")
        end1 = NewAtom("end1")
        end2 = NewAtom("end2")
        start.connect(pipe_)
        pipe_.connect(end1, end2)
        end1.feedbackEvent.emit("test_string")
        end2.feedbackEvent.emit("bla")
        assert start.onFeedback_checksum == 2

    def test_disconnect(self, atoms: tuple[NewAtom, NewAtom, NewAtom], pipe_: Pipe):
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

    def test_disconnect_feedback(
        self, atoms: tuple[NewAtom, NewAtom, NewAtom], pipe_: Pipe
    ):
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

    def test_pass_through_startEvent_checksum(
        self, pass_through_pipe: tuple[NewAtom, NewAtom, Pipe]
    ):
        """
        Each object in the chain should have been acted upon, except for the first.
        """
        start, end, pipe = pass_through_pipe
        assert start.onStart_checksum == 0
        assert end.onStart_checksum == 1
        assert pipe[0].onStart_checksum == 1  # type: ignore
        assert pipe[1].onStart_checksum == 1  # type: ignore
        assert pipe[2].onStart_checksum == 1  # type: ignore

    def test_pass_through_onDataEvent_checksum(
        self, pass_through_pipe: tuple[NewAtom, NewAtom, Pipe]
    ):
        """
        Each object in the chain should have been acted upon, except for the first.
        """
        start, end, pipe = pass_through_pipe
        assert start.onData_checksum == 0
        assert end.onData_checksum == 1
        assert pipe[0].onData_checksum == 1  # type: ignore
        assert pipe[1].onData_checksum == 1  # type: ignore
        assert pipe[2].onData_checksum == 1  # type: ignore

    def test_pass_through_onFeedback_checksum(
        self, pass_through_pipe: tuple[NewAtom, NewAtom, Pipe]
    ):
        """
        Each object in the chain should have been acted upon, except for the first.
        """
        start, end, pipe = pass_through_pipe
        assert end.onFeedback_checksum == 0
        assert start.onFeedback_checksum == 1
        assert pipe[0].onFeedback_checksum == 1  # type: ignore
        assert pipe[1].onFeedback_checksum == 1  # type: ignore
        assert pipe[2].onFeedback_checksum == 1  # type: ignore

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

    def test_union_pipe_onStart(
        self, pipe1: Pipe, pipe2: Pipe, atoms: tuple[NewAtom, NewAtom]
    ):
        start, end = atoms
        p1 = pipe1
        p2 = pipe2
        start.union(p1, p2)
        start.startEvent.emit("test_string")
        assert p1[-1].onStart_string == "test_string_x_y"  # type: ignore
        assert p2[-1].onStart_string == "test_string_a_b"  # type: ignore

    def test_union_pipe_onData(
        self, pipe1: Pipe, pipe2: Pipe, atoms: tuple[NewAtom, NewAtom]
    ):
        start, end = atoms
        p1 = pipe1
        p2 = pipe2
        start.union(p1, p2)
        start.dataEvent.emit("test_string")
        assert p1[-1].onData_string == "test_string_x_y"  # type: ignore
        assert p2[-1].onData_string == "test_string_a_b"  # type: ignore

    def test_union_pipe_onFeedback_1(
        self, pipe1: Pipe, pipe2: Pipe, atoms: tuple[NewAtom, NewAtom]
    ):
        start, end = atoms
        p1 = pipe1
        p2 = pipe2
        start.union(p1, p2)
        p1.feedbackEvent.emit("test_string")
        assert start.onFeedback_string == "test_string"

    def test_union_pipe_onFeedback_2(
        self, pipe1: Pipe, pipe2: Pipe, atoms: tuple[NewAtom, NewAtom]
    ):
        start, end = atoms
        p1 = pipe1
        p2 = pipe2
        start.union(p1, p2)
        p2.feedbackEvent.emit("test_string")
        assert start.onFeedback_string == "test_string"

    def test_union_pipe_output(
        self, pipe1: Pipe, pipe2: Pipe, atoms: tuple[NewAtom, NewAtom]
    ):
        start, end = atoms
        p1 = pipe1
        p2 = pipe2
        start.union(p1, p2)
        p1.connect(end)
        start.startEvent.emit("test_string")
        assert end.onStart_string == "test_string_x_y_z"

    def test_union_pipe_output_feedback(
        self, pipe1: Pipe, pipe2: Pipe, atoms: tuple[NewAtom, NewAtom]
    ):
        start, end = atoms
        p1 = pipe1
        p2 = pipe2
        start.union(p1, p2)
        p1.connect(end)
        end.feedbackEvent.emit("test_string")
        assert start.onFeedback_string == "test_string_z_y_x"


def test_pipe_is_not_bypassed_onData():
    """
    Want this test to be standalone, independent of all fixtures, to
    make sure no mistake is being replicated.
    """

    class atom(Atom):
        output = "NOTSET"

        def __init__(self, name: str) -> None:
            self.name = name
            super().__init__()

        def onData(self, data, *args):
            data[self.name] = "xxx"
            self.output = data
            self.dataEvent.emit(data)

        def __repr__(self):
            return self.__class__.__name__ + "()"

    x = atom("x")
    y = atom("y")
    z = atom("z")

    pipe = Pipe(x, y, z)

    start = atom("start")
    end = atom("end")

    start.connect(pipe)
    pipe.connect(end)

    start.onData({})
    assert "x" in y.output


def test_pipe_is_not_bypassed_onStart():
    """
    Want this test to be standalone, independent of all fixtures, to
    make sure no mistake is being replicated.
    """

    class atom(Atom):
        output = "NOTSET"

        def __init__(self, name: str) -> None:
            self.name = name
            super().__init__()

        def onStart(self, data, *args):
            super().onStart(data)
            data[self.name] = "xxx"
            self.output = data

        def __repr__(self):
            return self.__class__.__name__ + "()"

    x = atom("x")
    y = atom("y")
    z = atom("z")

    pipe = Pipe(x, y, z)

    start = atom("start")
    end = atom("end")

    start.connect(pipe)
    pipe.connect(end)

    start.onStart({})
    assert "x" in y.output


def test_pipe_is_not_bypassed_onFeedback():
    """
    Want this test to be standalone, independent of all fixtures, to
    make sure no mistake is being replicated.
    """

    class atom(Atom):
        output = "NOTSET"

        def __init__(self, name: str) -> None:
            self.name = name
            super().__init__()

        def onFeedback(self, data, *args):
            data[self.name] = "xxx"
            self.output = data
            super().onFeedback(data)

        def __repr__(self):
            return self.__class__.__name__ + "()"

    x = atom("x")
    y = atom("y")
    z = atom("z")

    pipe = Pipe(x, y, z)

    start = atom("start")
    end = atom("end")

    start.connect(pipe)
    pipe.connect(end)

    end.onFeedback({})
    assert "z" in y.output


def test_pipe_is_not_bypassed_onFeedback_correct_direction():
    """
    Want this test to be standalone, independent of all fixtures, to
    make sure no mistake is being replicated.
    """

    class atom(Atom):
        output = "NOTSET"

        def __init__(self, name: str) -> None:
            self.name = name
            super().__init__()

        def onFeedback(self, data, *args):
            data[self.name] = "xxx"
            self.output = data
            super().onFeedback(data)

        def __repr__(self):
            return self.__class__.__name__ + "()"

    x = atom("x")
    y = atom("y")
    z = atom("z")

    pipe = Pipe(x, y, z)

    start = atom("start")
    end = atom("end")

    start.connect(pipe)
    pipe.connect(end)

    start.onFeedback({})
    assert y.output == "NOTSET"


def test_pipe_is_connected_to_output_dataEvent():
    """
    Want this test to be standalone, independent of all fixtures, to
    make sure no mistake is being replicated.
    """

    class atom(Atom):
        output = "NOTSET"

        def __init__(self, name: str) -> None:
            self.name = name
            super().__init__()

        def onData(self, data, *args):
            data[self.name] = "xxx"
            self.output = data
            self.dataEvent.emit(data)

        def __repr__(self):
            return self.__class__.__name__ + "()"

    x = atom("x")
    y = atom("y")
    z = atom("z")

    pipe = Pipe(x, y, z)

    start = atom("start")
    end = atom("end")

    start += pipe
    pipe += end

    start.onData({})
    assert "x" in end.output
    assert "y" in end.output
    assert "z" in end.output
    assert len(end.output) == 5


def test_pipe_is_connected_to_output_startEvent():
    """
    Want this test to be standalone, independent of all fixtures, to
    make sure no mistake is being replicated.
    """

    class atom(Atom):
        output = "NOTSET"

        def __init__(self, name: str) -> None:
            self.name = name
            super().__init__()

        def onStart(self, data, *args):
            data[self.name] = "xxx"
            self.output = data
            super().onStart(data)

        def __repr__(self):
            return self.__class__.__name__ + "()"

    x = atom("x")
    y = atom("y")
    z = atom("z")

    pipe = Pipe(x, y, z)

    start = atom("start")
    end = atom("end")

    start += pipe
    pipe += end

    start.onStart({})
    assert "x" in end.output
    assert "y" in end.output
    assert "z" in end.output
    assert len(end.output) == 5


def test_pipe_is_connected_to_input_feedbackEvent():
    """
    Want this test to be standalone, independent of all fixtures, to
    make sure no mistake is being replicated.
    """

    class atom(Atom):
        output = "NOTSET"

        def __init__(self, name: str) -> None:
            self.name = name
            super().__init__()

        def onFeedback(self, data, *args):
            data[self.name] = "xxx"
            self.output = data
            super().onFeedback(data)

        def __repr__(self):
            return self.__class__.__name__ + "()"

    x = atom("x")
    y = atom("y")
    z = atom("z")

    pipe = Pipe(x, y, z)

    start = atom("start")
    end = atom("end")

    start.connect(pipe)
    pipe.connect(end)

    end.onFeedback({})
    assert "x" in end.output
    assert "y" in end.output
    assert "z" in end.output
    assert len(end.output) == 5


class AtomWithContract(Atom):

    def __init__(self, contract):
        super().__init__()
        self.contract = contract


class TestContract:
    @pytest.fixture
    def contract(self):
        return ibi.Contract(symbol="ES", exchange="CME")

    @pytest.fixture
    def atom(self):
        return AtomWithContract(ibi.Future(symbol="NQ", exchange="CME"))

    def test_can_assign_and_get_contract(self, atom: AtomWithContract):
        assert isinstance(atom.contract, ibi.Future)

    def test_same_contract_returned_as_assigned(self, contract: ibi.Contract):
        c = contract
        a = AtomWithContract(c)
        assert a.contract is c


class TestContractList:
    @pytest.fixture
    def contract(self):
        return ibi.Future(symbol="YM", exchange="NYMEX")

    @pytest.fixture
    def list_of_contracts(self, contract: ibi.Contract):
        return [
            contract,
            ibi.ContFuture(symbol="NQ", exchange="CME"),
            ibi.Stock(symbol="AAPL", exchange="NASDAQ"),
        ]

    @pytest.fixture
    def atom_with_contract(self, contract: ibi.Contract):
        class NewAtomWithContract(Atom):

            def __init__(self, contract):
                super().__init__()
                self.contract = contract

        a = NewAtomWithContract(contract)
        yield a
        a.contract_registry = ContractRegistry()  # type: ignore

    def test_newly_added_contract_in_Atom_registry(self, atom_with_contract: Atom):
        """
        All we're testing here is that the contract made it to the
        registry.  We're not checking if contract qualification and
        selectors work.
        """
        cont = ibi.Stock(symbol="AAPL", exchange="NASDAQ")
        atom_with_contract.contract = cont
        assert cont in atom_with_contract.contract_registry.blueprints


def test_all_contracts_from_many_atoms_in_registry():
    class A(Atom):

        def __init__(self, contract):
            self.contract = contract

    apple = ibi.Stock(symbol="AAPL", exchange="NASDAQ")
    nasdaq = ibi.ContFuture(symbol="NQ", exchange="CME")
    gold = ibi.Future(symbol="GC", exchange="COMEX")

    A(apple)
    A(nasdaq)
    A(gold)

    registry = A.contract_registry.blueprints

    assert apple in registry
    assert nasdaq in registry
    assert gold in registry


def test_onData_sets_attribute_if_dict_passed():
    class Source(Atom):
        def run(self):
            self.startEvent.emit({"strategy": "xxx"})

    class Output(Atom):

        def __init__(self):
            self.strategy = None
            super().__init__()

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
        _data = None
        args = None

        def onStart(self, data, *args):
            self._data = data
            self.args = args

    source = Source()
    inner = NewAtom()
    out = Output()
    Pipe(source, inner, out)
    source.run()
    assert isinstance(out.args[0], NewAtom)  # type: ignore


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

    assert out.strategy != "xxx"


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

    assert out.strategy == inner1.strategy == inner2.strategy == "xxx"  # type: ignore


def test_event_error_logged(caplog: pytest.LogCaptureFixture):
    class CustomException(Exception):
        pass

    class ErrorRaisingAtom(NewAtom):
        def onData(self, data, *args):
            raise CustomException("CustomError")

    a = NewAtom("a")
    b = ErrorRaisingAtom("b")
    a.connect(b)
    a.dataEvent.emit("xxx")
    assert "Event error dataEvent: CustomError" in caplog.messages


def test_event_error_logged_with_correct_logger(caplog: pytest.LogCaptureFixture):
    class CustomException(Exception):
        pass

    class ErrorRaisingAtom(NewAtom):
        def onData(self, data, *args):
            raise CustomException("CustomError")

    a = NewAtom("a")
    b = ErrorRaisingAtom("b")
    a.connect(b)
    a.dataEvent.emit("xxx")
    assert caplog.record_tuples == [
        ("strategy.NewAtom", logging.ERROR, "Event error dataEvent: CustomError")
    ]


def test_details_attr(details):
    """Only check if `details` on `Atom` properly linked to registry."""

    registry = ContractRegistry()

    class MockAtom(Atom):
        contract_registry = registry

        def __init__(self):
            super().__init__()
            self.contract = details.contract

    registry.details[details.contract] = details

    a = MockAtom()

    assert a.contract_details.contract == details.contract
    assert isinstance(a.contract_details, Details)


def test_if_no_contract_set_empty_details_returned():
    registry = ContractRegistry()

    class NewMockAtom(Atom):
        contract_registry = registry

    atom = NewMockAtom()

    assert atom.contract_details
    # no details stored
    assert len(registry.details) == 0
    # empty object created
    assert isinstance(atom.contract_details, Details)
    assert atom.contract_details.contract is None


def test_missing_details_log(
    caplog: pytest.LogCaptureFixture, details: ibi.ContractDetails
):
    caplog.set_level(logging.DEBUG)

    class NewMockAtom(Atom):

        def __init__(self, contract):
            self.contract = contract

    atom = NewMockAtom(details.contract)

    atom.contract_details

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

        a.strategy = "xxx"  # type: ignore
        b.strategy = "xxx"  # type: ignore
        c.strategy = "yyy"  # type: ignore
        d.strategy = "yyy"  # type: ignore

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

        a.strategy = "xxx"  # type: ignore
        b.strategy = "xxx"  # type: ignore
        c.strategy = "yyy"  # type: ignore
        d.strategy = "yyy"  # type: ignore

        a.data.position += 1
        b.data.position += 1
        c.data.position += 1
        assert a.data.position == 2
        assert d.data.position == 1


def test_provided_strategy_name_not_overriden():
    class A(Atom):
        def __init__(self, strategy: str):
            self.strategy = strategy
            super().__init__()

    a = A("xxx")

    assert a.strategy == "xxx"


def test_provided_strategy_name_not_overriden_onStart():
    class A(Atom):
        def __init__(self, strategy: str):
            self.strategy = strategy
            super().__init__()

    a = A("xxx")
    b = A("yyy")

    a += b

    a.startEvent.emit({"x": "y"})

    assert b.strategy == "yyy"


def test_strategy_updated_if_not_present():
    class A(Atom):
        def __init__(self, strategy: str):
            self.strategy = strategy
            super().__init__()

    a = A("xxx")
    b = A("")

    a += b

    a.startEvent.emit({"x": "y", "strategy": a.strategy})
    assert b.strategy == "xxx"


def test_startup_set():
    class A(Atom):
        def __init__(self, strategy: str):
            self.strategy = strategy
            super().__init__()

    a = A("xxx")
    b = A("yyy")

    a += b

    a.startEvent.emit({"startup": True})
    assert b.startup


def test_no_selector_if_no_contract_set():
    class FakeAtom(Atom):
        pass

    fake_atom = FakeAtom()

    with pytest.raises(KeyError):
        fake_atom.contract_selector


class Test_ActiveNext:
    """
    Test if correct contract requested.  Other modules determine if
    correct contract returned.
    """

    es = ibi.Future(symbol="ES", exchange="CME")

    def test_active_correct(self):
        mock_registry = Mock()

        class MyAtom(Atom):
            contract_registry = mock_registry

            def __init__(self, contract):
                super().__init__()
                self.contract = contract

        my_atom = MyAtom(self.es)

        # ACTIVE is the default
        # requesting contract should trigger lookup and call contract_registry
        contract = my_atom.contract  # noqa
        mock_registry.get_contract.assert_called_once_with(ANY, ActiveNext.ACTIVE)

    def test_next_correct(self):
        mock_registry = Mock()

        class MyAtom(Atom):
            contract_registry = mock_registry

            def __init__(self, contract):
                super().__init__()
                self.contract = contract
                self.which_contract = ActiveNext.NEXT

        my_atom = MyAtom(self.es)

        # ACTIVE is the default, but it was overriden to NEXT in __init__
        # requesting contract should trigger lookup and call contract_registry
        contract = my_atom.contract  # noqa
        mock_registry.get_contract.assert_called_once_with(ANY, ActiveNext.NEXT)
