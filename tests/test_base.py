import pytest

from ib_tools.base import Atom, Pipe


class NewAtom(Atom):
    onStart_string = None
    onData_string = None
    onStart_caller = None
    onData_caller = None
    onData_checksum = 0
    onStart_checksum = 0

    def __init__(self, name):
        self.name = name
        super().__init__()

    def onStart(self, data, source):
        self.onStart_string = data
        self.onStart_caller = source
        self.onStart_checksum += 1
        self.startEvent.emit(f"{data}_{self.name}")
        return data, source

    def onData(self, data, source):
        self.onData_string = data
        self.onData_caller = source
        self.onData_checksum += 1
        self.dataEvent.emit(f"{data}_{self.name}")
        return data, source


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

    def test_disconnect(self, atom1, atom2):
        atom1.connect(atom2)
        atom1.disconnect(atom2)
        assert len(atom1.startEvent) == 0
        assert len(atom2.dataEvent) == 0

    def test_clear(self, atom1, atom2):
        atom3 = NewAtom("atom3")
        atom3.connect(atom1, atom2)
        atom3.clear()
        assert len(atom3.startEvent) == 0
        assert len(atom3.dataEvent) == 0

    def test_iadd(self, atom1, atom2):
        atom1 += atom2
        atom1.startEvent.emit("test_string")
        atom1.dataEvent.emit("test_string")
        assert atom2.onStart_string == "test_string"
        assert atom2.onData_string == "test_string"

    def test_isub(self, atom1, atom2):
        atom1.connect(atom2)
        atom1 -= atom2
        assert len(atom1.startEvent) == 0
        assert len(atom2.dataEvent) == 0

    def test_union(self, atom1, atom2):
        atom3 = NewAtom("atom3")
        atom3.union(atom1, atom2)
        atom3.startEvent.emit("test_string")
        assert atom1.onStart_string == "test_string"
        assert atom2.onStart_string == "test_string"
        assert id(atom1.onStart_caller) == id(atom3)

    def test_unequality(self, atom1):
        ato = NewAtom("atom1")
        assert ato != atom1

    def test_equality(self, atom1, atom2):
        atom1.connect(atom2)
        atom1.startEvent.emit("test")
        assert atom2.onStart_caller == atom1

    def test_repr(self, atom1):
        assert repr(atom1) == "NewAtom(name=atom1)"

    def test_no_duplicate_connections(self, atom1, atom2):
        atom1.connect(atom2)
        atom1.connect(atom2)
        atom1.startEvent.emit("test_string")
        atom1.dataEvent.emit("test_string")
        assert len(atom1.startEvent) == 1
        assert len(atom1.dataEvent) == 1

    def test_no_duplicate_connections_1(self, atom1, atom2):
        atom1.connect(atom2)
        atom1.connect(atom2)
        atom1.startEvent.emit("test_string")
        atom1.dataEvent.emit("test_string")
        assert atom2.onStart_checksum == 1
        assert atom2.onData_checksum == 1


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

    def test_inside_atoms_end(self, pass_through_pipe):
        _, _, pipe = pass_through_pipe
        assert pipe[0].onData_string == "DataEvent"
        assert pipe[1].onData_string == "DataEvent_x"
        assert pipe[2].onData_string == "DataEvent_x_y"

    def test_inside_atoms_start_event_pass_through(self, pass_through_pipe):
        start, end, pipe = pass_through_pipe
        assert pipe[0].onStart_string == "StartEvent"
        assert pipe[1].onStart_string == "StartEvent_x"
        assert pipe[2].onStart_string == "StartEvent_x_y"

    def test_source_object_pass_through(self, pass_through_pipe):
        _, _, pipe = pass_through_pipe
        assert pipe[1].onStart_caller == pipe[0]
        assert pipe[2].onStart_caller == pipe[1]

    def test_source_object_pass_through_to_outside_of_pipe(self, pass_through_pipe):
        start, end, pipe = pass_through_pipe
        assert end.onStart_caller == pipe[-1]

    def test_object_pass_through_from_outside_of_pipe(self, pass_through_pipe):
        start, end, pipe = pass_through_pipe
        assert pipe[0].onStart_caller == start

    def test_pass_through_startEvent(self, pass_through_pipe):
        start, end, pipe = pass_through_pipe
        assert end.onStart_string == "StartEvent_x_y_z"

    def test_pass_through_dataEvent(self, pass_through_pipe):
        start, end, pipe = pass_through_pipe
        assert end.onData_string == "DataEvent_x_y_z"

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

    def test_union_pipe(self, pipe1, pipe2, atoms):
        start, end = atoms
        p1 = pipe1
        p2 = pipe2
        start.union(p1, p2)
        start.startEvent.emit("test_string")
        assert p1[-1].onStart_string == "test_string_x_y"
        assert p2[-1].onStart_string == "test_string_a_b"

    def test_union_pipe_output(self, pipe1, pipe2, atoms):
        start, end = atoms
        p1 = pipe1
        p2 = pipe2
        start.union(p1, p2)
        p1.connect(end)
        start.startEvent.emit("test_string")
        assert end.onStart_string == "test_string_x_y_z"
