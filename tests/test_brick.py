import ib_insync as ibi
import pandas as pd
import pytest

from ib_tools.base import Atom
from ib_tools.brick import AbstractBaseBrick, AbstractDfBrick


@pytest.fixture
def brick():
    class Brick(AbstractBaseBrick):
        def _signal(self, data):
            return {"signal": 1, "dupa": "xyz", "kamieni_kupa": 5}

    return Brick("eska_NQ", ibi.ContFuture("NQ", "CME"))


def test_brick_instantiates(brick):
    b = brick
    assert isinstance(b, AbstractBaseBrick)


def test_AbstractBaseBrick_is_abstract():
    with pytest.raises(TypeError):
        AbstractBaseBrick()


def test_data_passed_correct(brick):
    x = {}

    def recorder(a):
        nonlocal x
        x = a

    brick.dataEvent += recorder
    brick.onData(1)
    assert x == {
        "key": "eska_NQ",
        "contract": ibi.ContFuture("NQ", "CME"),
        "signal": 1,
        "dupa": "xyz",
        "kamieni_kupa": 5,
    }


# ========================================
# DfBrick from here
# ========================================


@pytest.fixture
def data_for_df():
    return {
        "price": [123, 123, 125, 124, 128],
        "position": [1, 0, 1, 1, 0],
        "signal": [1, 1, 1, 1, 1],
        "raw": [0, 0, 0, 0, -1],
    }


@pytest.fixture
def df_brick():
    class Brick(AbstractDfBrick):
        def df(self, data):
            df = pd.DataFrame(data)
            df["price_plus"] = df["price"] + 1
            return df

    return Brick(
        "eska_NQ",
        ibi.ContFuture("NQ", "CME"),
    )


@pytest.fixture
def df_connected_brick(df_brick):
    class NewAtom(Atom):
        def onData(self, data, *args):
            self.data = data

    new_atom = NewAtom()
    df_brick += new_atom
    return df_brick, new_atom


def test_df_brick_instantiates(df_brick):
    b = df_brick
    assert isinstance(b, AbstractBaseBrick)


def test_AbstractDfBrick_is_abstract():
    with pytest.raises(TypeError):
        AbstractDfBrick()


def test_signal_correct(df_connected_brick, data_for_df):
    brick, atom = df_connected_brick
    brick.onData(data_for_df)
    assert "signal" in atom.data.keys()
    assert atom.data["signal"] == 1


def test_signal_column_selection_works(df_connected_brick, data_for_df):
    brick, atom = df_connected_brick
    brick.signal_column = "raw"
    brick.onData(data_for_df)
    assert "signal" in atom.data.keys()
    assert atom.data["signal"] == -1


def test_column_selection_works(df_connected_brick, data_for_df):
    brick, atom = df_connected_brick
    brick.df_columns = ["price", "signal", "price_plus"]
    brick.onData(data_for_df)
    assert list(atom.data.keys()) == [
        "key",
        "contract",
        "price",
        "signal",
        "price_plus",
    ]
