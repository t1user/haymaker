import os
import pickle

import ib_insync as ibi
import pandas as pd
import pytest
from config import TEST_ROOT  # type: ignore

from ib_tools.base import Atom
from ib_tools.brick import AbstractBaseBrick, AbstractDfBrick


@pytest.fixture
def brick():
    class Brick(AbstractBaseBrick):
        def _signal(self, data):
            return {"signal": 1, "abc": "xyz", "pqz": 5}

    return Brick("eska_NQ", ibi.Future("NQ", "CME"))


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
    del x["Brick_ts"]
    assert x == {
        "strategy": "eska_NQ",
        "contract": ibi.Future("NQ", "CME"),
        "signal": 1,
        "abc": "xyz",
        "pqz": 5,
    }


# ========================================
# DfBrick from here
# ========================================


@pytest.fixture
def data_for_df():
    return {
        "date": ["2023-10-09", "2023-10-09", "2023-10-09", "2023-10-09", "2023-10-09"],
        "price": [123, 123, 125, 124, 128],
        "position": [1, 0, 1, 1, 0],
        "signal": [1, 1, 1, 1, 1],
        "raw": [0, 0, 0, 0, -1],
    }


@pytest.fixture
def df_brick():
    class Brick(AbstractDfBrick):
        def df(self, df):
            df["price_plus"] = df["price"] + 1
            return df

    return Brick(
        "eska_NQ", ibi.Future("NQ", "CME"), signal_column="signal", df_columns=None
    )


@pytest.fixture
def df_connected_brick(df_brick):
    class NewAtom(Atom):
        def onData(self, data, *args):
            self.memo = data

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
    assert "signal" in atom.memo.keys()
    assert atom.memo["signal"] == 1


def test_signal_column_selection_works(df_connected_brick, data_for_df):
    brick, atom = df_connected_brick
    brick.signal_column = "raw"
    brick.onData(data_for_df)
    assert "signal" in atom.memo.keys()
    assert atom.memo["signal"] == -1


def test_column_selection_works(df_connected_brick, data_for_df):
    brick, atom = df_connected_brick
    brick.df_columns = ["price", "signal", "price_plus"]
    brick.onData(data_for_df)
    assert list(atom.memo.keys()) == [
        "strategy",
        "contract",
        "price",
        "signal",
        "price_plus",
        "Brick_ts",
    ]


@pytest.fixture
def basic_df_brick():
    class Brick(AbstractDfBrick):
        def df(self, df):
            return df

    return Brick(
        "eska_NQ", ibi.Future("NQ", "CME"), signal_column="signal", df_columns=None
    )


def test_dispatchmethod_df(basic_df_brick, data_for_df):
    df = pd.DataFrame(data_for_df)
    row = basic_df_brick.df_row(df)
    # hard to compare two Series, easier with dicts, result the same
    assert row.to_dict() == df.iloc[-1].to_dict()


def test_dispatchmethod_barList(basic_df_brick):
    with open(os.path.join(TEST_ROOT, "data/data_from_streamer.pickle"), "rb") as f:
        data = pickle.loads(f.read())
    row_dict = data[-1].dict()
    row = basic_df_brick.df_row(data)
    assert row_dict == row.to_dict()


def test_dispatchmethod_dict(basic_df_brick, data_for_df):
    out = basic_df_brick.df_row(data_for_df)

    should_be = pd.DataFrame(data_for_df).iloc[-1]
    assert out.to_dict() == should_be.to_dict()
