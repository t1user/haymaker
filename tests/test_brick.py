import os
import pickle

import ib_insync as ibi
import pandas as pd
import pytest
from config import TEST_ROOT  # type: ignore

from haymaker.base import Atom
from haymaker.brick import AbstractBaseBrick, AbstractDfBrick


@pytest.fixture
def brick() -> AbstractBaseBrick:
    class Brick(AbstractBaseBrick):
        def _signal(self, data):
            return {"signal": 1, "abc": "xyz", "pqz": 5}

    return Brick("eska_NQ", ibi.Future("NQ", "CME"))


def test_brick_instantiates(brick: AbstractBaseBrick):
    b = brick
    assert isinstance(b, AbstractBaseBrick)


def test_AbstractBaseBrick_is_abstract():
    with pytest.raises(TypeError):
        AbstractBaseBrick()  # type: ignore


def test_data_passed_correct(brick: AbstractBaseBrick):
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
        def df(self, data):
            data["price_plus"] = data["price"] + 1
            return data

    return Brick("eska_NQ", ibi.Future("NQ", "CME"))


@pytest.fixture
def df_connected_brick(df_brick: AbstractDfBrick) -> tuple[AbstractDfBrick, Atom]:
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
        AbstractDfBrick()  # type: ignore


def test_signal_correct(
    df_connected_brick: tuple[AbstractDfBrick, Atom], data_for_df: dict
):
    """Test that df_brick correctly emits the last row of df."""
    brick, atom = df_connected_brick
    brick.onData(data_for_df)
    assert "signal" in atom.memo.keys()  # type: ignore
    assert atom.memo["signal"] == 1  # type: ignore


@pytest.fixture
def basic_df_brick():
    class Brick(AbstractDfBrick):
        def df(self, data):
            return data

    return Brick("eska_NQ", ibi.Future("NQ", "CME"))


def test_dispatchmethod_df(basic_df_brick: AbstractDfBrick, data_for_df: dict):
    df = pd.DataFrame(data_for_df)
    row = basic_df_brick.df_row(df)
    # hard to compare two Series, easier with dicts, result the same
    assert row.to_dict() == df.iloc[-1].to_dict()


def test_dispatchmethod_barList(basic_df_brick: AbstractDfBrick):
    with open(os.path.join(TEST_ROOT, "data/data_from_streamer.pickle"), "rb") as f:
        data = pickle.loads(f.read())
    row_dict = data[-1].dict()
    row = basic_df_brick.df_row(data)
    assert row_dict == row.to_dict()


def test_dispatchmethod_dict(basic_df_brick: AbstractDfBrick, data_for_df: dict):
    out = basic_df_brick.df_row(data_for_df)

    should_be = pd.DataFrame(data_for_df).iloc[-1]
    assert out.to_dict() == should_be.to_dict()
