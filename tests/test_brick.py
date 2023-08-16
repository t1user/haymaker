import ib_insync as ibi
import pandas as pd
import pytest

from ib_tools.brick import AbstractBaseBrick, AbstractDfBrick
from ib_tools.signals import BinarySignalProcessor


@pytest.fixture
def brick():
    class Brick(AbstractBaseBrick):
        def _signal(self, data):
            return (1, 1, "entry")

    return Brick(("eska", "NQ"), ibi.ContFuture("NQ", "CME"), None)


def test_brick_instantiates(brick):
    b = brick
    assert isinstance(b, AbstractBaseBrick)


def test_AbstractBaseBrick_is_abstract():
    with pytest.raises(TypeError):
        AbstractBaseBrick()


# ========================================
# DfBrick from here
# ========================================


@pytest.fixture
def df_data():
    return pd.DataFrame(
        {
            "price": [123, 123, 125, 124, 128],
            "position": [1, 0, 1, 1, 0],
            "signal": [1, 1, 1, 1, 1],
            "raw": [0, 0, 0, 0, -1],
        }
    )


@pytest.fixture
def state_checker():
    class FakeStateMachine:
        def position(self, key):
            return 1

        def locked(self, key):
            return 0


@pytest.fixture
def df_brick(df_data, state_checker):
    class Brick(AbstractDfBrick):
        def _signal(self, data):
            return (1, 1, "entry")

        def df(self, data):
            return df_data

    return Brick(
        ("eska", "NQ"),
        ibi.ContFuture("NQ", "CME"),
        None,
        True,
        False,
        BinarySignalProcessor(state_checker),
    )


def test_df_brick_instantiates(df_brick):
    b = df_brick
    assert isinstance(b, AbstractBaseBrick)


def test_AbstractDfBrick_is_abstract():
    with pytest.raises(TypeError):
        AbstractDfBrick()
