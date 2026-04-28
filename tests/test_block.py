import os
import pickle
from dataclasses import dataclass
from typing import ClassVar
from unittest.mock import ANY, Mock

import ib_insync as ibi
import pandas as pd
import pytest
from config import TEST_ROOT  # type: ignore

from haymaker.base import ActiveNext
from haymaker.base import Atom as BaseAtom
from haymaker.block import AbstractBaseBlock, AbstractDfBlock
from haymaker.contract_registry import ContractRegistry
from haymaker.datastore import CollectionNamerStrategySymbol


@pytest.fixture
def block() -> AbstractBaseBlock:
    class Block(AbstractBaseBlock):

        def _signal(self, data):
            return {"signal": 1, "abc": "xyz", "pqz": 5}

    return Block("eska_NQ", ibi.Future("NQ", "CME"))


def test_block_instantiates(block: AbstractBaseBlock):
    b = block
    assert isinstance(b, AbstractBaseBlock)


def test_AbstractBaseBlock_is_abstract():
    with pytest.raises(TypeError):
        AbstractBaseBlock()  # type: ignore


def test_data_passed_correct(block: AbstractBaseBlock):
    x = {}

    def recorder(a):
        nonlocal x
        x = a

    block.dataEvent += recorder
    block.onData(1)
    del x["Block_ts"]
    assert x == {
        "strategy": "eska_NQ",
        "contract": ibi.Future("NQ", "CME"),
        "signal": 1,
        "abc": "xyz",
        "pqz": 5,
    }


# ========================================
# DfBlock from here
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


# don't remove Atom dependency, it ensures contract_registry cleaned for every test
@pytest.fixture
def df_block(Atom):
    class Block(AbstractDfBlock):

        def df(self, data):
            data["price_plus"] = data["price"] + 1
            return data

    return Block("eska_NQ", ibi.Future("NQ", "CME"))


@pytest.fixture
def df_connected_block(
    df_block: AbstractDfBlock, Atom
) -> tuple[AbstractDfBlock, BaseAtom]:
    class NewAtom(Atom):
        def onData(self, data, *args):
            self.memo = data

    new_atom = NewAtom()
    df_block += new_atom
    return df_block, new_atom


def test_df_block_instantiates(df_block):
    b = df_block
    assert isinstance(b, AbstractBaseBlock)


def test_AbstractDfBlock_is_abstract():
    with pytest.raises(TypeError):
        AbstractDfBlock()  # type: ignore


async def test_signal_correct(
    df_connected_block: tuple[AbstractDfBlock, BaseAtom], data_for_df: dict
):
    """Test that df_block correctly emits the last row of df."""
    block, atom = df_connected_block
    block.onData(data_for_df)
    assert "signal" in atom.memo.keys()  # type: ignore
    assert atom.memo["signal"] == 1  # type: ignore


@pytest.fixture
def basic_df_block():
    class Block(AbstractDfBlock):

        def df(self, data):
            return data

    return Block("eska_NQ", ibi.Future("NQ", "CME"))


async def test_dispatchmethod_df(basic_df_block: AbstractDfBlock, data_for_df: dict):
    df = pd.DataFrame(data_for_df)
    row = basic_df_block.df_row(df)
    assert row.equals(df.iloc[-1])


async def test_dispatchmethod_barList(basic_df_block: AbstractDfBlock):
    with open(os.path.join(TEST_ROOT, "data/data_from_streamer.pickle"), "rb") as f:
        data = pickle.loads(f.read())
    row_dict = data[-1].dict()
    row = basic_df_block.df_row(data)
    assert row_dict == row.to_dict()


async def test_dispatchmethod_dict(basic_df_block: AbstractDfBlock, data_for_df: dict):
    out = basic_df_block.df_row(data_for_df)

    should_be = pd.DataFrame(data_for_df).iloc[-1]
    assert out.equals(should_be)


async def test_df_block_accepts_df(
    basic_df_block: AbstractDfBlock, data_for_df: dict, Atom
):

    class OutputAtom(Atom):
        output = None

        def onData(self, data, *args):
            self.output = data

    output_atom = OutputAtom()
    basic_df_block += output_atom
    df = pd.DataFrame(data_for_df)
    basic_df_block.onData(df)
    assert output_atom.output is not None
    last_row = df.iloc[-1].to_dict()
    assert {k: v for k, v in output_atom.output.items() if k in last_row} == last_row


def test_DfBlock_has_correct_collection_namer():
    class Block(AbstractDfBlock):

        def df(self, data):
            return pd.DataFrame(data)

    Block.set_datastore(Mock())

    block = Block("test_strategy", ibi.Future(symbol="NQ", exchange="CME"))

    store = block._datastore

    store.override_collection_namer.assert_called_once()
    store.override_collection_namer.assert_called_once_with(
        CollectionNamerStrategySymbol("test_strategy")
    )


def test_next_correct() -> None:

    es = ibi.Future(symbol="ES", exchange="CME")

    mock_registry = Mock()

    @dataclass
    class MyBlock(AbstractDfBlock):
        contract_registry: ClassVar[ContractRegistry] = mock_registry

        strategy: str
        contract: ibi.Contract
        which_contract: ActiveNext = ActiveNext.NEXT

        def df(self, df: pd.DataFrame) -> pd.DataFrame:
            return df

    my_block = MyBlock("my_strategy_name", es)

    # just make sure `Block` asks for correct contract
    # class default is ACTIVE, we've changed it to NEXT
    # wheather correct object actually returned is tested elsewhere
    contract = my_block.contract  # noqa
    mock_registry.get_contract.assert_called_once_with(ANY, ActiveNext.NEXT)


def test_active_correct() -> None:

    es = ibi.Future(symbol="ES", exchange="CME")

    mock_registry = Mock()

    @dataclass
    class MyBlock(AbstractDfBlock):
        contract_registry: ClassVar[ContractRegistry] = mock_registry

        strategy: str
        contract: ibi.Contract

        def df(self, df: pd.DataFrame) -> pd.DataFrame:
            return df

    my_block = MyBlock("my_strategy_name", es)

    # just make sure correct contract requested in line with attr
    contract = my_block.contract  # noqa
    mock_registry.get_contract.assert_called_once_with(ANY, ActiveNext.ACTIVE)
