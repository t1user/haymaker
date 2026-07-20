import os
import pickle
from dataclasses import dataclass
from unittest.mock import ANY, Mock

import ib_insync as ibi
import pandas as pd
import pytest
from config import TEST_ROOT  # type: ignore

from haymaker.base import ActiveNext
from haymaker.base import Atom as BaseAtom
from haymaker.block import AbstractBaseBlock, AbstractDfBlock
from haymaker.config.settings import StorageSettings
from haymaker.databases import StoreFactory
from haymaker.datastore import AsyncAbstractBaseStore, CollectionNamerStrategySymbol


@pytest.fixture
def block(atom_runtime) -> AbstractBaseBlock:
    class Block(AbstractBaseBlock):

        def _signal(self, data):
            return {"signal": 1, "abc": "xyz", "pqz": 5}

    return Block("eska_NQ", ibi.Future("NQ", "CME"))


def test_block_instantiates(block: AbstractBaseBlock):
    b = block
    assert isinstance(b, AbstractBaseBlock)


def test_block_registers_default_future_roll_policy(atom_runtime) -> None:
    """Blocks should opt into automatic futures rolling by default."""

    class Block(AbstractBaseBlock):
        def _signal(self, data):
            return {}

    Block("default-roll", ibi.Future("NQ", "CME"))

    assert atom_runtime.future_roll_policies == {"default-roll": True}


def test_block_registers_explicit_future_roll_opt_out(atom_runtime) -> None:
    """A Block keyword should declare a strategy-level roll exclusion."""

    class Block(AbstractBaseBlock):
        def _signal(self, data):
            return {}

    Block(
        "manual-roll",
        ibi.Future("NQ", "CME"),
        auto_roll_futures=False,
    )

    assert atom_runtime.future_roll_policies == {"manual-roll": False}


def test_block_allows_repeated_consistent_future_roll_policy(atom_runtime) -> None:
    """Multiple Blocks may share a strategy when their policies agree."""

    class Block(AbstractBaseBlock):
        def _signal(self, data):
            return {}

    Block("shared", ibi.Future("NQ", "CME"), auto_roll_futures=False)
    Block("shared", ibi.Future("ES", "CME"), auto_roll_futures=False)

    assert atom_runtime.future_roll_policies == {"shared": False}


def test_block_rejects_conflicting_future_roll_policy(atom_runtime) -> None:
    """Conflicting declarations for one persisted strategy should fail."""

    class Block(AbstractBaseBlock):
        def _signal(self, data):
            return {}

    Block("shared", ibi.Future("NQ", "CME"), auto_roll_futures=False)

    with pytest.raises(ValueError, match="Conflicting auto_roll_futures"):
        Block("shared", ibi.Future("ES", "CME"), auto_roll_futures=True)


def test_future_roll_policy_is_keyword_only_for_block_subclasses(atom_runtime) -> None:
    """Default policy must not prevent subclasses from adding required fields."""

    @dataclass
    class ParameterizedBlock(AbstractBaseBlock):
        lookback: int

        def _signal(self, data):
            return {}

    block = ParameterizedBlock(
        "parameterized",
        ibi.Future("NQ", "CME"),
        20,
        auto_roll_futures=False,
    )

    assert block.lookback == 20
    assert atom_runtime.future_roll_policies == {"parameterized": False}


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
def basic_df_block(atom_runtime):
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


def test_df_block_runtime_store_is_constructed_with_strategy_namer(
    atom_runtime, monkeypatch
):
    class Block(AbstractDfBlock):
        def df(self, data):
            return pd.DataFrame(data)

    store = Mock(spec=AsyncAbstractBaseStore)
    factory = StoreFactory(StorageSettings(block_library="block_data"))
    arctic_store = Mock(return_value=store)
    monkeypatch.setattr(factory, "arctic_store", arctic_store)
    atom_runtime.store_factory = factory

    block = Block("test_strategy", ibi.Future(symbol="NQ", exchange="CME"))

    assert block._datastore is store
    arctic_store.assert_called_once_with(
        "block_data",
        collection_namer=CollectionNamerStrategySymbol("test_strategy"),
    )


def test_df_blocks_keep_explicit_datastores_isolated(atom_runtime, monkeypatch):
    class Block(AbstractDfBlock):
        def df(self, data):
            return pd.DataFrame(data)

    factory = Mock()
    monkeypatch.setattr(atom_runtime.store_factory, "arctic_store", factory)
    first_store = Mock(spec=AsyncAbstractBaseStore)
    second_store = Mock(spec=AsyncAbstractBaseStore)

    first = Block(
        "first",
        ibi.Future(symbol="NQ", exchange="CME"),
        datastore=first_store,
    )
    second = Block(
        "second",
        ibi.Future(symbol="ES", exchange="CME"),
        datastore=second_store,
    )

    assert first.store is first_store
    assert second.store is second_store
    factory.assert_not_called()


def test_next_correct(atom_runtime_factory) -> None:

    es = ibi.Future(symbol="ES", exchange="CME")

    mock_registry = Mock()
    atom_runtime_factory(contract_registry=mock_registry)

    @dataclass
    class MyBlock(AbstractDfBlock):
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


def test_active_correct(atom_runtime_factory) -> None:

    es = ibi.Future(symbol="ES", exchange="CME")

    mock_registry = Mock()
    atom_runtime_factory(contract_registry=mock_registry)

    @dataclass
    class MyBlock(AbstractDfBlock):
        strategy: str
        contract: ibi.Contract

        def df(self, df: pd.DataFrame) -> pd.DataFrame:
            return df

    my_block = MyBlock("my_strategy_name", es)

    # just make sure correct contract requested in line with attr
    contract = my_block.contract  # noqa
    mock_registry.get_contract.assert_called_once_with(ANY, ActiveNext.ACTIVE)
