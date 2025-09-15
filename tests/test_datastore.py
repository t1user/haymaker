from unittest.mock import MagicMock, patch

import ib_insync as ibi
import pandas as pd
import pytest
from arctic.exceptions import NoDataFoundException  # type: ignore

from haymaker.datastore import AbstractBaseStore, ArcticStore

# these are llm generated tests that don't really test much


@pytest.fixture
def dummy_contract():
    return ibi.Contract(
        localSymbol="EURUSD",
        secType="FX",
        conId=0,
        symbol="EURUSD",
        lastTradeDateOrContractMonth="",
        multiplier="",
        exchange="",
        currency="USD",
        tradingClass="",
    )


@pytest.fixture
def dummy_df():
    return pd.DataFrame(
        {"open": [1, 2], "close": [3, 4]}, index=pd.date_range("2025-01-01", periods=2)
    )


@pytest.fixture
def arctic_store_mock():
    with patch("haymaker.datastore.datastore.Arctic") as mock_arctic:
        db_instance = MagicMock()
        mock_arctic.return_value = db_instance
        db_instance.__getitem__.return_value = MagicMock()  # store
        yield db_instance


def test_write_calls_store_write(dummy_contract, dummy_df, arctic_store_mock):
    store = ArcticStore("test_lib", host="localhost")
    store.store.write = MagicMock(return_value=MagicMock(symbol="EURUSD", version=1))
    res = store.write(dummy_contract, dummy_df, meta={"foo": "bar"})
    assert "symbol: EURUSD version: 1" == res
    store.store.write.assert_called_once()


def test_read_returns_data_frame(dummy_contract, dummy_df, arctic_store_mock):
    store = ArcticStore("test_lib", host="localhost")
    versioned_item = MagicMock()
    versioned_item.data = dummy_df
    store.read_object = MagicMock(return_value=versioned_item)
    result = store.read(dummy_contract)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, dummy_df)


def test_read_returns_none_if_no_data(dummy_contract, arctic_store_mock):
    store = ArcticStore("test_lib", host="localhost")
    store.read_object = MagicMock(return_value=None)
    result = store.read(dummy_contract)
    assert result is None


def test_delete_calls_store_delete(dummy_contract, arctic_store_mock):
    store = ArcticStore("test_lib", host="localhost")
    store.store.delete = MagicMock()
    store.delete(dummy_contract)
    store.store.delete.assert_called_once()


def test_keys_returns_list(arctic_store_mock):
    store = ArcticStore("test_lib", host="localhost")
    store.store.list_symbols = MagicMock(return_value=["A", "B"])
    keys = store.keys()
    assert keys == ["A", "B"]


def test_read_metadata_returns_dict(arctic_store_mock):
    store = ArcticStore("test_lib", host="localhost")
    store.store.read_metadata = MagicMock()
    md = MagicMock()
    md.metadata = {"foo": "bar"}
    store.store.read_metadata.return_value = md
    result = store.read_metadata("EURUSD")
    assert result == {"foo": "bar"}


def test_read_metadata_handles_no_data(arctic_store_mock):
    store = ArcticStore("test_lib", host="localhost")
    store.store.read_metadata = MagicMock(side_effect=NoDataFoundException)
    result = store.read_metadata("EURUSD")
    assert result == {}


def test_read_metadata_handles_no_data_1(arctic_store_mock):
    store = ArcticStore("test_lib", host="localhost")
    store.store.read_metadata = MagicMock(side_effect=AttributeError)
    result = store.read_metadata("EURUSD")
    assert result == {}


def test_write_metadata_calls_store_write_metadata(dummy_contract, arctic_store_mock):
    store = ArcticStore("test_lib", host="localhost")
    store.store.write_metadata = MagicMock()
    store.write_metadata(dummy_contract, {"foo": "bar"})
    store.store.write_metadata.assert_called_once()


def test_override_metadata_calls_store_write_metadata(
    dummy_contract, arctic_store_mock
):
    store = ArcticStore("test_lib", host="localhost")
    store.store.write_metadata = MagicMock()
    store.override_metadata("EURUSD", {"baz": "qux"})
    store.store.write_metadata.assert_called_once()


def test_symbol_conversion(dummy_contract):
    store = ArcticStore("test_lib", host="localhost")
    assert store._symbol(dummy_contract) == "EURUSD_FX"
    assert store._symbol("USDJPY") == "USDJPY"


def test_clean_removes_duplicates():
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3]}, index=[1, 1, 2])
    cleaned = AbstractBaseStore._clean(df)
    assert cleaned.index.is_unique
