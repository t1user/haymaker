import importlib
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import ib_insync as ibi
import pandas as pd
import pytest
from sample_barDataList import sample_barDataList

from haymaker.streamers import (
    HistoricalDataStreamer,
    bar_filter,
)


def test_bar_filter():
    bar = ibi.BarData(
        datetime(2025, 9, 25, 9, 6, 0, 0),
        high=21,
        low=20,
        close=-1,
        volume=100,
        average=0,
        barCount=4,
    )
    assert bar_filter(bar)


def test_Streamer_is_abstract(Streamer):
    with pytest.raises(TypeError):
        Streamer()  # type: ignore


def test_Streamer_keeps_instances(Streamer):
    class ConcreteStreamer(Streamer):
        def streaming_func(self):
            pass

    s0 = ConcreteStreamer()
    s1 = ConcreteStreamer()
    assert Streamer.instances == [s0, s1]


def test_StreamerId(Streamer):
    # make sure module level variable is not carried over from previous runs
    importlib.reload(importlib.import_module("haymaker.streamers"))

    class ConcreteStreamer(Streamer):

        def streaming_func(self):
            pass

    s0 = ConcreteStreamer()
    s1 = ConcreteStreamer()
    s2 = ConcreteStreamer()
    s2.name = "my_streamer"
    assert str(s0) == "ConcreteStreamer<0>"
    assert str(s1) == "ConcreteStreamer<1>"
    assert str(s2) == "ConcreteStreamer<2><my_streamer>"
    # repeated calls should not create another id
    assert str(s0) == "ConcreteStreamer<0>"
    # with contract set
    s2.contract = ibi.Future(symbol="NQ")
    assert str(s2) == "ConcreteStreamer<2><NQ><my_streamer>"


def test_StreamerId_dataclass():
    # make sure module level variable is not carried over from previous runs
    importlib.reload(importlib.import_module("haymaker.streamers"))

    s0 = HistoricalDataStreamer(ibi.Contract(symbol="XXX"), "x", "x", "x")
    assert str(s0) == "HistoricalDataStreamer<0><XXX>"


class FakeStore:
    async def read(self, x, *y):
        return None

    async def read_metadata(self, x, *y):
        return {}


@pytest.fixture()
def mock_arctic_store():
    with patch("haymaker.streamers.AsyncArcticStore") as mock_store_class:
        instance = mock_store_class.return_value
        instance.read = AsyncMock()
        instance.read_metadata = AsyncMock(return_value={})
        yield mock_store_class


@pytest.mark.parametrize(
    "datastore_value,expected",
    [(True, "mock"), (False, None), (fakestore := FakeStore(), fakestore)],
)
def test_HistoricalDataStreamer_datastore_property(
    mock_arctic_store, datastore_value, expected
):
    streamer = HistoricalDataStreamer(
        ibi.Future(symbol="NQ", exchange="CME"),
        10000,
        "1 min",
        "TRADES",
        datastore=datastore_value,
    )
    expected_store = mock_arctic_store.return_value if expected == "mock" else expected
    assert streamer.store == expected_store


def test_HistoricalDataStreamer_durationStr_given_as_int():
    streamer = HistoricalDataStreamer(
        ibi.Future(symbol="NQ", exchange="CME"), 10000, "1 min", "TRADES"
    )
    with patch(
        "haymaker.streamers.typical_session_length", return_value=timedelta(hours=23)
    ):
        assert streamer._durationStr == "8 D"


def test_HistoricalDataStreamer_durationStr_given_as_str():
    streamer = HistoricalDataStreamer(
        ibi.Future(symbol="NQ", exchange="CME"), "5 D", "1 min", "TRADES"
    )
    with patch(
        "haymaker.streamers.typical_session_length", return_value=timedelta(hours=23)
    ):
        assert streamer._durationStr == "5 D"


def test_HistoricalDataStreamer_durationStr_with_last_bar_date():
    streamer = HistoricalDataStreamer(
        ibi.Future(symbol="NQ", exchange="CME"),
        10000,
        "1 min",
        "TRADES",
        _last_bar_date=datetime(2026, 1, 26, 10, 0),
    )
    with patch(
        "haymaker.streamers.typical_session_length", return_value=timedelta(hours=23)
    ):
        with patch("haymaker.durationStr_converters.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2026, 1, 26, 10, 10)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            # 10 min elapsed (600 S) + 2 bars as standard margin: (10 + 2) * 60 = 720 S
            assert streamer._durationStr == "720 S"


@pytest.mark.asyncio
async def test_HistoricalDataStreamer_sync_last_bar_date_last_bar_date_given():
    streamer = HistoricalDataStreamer(
        ibi.Future(symbol="NQ", exchange="CME"),
        10000,
        "1 min",
        "TRADES",
        _last_bar_date=datetime(2026, 1, 26, 10, 0),
    )
    await streamer.sync_last_bar_date()
    assert streamer._last_bar_date == datetime(2026, 1, 26, 10, 0)


@pytest.mark.asyncio
async def test_HistoricalDataStreamer_sync_last_bar_date_store_True():
    """
    If HistoricalDataStreamer instantiated with `datastore = True`,
    default AsyncArcticStore should be instantiated and later used to
    read last data point.
    """
    contract = ibi.Future(symbol="NQ", exchange="CME")
    df = pd.DataFrame(sample_barDataList).set_index("date")

    with patch("haymaker.streamers.AsyncArcticStore") as mock_store_class:
        streamer = HistoricalDataStreamer(
            contract, 10000, "1 min", "TRADES", datastore=True
        )
        mock_store_instance = mock_store_class.return_value
        mock_store_instance.read = AsyncMock(return_value=df)
        mock_store_instance.read_metadata = AsyncMock(return_value={})
        assert streamer.store == mock_store_instance
        assert await streamer.last_db_point() == df.index[-1]

        await streamer.sync_last_bar_date()
        assert streamer._last_bar_date == df.index[-1]

        mock_store_instance.read.assert_awaited_with(contract)


@pytest.mark.asyncio
async def test_HistoricalDataStreamer_sync_last_bar_date_store_False():
    """
    If HistoricalDataStreamer instantiated with `datastore = False`,
    no data reads from datastore should be attempted.
    """
    streamer = HistoricalDataStreamer(
        ibi.Future(symbol="NQ", exchange="CME"),
        10000,
        "1 min",
        "TRADES",
        datastore=False,
    )

    with patch("haymaker.streamers.AsyncArcticStore") as mock_store_class:
        assert streamer.store is None
        assert await streamer.last_db_point() is None
        mock_store_class.assert_not_called()


@pytest.mark.asyncio
async def test_HistoricalDataStreamer_sync_last_bar_date_store_datastore_given():
    """
    If HistoricalDataStreamer instantiated with `_store = store_instance`,
    it should be using this store_instance to read last datapoint.
    """
    df = pd.DataFrame(sample_barDataList).set_index("date")

    class FakeStore:
        saved_contract = None
        call_counter = 0

        async def read(self, symbol, *args):
            self.saved_contract = symbol
            self.call_counter += 1
            return df

        async def read_metadata(self, symbol: ibi.Contract) -> dict:
            return {}

    fake_store = FakeStore()

    contract = ibi.Future(symbol="NQ", exchange="CME")
    streamer = HistoricalDataStreamer(
        contract, 10000, "1 min", "TRADES", datastore=fake_store
    )

    assert streamer.store == fake_store
    assert await streamer.last_db_point() == df.index[-1]
    assert fake_store.call_counter == 1

    await streamer.sync_last_bar_date()
    assert streamer._last_bar_date == df.index[-1]

    assert fake_store.saved_contract == contract
    assert fake_store.call_counter == 2
