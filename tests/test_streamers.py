import importlib
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import ib_insync as ibi
import pandas as pd
import pytest
from sample_barDataList import sample_barDataList

from haymaker.streamers import (
    HistoricalDataStreamer,
    bar_filter,
)


@pytest.fixture(autouse=True)
def install_atom_runtime(atom_runtime):
    """Install default Atom runtime for streamer tests."""

    return atom_runtime


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


def test_timer_true():
    with patch("haymaker.streamers.Timeout.from_atom") as MockTimeout:
        streamer = HistoricalDataStreamer(
            ibi.Future(symbol="NQ", exchange="CME"),
            10000,
            "1 min",
            "TRADES",
            timeout=True,
        )
        event = ibi.Event()
        name = "my_test_timeout"
        streamer._set_timeout(event, name)
        MockTimeout.assert_called_once_with(streamer, event, name)


def test_timer_float():
    with patch("haymaker.streamers.Timeout.from_atom") as MockTimeout:
        streamer = HistoricalDataStreamer(
            ibi.Future(symbol="NQ", exchange="CME"),
            10000,
            "1 min",
            "TRADES",
            timeout=100,
        )
        event = ibi.Event()
        name = "my_test_timeout"
        streamer._set_timeout(event, name)
        MockTimeout.assert_called_once_with(streamer, event, name, 100)


@pytest.mark.parametrize(
    "datastore",
    [None, FakeStore()],
)
def test_HistoricalDataStreamer_keeps_injected_datastore(datastore):
    """A streamer should retain its explicit datastore dependency."""

    streamer = HistoricalDataStreamer(
        ibi.Future(symbol="NQ", exchange="CME"),
        10000,
        "1 min",
        "TRADES",
        datastore=datastore,
    )

    assert streamer.datastore is datastore


@pytest.mark.parametrize("datastore", [True, False])
def test_HistoricalDataStreamer_rejects_boolean_datastore_shortcuts(datastore):
    """Legacy boolean service-locator shortcuts should fail clearly."""

    with pytest.raises(TypeError, match="boolean shortcuts"):
        HistoricalDataStreamer(
            ibi.Future(symbol="NQ", exchange="CME"),
            10000,
            "1 min",
            "TRADES",
            datastore=datastore,  # type: ignore[arg-type]
        )


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
async def test_HistoricalDataStreamer_sync_last_bar_date_uses_injected_store():
    """Incremental startup should read through the injected datastore."""

    contract = ibi.Future(symbol="NQ", exchange="CME")
    df = pd.DataFrame(sample_barDataList).set_index("date")
    store = Mock()
    store.read = AsyncMock(return_value=df)
    store.read_metadata = AsyncMock(return_value={})

    streamer = HistoricalDataStreamer(
        contract, 10000, "1 min", "TRADES", datastore=store
    )
    assert await streamer.last_db_point() == df.index[-1]

    await streamer.sync_last_bar_date()
    assert streamer._last_bar_date == df.index[-1]

    store.read.assert_awaited_with(contract)


@pytest.mark.asyncio
async def test_HistoricalDataStreamer_sync_last_bar_date_store_none():
    """A streamer without a datastore should skip database reads."""

    streamer = HistoricalDataStreamer(
        ibi.Future(symbol="NQ", exchange="CME"),
        10000,
        "1 min",
        "TRADES",
    )

    assert streamer.datastore is None
    assert await streamer.last_db_point() is None


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

    assert streamer.datastore == fake_store
    assert await streamer.last_db_point() == df.index[-1]
    assert fake_store.call_counter == 1

    await streamer.sync_last_bar_date()
    assert streamer._last_bar_date == df.index[-1]

    assert fake_store.saved_contract == contract
    assert fake_store.call_counter == 2
