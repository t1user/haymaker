import asyncio
import importlib
import math
from datetime import date, datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import ib_insync as ibi
import pandas as pd
import pytest
from sample_barDataList import sample_barDataList

from haymaker.components.streamers import (
    HistoricalDataStreamer,
    MktDataStreamer,
    RealTimeBarsStreamer,
)


@pytest.fixture(autouse=True)
def install_atom_runtime(atom_runtime):
    """Install default Atom runtime for streamer tests."""

    return atom_runtime


def make_bar(bar_date: date | datetime, price: float = 20) -> ibi.BarData:
    """Return a valid historical bar for streamer tests."""
    return ibi.BarData(
        date=bar_date,
        open=price,
        high=price + 1,
        low=price - 1,
        close=price,
        volume=100,
        average=price,
        barCount=4,
    )


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
    importlib.reload(importlib.import_module("haymaker.components.streamers"))

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
    importlib.reload(importlib.import_module("haymaker.components.streamers"))

    s0 = HistoricalDataStreamer(ibi.Contract(symbol="XXX"), "1 D", "1 min", "TRADES")
    assert str(s0) == "HistoricalDataStreamer<0><XXX>"


class FakeStore:
    async def read(self, x, *y):
        return None

    async def read_metadata(self, x, *y):
        return {}


def test_timer_true():
    with patch(
        "haymaker.components.streamers.MarketDataTimeout.from_atom"
    ) as MockTimeout:
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
    with patch(
        "haymaker.components.streamers.MarketDataTimeout.from_atom"
    ) as MockTimeout:
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
        "haymaker.components.streamers.typical_session_length",
        return_value=timedelta(hours=23),
    ):
        assert streamer._durationStr == "8 D"


def test_HistoricalDataStreamer_durationStr_given_as_str():
    streamer = HistoricalDataStreamer(
        ibi.Future(symbol="NQ", exchange="CME"), "5 D", "1 min", "TRADES"
    )
    with patch(
        "haymaker.components.streamers.typical_session_length",
        return_value=timedelta(hours=23),
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
        "haymaker.components.streamers.typical_session_length",
        return_value=timedelta(hours=23),
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
async def test_HistoricalDataStreamer_restores_persisted_daily_bar_date():
    """Daily metadata should retain the date category received from IB."""
    contract = ibi.Future(symbol="NQ", exchange="CME")
    store = Mock()
    store.read_metadata = AsyncMock(return_value={"up_to": "2026-01-26"})

    streamer = HistoricalDataStreamer(
        contract, 10000, "1 day", "TRADES", datastore=store
    )

    assert await streamer.last_db_point() == date(2026, 1, 26)


def test_HistoricalDataStreamer_durationStr_with_daily_last_bar_date():
    """A date watermark should be converted only for duration calculation."""
    streamer = HistoricalDataStreamer(
        ibi.Future(symbol="NQ", exchange="CME"),
        10000,
        "1 day",
        "TRADES",
        _last_bar_date=date(2026, 1, 26),
    )
    with patch("haymaker.durationStr_converters.datetime") as mock_datetime:
        mock_datetime.now.return_value = datetime(2026, 1, 27)
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

        assert streamer._durationStr == "3 D"
        assert streamer._last_bar_date == date(2026, 1, 26)


def test_HistoricalDataStreamer_removes_invalid_bars_from_emitted_history():
    """A rejected historical bar should not reappear in a later snapshot."""
    streamer = HistoricalDataStreamer(
        ibi.Future(symbol="NQ", exchange="CME"),
        10000,
        "1 min",
        "TRADES",
    )
    first = make_bar(datetime(2026, 1, 26, 10, 0))
    invalid = make_bar(datetime(2026, 1, 26, 10, 1), price=math.nan)
    latest = make_bar(datetime(2026, 1, 26, 10, 2))
    emitted: list[list[ibi.BarData]] = []
    streamer.dataEvent += emitted.append

    streamer.on_new_bar([first, invalid, latest])

    assert emitted == [[first, latest]]


def test_HistoricalDataStreamer_accepts_finite_nonpositive_prices():
    """Finite zero or negative historical prices are not inherently invalid."""
    streamer = HistoricalDataStreamer(
        ibi.Future(symbol="NQ", exchange="CME"),
        10000,
        "1 min",
        "TRADES",
    )

    assert streamer._is_valid_bar(make_bar(datetime(2026, 1, 26), price=0))
    assert streamer._is_valid_bar(make_bar(datetime(2026, 1, 27), price=-20))


def test_HistoricalDataStreamer_ignores_unavailable_average_for_midpoint():
    """Non-trade data should not require a finite trade-average field."""
    streamer = HistoricalDataStreamer(
        ibi.Future(symbol="NQ", exchange="CME"),
        10000,
        "1 min",
        "MIDPOINT",
    )
    bar = make_bar(datetime(2026, 1, 26))
    bar.average = math.nan

    assert streamer._is_valid_bar(bar)


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


def test_RealTimeBarsStreamer_forwards_request_options():
    """Real-time bar request options should reach IB unchanged."""
    contract = ibi.Future(symbol="NQ", exchange="CME")
    options = [ibi.TagValue("test", "value")]
    streamer = RealTimeBarsStreamer(
        contract,
        whatToShow="TRADES",
        useRTH=False,
        realTimeBarsOptions=options,
    )
    streamer.ib.reqRealTimeBars = Mock(return_value=ibi.RealTimeBarList())

    streamer.streaming_func()

    streamer.ib.reqRealTimeBars.assert_called_once_with(
        contract,
        5,
        "TRADES",
        False,
        realTimeBarsOptions=options,
    )


def test_RealTimeBarsStreamer_validates_its_own_bar_schema():
    """Real-time validation should use RealTimeBar field names."""
    streamer = RealTimeBarsStreamer(
        ibi.Future(symbol="NQ", exchange="CME"),
        whatToShow="TRADES",
        useRTH=False,
    )
    bar = ibi.RealTimeBar(
        open_=20,
        high=21,
        low=19,
        close=20,
        wap=20,
    )
    bars = ibi.RealTimeBarList([bar])
    emitted: list[ibi.RealTimeBarList] = []
    streamer.dataEvent += emitted.append

    streamer.onUpdateEvent(bars, True)

    assert emitted == [bars]


def test_RealTimeBarsStreamer_rejects_historical_only_data_type():
    """Real-time bar requests should reject historical-only data types."""
    with pytest.raises(ValueError, match="Real-time bar whatToShow"):
        RealTimeBarsStreamer(
            ibi.Future(symbol="NQ", exchange="CME"),
            whatToShow="BID_ASK",
            useRTH=False,
        )


@pytest.mark.asyncio
async def test_MktDataStreamer_preserves_shared_ticker_listeners():
    """Starting a streamer should not clear another ticker listener."""
    contract = ibi.Future(symbol="NQ", exchange="CME")
    streamer = MktDataStreamer(contract, tickList="221")
    ticker = ibi.Ticker(contract=contract)
    existing_updates: list[ibi.Ticker] = []
    streamer_updates: list[ibi.Ticker] = []
    ticker.updateEvent += existing_updates.append
    streamer.dataEvent += streamer_updates.append
    streamer.streaming_func = Mock(return_value=ticker)  # type: ignore[method-assign]
    streamer._set_timeout = Mock()  # type: ignore[method-assign]
    streamer.ib.isConnected = Mock(return_value=True)

    task = asyncio.create_task(streamer.run())
    await asyncio.sleep(0)
    ticker.updateEvent.emit(ticker)

    assert existing_updates == [ticker]
    assert streamer_updates == [ticker]

    streamer.ib.disconnectedEvent.emit()
    await task
    ticker.updateEvent.emit(ticker)

    assert existing_updates == [ticker, ticker]
    assert streamer_updates == [ticker]
