import importlib
from datetime import datetime, timedelta
from unittest.mock import patch

import ib_insync as ibi
import pytest

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
