import importlib

import ib_insync as ibi
import pytest

from haymaker.streamers import HistoricalDataStreamer, Streamer


def test_Streamer_is_abstract():
    with pytest.raises(TypeError):
        Streamer()  # type: ignore


def test_Streamer_keeps_instances():
    class ConcreteStreamer(Streamer):
        def streaming_func(self):
            pass

    s0 = ConcreteStreamer()
    s1 = ConcreteStreamer()
    assert Streamer.instances == [s0, s1]


def test_StreamerId():
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
