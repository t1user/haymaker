import ib_insync as ibi
import pytest

from ib_tools.streamers import HistoricalDataStreamer, Streamer


def test_Streamer_is_abstract():
    with pytest.raises(TypeError):
        Streamer()


def test_Streamer_keeps_instances():
    class ConcreteStreamer(Streamer):
        def streaming_func(self):
            pass

    s0 = ConcreteStreamer()
    s1 = ConcreteStreamer()
    assert Streamer.instances == [s0, s1]


def test_StreamerId():
    class ConcreteStreamer(Streamer):
        def streaming_func(self):
            pass

    s0 = ConcreteStreamer()
    s1 = ConcreteStreamer()
    s2 = ConcreteStreamer()
    s2.name = "my_streamer"
    assert s0.name == "ConcreteStreamer<0>"
    assert s1.name == "ConcreteStreamer<1>"
    assert s2.name == "my_streamer"


def test_StreamerId_dataclass():
    s0 = HistoricalDataStreamer(ibi.Contract(symbol="XXX"), "x", "x", "x")
    assert s0.name == "HistoricalDataStreamer<XXX>"
