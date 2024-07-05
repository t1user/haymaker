import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Type

import eventkit as ev  # type: ignore
import ib_insync as ibi
import pytest

from haymaker.streamers import HistoricalDataStreamer, Streamer, Timeout


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


def test_timer_containers():
    class ConcreteStreamer(Streamer):
        def streaming_func(self):
            pass

    s = ConcreteStreamer()

    s.timers["bar"] = "xxx"

    assert s.timers["bar"] == "xxx"


def test_timer_containers_from_within():
    class ConcreteStreamer(Streamer):
        def __init__(self):
            super().__init__()
            self.process_timer()

        def streaming_func(self):
            pass

        def process_timer(self):
            self.timers["bar"] = "xxx"

    s = ConcreteStreamer()

    assert s.timers == {"bar": "xxx"}


def test_timer_cannot_be_overriden():
    class ConcreteStreamer(Streamer):
        def __init__(self):
            super().__init__()
            self.process_timer()

        def streaming_func(self):
            pass

        def process_timer(self):
            self.timers = "xxx"

    with pytest.raises(ValueError):
        ConcreteStreamer()


@pytest.fixture
def timeout() -> Type[Timeout]:
    @dataclass
    class TimerForTesting(Timeout):
        triggered: bool = False

        def triggered_action(self) -> None:
            self.triggered = True

    return TimerForTesting


@pytest.mark.asyncio
async def test_timer_not_triggered(timeout, Atom, details):
    t = timeout(
        time=0.15,
        event=ev.Event(),
        ib=ibi.IB(),
        details=Atom.contract_details[details.contract],
        name="xxx",
        debug=True,
        _now=datetime(2024, 3, 4, 14, 00, tzinfo=timezone.utc),
    )

    await asyncio.sleep(0.1)
    assert not t.triggered


@pytest.mark.asyncio
async def test_timer_triggered(timeout, Atom, details):
    t = timeout(
        time=0.1,
        event=ev.Event(),
        ib=ibi.IB(),
        details=Atom.contract_details[details.contract],
        name="xxx",
        debug=True,
        _now=datetime(2024, 3, 4, 14, 00, tzinfo=timezone.utc),
    )
    await asyncio.sleep(0.2)
    assert t.triggered
