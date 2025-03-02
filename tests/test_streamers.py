import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Type

import eventkit as ev  # type: ignore
import ib_insync as ibi
import pytest
from helpers import wait_for_condition

from haymaker.base import Atom
from haymaker.streamers import HistoricalDataStreamer, Streamer, Timeout


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
    """This tests only container, not instantiation of timers."""

    class ConcreteStreamer(Streamer):
        def streaming_func(self):
            pass

    s = ConcreteStreamer()

    s.timers["bar"] = "xxx"  # type: ignore

    assert s.timers["bar"] == "xxx"


def test_timer_containers_from_within():
    """This tests only container, not instantiation of timers."""

    class ConcreteStreamer(Streamer):
        def __init__(self):
            super().__init__()
            self.process_timer()

        def streaming_func(self):
            pass

        def process_timer(self):
            self.timers["bar"] = "xxx"  # type: ignore

    s = ConcreteStreamer()

    assert s.timers == {"bar": "xxx"}


def test_timer_containers_single_for_all_streamers():
    """This tests only container, not instantiation of timers."""

    class ConcreteStreamer(Streamer):
        def __init__(self):
            super().__init__()
            self.process_timer()

        def streaming_func(self):
            pass

        def process_timer(self):
            self.timers["bar"] = "xxx"  # type: ignore

    s1 = ConcreteStreamer()
    s2 = ConcreteStreamer()

    assert s1._timers is s2._timers


def test_timer_cannot_be_overriden():
    class ConcreteStreamer(Streamer):
        def __init__(self):
            super().__init__()
            self.process_timer()

        def streaming_func(self):
            pass

        def process_timer(self):
            self.timers = "xxx"  # type: ignore

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
async def test_timer_not_triggered(
    timeout: type[Timeout], Atom: type[Atom], details: ibi.ContractDetails
):
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
    assert not t.triggered  # type: ignore


@pytest.mark.asyncio
async def test_timer_triggered(
    timeout: type[Timeout], Atom: Atom, details: ibi.ContractDetails
):
    t = timeout(
        time=0.1,
        event=ev.Event(),
        ib=ibi.IB(),
        details=Atom.contract_details[details.contract],
        name="xxx",
        debug=True,
        _now=datetime(2024, 3, 4, 14, 00, tzinfo=timezone.utc),
    )
    assert await wait_for_condition(lambda: t.triggered)  # type: ignore
