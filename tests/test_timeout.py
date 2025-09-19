import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Type

import eventkit as ev  # type: ignore
import ib_insync as ibi
import pytest
from helpers import wait_for_condition

from haymaker.base import Atom, Details
from haymaker.timeout import Timeout as _Timeout


@pytest.fixture
def Timeout():
    yield _Timeout
    _Timeout.instances = []


def test_all_timouts_stored(Timeout):
    t1 = Timeout(ev.Event(), 0.1, "mytimeout1")
    t2 = Timeout(ev.Event(), 0.2, "mytimout2")
    assert Timeout.instances == [t1, t2]


def test_all_timouts_cleared_on_reset(Timeout):
    Timeout(ev.Event(), 0.1, "mytimeout1")
    Timeout(ev.Event(), 0.2, "mytimout2")
    assert len(Timeout.instances) == 2
    Timeout.reset()
    assert Timeout.instances == []


def test_timeout_created_from_atom(Timeout):

    class FakeDetails(Details):
        def __post_init__(self):
            pass

    class MyAtom(Atom):
        def __str__(self):
            return "MyAtom"

        @property
        def details(self):
            return FakeDetails(ibi.ContractDetails())

    a = MyAtom()

    t = Timeout.from_atom(a, ev.Event(), "my_key")
    assert t
    assert "my_key" in str(t)
    assert "MyAtom" in str(t)


def test_timeout_from_atom_raises_when_no_details(Timeout):
    class MyAtom(Atom):
        def __str__(self):
            return "MyAtom"

    a = MyAtom()

    with pytest.raises(AssertionError):
        Timeout.from_atom(a, ev.Event(), "my_key")


def test_timeout_raises_when_no_event(Timeout):
    with pytest.raises(AssertionError):
        Timeout("some random object")


def test_timeout_with_no_name_gets_a_number(Timeout):
    t0 = Timeout(ev.Event(), 0.1)
    t1 = Timeout(ev.Event(), 0.2)
    assert str(t0).startswith("Timeout <0.1s> for <0>  event id:")
    assert str(t1).startswith("Timeout <0.2s> for <1>  event id:")


@pytest.fixture
def timeout(Timeout) -> Type[_Timeout]:
    @dataclass
    class TimerForTesting(Timeout):
        triggered: bool = False

        def triggered_action(self) -> None:
            self.triggered = True

    return TimerForTesting


@pytest.mark.asyncio
async def test_timer_not_triggered(
    timeout: type[_Timeout], Atom: type[Atom], details: ibi.ContractDetails
):
    t = timeout(
        event=ev.Event(),
        time=0.15,
        name="xxx",
        details=Atom.contract_details[details.contract],
        debug=True,
        _now=datetime(2024, 3, 4, 14, 00, tzinfo=timezone.utc),
    )

    await asyncio.sleep(0.1)
    assert not t.triggered  # type: ignore


@pytest.mark.asyncio
async def test_timer_triggered(
    timeout: type[_Timeout], Atom: Atom, details: ibi.ContractDetails
):
    t = timeout(
        event=ev.Event(),
        time=0.1,
        name="xxx",
        details=Atom.contract_details[details.contract],
        debug=True,
        _now=datetime(2024, 3, 4, 14, 00, tzinfo=timezone.utc),
    )
    assert await wait_for_condition(lambda: t.triggered)  # type: ignore


@pytest.mark.asyncio
async def test_timer_works_with_no_details(
    timeout: type[_Timeout], Atom: Atom, details: ibi.ContractDetails
):
    t = timeout(
        event=ev.Event(),
        time=0.1,
        debug=True,
        _now=datetime(2024, 3, 4, 14, 00, tzinfo=timezone.utc),
    )
    assert await wait_for_condition(lambda: t.triggered)  # type: ignore
