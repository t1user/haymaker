"""Tests for generic and market-data event inactivity monitoring."""

from __future__ import annotations

import asyncio
import importlib.util
import logging
from datetime import datetime, timedelta, timezone

import eventkit as ev  # type: ignore
import ib_insync as ibi
import pytest
from helpers import wait_for_condition

import haymaker.components as components_package
from haymaker.base import Atom
from haymaker.components import EventTimeout, MarketDataTimeout
from haymaker.config import TimeoutPolicy
from haymaker.details_processor import Details


def test_timeout_policy_belongs_to_config_without_compatibility_module() -> None:
    """TimeoutPolicy is configuration, not a public trading component."""

    assert TimeoutPolicy.__module__ == "haymaker.config.settings"
    assert "TimeoutPolicy" not in components_package.__all__
    assert importlib.util.find_spec("haymaker.timeout") is None


@pytest.fixture(autouse=True)
def cancel_market_data_timeouts():
    """Keep workload-scoped timeout ownership isolated between tests."""

    yield
    MarketDataTimeout._cancel_all()


def test_event_timeout_validates_inputs() -> None:
    """Invalid generic timeout inputs should fail with actionable errors."""

    callback = lambda: None

    with pytest.raises(TypeError, match="eventkit.Event"):
        EventTimeout(object(), 0, callback=callback)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="real number"):
        EventTimeout(ev.Event(), True, callback=callback)
    with pytest.raises(ValueError, match="finite and non-negative"):
        EventTimeout(ev.Event(), -1, callback=callback)
    with pytest.raises(ValueError, match="finite and non-negative"):
        EventTimeout(ev.Event(), float("nan"), callback=callback)
    with pytest.raises(TypeError, match="callback"):
        EventTimeout(ev.Event(), 0, callback=None)  # type: ignore[arg-type]


def test_positive_event_timeout_requires_running_loop() -> None:
    """A scheduled generic timeout must bind to the loop that will run it."""

    with pytest.raises(RuntimeError, match="event loop"):
        EventTimeout(ev.Event(), 1, callback=lambda: None)


def test_zero_event_timeout_can_be_created_without_running_loop() -> None:
    """A disabled timeout may be composed before application startup."""

    timeout = EventTimeout(ev.Event(), 0, callback=lambda: None)

    assert not timeout.armed
    assert not timeout.cancelled
    assert str(timeout).startswith("EventTimeout<0s:<")

    timeout.cancel()


@pytest.mark.asyncio
async def test_event_emissions_restart_inactivity_interval() -> None:
    """Fresh data should move the deadline rather than create another timer."""

    source = ev.Event()
    fired = asyncio.Event()
    timeout = EventTimeout(source, 0.04, callback=fired.set, name="updates")

    await asyncio.sleep(0.025)
    source.emit("fresh")
    await asyncio.sleep(0.025)

    assert not fired.is_set()
    assert await wait_for_condition(fired.is_set)
    timeout.cancel()


@pytest.mark.asyncio
async def test_event_timeout_fires_once_until_source_recovers() -> None:
    """One stale episode should produce one callback and no repeated spam."""

    source = ev.Event()
    calls: list[str] = []
    timeout = EventTimeout(
        source,
        0.01,
        callback=lambda: calls.append("stale"),
        name="updates",
    )

    assert await wait_for_condition(lambda: calls == ["stale"])
    await asyncio.sleep(0.03)
    assert calls == ["stale"]
    assert timeout.triggered
    assert not timeout.armed

    source.emit("recovered")

    assert not timeout.triggered
    assert timeout.armed
    assert await wait_for_condition(lambda: calls == ["stale", "stale"])
    timeout.cancel()


@pytest.mark.asyncio
async def test_event_timeout_awaits_async_callback() -> None:
    """Asynchronous callbacks should execute on the active event loop."""

    callback_finished = asyncio.Event()

    async def callback() -> None:
        await asyncio.sleep(0)
        callback_finished.set()

    timeout = EventTimeout(ev.Event(), 0.01, callback=callback)

    assert await wait_for_condition(callback_finished.is_set)
    timeout.cancel()


@pytest.mark.asyncio
async def test_event_timeout_cancel_stops_running_async_callback() -> None:
    """Owner cancellation should also stop callback work still in progress."""

    callback_started = asyncio.Event()
    callback_cancelled = asyncio.Event()

    async def callback() -> None:
        callback_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            callback_cancelled.set()

    timeout = EventTimeout(ev.Event(), 0.01, callback=callback)
    assert await wait_for_condition(callback_started.is_set)

    timeout.cancel()

    assert await wait_for_condition(callback_cancelled.is_set)


@pytest.mark.asyncio
async def test_event_timeout_cancel_stops_deadline_and_disconnects() -> None:
    """Owner cancellation should prevent future callbacks and event rearming."""

    source = ev.Event()
    calls: list[str] = []
    timeout = EventTimeout(source, 0.02, callback=lambda: calls.append("stale"))

    timeout.cancel()
    source.emit("ignored")
    await asyncio.sleep(0.04)

    assert calls == []
    assert timeout.cancelled
    assert not timeout.armed


@pytest.mark.asyncio
async def test_source_completion_cancels_event_timeout() -> None:
    """An ended source should release its timeout automatically."""

    source = ev.Event()
    calls: list[str] = []
    timeout = EventTimeout(source, 0.02, callback=lambda: calls.append("stale"))

    source.set_done()
    await asyncio.sleep(0.04)

    assert timeout.cancelled
    assert calls == []


@pytest.mark.asyncio
async def test_market_timeout_cleanup_does_not_cancel_general_timeout() -> None:
    """Workload cleanup must leave user-owned general monitors alone."""

    fired = asyncio.Event()
    timeout = EventTimeout(ev.Event(), 0.01, callback=fired.set)

    MarketDataTimeout._cancel_all()

    assert await wait_for_condition(fired.is_set)
    assert not timeout.cancelled
    timeout.cancel()


def _atom_with_details(details: ibi.ContractDetails) -> Atom:
    atom = Atom()
    assert details.contract is not None
    atom.contract = details.contract
    return atom


def test_market_data_timeout_created_from_atom(
    atom_runtime, details: ibi.ContractDetails
) -> None:
    """Atom construction should supply policy, details, restart, and naming."""

    atom_runtime.timeout_policy = TimeoutPolicy(seconds=0, action="restart")
    atom = _atom_with_details(details)

    timeout = MarketDataTimeout.from_atom(atom, ev.Event(), "ticks")

    assert timeout.seconds == 0
    assert timeout.details is atom_runtime.contract_registry.get_details(
        details.contract
    )
    assert timeout.request_restart == atom_runtime.request_restart
    assert "ticks" in timeout.name
    assert str(atom) in timeout.name


def test_market_data_timeout_from_atom_requires_details(atom_runtime) -> None:
    """A contractless Atom cannot supply market-session behavior."""

    with pytest.raises(ValueError, match="Contract details"):
        MarketDataTimeout.from_atom(Atom(), ev.Event(), "ticks")


def test_restart_market_timeout_requires_bound_supervisor(
    atom_runtime, details: ibi.ContractDetails
) -> None:
    """Positive restart mode cannot be created before supervisor binding."""

    atom_runtime.timeout_policy = TimeoutPolicy(seconds=1, action="restart")
    atom_runtime.request_restart = None
    atom = _atom_with_details(details)

    with pytest.raises(RuntimeError, match="supervisor"):
        MarketDataTimeout.from_atom(atom, ev.Event(), "ticks")


@pytest.mark.asyncio
async def test_open_market_timeout_requests_one_restart_even_when_rejected(
    details: ibi.ContractDetails,
) -> None:
    """A rejected request already means lifecycle work is in progress."""

    source = ev.Event()
    reasons: list[str] = []

    def reject_restart(reason: str) -> bool:
        reasons.append(reason)
        return False

    timeout = MarketDataTimeout(
        source,
        0.01,
        details=Details(details),
        request_restart=reject_restart,
        name="quotes",
    )
    timeout._now = datetime(2024, 3, 4, 14, 0, tzinfo=timezone.utc)

    assert await wait_for_condition(lambda: len(reasons) == 1)
    source.emit("late data")
    await asyncio.sleep(0.03)

    assert len(reasons) == 1
    assert timeout.triggered
    assert not timeout.armed


@pytest.mark.asyncio
async def test_log_only_market_timeout_rearms_after_fresh_data(
    caplog, details: ibi.ContractDetails
) -> None:
    """Log mode should report once per stale-data episode."""

    source = ev.Event()
    with caplog.at_level(logging.ERROR, logger="haymaker.components.timeouts"):
        timeout = MarketDataTimeout(
            source,
            0.01,
            details=Details(details),
            log_only=True,
            name="quotes",
        )
        timeout._now = datetime(2024, 3, 4, 14, 0, tzinfo=timezone.utc)

        assert await wait_for_condition(
            lambda: sum("market data may be stale" in r.message for r in caplog.records)
            == 1
        )
        await asyncio.sleep(0.02)
        assert sum("market data may be stale" in r.message for r in caplog.records) == 1

        source.emit("recovered")

        assert await wait_for_condition(
            lambda: sum("market data may be stale" in r.message for r in caplog.records)
            == 2
        )
        timeout.cancel()


@pytest.mark.asyncio
async def test_closed_market_timeout_pauses_until_session_open(
    details: ibi.ContractDetails,
) -> None:
    """Closed sessions should not log stale data or request a restart."""

    reasons: list[str] = []
    timeout = MarketDataTimeout(
        ev.Event(),
        0.01,
        details=Details(details),
        request_restart=lambda reason: reasons.append(reason),
        name="closed",
    )
    timeout._now = datetime(2024, 3, 4, 22, 0, tzinfo=timezone.utc)

    assert await wait_for_condition(lambda: timeout.triggered)

    assert reasons == []
    assert not timeout.armed
    assert not timeout.cancelled

    timeout.cancel()


@pytest.mark.asyncio
async def test_closed_market_timeout_arms_full_interval_after_reopen() -> None:
    """Reopening should begin a fresh interval rather than fire immediately."""

    class ReopeningDetails(Details):
        def __init__(self) -> None:
            pass

        def is_open(self, _now: datetime | None = None) -> bool:
            return False

        def next_open(self, _now: datetime | None = None) -> datetime | None:
            assert _now is not None
            return _now + timedelta(seconds=0.02)

    timeout = MarketDataTimeout(
        ev.Event(),
        0.04,
        details=ReopeningDetails(),
        log_only=True,
        name="reopening",
    )
    timeout._now = datetime(2024, 3, 4, 22, 0, tzinfo=timezone.utc)

    assert await wait_for_condition(lambda: timeout.triggered)
    assert await wait_for_condition(lambda: timeout.armed and not timeout.triggered)
    timeout.cancel()


@pytest.mark.asyncio
async def test_workload_cleanup_stops_armed_and_market_reopen_timeouts(
    details: ibi.ContractDetails,
) -> None:
    """Workload cleanup must cancel every timer and pending reopen wait."""

    open_timeout = MarketDataTimeout(
        ev.Event(),
        0.05,
        details=Details(details),
        log_only=True,
    )
    open_timeout._now = datetime(2024, 3, 4, 14, 0, tzinfo=timezone.utc)
    closed_timeout = MarketDataTimeout(
        ev.Event(),
        0.01,
        details=Details(details),
        log_only=True,
    )
    closed_timeout._now = datetime(2024, 3, 4, 22, 0, tzinfo=timezone.utc)
    assert await wait_for_condition(lambda: closed_timeout.triggered)

    MarketDataTimeout._cancel_all()
    await asyncio.sleep(0.06)

    assert open_timeout.cancelled
    assert closed_timeout.cancelled
    assert not open_timeout.armed
    assert not closed_timeout.armed
    assert MarketDataTimeout._instances == []
