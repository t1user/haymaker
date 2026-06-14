import asyncio
import importlib
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def dataloader_module(monkeypatch):
    """Import dataloader with minimal hermetic config for lifecycle tests."""

    from haymaker.config import CONFIG
    import haymaker.logging as logging_package

    class FakeStore:
        def read(self, contract):
            return None

        def write(self, contract, data):
            return "version"

    config_values = {
        "logging_config": None,
        "barSize": "30 secs",
        "wts": "TRADES",
        "max_bars": 100_000,
        "fill_gaps": False,
        "auto_save_interval": 0,
        "number_of_workers": 2,
        "datastore": FakeStore(),
        "run_mode": "reconnect",
        "source": "contracts.csv",
        "pacer_no_restriction": False,
        "pacer_restrictions": [(5, 2)],
        "max_period": 120,
    }
    for key, value in config_values.items():
        monkeypatch.setitem(CONFIG.maps[0], key, value)
    monkeypatch.setattr(logging_package, "setup_logging", lambda config: None)
    sys.modules.pop("haymaker.dataloader.dataloader", None)
    return importlib.import_module("haymaker.dataloader.dataloader")


@pytest.mark.asyncio
async def test_main_cleans_up_producer_and_workers_on_cancellation(
    monkeypatch, dataloader_module
):
    """Cancelling a managed run should not leave producer or worker tasks alive."""

    dataloader = dataloader_module
    started_workers = 0
    cancelled_workers = 0
    producer_cancelled = asyncio.Event()
    workers_started = asyncio.Event()

    async def fake_producer(self, queue):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            producer_cancelled.set()
            raise

    async def fake_worker(self, name, queue):
        nonlocal started_workers, cancelled_workers
        started_workers += 1
        if started_workers == 2:
            workers_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled_workers += 1
            raise

    monkeypatch.setattr(dataloader.DataloaderSession, "producer", fake_producer)
    monkeypatch.setattr(dataloader.DataloaderSession, "worker", fake_worker)

    task = asyncio.create_task(
        dataloader.DataloaderSession(object(), number_of_workers=2).run()
    )
    await asyncio.wait_for(workers_started.wait(), timeout=1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)

    assert producer_cancelled.is_set()
    assert cancelled_workers == 2


def test_sessions_own_separate_pacing_state(dataloader_module):
    """Separate sessions should not share pacing violation or limiter state."""

    dataloader = dataloader_module
    first = dataloader.DataloaderSession(object())
    second = dataloader.DataloaderSession(object())
    contract = object()

    first.pacing.registry.data.add(contract)

    assert first.pacing.verify(contract)
    assert not second.pacing.verify(contract)
    assert first.pacing.limiter.restrictions[0].holder is not (
        second.pacing.limiter.restrictions[0].holder
    )


@pytest.mark.asyncio
async def test_session_producer_requeues_active_writers_before_new_discovery(
    dataloader_module,
):
    """Restarted sessions should resume active writers before new discovery."""

    dataloader = dataloader_module

    class FakeWriter:
        def __init__(self, name):
            self.name = name

        def is_done(self):
            return False

    async def new_writers():
        yield new_writer

    active_writer = FakeWriter("active")
    new_writer = FakeWriter("new")
    manager = SimpleNamespace(
        ib=None,
        active_writers=[active_writer],
        new_writer_generator=new_writers(),
        pacing=dataloader.request_pacing_factory(),
    )
    session = dataloader.DataloaderSession(object(), manager=manager)
    queue = asyncio.Queue()

    await session.producer(queue)

    assert queue.get_nowait() is active_writer
    assert queue.get_nowait() is new_writer
