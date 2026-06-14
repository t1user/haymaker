import asyncio
import importlib
import sys

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

    async def fake_producer(manager, queue):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            producer_cancelled.set()
            raise

    async def fake_worker(name, queue, ib):
        nonlocal started_workers, cancelled_workers
        started_workers += 1
        if started_workers == 2:
            workers_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled_workers += 1
            raise

    monkeypatch.setattr(dataloader, "NUMBER_OF_WORKERS", 2)
    monkeypatch.setattr(dataloader, "producer", fake_producer)
    monkeypatch.setattr(dataloader, "worker", fake_worker)

    task = asyncio.create_task(dataloader.main(object(), object()))
    await asyncio.wait_for(workers_started.wait(), timeout=1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)

    assert producer_cancelled.is_set()
    assert cancelled_workers == 2
