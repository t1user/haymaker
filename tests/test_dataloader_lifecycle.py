import asyncio
import importlib
import sys
from datetime import datetime, timezone
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
        "clientId": 1,
        "source": "contracts.csv",
        "pacer_no_restriction": False,
        "pacer_allowance_fraction": 1.0,
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

    task = asyncio.create_task(dataloader.DataloaderSession(object()).run())
    await asyncio.wait_for(workers_started.wait(), timeout=1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)

    assert producer_cancelled.is_set()
    assert cancelled_workers == 2


def test_sessions_own_separate_pacing_state(dataloader_module):
    """Separate sessions should not share pacing violation or restriction state."""

    dataloader = dataloader_module
    first = dataloader.DataloaderSession(object())
    second = dataloader.DataloaderSession(object())
    contract = object()

    first.pacing.registry.register(1, 162, "pacing violation", contract)

    assert not second.pacing.verify(contract)
    assert first.pacing.verify(contract)
    assert first.pacing.historical.rules[0].history is not (
        second.pacing.historical.rules[0].history
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


def test_dataloader_config_validates_pacer_allowance_fraction(
    monkeypatch,
    dataloader_module,
):
    """Pacing scale should come from config and allow values above one."""

    from haymaker.config import CONFIG

    assert dataloader_module.PACER_ALLOWANCE_FRACTION == 1.0

    monkeypatch.setitem(CONFIG.maps[0], "pacer_allowance_fraction", 2)
    sys.modules.pop("haymaker.dataloader.dataloader", None)
    assert (
        importlib.import_module(
            "haymaker.dataloader.dataloader"
        ).PACER_ALLOWANCE_FRACTION
        == 2
    )

    monkeypatch.setitem(CONFIG.maps[0], "pacer_allowance_fraction", 0)
    sys.modules.pop("haymaker.dataloader.dataloader", None)
    with pytest.raises(ValueError, match="pacer_allowance_fraction"):
        importlib.import_module("haymaker.dataloader.dataloader")


class FakeContract:
    symbol = "ES"
    localSymbol = "ESZ5"


@pytest.mark.asyncio
async def test_headstamp_unexpected_request_error_propagates(dataloader_module):
    """Unexpected headstamp errors should not be swallowed in an infinite loop."""

    dataloader = dataloader_module

    class FakeIB:
        async def reqHeadTimeStampAsync(self, *args, **kwargs):
            raise RuntimeError("bad headstamp")

    manager = dataloader.Manager(FakeIB())

    with pytest.raises(RuntimeError, match="bad headstamp"):
        await manager.headstamp(FakeContract())


@pytest.mark.asyncio
async def test_headstamp_retries_pacing_violation(dataloader_module):
    """Pacing violations should retry the same headstamp request."""

    dataloader = dataloader_module
    contract = FakeContract()

    class FakeIB:
        def __init__(self):
            self.requests = 0

        async def reqHeadTimeStampAsync(self, *args, **kwargs):
            self.requests += 1
            if self.requests == 1:
                manager.pacing.registry.register(1, 162, "pacing violation", contract)
                return None
            return datetime(2025, 1, 1, tzinfo=timezone.utc)

    ib = FakeIB()
    manager = dataloader.Manager(ib)
    manager.pacing.no_restriction = True
    manager.pacing.pacing_retry_delay = 0

    assert await manager.headstamp(contract) == datetime(
        2025, 1, 1, tzinfo=timezone.utc
    )
    assert ib.requests == 2


@pytest.mark.asyncio
async def test_worker_connection_loss_is_recorded(dataloader_module):
    """Connection loss should record the failed writer and keep the queue moving."""

    dataloader = dataloader_module

    class FakeIB:
        async def reqHistoricalDataAsync(self, *args, **kwargs):
            raise ConnectionError("socket down")

    class FakeWriter:
        contract = FakeContract()
        next_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
        saved = False

        @property
        def params(self):
            return {
                "contract": self.contract,
                "endDateTime": self.next_date,
                "durationStr": "1 D",
            }

        async def save_chunk(self, chunk):
            self.saved = True

        def __str__(self):
            return "<FakeWriter>"

    session = dataloader.DataloaderSession(FakeIB())
    session.pacing.no_restriction = True
    queue = asyncio.Queue()
    writer = FakeWriter()
    await queue.put(writer)

    task = asyncio.create_task(session.worker("worker", queue))
    await asyncio.wait_for(queue.join(), timeout=1)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert not writer.saved
    assert len(session.failures.failures) == 1
    assert isinstance(session.failures.failures[0].error, ConnectionError)


@pytest.mark.asyncio
async def test_worker_failure_does_not_stop_next_writer(dataloader_module):
    """A failed writer should be recorded without stopping the worker."""

    dataloader = dataloader_module

    class FakeIB:
        async def reqHistoricalDataAsync(self, *args, **kwargs):
            return []

    class FakeWriter:
        contract = FakeContract()
        next_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
        bar_size = dataloader.BARSIZE

        def __init__(self, name, *, fail=False):
            self.name = name
            self.fail = fail
            self.saved = False

        @property
        def params(self):
            return {
                "contract": self.contract,
                "endDateTime": self.next_date,
                "durationStr": "1 D",
            }

        async def save_chunk(self, chunk):
            if self.fail:
                raise RuntimeError("writer failed")
            self.saved = True
            self.next_date = None

        def __str__(self):
            return f"<FakeWriter {self.name}>"

    session = dataloader.DataloaderSession(FakeIB())
    session.pacing.no_restriction = True
    queue = asyncio.Queue()
    failed_writer = FakeWriter("failed", fail=True)
    good_writer = FakeWriter("good")
    await queue.put(failed_writer)
    await queue.put(good_writer)

    task = asyncio.create_task(session.worker("worker", queue))
    await asyncio.wait_for(queue.join(), timeout=1)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert not failed_writer.saved
    assert good_writer.saved
    assert len(session.failures.failures) == 1
    assert session.failures.failures[0].writer is failed_writer
    assert isinstance(session.failures.failures[0].error, RuntimeError)


@pytest.mark.asyncio
async def test_worker_pacing_violation_retries_same_writer(
    dataloader_module,
):
    """Pacing retry should preserve writer state until a non-pacing result arrives."""

    dataloader = dataloader_module
    contract = FakeContract()

    class FakeWriter:
        def __init__(self):
            self.contract = contract
            self.next_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
            self.saved_chunks = []
            self.bar_size = dataloader.BARSIZE

        @property
        def params(self):
            return {
                "contract": self.contract,
                "endDateTime": self.next_date,
                "durationStr": "1 D",
            }

        async def save_chunk(self, chunk):
            self.saved_chunks.append(chunk)
            self.next_date = None

        def __str__(self):
            return "<FakeWriter>"

    class FakeIB:
        def __init__(self):
            self.requests = 0

        async def reqHistoricalDataAsync(self, *args, **kwargs):
            self.requests += 1
            if self.requests == 1:
                session.pacing.registry.register(1, 162, "pacing violation", contract)
            return []

    ib = FakeIB()
    session = dataloader.DataloaderSession(ib)
    session.pacing.no_restriction = True
    session.pacing.pacing_retry_delay = 0
    queue = asyncio.Queue()
    writer = FakeWriter()
    await queue.put(writer)

    task = asyncio.create_task(session.worker("worker", queue))
    await asyncio.wait_for(queue.join(), timeout=1)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert ib.requests == 2
    assert writer.saved_chunks == [[]]
