import asyncio
import importlib
import sys
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest


@pytest.fixture
def dataloader_module(monkeypatch):
    """Import dataloader with minimal hermetic config for lifecycle tests."""

    from haymaker.config import CONFIG
    import haymaker.logging as logging_package

    config_values = {
        "logging_config": None,
        "barSize": "30 secs",
        "wts": "TRADES",
        "max_bars": 100_000,
        "fill_gaps": False,
        "auto_save_interval": 0,
        "number_of_workers": 2,
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
async def test_session_producer_requeues_active_jobs_before_new_discovery(
    dataloader_module,
):
    """Restarted sessions should resume active jobs before new discovery."""

    dataloader = dataloader_module

    class FakeJob:
        def __init__(self, name):
            self.name = name

        def is_done(self):
            return False

    async def new_jobs():
        yield new_job

    active_job = FakeJob("active")
    new_job = FakeJob("new")
    manager = SimpleNamespace(
        ib=None,
        active_jobs=[active_job],
        new_job_generator=new_jobs(),
        pacing=dataloader.request_pacing_factory(object()),
    )
    session = dataloader.DataloaderSession(object(), manager=manager)
    queue = asyncio.Queue()

    await session.producer(queue)

    assert queue.get_nowait() is active_job
    assert queue.get_nowait() is new_job


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


def test_download_container_rejects_mixed_intraday_points(dataloader_module):
    """Intraday download ranges should fail before mixed-type comparisons."""

    dataloader = dataloader_module

    with pytest.raises(TypeError, match="Intraday bars require datetime"):
        dataloader.DownloadContainer(
            date(2025, 1, 1),
            datetime(2025, 1, 2, tzinfo=timezone.utc),
            bar_size="30 secs",
        )


def test_download_container_rejects_naive_intraday_datetime(dataloader_module):
    """Intraday download ranges should never use naive datetimes."""

    dataloader = dataloader_module

    with pytest.raises(ValueError, match="timezone-aware"):
        dataloader.DownloadContainer(
            datetime(2025, 1, 1),
            datetime(2025, 1, 2, tzinfo=timezone.utc),
            bar_size="30 secs",
        )


def test_daily_download_job_uses_date_end_datetime(dataloader_module):
    """Daily-like bars should pass date endDateTime values to IB."""

    dataloader = dataloader_module

    class FakeSink:
        async def write(self, data):
            return "version"

    container = dataloader.DownloadContainer(
        datetime(2025, 1, 1, tzinfo=timezone.utc),
        datetime(2025, 1, 5, tzinfo=timezone.utc),
        bar_size="1 day",
    )
    job = dataloader.DownloadJob(
        FakeContract(),
        sink=FakeSink(),
        queue=[container],
        bar_size="1 day",
    )

    assert job.params["endDateTime"] == date(2025, 1, 5)


@pytest.mark.asyncio
async def test_download_job_marks_backfill_exhausted_on_empty_backfill(
    dataloader_module,
):
    """Empty backfill chunks should mark older history as exhausted."""

    dataloader = dataloader_module

    class FakeSink:
        def __init__(self):
            self.marked = 0

        async def write(self, data):
            return "version"

        async def mark_backfill_exhausted(self):
            self.marked += 1

    sink = FakeSink()
    job = dataloader.DownloadJob(
        FakeContract(),
        sink=sink,
        queue=[
            dataloader.DownloadContainer(
                datetime(2025, 1, 1, tzinfo=timezone.utc),
                datetime(2025, 1, 2, tzinfo=timezone.utc),
                bar_size="30 secs",
                kind="backfill",
            )
        ],
        bar_size="30 secs",
    )

    await job.save_chunk([])

    assert sink.marked == 1
    assert job.is_done()


@pytest.mark.asyncio
async def test_download_job_does_not_mark_update_or_gap_exhausted(
    dataloader_module,
):
    """Only backfill misses should persist the exhaustion marker."""

    dataloader = dataloader_module

    class FakeSink:
        def __init__(self):
            self.marked = 0

        async def write(self, data):
            return "version"

        async def mark_backfill_exhausted(self):
            self.marked += 1

    for kind in ("update", "gap"):
        sink = FakeSink()
        job = dataloader.DownloadJob(
            FakeContract(),
            sink=sink,
            queue=[
                dataloader.DownloadContainer(
                    datetime(2025, 1, 1, tzinfo=timezone.utc),
                    datetime(2025, 1, 2, tzinfo=timezone.utc),
                    bar_size="30 secs",
                    kind=kind,
                )
            ],
            bar_size="30 secs",
        )

        await job.save_chunk([])

        assert sink.marked == 0
        assert job.is_done()


def test_validate_age_uses_run_scoped_now(dataloader_module):
    """Age validation should use the run snapshot supplied by the caller."""

    dataloader = dataloader_module

    class FakeJob:
        bar_size = "1 secs"
        next_date = datetime(2025, 1, 1, tzinfo=timezone.utc)

    assert not dataloader.validate_age(
        FakeJob(),
        FakeJob.next_date + timedelta(days=181),
    )
    assert dataloader.validate_age(
        FakeJob(),
        FakeJob.next_date + timedelta(days=1),
    )


def test_manager_syncs_injected_pacing_to_request_policy(dataloader_module):
    """Injected pacing should not keep stale request-policy metadata."""

    dataloader = dataloader_module
    pacing = dataloader.request_pacing_factory(
        object(), bar_size="30 secs", wts="TRADES"
    )

    manager = dataloader.Manager(
        object(),
        pacing=pacing,
        bar_size="1 day",
        wts="MIDPOINT",
    )

    assert manager.pacing is pacing
    assert pacing.bar_size == "1 day"
    assert pacing.wts == "MIDPOINT"


@pytest.mark.asyncio
async def test_manager_policy_flows_to_store_and_download_job(
    monkeypatch, dataloader_module
):
    """Manager request policy should reach store naming and generated jobs."""

    dataloader = dataloader_module
    contract = FakeContract()
    created_stores = []

    class FakeStore:
        async def read(self, requested_contract):
            assert requested_contract is contract
            return None

        async def read_metadata(self, requested_contract):
            assert requested_contract is contract
            return {}

    class FakeSink:
        async def write(self, data):
            return "version"

        async def mark_backfill_exhausted(self):
            return None

    async def fake_headstamps(self):
        yield contract, datetime(2025, 1, 1, tzinfo=timezone.utc)

    def fake_store(wts, bar_size):
        created_stores.append((wts, bar_size))
        return FakeStore()

    monkeypatch.setattr(dataloader.Manager, "headstamps", fake_headstamps)
    monkeypatch.setattr(dataloader, "create_dataloader_store", fake_store)
    monkeypatch.setattr(dataloader, "HistorySink", lambda contract, store: FakeSink())

    manager = dataloader.Manager(
        object(),
        bar_size="1 day",
        wts="MIDPOINT",
        now=datetime(2025, 1, 10, tzinfo=timezone.utc),
        max_period=120,
        max_bars=12,
    )
    job = await anext(manager._job_generator())

    assert created_stores == [("MIDPOINT", "1 day")]
    assert manager.pacing.bar_size == "1 day"
    assert manager.pacing.wts == "MIDPOINT"
    assert job.bar_size == "1 day"
    assert job.max_bars == 12
    assert job.params["endDateTime"] == date(2025, 1, 10)


class FakeContract:
    symbol = "ES"
    localSymbol = "ESZ5"
    lastTradeDateOrContractMonth = ""


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
    """Connection loss should record the failed job and keep the queue moving."""

    dataloader = dataloader_module

    class FakeIB:
        async def reqHistoricalDataAsync(self, *args, **kwargs):
            raise ConnectionError("socket down")

    class FakeJob:
        contract = FakeContract()
        next_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
        bar_size = dataloader.BARSIZE
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
            return "<FakeJob>"

    session = dataloader.DataloaderSession(FakeIB())
    session.pacing.no_restriction = True
    queue = asyncio.Queue()
    job = FakeJob()
    await queue.put(job)

    task = asyncio.create_task(session.worker("worker", queue))
    await asyncio.wait_for(queue.join(), timeout=1)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert not job.saved
    assert len(session.failures.failures) == 1
    assert isinstance(session.failures.failures[0].error, ConnectionError)


@pytest.mark.asyncio
async def test_worker_failure_does_not_stop_next_job(dataloader_module):
    """A failed job should be recorded without stopping the worker."""

    dataloader = dataloader_module

    class FakeIB:
        async def reqHistoricalDataAsync(self, *args, **kwargs):
            return []

    class FakeJob:
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
                raise RuntimeError("job failed")
            self.saved = True
            self.next_date = None

        def __str__(self):
            return f"<FakeJob {self.name}>"

    session = dataloader.DataloaderSession(FakeIB())
    session.pacing.no_restriction = True
    queue = asyncio.Queue()
    failed_job = FakeJob("failed", fail=True)
    good_job = FakeJob("good")
    await queue.put(failed_job)
    await queue.put(good_job)

    task = asyncio.create_task(session.worker("worker", queue))
    await asyncio.wait_for(queue.join(), timeout=1)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert not failed_job.saved
    assert good_job.saved
    assert len(session.failures.failures) == 1
    assert session.failures.failures[0].job is failed_job
    assert isinstance(session.failures.failures[0].error, RuntimeError)


@pytest.mark.asyncio
async def test_worker_pacing_violation_retries_same_job(
    dataloader_module,
):
    """Pacing retry should preserve job state until a non-pacing result arrives."""

    dataloader = dataloader_module
    contract = FakeContract()

    class FakeJob:
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
            return "<FakeJob>"

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
    job = FakeJob()
    await queue.put(job)

    task = asyncio.create_task(session.worker("worker", queue))
    await asyncio.wait_for(queue.join(), timeout=1)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert ib.requests == 2
    assert job.saved_chunks == [[]]


@pytest.mark.asyncio
async def test_worker_hardcodes_historical_format_date_two(dataloader_module):
    """Dataloader workers should keep the UTC-aware formatDate=2 policy."""

    dataloader = dataloader_module

    class FakeIB:
        def __init__(self):
            self.kwargs = None

        async def reqHistoricalDataAsync(self, *args, **kwargs):
            self.kwargs = {
                "endDateTime": args[1],
                "durationStr": args[2],
                "barSizeSetting": args[3],
                "whatToShow": args[4],
                "useRTH": args[5],
                "formatDate": args[6],
            }
            return []

    class FakeJob:
        contract = FakeContract()
        next_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
        bar_size = "1 day"

        @property
        def params(self):
            return {
                "contract": self.contract,
                "endDateTime": self.next_date,
                "durationStr": "1 D",
            }

        async def save_chunk(self, chunk):
            self.next_date = None

        def __str__(self):
            return "<FakeJob>"

    ib = FakeIB()
    manager = dataloader.Manager(ib, bar_size="30 secs", wts="MIDPOINT")
    session = dataloader.DataloaderSession(ib, manager=manager)
    session.pacing.no_restriction = True
    queue = asyncio.Queue()
    await queue.put(FakeJob())

    task = asyncio.create_task(session.worker("worker", queue))
    await asyncio.wait_for(queue.join(), timeout=1)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert ib.kwargs["formatDate"] == 2
    assert ib.kwargs["barSizeSetting"] == "1 day"
    assert ib.kwargs["whatToShow"] == "MIDPOINT"
