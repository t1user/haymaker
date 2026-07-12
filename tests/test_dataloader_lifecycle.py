import asyncio
import importlib
import sys
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import ib_insync as ibi
import pandas as pd
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
        "gap_fill_mode": "off",
        "useRTH": False,
        "auto_save_interval": 0,
        "number_of_workers": 2,
        "clientId": 1,
        "source": "contracts.csv",
        "pacer_no_restriction": False,
        "pacer_allowance_fraction": 1.0,
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
        pacing=dataloader.RequestPacing(object()),
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
async def test_contfuture_download_job_uses_empty_end_datetime_once(dataloader_module):
    """Continuous futures should use IB's latest-ended request shape only once."""

    dataloader = dataloader_module
    contract = ibi.ContFuture(
        symbol="ES",
        exchange="CME",
        currency="USD",
        conId=123,
        localSymbol="ES",
    )

    class FakeSink:
        def __init__(self):
            self.writes = []

        async def write(self, data):
            self.writes.append(data)
            return "version"

    sink = FakeSink()
    container = dataloader.DownloadContainer(
        datetime(2025, 1, 1, tzinfo=timezone.utc),
        datetime(2025, 1, 10, tzinfo=timezone.utc),
        bar_size="30 secs",
    )
    job = dataloader.DownloadJob(
        contract,
        sink=sink,
        queue=[container],
        bar_size="30 secs",
    )

    assert job.params["endDateTime"] == ""

    bars = ibi.BarDataList(
        [
            ibi.BarData(
                date=datetime(2025, 1, 9, tzinfo=timezone.utc),
                open=1,
                high=1,
                low=1,
                close=1,
            )
        ]
    )
    await job.save_chunk(bars)

    assert job.is_done()
    assert len(sink.writes) == 1


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


def test_request_age_available_uses_run_scoped_now(dataloader_module):
    """Age validation should use the run snapshot supplied by the caller."""

    dataloader = dataloader_module

    class FakeJob:
        bar_size = "1 secs"
        next_date = datetime(2025, 1, 1, tzinfo=timezone.utc)

    assert not dataloader.request_age_available(
        FakeJob(),
        FakeJob.next_date + timedelta(days=181),
    )
    assert dataloader.request_age_available(
        FakeJob(),
        FakeJob.next_date + timedelta(days=1),
    )


def test_request_age_available_applies_to_30_second_bars(dataloader_module):
    """IB's six-month hard limit applies to 30-second bars too."""

    dataloader = dataloader_module

    class FakeJob:
        bar_size = "30 secs"
        next_date = datetime(2025, 1, 1, tzinfo=timezone.utc)

    assert not dataloader.request_age_available(
        FakeJob(),
        FakeJob.next_date + timedelta(days=181),
    )


def test_request_age_available_does_not_apply_to_one_minute_bars(dataloader_module):
    """Bars larger than 30 seconds should not use the six-month hard limit."""

    dataloader = dataloader_module

    class FakeJob:
        bar_size = "1 min"
        next_date = datetime(2025, 1, 1, tzinfo=timezone.utc)

    assert dataloader.request_age_available(
        FakeJob(),
        FakeJob.next_date + timedelta(days=1000),
    )


@pytest.mark.asyncio
async def test_worker_skips_small_bar_request_past_age_limit(dataloader_module):
    """Known unavailable small-bar ranges should be dropped before IB request."""

    dataloader = dataloader_module

    class FakeIB:
        def __init__(self):
            self.requests = 0

        async def reqHistoricalDataAsync(self, *args, **kwargs):
            self.requests += 1
            return ["unexpected"]

    class FakeJob:
        contract = FakeContract()
        next_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
        bar_size = "30 secs"

        def __init__(self):
            self.saved_chunks = []

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

    ib = FakeIB()
    manager = dataloader.Manager(
        ib,
        bar_size="30 secs",
        now=datetime(2025, 7, 1, tzinfo=timezone.utc),
    )
    session = dataloader.DataloaderSession(ib, manager=manager)
    job = FakeJob()

    await session.download_job("worker", job)

    assert ib.requests == 0
    assert job.saved_chunks == [None]


def test_manager_preserves_injected_pacing_policy(dataloader_module):
    """Injected pacing should not be silently mutated by Manager setup."""

    dataloader = dataloader_module
    pacing = dataloader.RequestPacing(object())

    manager = dataloader.Manager(
        object(),
        pacing=pacing,
        bar_size="1 day",
        wts="MIDPOINT",
    )

    assert manager.pacing is pacing
    assert manager.bar_size == "1 day"
    assert manager.wts == "MIDPOINT"
    assert not hasattr(pacing, "bar_size")
    assert not hasattr(pacing, "wts")


@pytest.mark.parametrize("max_lookback_days", [0, -1])
def test_manager_rejects_non_positive_max_lookback(
    dataloader_module, max_lookback_days
):
    """A configured lookback must be absent or a positive number of days."""

    with pytest.raises(ValueError, match="max_lookback_days"):
        dataloader_module.Manager(
            object(),
            max_lookback_days=max_lookback_days,
            bar_size="1 day",
        )


def test_manager_default_now_is_normalized_without_optional_sentinel(dataloader_module):
    """Manager should expose a concrete normalized now value after construction."""

    dataloader = dataloader_module

    intraday = dataloader.Manager(object(), bar_size="30 secs")
    daily = dataloader.Manager(object(), bar_size="1 day")

    assert isinstance(intraday.now, datetime)
    assert intraday.now.tzinfo is timezone.utc
    assert isinstance(daily.now, date)
    assert not isinstance(daily.now, datetime)


@pytest.mark.asyncio
async def test_manager_policy_flows_to_store_and_download_job(
    monkeypatch, dataloader_module
):
    """Manager request policy should reach store naming and generated jobs."""

    dataloader = dataloader_module
    contract = FakeContract()
    created_stores = []

    class FakeStore:
        def __init__(self, lib, host):
            created_stores.append((lib, host))

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

    monkeypatch.setattr(dataloader.Manager, "headstamps", fake_headstamps)
    monkeypatch.setattr(dataloader, "get_mongo_client", lambda: "mongo")
    monkeypatch.setattr(dataloader, "AsyncArcticStore", FakeStore)
    monkeypatch.setattr(dataloader, "HistorySink", lambda contract, store: FakeSink())

    manager = dataloader.Manager(
        object(),
        bar_size="1 day",
        wts="MIDPOINT",
        now=datetime(2025, 1, 10, tzinfo=timezone.utc),
        max_lookback_days=120,
    )
    job = await anext(manager._job_generator())

    assert created_stores == [("MIDPOINT_1_day", "mongo")]
    assert manager.bar_size == "1 day"
    assert manager.wts == "MIDPOINT"
    assert not hasattr(manager.pacing, "bar_size")
    assert not hasattr(manager.pacing, "wts")
    assert job.bar_size == "1 day"
    assert not hasattr(job, "max_bars")
    assert (
        job.target_bars_per_request
        == dataloader.helpers.DEFAULT_TARGET_BARS_PER_REQUEST
    )
    assert job.params["endDateTime"] == date(2025, 1, 10)


@pytest.mark.asyncio
async def test_schedule_gap_mode_uses_matching_use_rth(dataloader_module):
    """Schedule gap planning should pass the manager's historical-data RTH mode."""

    dataloader = dataloader_module
    contract = ibi.Contract(
        secType="FUT",
        symbol="NQ",
        exchange="CME",
        currency="USD",
        localSymbol="NQM6",
        tradingClass="NQ",
    )
    index = pd.DatetimeIndex(
        [
            datetime(2025, 1, 1, 10, tzinfo=timezone.utc),
            datetime(2025, 1, 1, 10, 0, 30, tzinfo=timezone.utc),
            datetime(2025, 1, 1, 10, 2, tzinfo=timezone.utc),
            datetime(2025, 1, 1, 10, 2, 30, tzinfo=timezone.utc),
        ]
    )

    class FakeStore:
        async def read(self, requested_contract):
            return pd.DataFrame({"close": range(len(index))}, index=index)

        async def read_metadata(self, requested_contract):
            return {}

    class FakePacing:
        def __init__(self):
            self.schedule_calls = []

        def contract_timezone(self, contract):
            return "UTC"

        async def historical_schedule(self, contract, *, numDays, endDateTime, useRTH):
            self.schedule_calls.append((contract, numDays, endDateTime, useRTH))
            return SimpleNamespace(
                timeZone="UTC",
                sessions=[
                    SimpleNamespace(
                        startDateTime="20250101-09:00:00",
                        endDateTime="20250101-11:00:00",
                    )
                ],
            )

    pacing = FakePacing()
    manager = dataloader.Manager(
        object(),
        pacing=pacing,
        bar_size="30 secs",
        gap_fill_mode="schedule",
        use_rth=True,
        now=datetime(2025, 1, 1, 11, tzinfo=timezone.utc),
    )
    store = await dataloader.AsyncStoreView.create(
        contract, FakeStore(), manager.now, manager.bar_size
    )

    tasks = await manager.tasks(store, datetime(2025, 1, 1, 9, tzinfo=timezone.utc))

    assert pacing.schedule_calls == [
        (contract, 2, datetime(2025, 1, 1, 10, 2, 30, tzinfo=timezone.utc), True)
    ]
    assert [(task.kind, task.from_date, task.to_date) for task in tasks] == [
        (
            "update",
            datetime(2025, 1, 1, 10, 2, 30, tzinfo=timezone.utc),
            datetime(2025, 1, 1, 11, tzinfo=timezone.utc),
        ),
        (
            "backfill",
            datetime(2025, 1, 1, 9, tzinfo=timezone.utc),
            datetime(2025, 1, 1, 10, 0, 30, tzinfo=timezone.utc),
        ),
        (
            "gap",
            datetime(2025, 1, 1, 10, 0, 30, tzinfo=timezone.utc),
            datetime(2025, 1, 1, 10, 2, 30, tzinfo=timezone.utc),
        ),
    ]


@pytest.mark.asyncio
async def test_schedule_gap_mode_fails_when_schedule_empty(dataloader_module):
    """Explicit schedule mode should fail instead of silently using heuristics."""

    dataloader = dataloader_module
    contract = ibi.Contract(secType="FUT", symbol="NQ", exchange="CME")
    index = pd.DatetimeIndex(
        [
            datetime(2025, 1, 1, 10, tzinfo=timezone.utc),
            datetime(2025, 1, 1, 10, 0, 30, tzinfo=timezone.utc),
            datetime(2025, 1, 1, 10, 2, tzinfo=timezone.utc),
        ]
    )

    class FakeStore:
        async def read(self, requested_contract):
            return pd.DataFrame({"close": range(len(index))}, index=index)

        async def read_metadata(self, requested_contract):
            return {}

    class FakePacing:
        def contract_timezone(self, contract):
            return "UTC"

        async def historical_schedule(self, contract, *, numDays, endDateTime, useRTH):
            return SimpleNamespace(timeZone="UTC", sessions=[])

    manager = dataloader.Manager(
        object(),
        pacing=FakePacing(),
        bar_size="30 secs",
        gap_fill_mode="schedule",
        now=datetime(2025, 1, 1, 11, tzinfo=timezone.utc),
    )
    store = await dataloader.AsyncStoreView.create(
        contract, FakeStore(), manager.now, manager.bar_size
    )

    with pytest.raises(RuntimeError, match="No historical schedule"):
        await manager.tasks(store, datetime(2025, 1, 1, 9, tzinfo=timezone.utc))


@pytest.mark.asyncio
async def test_gap_learner_is_run_local_and_does_not_touch_sink(dataloader_module):
    """No-data gap patterns should be learned in memory only."""

    dataloader = dataloader_module
    pattern = dataloader.GapPattern(
        datetime(2025, 1, 1, 10).time(),
        timedelta(minutes=15, seconds=30),
        "UTC",
    )

    class FakeSink:
        def __init__(self):
            self.marked = False

        async def write(self, data):
            return "version"

        async def mark_backfill_exhausted(self):
            self.marked = True

    sink = FakeSink()
    learner = dataloader.RunGapLearner()
    job = dataloader.DownloadJob(
        FakeContract(),
        sink,
        [
            dataloader.DownloadContainer(
                datetime(2025, 1, 1, 10, tzinfo=timezone.utc),
                datetime(2025, 1, 1, 10, 16, tzinfo=timezone.utc),
                kind="gap",
                gap_pattern=pattern,
            )
        ],
        gap_learner=learner,
    )

    await job.save_chunk(None)
    learner.record_empty_gap(pattern)

    assert learner.typical_patterns == {pattern}
    assert sink.marked is False


@pytest.mark.asyncio
async def test_gap_learner_prunes_same_job_pending_typical_gaps(dataloader_module):
    """Repeated empty gap patterns should drop remaining matching gap ranges."""

    dataloader = dataloader_module
    repeated = dataloader.GapPattern(
        datetime(2025, 1, 1, 10).time(),
        timedelta(minutes=15, seconds=30),
        "UTC",
    )
    different = dataloader.GapPattern(
        datetime(2025, 1, 1, 11).time(),
        timedelta(minutes=15, seconds=30),
        "UTC",
    )

    class FakeSink:
        async def write(self, data):
            return "version"

        async def mark_backfill_exhausted(self):
            return None

    learner = dataloader.RunGapLearner()
    job = dataloader.DownloadJob(
        FakeContract(),
        FakeSink(),
        [
            dataloader.DownloadContainer(
                datetime(2025, 1, 1, 10, tzinfo=timezone.utc),
                datetime(2025, 1, 1, 10, 16, tzinfo=timezone.utc),
                kind="gap",
                gap_pattern=repeated,
            ),
            dataloader.DownloadContainer(
                datetime(2025, 1, 2, 10, tzinfo=timezone.utc),
                datetime(2025, 1, 2, 10, 16, tzinfo=timezone.utc),
                kind="gap",
                gap_pattern=repeated,
            ),
            dataloader.DownloadContainer(
                datetime(2025, 1, 3, 10, tzinfo=timezone.utc),
                datetime(2025, 1, 3, 10, 16, tzinfo=timezone.utc),
                kind="gap",
                gap_pattern=repeated,
            ),
            dataloader.DownloadContainer(
                datetime(2025, 1, 3, 11, tzinfo=timezone.utc),
                datetime(2025, 1, 3, 11, 16, tzinfo=timezone.utc),
                kind="gap",
                gap_pattern=different,
            ),
        ],
        gap_learner=learner,
    )

    await job.save_chunk(None)
    assert [container.gap_pattern for container in job.queue] == [
        repeated,
        repeated,
        different,
    ]

    await job.save_chunk(None)

    assert learner.typical_patterns == {repeated}
    assert [container.gap_pattern for container in job.queue] == [different]


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
async def test_headstamp_empty_response_uses_fallback_once(dataloader_module):
    """A non-retryable empty headstamp should use fallback without local looping."""

    dataloader = dataloader_module

    class FakeIB:
        def __init__(self):
            self.requests = 0

        async def reqHeadTimeStampAsync(self, *args, **kwargs):
            self.requests += 1
            return None

    ib = FakeIB()
    manager = dataloader.Manager(ib)
    manager.pacing.no_restriction = True

    headstamp = await manager.headstamp(FakeContract())

    assert ib.requests == 1
    assert isinstance(headstamp, datetime)
    assert headstamp.tzinfo is timezone.utc


@pytest.mark.asyncio
async def test_worker_connection_loss_aborts_worker(dataloader_module):
    """Connection loss should escape instead of becoming an ordinary job failure."""

    dataloader = dataloader_module

    class FakeIB:
        async def reqHistoricalDataAsync(self, *args, **kwargs):
            raise ConnectionError("socket down")

    class FakeJob:
        contract = FakeContract()
        next_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
        bar_size = "1 min"
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

    assert not job.saved
    with pytest.raises(ConnectionError, match="socket down"):
        await task
    assert session.fatal_error is not None
    assert len(session.failures.failures) == 0


@pytest.mark.asyncio
async def test_session_run_propagates_worker_connection_loss(
    monkeypatch, dataloader_module
):
    """Session-level workload should fail when a worker hits connection loss."""

    dataloader = dataloader_module

    class FakeIB:
        async def reqHistoricalDataAsync(self, *args, **kwargs):
            raise ConnectionError("socket down")

    class FakeJob:
        contract = FakeContract()
        next_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
        bar_size = "1 min"

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

    async def fake_producer(self, queue):
        await queue.put(FakeJob())

    monkeypatch.setattr(dataloader.DataloaderSession, "producer", fake_producer)
    session = dataloader.DataloaderSession(FakeIB(), number_of_workers=1)
    session.pacing.no_restriction = True

    with pytest.raises(ConnectionError, match="socket down"):
        await session.run()
    assert len(session.failures.failures) == 0


@pytest.mark.asyncio
async def test_worker_request_failure_does_not_stop_next_job(dataloader_module):
    """A failed broker request should be recorded without stopping the worker."""

    dataloader = dataloader_module

    class FakeIB:
        async def reqHistoricalDataAsync(self, contract, *args, **kwargs):
            if contract.localSymbol == "failed":
                raise RuntimeError("request failed")
            return []

    class FakeRequestContract:
        def __init__(self, local_symbol):
            self.localSymbol = local_symbol

        def __hash__(self):
            return hash(self.localSymbol)

    class FakeJob:
        next_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
        bar_size = "1 min"

        def __init__(self, name):
            self.name = name
            self.saved = False
            self.contract = FakeRequestContract(name)

        @property
        def params(self):
            return {
                "contract": self.contract,
                "endDateTime": self.next_date,
                "durationStr": "1 D",
            }

        async def save_chunk(self, chunk):
            self.saved = True
            self.next_date = None

        def __str__(self):
            return f"<FakeJob {self.name}>"

    session = dataloader.DataloaderSession(FakeIB())
    session.pacing.no_restriction = True
    queue = asyncio.Queue()
    failed_job = FakeJob("failed")
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
async def test_worker_local_failure_aborts_worker(dataloader_module):
    """Local processing failures should abort the session instead of being hidden."""

    dataloader = dataloader_module

    class FakeIB:
        async def reqHistoricalDataAsync(self, *args, **kwargs):
            return []

    class FakeJob:
        contract = FakeContract()
        next_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
        bar_size = "1 min"

        @property
        def params(self):
            return {
                "contract": self.contract,
                "endDateTime": self.next_date,
                "durationStr": "1 D",
            }

        async def save_chunk(self, chunk):
            raise RuntimeError("store failed")

        def __str__(self):
            return "<FakeJob>"

    session = dataloader.DataloaderSession(FakeIB())
    session.pacing.no_restriction = True
    queue = asyncio.Queue()
    await queue.put(FakeJob())

    task = asyncio.create_task(session.worker("worker", queue))
    await asyncio.wait_for(queue.join(), timeout=1)
    task.cancel()

    with pytest.raises(RuntimeError, match="store failed"):
        await task
    assert isinstance(session.fatal_error, RuntimeError)
    assert len(session.failures.failures) == 0


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
            self.bar_size = "1 min"

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
