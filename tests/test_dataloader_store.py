from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import ib_insync as ibi
import pandas as pd
import pytest

from haymaker.async_wrappers import QueueShutdownPolicy
from haymaker.dataloader.scheduling import (
    GapCandidate,
    GapPattern,
    PlannedRange,
    SessionRange,
    TaskPlanner,
    heuristic_gap_candidates,
    scheduled_gap_candidates,
    sessions_from_historical_schedule,
    historical_unavailability_reason,
)
from haymaker.dataloader.helpers import duration_in_secs
from haymaker.dataloader.store_wrapper import AsyncStoreView, HistorySink


@pytest.fixture
def dataloader_module():
    """Return the side-effect-free dataloader module."""

    from haymaker.dataloader import dataloader

    return dataloader


class FakeAsyncStore:
    """Minimal async datastore used by dataloader store-wrapper tests."""

    def __init__(
        self, data: pd.DataFrame | None = None, metadata: dict | None = None
    ) -> None:
        self.data = data
        self.metadata = metadata or {}
        self.reads: list[ibi.Contract] = []
        self.writes: list[tuple[ibi.Contract, pd.DataFrame]] = []
        self.metadata_writes: list[tuple[ibi.Contract, dict]] = []

    async def read(self, contract: ibi.Contract) -> pd.DataFrame | None:
        """Return configured data and record the requested contract."""

        self.reads.append(contract)
        return self.data

    async def read_metadata(self, contract: ibi.Contract) -> dict:
        """Return configured metadata."""

        return self.metadata

    def write_metadata(self, contract: ibi.Contract, metadata: dict) -> str:
        """Record a metadata write and merge it into latest metadata."""

        self.metadata_writes.append((contract, metadata))
        self.metadata.update(metadata)
        return "metadata-version"

    async def async_write(self, contract: ibi.Contract, data: pd.DataFrame) -> str:
        """Record a write and keep the dataframe as latest store state."""

        self.writes.append((contract, data))
        self.data = data
        return "version"


@pytest.fixture
def contract() -> ibi.Contract:
    """Return a concrete contract suitable for datastore naming."""

    return ibi.Contract(secType="FUT", localSymbol="ESZ5")


@pytest.fixture
def store_index() -> pd.DatetimeIndex:
    """Return a short UTC datetime index for store-wrapper boundaries."""

    return pd.DatetimeIndex(
        [
            datetime(2025, 1, 1, tzinfo=timezone.utc),
            datetime(2025, 1, 2, tzinfo=timezone.utc),
            datetime(2025, 1, 3, tzinfo=timezone.utc),
        ]
    )


@pytest.mark.asyncio
async def test_async_store_view_preloads_existing_data(contract, store_index):
    """AsyncStoreView should preload data through the async datastore API."""

    data = pd.DataFrame({"close": [1, 2, 3]}, index=store_index)
    store = FakeAsyncStore(data)

    wrapper = await AsyncStoreView.create(contract, store, store_index[-1], "30 secs")

    assert store.reads == [contract]
    assert wrapper.backfill_boundary == store_index[1]
    assert wrapper.to_date == store_index[-1]


@pytest.mark.asyncio
async def test_async_store_view_one_row_uses_single_timestamp_boundary(
    contract, store_index
):
    """One stored row should use that point as the stored-data boundary."""

    data = pd.DataFrame({"close": [1]}, index=store_index[:1])

    wrapper = await AsyncStoreView.create(
        contract, FakeAsyncStore(data), store_index[-1], "30 secs"
    )

    assert wrapper.backfill_boundary == store_index[0]
    assert wrapper.to_date == store_index[0]


@pytest.mark.asyncio
async def test_task_planner_one_row_store_does_not_schedule_full_duplicate(
    contract, store_index
):
    """A one-row collection should schedule around the stored boundary."""

    data = pd.DataFrame({"close": [1]}, index=store_index[:1])
    wrapper = await AsyncStoreView.create(
        contract, FakeAsyncStore(data), store_index[-1], "30 secs"
    )

    tasks = TaskPlanner(
        wrapper,
        store_index[0],
        max_lookback_days=100,
    ).planned_ranges()

    assert tasks == [PlannedRange(store_index[0], store_index[-1], "update")]


@pytest.mark.asyncio
async def test_task_planner_clamps_start_to_max_lookback(contract):
    """TaskPlanner should own the run lookback clamp for download ranges."""

    now = datetime(2025, 1, 10, tzinfo=timezone.utc)
    headstamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
    wrapper = await AsyncStoreView.create(contract, FakeAsyncStore(), now, "30 secs")

    tasks = TaskPlanner(
        wrapper,
        headstamp,
        max_lookback_days=3,
    ).planned_ranges()

    assert tasks == [
        PlannedRange(datetime(2025, 1, 7, tzinfo=timezone.utc), now, "backfill")
    ]


@pytest.mark.asyncio
async def test_task_planner_without_max_lookback_uses_headstamp(contract):
    """No user lookback should leave headTimestamp as the start candidate."""

    now = datetime(2025, 1, 10, tzinfo=timezone.utc)
    headstamp = datetime(2020, 1, 1, tzinfo=timezone.utc)
    wrapper = await AsyncStoreView.create(contract, FakeAsyncStore(), now, "1 min")

    tasks = TaskPlanner(
        wrapper,
        headstamp,
        max_lookback_days=None,
    ).planned_ranges()

    assert tasks == [PlannedRange(headstamp, now, "backfill")]


@pytest.mark.asyncio
async def test_task_planner_clamps_small_bars_to_six_month_limit(contract):
    """Bars 30 seconds or smaller should not request past IB's age limit."""

    now = datetime(2025, 7, 1, tzinfo=timezone.utc)
    headstamp = datetime(2020, 1, 1, tzinfo=timezone.utc)
    wrapper = await AsyncStoreView.create(contract, FakeAsyncStore(), now, "30 secs")

    tasks = TaskPlanner(
        wrapper,
        headstamp,
        max_lookback_days=None,
    ).planned_ranges()

    assert tasks == [PlannedRange(now - timedelta(days=180), now, "backfill")]


@pytest.mark.asyncio
async def test_task_planner_does_not_apply_six_month_limit_to_one_min(contract):
    """The six-month hard limit does not apply to bars larger than 30 seconds."""

    now = datetime(2025, 7, 1, tzinfo=timezone.utc)
    headstamp = datetime(2020, 1, 1, tzinfo=timezone.utc)
    wrapper = await AsyncStoreView.create(contract, FakeAsyncStore(), now, "1 min")

    tasks = TaskPlanner(
        wrapper,
        headstamp,
        max_lookback_days=None,
    ).planned_ranges()

    assert tasks == [PlannedRange(headstamp, now, "backfill")]


@pytest.mark.asyncio
async def test_task_planner_clamps_expired_future_to_two_year_limit():
    """Expired futures should not request older than two years from expiry."""

    contract = ibi.Contract(
        secType="FUT",
        localSymbol="NQZ5",
        lastTradeDateOrContractMonth="20250102",
    )
    now = datetime(2025, 1, 3, tzinfo=timezone.utc)
    wrapper = await AsyncStoreView.create(contract, FakeAsyncStore(), now, "1 day")

    tasks = TaskPlanner(
        wrapper,
        date(2020, 1, 1),
        max_lookback_days=None,
    ).planned_ranges()

    assert tasks == [PlannedRange(date(2023, 1, 3), date(2025, 1, 2), "backfill")]


@pytest.mark.asyncio
async def test_task_planner_keeps_active_future_unclamped():
    """The two-year expired-future limit should not clamp active contracts."""

    contract = ibi.Contract(
        secType="FUT",
        localSymbol="NQZ9",
        lastTradeDateOrContractMonth="20290102",
    )
    now = datetime(2025, 1, 3, tzinfo=timezone.utc)
    wrapper = await AsyncStoreView.create(contract, FakeAsyncStore(), now, "1 day")

    tasks = TaskPlanner(
        wrapper,
        date(2020, 1, 1),
        max_lookback_days=None,
    ).planned_ranges()

    assert tasks == [PlannedRange(date(2020, 1, 1), date(2025, 1, 3), "backfill")]


@pytest.mark.asyncio
async def test_task_planner_skips_expired_options():
    """IB documents expired options as unavailable historical data."""

    contract = ibi.Contract(
        secType="OPT",
        localSymbol="AAPL  250102C00100000",
        lastTradeDateOrContractMonth="20250102",
    )
    now = datetime(2025, 1, 3, tzinfo=timezone.utc)
    wrapper = await AsyncStoreView.create(contract, FakeAsyncStore(), now, "1 day")

    assert (
        TaskPlanner(
            wrapper,
            date(2020, 1, 1),
            max_lookback_days=None,
        ).planned_ranges()
        == []
    )
    reason = historical_unavailability_reason(wrapper)
    assert reason is not None
    assert "documented IB rule" in reason


@pytest.mark.asyncio
async def test_task_planner_missing_metadata_still_schedules_backfill(contract):
    """Missing optional metadata should not prevent normal backfill planning."""

    now = datetime(2025, 1, 10, tzinfo=timezone.utc)
    headstamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
    wrapper = await AsyncStoreView.create(contract, FakeAsyncStore(), now, "30 secs")

    tasks = TaskPlanner(
        wrapper,
        headstamp,
        max_lookback_days=3,
    ).planned_ranges()

    assert tasks == [
        PlannedRange(
            datetime(2025, 1, 7, tzinfo=timezone.utc),
            now,
            "backfill",
        )
    ]


@pytest.mark.asyncio
async def test_task_planner_skips_exhausted_backfill_but_keeps_update(
    contract, store_index
):
    """Backfill exhaustion should not block update ranges."""

    data = pd.DataFrame({"close": [1, 2, 3]}, index=store_index)
    now = datetime(2025, 1, 5, tzinfo=timezone.utc)
    wrapper = await AsyncStoreView.create(
        contract,
        FakeAsyncStore(data, metadata={"backfill_exhausted": True}),
        now,
        "30 secs",
    )

    tasks = TaskPlanner(
        wrapper,
        datetime(2025, 1, 1, tzinfo=timezone.utc),
        max_lookback_days=100,
    ).planned_ranges()

    assert tasks == [PlannedRange(store_index[-1], now, "update")]
    assert TaskPlanner(
        wrapper,
        datetime(2025, 1, 1, tzinfo=timezone.utc),
        max_lookback_days=100,
    ).planned_ranges() == [PlannedRange(store_index[-1], now, "update")]


@pytest.mark.asyncio
async def test_task_planner_exhausted_backfill_without_update_has_no_range(
    contract, store_index
):
    """Backfill exhaustion should make a current store a no-op."""

    data = pd.DataFrame({"close": [1, 2, 3]}, index=store_index)
    wrapper = await AsyncStoreView.create(
        contract,
        FakeAsyncStore(data, metadata={"backfill_exhausted": True}),
        store_index[-1],
        "30 secs",
    )

    assert (
        TaskPlanner(
            wrapper,
            datetime(2025, 1, 1, tzinfo=timezone.utc),
            max_lookback_days=100,
        ).planned_ranges()
        == []
    )


@pytest.mark.asyncio
async def test_task_planner_completes_one_intraday_bar_without_request():
    """One bar before exact past expiry should be completed locally."""

    contract = ibi.Future(localSymbol="ESH5", lastTradeDateOrContractMonth="20250102")
    expiry = datetime(2025, 1, 2, tzinfo=timezone.utc)
    last_bar = expiry - timedelta(seconds=30)
    data = pd.DataFrame({"close": [1]}, index=pd.DatetimeIndex([last_bar]))
    wrapper = await AsyncStoreView.create(
        contract,
        FakeAsyncStore(data, metadata={"backfill_exhausted": True}),
        datetime(2025, 1, 3, tzinfo=timezone.utc),
        "30 secs",
    )
    planner = TaskPlanner(wrapper, None, max_lookback_days=None)

    assert planner.completes_update_without_request is True
    assert planner.planned_ranges() == []


@pytest.mark.asyncio
async def test_task_planner_tags_larger_past_expiry_update():
    """A larger expired update should be requested and mark on completion."""

    contract = ibi.Future(localSymbol="ESH5", lastTradeDateOrContractMonth="20250102")
    last_bar = datetime(2025, 1, 1, 12, tzinfo=timezone.utc)
    expiry = datetime(2025, 1, 2, tzinfo=timezone.utc)
    data = pd.DataFrame({"close": [1]}, index=pd.DatetimeIndex([last_bar]))
    wrapper = await AsyncStoreView.create(
        contract,
        FakeAsyncStore(data, metadata={"backfill_exhausted": True}),
        datetime(2025, 1, 3, tzinfo=timezone.utc),
        "30 secs",
    )

    assert TaskPlanner(wrapper, None, max_lookback_days=None).planned_ranges() == [
        PlannedRange(last_bar, expiry, "update", exhausts_update=True)
    ]


@pytest.mark.asyncio
async def test_task_planner_uses_update_marker_only_after_expiry():
    """Update metadata must never suppress a live contract update."""

    contract = ibi.Future(localSymbol="ESM5", lastTradeDateOrContractMonth="20250103")
    last_bar = datetime(2025, 1, 1, tzinfo=timezone.utc)
    now = datetime(2025, 1, 2, tzinfo=timezone.utc)
    data = pd.DataFrame({"close": [1]}, index=pd.DatetimeIndex([last_bar]))
    wrapper = await AsyncStoreView.create(
        contract,
        FakeAsyncStore(
            data,
            metadata={"backfill_exhausted": True, "update_exhausted": True},
        ),
        now,
        "30 secs",
    )

    assert wrapper.update_exhausted is True
    assert TaskPlanner(wrapper, None, max_lookback_days=None).planned_ranges() == [
        PlannedRange(last_bar, now, "update")
    ]


@pytest.mark.asyncio
async def test_task_planner_update_marker_preserves_backfill():
    """Past-expiry update completion should not suppress older backfill."""

    contract = ibi.Future(localSymbol="ESH5", lastTradeDateOrContractMonth="20250102")
    head = datetime(2025, 1, 1, tzinfo=timezone.utc)
    last_bar = datetime(2025, 1, 1, 12, tzinfo=timezone.utc)
    data = pd.DataFrame({"close": [1]}, index=pd.DatetimeIndex([last_bar]))
    wrapper = await AsyncStoreView.create(
        contract,
        FakeAsyncStore(data, metadata={"update_exhausted": True}),
        datetime(2025, 1, 3, tzinfo=timezone.utc),
        "30 secs",
    )

    assert TaskPlanner(wrapper, head, max_lookback_days=None).planned_ranges() == [
        PlannedRange(head, last_bar, "backfill")
    ]


@pytest.mark.asyncio
async def test_task_planner_update_marker_preserves_gap_filling():
    """Past-expiry update completion should not suppress internal gaps."""

    contract = ibi.Future(localSymbol="ESH5", lastTradeDateOrContractMonth="20250102")
    index = pd.DatetimeIndex(
        [
            datetime(2025, 1, 1, 10, tzinfo=timezone.utc),
            datetime(2025, 1, 1, 10, 0, 30, tzinfo=timezone.utc),
            datetime(2025, 1, 1, 10, 2, tzinfo=timezone.utc),
            datetime(2025, 1, 1, 10, 2, 30, tzinfo=timezone.utc),
        ]
    )
    data = pd.DataFrame({"close": range(len(index))}, index=index)
    wrapper = await AsyncStoreView.create(
        contract,
        FakeAsyncStore(
            data,
            metadata={"backfill_exhausted": True, "update_exhausted": True},
        ),
        datetime(2025, 1, 3, tzinfo=timezone.utc),
        "30 secs",
    )

    assert TaskPlanner(
        wrapper,
        None,
        max_lookback_days=None,
        gap_fill_mode="heuristic",
        timezone_name="UTC",
    ).planned_ranges() == [
        PlannedRange(
            index[1],
            index[3],
            "gap",
            GapPattern(index[1].time(), timedelta(minutes=1, seconds=30), "UTC"),
        )
    ]


@pytest.mark.asyncio
async def test_task_planner_does_not_exhaust_update_at_exact_expiry():
    """Expiry equal to the run boundary is not strictly in the past."""

    contract = ibi.Future(localSymbol="ESH5", lastTradeDateOrContractMonth="20250102")
    last_bar = datetime(2025, 1, 1, 23, 59, 30, tzinfo=timezone.utc)
    expiry = datetime(2025, 1, 2, tzinfo=timezone.utc)
    data = pd.DataFrame({"close": [1]}, index=pd.DatetimeIndex([last_bar]))
    wrapper = await AsyncStoreView.create(
        contract,
        FakeAsyncStore(
            data,
            metadata={"backfill_exhausted": True, "update_exhausted": True},
        ),
        expiry,
        "30 secs",
    )
    planner = TaskPlanner(wrapper, None, max_lookback_days=None)

    assert planner.completes_update_without_request is False
    assert planner.planned_ranges() == [PlannedRange(last_bar, expiry, "update")]


@pytest.mark.asyncio
async def test_task_planner_requests_one_daily_period_before_past_expiry():
    """The local one-bar shortcut should remain intraday-only."""

    contract = ibi.Future(localSymbol="ESH5", lastTradeDateOrContractMonth="20250102")
    data = pd.DataFrame({"close": [1]}, index=pd.Index([date(2025, 1, 1)]))
    wrapper = await AsyncStoreView.create(
        contract,
        FakeAsyncStore(data, metadata={"backfill_exhausted": True}),
        date(2025, 1, 3),
        "1 day",
    )
    planner = TaskPlanner(wrapper, None, max_lookback_days=None)

    assert planner.completes_update_without_request is False
    assert planner.planned_ranges() == [
        PlannedRange(
            date(2025, 1, 1),
            date(2025, 1, 2),
            "update",
            exhausts_update=True,
        )
    ]


@pytest.mark.asyncio
async def test_gap_planning_preserves_execution_order(contract):
    """Heuristic gap planning should run update, backfill, then gaps."""

    index = pd.DatetimeIndex(
        [
            datetime(2025, 1, 1, tzinfo=timezone.utc),
            datetime(2025, 1, 2, tzinfo=timezone.utc),
            datetime(2025, 1, 5, tzinfo=timezone.utc),
            datetime(2025, 1, 6, tzinfo=timezone.utc),
            datetime(2025, 1, 9, tzinfo=timezone.utc),
            datetime(2025, 1, 10, tzinfo=timezone.utc),
            datetime(2025, 1, 13, tzinfo=timezone.utc),
            datetime(2025, 1, 14, tzinfo=timezone.utc),
        ]
    )
    data = pd.DataFrame({"close": range(len(index))}, index=index)
    now = datetime(2025, 1, 15, tzinfo=timezone.utc)
    headstamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
    wrapper = await AsyncStoreView.create(
        contract, FakeAsyncStore(data), now, "30 secs"
    )
    planner = TaskPlanner(
        wrapper,
        headstamp,
        max_lookback_days=100,
        gap_fill_mode="heuristic",
        timezone_name="UTC",
    )

    tasks = planner.planned_ranges()

    assert [task.kind for task in tasks] == ["update", "backfill", "gap", "gap"]
    assert tasks[:2] == planner.base_ranges()


@pytest.mark.asyncio
async def test_task_planner_skips_gap_fill_for_continuous_futures():
    """Continuous futures cannot target historical gap windows with endDateTime."""

    contract = ibi.ContFuture(
        symbol="ES",
        exchange="CME",
        currency="USD",
        conId=123,
        localSymbol="ES",
    )
    index = pd.DatetimeIndex(
        [
            datetime(2025, 1, 1, 10, tzinfo=timezone.utc),
            datetime(2025, 1, 1, 10, 0, 30, tzinfo=timezone.utc),
            datetime(2025, 1, 1, 10, 2, tzinfo=timezone.utc),
        ]
    )
    data = pd.DataFrame({"close": range(len(index))}, index=index)
    now = datetime(2025, 1, 1, 11, tzinfo=timezone.utc)
    wrapper = await AsyncStoreView.create(
        contract, FakeAsyncStore(data), now, "30 secs"
    )

    tasks = TaskPlanner(
        wrapper,
        datetime(2025, 1, 1, 9, tzinfo=timezone.utc),
        max_lookback_days=100,
        gap_fill_mode="heuristic",
    ).planned_ranges()

    assert [task.kind for task in tasks] == ["update"]


@pytest.mark.asyncio
async def test_task_planner_backfills_empty_continuous_future_once():
    """An empty continuous-future series can make one latest-ended backfill."""

    contract = ibi.ContFuture(
        symbol="ES",
        exchange="CME",
        currency="USD",
        conId=123,
        localSymbol="ES",
    )
    now = datetime(2025, 1, 10, tzinfo=timezone.utc)
    wrapper = await AsyncStoreView.create(contract, FakeAsyncStore(), now, "30 secs")

    tasks = TaskPlanner(
        wrapper,
        datetime(2025, 1, 1, tzinfo=timezone.utc),
        max_lookback_days=100,
        gap_fill_mode="heuristic",
    ).planned_ranges()

    assert tasks == [
        PlannedRange(
            datetime(2025, 1, 1, tzinfo=timezone.utc),
            now,
            "backfill",
        )
    ]


@pytest.mark.asyncio
async def test_async_store_view_dates_update_after_reload(contract, store_index):
    """Date boundaries should reflect the latest loaded store data."""

    store = FakeAsyncStore(pd.DataFrame({"close": [1]}, index=store_index[:1]))
    wrapper = await AsyncStoreView.create(contract, store, store_index[-1], "30 secs")
    store.data = pd.DataFrame({"close": [1, 2, 3]}, index=store_index)

    await wrapper.read()

    assert wrapper.backfill_boundary == store_index[1]
    assert wrapper.to_date == store_index[-1]


@pytest.mark.asyncio
async def test_async_store_view_ignores_month_only_expiry(store_index):
    """Month-only contract months are not treated as precise expiry dates."""

    contract = ibi.Contract(
        secType="FUT",
        localSymbol="ESM5",
        lastTradeDateOrContractMonth="202506",
    )
    wrapper = await AsyncStoreView.create(
        contract, FakeAsyncStore(), store_index[-1], "30 secs"
    )

    assert wrapper.expiry is None
    assert wrapper.expiry_or_now() == store_index[-1]


@pytest.mark.asyncio
async def test_async_store_view_exact_expiry_caps_now(store_index):
    """Exact expiry dates should cap the latest downloadable point."""

    contract = ibi.Contract(
        secType="FUT",
        localSymbol="ESM5",
        lastTradeDateOrContractMonth="20250102",
    )
    now = datetime(2025, 1, 3, tzinfo=timezone.utc)
    wrapper = await AsyncStoreView.create(contract, FakeAsyncStore(), now, "30 secs")

    assert wrapper.expiry_or_now() == datetime(2025, 1, 2, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_async_store_view_date_bar_normalizes_boundaries_to_dates(
    contract, store_index
):
    """Daily-like bars should schedule with dates, not midnight datetimes."""

    data = pd.DataFrame({"close": [1, 2, 3]}, index=store_index)
    wrapper = await AsyncStoreView.create(
        contract,
        FakeAsyncStore(data),
        datetime(2025, 1, 10, 12, tzinfo=timezone.utc),
        "1 day",
    )

    assert wrapper.backfill_boundary == date(2025, 1, 2)
    assert wrapper.to_date == date(2025, 1, 3)
    assert wrapper.expiry_or_now() == date(2025, 1, 10)


@pytest.mark.asyncio
async def test_async_store_view_intraday_rejects_naive_datastore_index(contract):
    """Intraday stored timestamps must be timezone-aware before scheduling."""

    data = pd.DataFrame(
        {"close": [1]},
        index=pd.DatetimeIndex([datetime(2025, 1, 1)]),
    )

    with pytest.raises(ValueError, match="timezone-aware"):
        await AsyncStoreView.create(
            contract,
            FakeAsyncStore(data),
            datetime(2025, 1, 2, tzinfo=timezone.utc),
            "30 secs",
        )


def test_month_bar_size_has_duration() -> None:
    """Monthly bars are valid IB historical requests and need duration support."""

    assert duration_in_secs("1 month") > duration_in_secs("1 week")


@pytest.mark.asyncio
async def test_history_sink_concats_existing_data_and_writes(contract, store_index):
    """HistorySink should preserve current full read, concat, write behavior."""

    initial = pd.DataFrame({"close": [1]}, index=store_index[:1])
    updated = pd.DataFrame({"close": [1, 2]}, index=store_index[:2])
    store = FakeAsyncStore(initial)
    sink = HistorySink(contract, store)

    version = await sink.write(updated.iloc[1:])

    assert version == "version"
    assert store.writes[0][0] is contract
    pd.testing.assert_frame_equal(store.writes[0][1], updated)


@pytest.mark.asyncio
async def test_history_sink_preserves_raw_downloaded_index(contract):
    """HistorySink should not normalize the dataframe received from callers."""

    raw_dates = pd.Index([date(2025, 1, 1), date(2025, 1, 2)], name="date")
    new_data = pd.DataFrame({"close": [1, 2]}, index=raw_dates)
    store = FakeAsyncStore()
    sink = HistorySink(contract, store)

    await sink.write(new_data)

    pd.testing.assert_frame_equal(store.writes[0][1], new_data)


@pytest.mark.asyncio
async def test_history_sink_marks_backfill_exhausted_for_existing_series(
    contract, store_index
):
    """HistorySink should persist the backfill exhaustion marker only."""

    data = pd.DataFrame({"close": [1]}, index=store_index[:1])
    store = FakeAsyncStore(data, metadata={"up_to": store_index[0].isoformat()})
    sink = HistorySink(contract, store)

    version = await sink.mark_backfill_exhausted()

    assert version == "metadata-version"
    assert store.metadata_writes == [(contract, {"backfill_exhausted": True})]
    assert "from" not in store.metadata
    assert store.metadata["up_to"] == store_index[0].isoformat()


@pytest.mark.asyncio
async def test_history_sink_marks_update_exhausted_for_existing_series(
    contract, store_index
):
    """HistorySink should preserve metadata when marking terminal updates."""

    data = pd.DataFrame({"close": [1]}, index=store_index[:1])
    store = FakeAsyncStore(data, metadata={"up_to": store_index[0].isoformat()})
    sink = HistorySink(contract, store)

    version = await sink.mark_update_exhausted()

    assert version == "metadata-version"
    assert store.metadata_writes == [(contract, {"update_exhausted": True})]
    assert store.metadata["up_to"] == store_index[0].isoformat()


@pytest.mark.asyncio
async def test_history_sink_does_not_mark_empty_series(contract):
    """Empty series should not be marked as exhausted without a data anchor."""

    store = FakeAsyncStore()
    sink = HistorySink(contract, store)

    assert await sink.mark_backfill_exhausted() is None
    assert await sink.mark_update_exhausted() is None
    assert store.metadata_writes == []


def test_manager_datastore_builds_async_arctic_store(monkeypatch, dataloader_module):
    """Manager should lazily build the only supported dataloader backend."""

    dataloader = dataloader_module
    created: dict[str, object] = {}

    class FakeStoreFactory:
        """Capture store construction without opening Arctic."""

        def arctic_store(
            self,
            library: str,
            *,
            shutdown_policy: QueueShutdownPolicy,
        ) -> object:
            created["lib"] = library
            created["host"] = "mongo"
            created["shutdown_policy"] = shutdown_policy
            return self

    manager = dataloader.Manager(
        object(),
        store_factory=FakeStoreFactory(),
        wts="TRADES",
        bar_size="30 secs",
    )
    store = manager.datastore

    assert isinstance(store, FakeStoreFactory)
    assert created == {
        "lib": "TRADES_30_secs",
        "host": "mongo",
        "shutdown_policy": QueueShutdownPolicy.DRAIN,
    }


def test_heuristic_suppresses_repeated_short_pair_but_keeps_long_gap():
    """Repeated short gaps should not hide long missing intervals at same time."""

    start = datetime(2025, 1, 1, 21, tzinfo=timezone.utc)
    short_one = GapCandidate(
        start,
        start + timedelta(hours=1),
        start + timedelta(seconds=30),
        start + timedelta(minutes=59, seconds=30),
        timedelta(hours=1),
    )
    short_two = GapCandidate(
        start + timedelta(days=1),
        start + timedelta(days=1, hours=1),
        start + timedelta(days=1, seconds=30),
        start + timedelta(days=1, minutes=59, seconds=30),
        timedelta(hours=1),
    )
    long_gap = GapCandidate(
        start + timedelta(days=2),
        start + timedelta(days=2, hours=23),
        start + timedelta(days=2, seconds=30),
        start + timedelta(days=2, hours=22, minutes=59, seconds=30),
        timedelta(hours=23),
    )

    assert heuristic_gap_candidates(
        [short_one, short_two, long_gap], timezone_name="UTC"
    ) == [long_gap]


def test_heuristic_suppresses_simple_weekend_gap():
    """A gap whose missing interval is Saturday/Sunday is typical by calendar."""

    candidate = GapCandidate(
        datetime(2025, 1, 3, 22, tzinfo=timezone.utc),
        datetime(2025, 1, 6, 1, tzinfo=timezone.utc),
        datetime(2025, 1, 4, 0, tzinfo=timezone.utc),
        datetime(2025, 1, 5, 23, tzinfo=timezone.utc),
        timedelta(days=2, hours=3),
    )

    assert heuristic_gap_candidates([candidate], timezone_name="UTC") == []


def test_schedule_filter_keeps_only_session_overlap():
    """Schedule filtering should suppress closed-market gaps."""

    open_gap = GapCandidate(
        datetime(2025, 1, 1, 10, tzinfo=timezone.utc),
        datetime(2025, 1, 1, 10, 2, tzinfo=timezone.utc),
        datetime(2025, 1, 1, 10, 0, 30, tzinfo=timezone.utc),
        datetime(2025, 1, 1, 10, 1, 30, tzinfo=timezone.utc),
        timedelta(minutes=2),
    )
    closed_gap = GapCandidate(
        datetime(2025, 1, 1, 21, tzinfo=timezone.utc),
        datetime(2025, 1, 1, 22, tzinfo=timezone.utc),
        datetime(2025, 1, 1, 21, 0, 30, tzinfo=timezone.utc),
        datetime(2025, 1, 1, 21, 59, 30, tzinfo=timezone.utc),
        timedelta(hours=1),
    )
    sessions = [
        SessionRange(
            datetime(2025, 1, 1, 9, tzinfo=timezone.utc),
            datetime(2025, 1, 1, 16, tzinfo=timezone.utc),
        )
    ]

    assert scheduled_gap_candidates(
        [open_gap, closed_gap], sessions, timezone_name="UTC"
    ) == [open_gap]


def test_sessions_from_historical_schedule_uses_schedule_timezone():
    """IB schedule strings should be interpreted in the schedule timezone."""

    schedule = SimpleNamespace(
        timeZone="US/Central",
        sessions=[
            SimpleNamespace(
                startDateTime="20250101-17:00:00",
                endDateTime="20250102-16:00:00",
            )
        ],
    )

    assert sessions_from_historical_schedule(schedule) == [
        SessionRange(
            datetime(2025, 1, 1, 23, tzinfo=timezone.utc),
            datetime(2025, 1, 2, 22, tzinfo=timezone.utc),
        )
    ]
