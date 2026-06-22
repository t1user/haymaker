import importlib
import sys
from datetime import date, datetime, timezone

import ib_insync as ibi
import pandas as pd
import pytest

from haymaker.dataloader.scheduling import (
    TaskPlanner,
    task_factory,
    task_factory_with_gaps,
)
from haymaker.dataloader.helpers import duration_in_secs
from haymaker.dataloader.store_wrapper import AsyncStoreView, HistorySink


@pytest.fixture
def dataloader_module(monkeypatch):
    """Import dataloader with hermetic config for store-factory tests."""

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
    assert wrapper.from_date == store_index[1]
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

    assert wrapper.from_date == store_index[0]
    assert wrapper.to_date == store_index[0]


@pytest.mark.asyncio
async def test_task_factory_one_row_store_does_not_schedule_full_duplicate(
    contract, store_index
):
    """A one-row collection should schedule around the stored boundary."""

    data = pd.DataFrame({"close": [1]}, index=store_index[:1])
    wrapper = await AsyncStoreView.create(
        contract, FakeAsyncStore(data), store_index[-1], "30 secs"
    )

    tasks = task_factory(wrapper, store_index[0])

    assert tasks == [(store_index[0], store_index[-1])]


@pytest.mark.asyncio
async def test_task_planner_clamps_start_to_max_period(contract):
    """TaskPlanner should own the run lookback clamp for download ranges."""

    now = datetime(2025, 1, 10, tzinfo=timezone.utc)
    headstamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
    wrapper = await AsyncStoreView.create(contract, FakeAsyncStore(), now, "30 secs")

    tasks = TaskPlanner(
        wrapper,
        headstamp,
        max_period_days=3,
        fill_gaps=False,
    ).ranges()

    assert tasks == [(datetime(2025, 1, 7, tzinfo=timezone.utc), now)]


@pytest.mark.asyncio
async def test_task_planner_missing_metadata_still_schedules_backfill(contract):
    """Missing optional metadata should not prevent normal backfill planning."""

    now = datetime(2025, 1, 10, tzinfo=timezone.utc)
    headstamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
    wrapper = await AsyncStoreView.create(contract, FakeAsyncStore(), now, "30 secs")

    tasks = TaskPlanner(
        wrapper,
        headstamp,
        max_period_days=3,
        fill_gaps=False,
    ).planned_ranges()

    assert tasks == [
        (
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
        max_period_days=100,
        fill_gaps=False,
    ).planned_ranges()

    assert tasks == [(store_index[-1], now, "update")]
    assert task_factory(wrapper, datetime(2025, 1, 1, tzinfo=timezone.utc)) == [
        (store_index[-1], now)
    ]


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
            max_period_days=100,
            fill_gaps=False,
        ).planned_ranges()
        == []
    )


@pytest.mark.asyncio
async def test_task_planner_preserves_gap_factory_order(contract):
    """TaskPlanner should preserve legacy gap-first scheduling behavior."""

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

    tasks = TaskPlanner(
        wrapper,
        headstamp,
        max_period_days=100,
        fill_gaps=True,
    ).ranges()

    assert tasks == task_factory_with_gaps(wrapper, headstamp)


@pytest.mark.asyncio
async def test_async_store_view_dates_update_after_reload(contract, store_index):
    """Date boundaries should reflect the latest loaded store data."""

    store = FakeAsyncStore(pd.DataFrame({"close": [1]}, index=store_index[:1]))
    wrapper = await AsyncStoreView.create(contract, store, store_index[-1], "30 secs")
    store.data = pd.DataFrame({"close": [1, 2, 3]}, index=store_index)

    await wrapper.read()

    assert wrapper.from_date == store_index[1]
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

    assert wrapper.from_date == date(2025, 1, 2)
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
async def test_history_sink_does_not_mark_empty_series(contract):
    """Empty series should not be marked as exhausted without a data anchor."""

    store = FakeAsyncStore()
    sink = HistorySink(contract, store)

    assert await sink.mark_backfill_exhausted() is None
    assert store.metadata_writes == []


def test_create_dataloader_store_builds_async_arctic_store(
    monkeypatch, dataloader_module
):
    """The dataloader store factory should build the only supported backend."""

    created: dict[str, object] = {}

    class FakeAsyncArcticStore:
        """Capture constructor values without opening Arctic."""

        def __init__(self, lib: str, host: object) -> None:
            created["lib"] = lib
            created["host"] = host

    monkeypatch.setattr(dataloader_module, "get_mongo_client", lambda: "mongo")
    monkeypatch.setattr(dataloader_module, "AsyncArcticStore", FakeAsyncArcticStore)

    store = dataloader_module.create_dataloader_store(wts="TRADES", bar_size="30 secs")

    assert isinstance(store, FakeAsyncArcticStore)
    assert created == {"lib": "TRADES_30_secs", "host": "mongo"}
