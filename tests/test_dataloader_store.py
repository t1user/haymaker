import importlib
import sys
from datetime import datetime, timezone

import ib_insync as ibi
import pandas as pd
import pytest

from haymaker.dataloader.store_wrapper import StoreWrapper
from haymaker.dataloader.scheduling import task_factory


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

    def __init__(self, data: pd.DataFrame | None = None) -> None:
        self.data = data
        self.reads: list[ibi.Contract] = []
        self.writes: list[tuple[ibi.Contract, pd.DataFrame]] = []

    async def read(self, contract: ibi.Contract) -> pd.DataFrame | None:
        """Return configured data and record the requested contract."""

        self.reads.append(contract)
        return self.data

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
async def test_store_wrapper_preloads_existing_data(contract, store_index):
    """StoreWrapper should preload data through the async datastore API."""

    data = pd.DataFrame({"close": [1, 2, 3]}, index=store_index)
    store = FakeAsyncStore(data)

    wrapper = await StoreWrapper.create(contract, store, store_index[-1])

    assert store.reads == [contract]
    assert wrapper.from_date == store_index[1]
    assert wrapper.to_date == store_index[-1]


@pytest.mark.asyncio
async def test_store_wrapper_one_row_uses_single_timestamp_boundary(
    contract, store_index
):
    """One stored row should use that point as the stored-data boundary."""

    data = pd.DataFrame({"close": [1]}, index=store_index[:1])

    wrapper = await StoreWrapper.create(contract, FakeAsyncStore(data), store_index[-1])

    assert wrapper.from_date == store_index[0]
    assert wrapper.to_date == store_index[0]


@pytest.mark.asyncio
async def test_task_factory_one_row_store_does_not_schedule_full_duplicate(
    contract, store_index
):
    """A one-row collection should schedule around the stored boundary."""

    data = pd.DataFrame({"close": [1]}, index=store_index[:1])
    wrapper = await StoreWrapper.create(contract, FakeAsyncStore(data), store_index[-1])

    tasks = task_factory(wrapper, store_index[0])

    assert tasks == [(store_index[0], store_index[-1])]


@pytest.mark.asyncio
async def test_store_wrapper_writes_through_async_store(contract, store_index):
    """StoreWrapper writes should use async_write and refresh cached data."""

    initial = pd.DataFrame({"close": [1]}, index=store_index[:1])
    updated = pd.DataFrame({"close": [1, 2]}, index=store_index[:2])
    store = FakeAsyncStore(initial)
    wrapper = await StoreWrapper.create(contract, store, store_index[-1])

    version = await wrapper.write_async(contract, updated)

    assert version == "version"
    assert store.writes == [(contract, updated)]
    assert wrapper.data is updated


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

    store = dataloader_module.create_dataloader_store(
        wts="TRADES", bar_size="30 secs"
    )

    assert isinstance(store, FakeAsyncArcticStore)
    assert created == {"lib": "TRADES_30_secs", "host": "mongo"}
