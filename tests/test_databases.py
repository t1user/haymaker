"""Tests for runtime-owned persistence services."""

from unittest.mock import MagicMock

import ib_insync as ibi
import pandas as pd
import pytest

from haymaker import databases
from haymaker.async_wrappers import QueueProcessingError, QueueShutdownPolicy
from haymaker.config.settings import MongoSettings, StorageSettings
from haymaker.databases import StoreFactory
from haymaker.datastore import AbstractBaseStore, ArcticStore, AsyncArcticStore


def storage_settings(**client: object) -> StorageSettings:
    """Return storage settings with one test Mongo database."""

    return StorageSettings(mongodb=MongoSettings(client=client, database="test_data"))


def test_real_mongo_client_blocked_by_default() -> None:
    factory = StoreFactory(storage_settings())

    with pytest.raises(AssertionError, match="real MongoDB"):
        factory.mongo_client()


def test_store_factory_creates_and_reuses_one_probed_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    created_with: dict[str, object] = {}

    def fake_mongo_client(**kwargs: object) -> object:
        created_with.update(kwargs)
        return client

    monkeypatch.setattr(databases, "MongoClient", fake_mongo_client)
    factory = StoreFactory(storage_settings(host="mongodb://example"))

    assert factory.mongo_client() is client
    assert factory.mongo_client() is client

    assert created_with == {"host": "mongodb://example"}
    client.admin.command.assert_called_once_with("ping")
    assert factory.health_checks == [factory.mongodb_health_check]


def test_store_factory_reraises_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_mongo_client(**kwargs: object) -> object:
        raise databases.ConfigurationError("bad mongo configuration")

    monkeypatch.setattr(databases, "MongoClient", fake_mongo_client)
    factory = StoreFactory(storage_settings())

    with pytest.raises(databases.ConfigurationError):
        factory.mongo_client()


def test_database_name_is_required_only_for_savers() -> None:
    factory = StoreFactory(StorageSettings())

    with pytest.raises(ValueError, match="database"):
        _ = factory.database


@pytest.mark.asyncio
async def test_awaited_arctic_mutations_return_backend_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Awaited mutations should finish inline and return backend results."""

    calls: list[tuple[str, tuple[object, ...]]] = []

    class FakeArcticStore:
        def __init__(self, lib, host, collection_namer) -> None:
            self.lib = lib
            self.host = host

        def write(self, *args):
            calls.append(("write", args))
            return "write-version"

        def append(self, *args):
            calls.append(("append", args))
            return "append-version"

        def write_metadata(self, *args):
            calls.append(("metadata", args))
            return "metadata-version"

    async def inline_make_async(func, *args):
        return func(*args)

    monkeypatch.setattr(AsyncArcticStore, "_sync_class", FakeArcticStore)
    monkeypatch.setattr(
        "haymaker.datastore.async_datastore.make_async", inline_make_async
    )
    store = AsyncArcticStore("awaited")
    data = pd.DataFrame({"close": [1]})

    assert await store.async_write("ES", data) == "write-version"
    assert await store.async_append("ES", data) == "append-version"
    assert (
        await store.async_write_metadata("ES", {"complete": True}) == "metadata-version"
    )
    assert [name for name, _ in calls] == ["write", "append", "metadata"]


@pytest.mark.asyncio
async def test_awaited_metadata_failure_is_reported_at_call_site(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Awaited metadata writes should not defer failures until queue shutdown."""

    class FailingArcticStore:
        def __init__(self, lib, host, collection_namer) -> None:
            self.lib = lib
            self.host = host

        def write_metadata(self, symbol, metadata) -> None:
            raise RuntimeError("metadata write failed")

    async def inline_make_async(func, *args):
        return func(*args)

    monkeypatch.setattr(AsyncArcticStore, "_sync_class", FailingArcticStore)
    monkeypatch.setattr(
        "haymaker.datastore.async_datastore.make_async", inline_make_async
    )
    store = AsyncArcticStore("awaited-failure")

    with pytest.raises(RuntimeError, match="metadata write failed"):
        await store.async_write_metadata("ES", {"complete": True})


@pytest.mark.asyncio
async def test_critical_arctic_store_uses_a_dedicated_draining_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A critical store must drain its own queued metadata before closing."""

    writes: list[tuple[str, dict[str, bool]]] = []

    class FakeArcticStore:
        def __init__(self, lib, host, collection_namer) -> None:
            self.lib = lib
            self.host = host

        def write_metadata(self, symbol, metadata) -> None:
            writes.append((symbol, metadata))

    async def inline_to_thread(func, *args):
        return func(*args)

    monkeypatch.setattr(AsyncArcticStore, "_sync_class", FakeArcticStore)
    monkeypatch.setattr("haymaker.async_wrappers.asyncio.to_thread", inline_to_thread)

    store = AsyncArcticStore(
        "critical",
        shutdown_policy=QueueShutdownPolicy.DRAIN,
    )

    assert store._queue is not AsyncArcticStore._queue
    assert store._queue.shutdown_policy is QueueShutdownPolicy.DRAIN

    store.write_metadata("ES", {"complete": True})
    await store.close()

    assert writes == [("ES", {"complete": True})]


@pytest.mark.asyncio
async def test_critical_arctic_store_propagates_queued_write_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A draining store must report queued write failures during shutdown."""

    class FailingArcticStore:
        def __init__(self, lib, host, collection_namer) -> None:
            self.lib = lib
            self.host = host

        def write_metadata(self, symbol, metadata) -> None:
            raise RuntimeError("write failed")

    async def inline_to_thread(func, *args):
        return func(*args)

    monkeypatch.setattr(AsyncArcticStore, "_sync_class", FailingArcticStore)
    monkeypatch.setattr("haymaker.async_wrappers.asyncio.to_thread", inline_to_thread)

    store = AsyncArcticStore(
        "critical-failure",
        shutdown_policy=QueueShutdownPolicy.DRAIN,
    )
    store.write_metadata("ES", {"complete": True})

    with pytest.raises(QueueProcessingError, match="failed to process queued work"):
        await store.close()


@pytest.mark.asyncio
async def test_queued_writes_keep_each_store_symbol_namer_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Queued contract writes must use each store's construction-time namer."""

    writes: dict[str, list[str]] = {"alpha": [], "beta": []}

    class RecordingArcticStore(ArcticStore):
        def __init__(self, lib, host, collection_namer) -> None:
            AbstractBaseStore.__init__(self, collection_namer)
            self.lib = lib
            self.host = host

        def write(self, symbol, data, meta=None) -> str:
            persisted_symbol = self._symbol(symbol)
            writes[self.lib].append(persisted_symbol)
            return persisted_symbol

    async def inline_to_thread(func, *args):
        return func(*args)

    def alpha_namer(contract: ibi.Contract) -> str:
        return f"alpha_{contract.localSymbol}"

    def beta_namer(contract: ibi.Contract) -> str:
        return f"beta_{contract.localSymbol}"

    monkeypatch.setattr(AsyncArcticStore, "_sync_class", RecordingArcticStore)
    monkeypatch.setattr("haymaker.async_wrappers.asyncio.to_thread", inline_to_thread)

    alpha_store = AsyncArcticStore(
        "alpha",
        collection_namer=alpha_namer,
        shutdown_policy=QueueShutdownPolicy.DRAIN,
    )
    beta_store = AsyncArcticStore(
        "beta",
        collection_namer=beta_namer,
        shutdown_policy=QueueShutdownPolicy.DRAIN,
    )
    contract = ibi.Future(localSymbol="NQH6")
    data = pd.DataFrame({"close": [1]})

    alpha_store.write(contract, data)
    beta_store.write(contract, data)
    await alpha_store.close()
    await beta_store.close()

    assert writes == {"alpha": ["alpha_NQH6"], "beta": ["beta_NQH6"]}
    assert alpha_store.symbol_namer is alpha_namer
    assert beta_store.symbol_namer is beta_namer
    assert not hasattr(alpha_store, "override_collection_namer")
    with pytest.raises(AttributeError):
        alpha_store.symbol_namer = beta_namer  # type: ignore[misc]
    with pytest.raises(AttributeError):
        alpha_store.store.symbol_namer = beta_namer  # type: ignore[misc]
