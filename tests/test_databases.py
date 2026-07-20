"""Tests for runtime-owned persistence services."""

from unittest.mock import MagicMock

import pytest

from haymaker import databases
from haymaker.async_wrappers import QueueProcessingError, QueueShutdownPolicy
from haymaker.config.settings import MongoSettings, StorageSettings
from haymaker.databases import StoreFactory
from haymaker.datastore import AsyncArcticStore


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
