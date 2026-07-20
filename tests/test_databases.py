"""Tests for runtime-owned persistence services."""

import asyncio

from unittest.mock import MagicMock, Mock, call

import ib_insync as ibi
import pandas as pd
import pytest

from haymaker import databases
from haymaker.async_wrappers import (
    QueueProcessingError,
    QueueShutdownPolicy,
    SyncQueueRunner,
)
from haymaker.databases import MongoService, create_frame_store_provider
from haymaker.datastore import (
    AbstractBaseStore,
    ArcticStore,
    AsyncArcticStore,
    AsyncDataStore,
    BarSizeSymbolNamer,
    FrameStoreProvider,
    QueuedDataSink,
)


def accepts_async_datastore(store: AsyncDataStore) -> None:
    """Type-check one structural async datastore implementation."""


def accepts_queued_sink(store: QueuedDataSink) -> None:
    """Type-check one structural queued datastore implementation."""


def accepts_frame_store_provider(provider: FrameStoreProvider) -> None:
    """Type-check the Arctic strategy-composition provider."""


def test_real_mongo_client_blocked_by_default() -> None:
    service = MongoService()

    with pytest.raises(AssertionError, match="real MongoDB"):
        service.mongo_client()


def test_mongo_service_creates_and_reuses_one_probed_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    created_with: dict[str, object] = {}

    def fake_mongo_client(**kwargs: object) -> object:
        created_with.update(kwargs)
        return client

    monkeypatch.setattr(databases, "MongoClient", fake_mongo_client)
    service = MongoService({"host": "mongodb://example"})

    assert service.mongo_client() is client
    assert service.mongo_client() is client

    assert created_with == {"host": "mongodb://example"}
    client.admin.command.assert_called_once_with("ping")
    assert service.health_checks == [service.mongodb_health_check]


def test_mongo_service_reraises_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_mongo_client(**kwargs: object) -> object:
        raise databases.ConfigurationError("bad mongo configuration")

    monkeypatch.setattr(databases, "MongoClient", fake_mongo_client)
    service = MongoService()

    with pytest.raises(databases.ConfigurationError):
        service.mongo_client()


def test_mongo_service_exposes_no_storage_policy() -> None:
    service = MongoService()

    assert not hasattr(service, "database")
    assert not hasattr(service, "path")
    assert not hasattr(service, "arctic_store")


def test_frame_store_provider_exposes_only_narrow_store_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The strategy provider should construct stores without exposing Mongo."""

    client = object()
    mongo_client = Mock(return_value=client)
    stores = [Mock(), Mock()]
    arctic_store = Mock(side_effect=stores)
    monkeypatch.setattr("haymaker.datastore.AsyncArcticStore", arctic_store)
    provider = create_frame_store_provider(mongo_client)
    namer = BarSizeSymbolNamer("30 secs")

    accepts_frame_store_provider(provider)
    assert provider.datastore("market_data", symbol_namer=namer) is stores[0]
    assert provider.queued_sink("block_data", symbol_namer=namer) is stores[1]
    assert arctic_store.call_args_list == [
        call(lib="market_data", host=client, symbol_namer=namer),
        call(lib="block_data", host=client, symbol_namer=namer),
    ]
    assert mongo_client.call_count == 2
    assert not hasattr(provider, "mongo_client")
    assert not hasattr(provider, "database")
    assert not hasattr(provider, "settings")
    assert not hasattr(provider, "store_factory")


def test_arctic_store_initializes_a_shared_library_only_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Separate store wrappers should not repeatedly initialize one library."""

    libraries: set[str] = set()
    existence_checks: list[str] = []
    initializations: list[str] = []
    instances: list[object] = []
    handles: list[object] = []

    class FakeArctic:
        """Track library lifecycle calls without opening MongoDB."""

        def __init__(self, host: object) -> None:
            self.host = host
            instances.append(self)

        def library_exists(self, library: str) -> bool:
            existence_checks.append(library)
            return library in libraries

        def initialize_library(self, library: str) -> None:
            initializations.append(library)
            libraries.add(library)

        def __getitem__(self, library: str) -> object:
            assert library in libraries
            handle = object()
            handles.append(handle)
            return handle

    monkeypatch.setattr("haymaker.datastore.datastore.Arctic", FakeArctic)

    first = ArcticStore("shared library", host="mongo.example")
    second = ArcticStore("shared library", host="mongo.example")

    assert existence_checks == ["shared_library", "shared_library"]
    assert initializations == ["shared_library"]
    assert instances == [first.db, second.db]
    assert first.db is not second.db
    assert handles == [first.store, second.store]
    assert first.store is not second.store


@pytest.mark.asyncio
async def test_awaited_arctic_mutations_return_backend_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Awaited mutations should finish inline and return backend results."""

    calls: list[tuple[str, tuple[object, ...]]] = []

    class FakeArcticStore:
        def __init__(self, lib, host, symbol_namer) -> None:
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

    accepts_async_datastore(store)
    accepts_queued_sink(store)
    assert await store.write("ES", data) == "write-version"
    assert await store.append("ES", data) == "append-version"
    assert await store.write_metadata("ES", {"complete": True}) == "metadata-version"
    assert [name for name, _ in calls] == ["write", "append", "metadata"]
    assert not hasattr(store, "async_write")


@pytest.mark.asyncio
async def test_awaited_metadata_failure_is_reported_at_call_site(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Awaited metadata writes should not defer failures until queue shutdown."""

    class FailingArcticStore:
        def __init__(self, lib, host, symbol_namer) -> None:
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
        await store.write_metadata("ES", {"complete": True})


@pytest.mark.asyncio
async def test_critical_arctic_store_uses_a_dedicated_draining_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A critical store must drain its own queued metadata before closing."""

    writes: list[tuple[str, dict[str, bool]]] = []

    class FakeArcticStore:
        def __init__(self, lib, host, symbol_namer) -> None:
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
    assert store.shutdown_policy is QueueShutdownPolicy.DRAIN

    store.enqueue_write_metadata("ES", {"complete": True})
    await store.close()

    assert writes == [("ES", {"complete": True})]


@pytest.mark.asyncio
async def test_critical_arctic_store_propagates_queued_write_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A draining store must report queued write failures during shutdown."""

    class FailingArcticStore:
        def __init__(self, lib, host, symbol_namer) -> None:
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
    store.enqueue_write_metadata("ES", {"complete": True})

    with pytest.raises(QueueProcessingError, match="failed to process queued work"):
        await store.close()


@pytest.mark.asyncio
async def test_queued_writes_keep_each_store_symbol_namer_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Queued contract writes must use each store's construction-time namer."""

    writes: dict[str, list[str]] = {"alpha": [], "beta": []}

    class RecordingArcticStore(ArcticStore):
        def __init__(self, lib, host, symbol_namer) -> None:
            AbstractBaseStore.__init__(self, symbol_namer)
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
        symbol_namer=alpha_namer,
        shutdown_policy=QueueShutdownPolicy.DRAIN,
    )
    beta_store = AsyncArcticStore(
        "beta",
        symbol_namer=beta_namer,
        shutdown_policy=QueueShutdownPolicy.DRAIN,
    )
    contract = ibi.Future(localSymbol="NQH6")
    data = pd.DataFrame({"close": [1]})

    alpha_store.enqueue_write(contract, data)
    beta_store.enqueue_write(contract, data)
    await alpha_store.close()
    await beta_store.close()

    assert writes == {"alpha": ["alpha_NQH6"], "beta": ["beta_NQH6"]}
    assert alpha_store.symbol_namer is alpha_namer
    assert beta_store.symbol_namer is beta_namer
    assert not hasattr(alpha_store, "override_symbol_namer")
    with pytest.raises(AttributeError):
        alpha_store.symbol_namer = beta_namer  # type: ignore[misc]
    with pytest.raises(AttributeError):
        alpha_store.store.symbol_namer = beta_namer  # type: ignore[misc]


@pytest.mark.asyncio
async def test_shared_default_queue_keeps_store_symbol_namers_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default queued sinks must retain their own symbol-naming policies."""

    writes: list[str] = []
    writes_complete = asyncio.Event()

    class RecordingArcticStore(ArcticStore):
        """Record symbols produced by each immutable store configuration."""

        def __init__(self, lib, host, symbol_namer) -> None:
            AbstractBaseStore.__init__(self, symbol_namer)
            self.lib = lib
            self.host = host

        def write(self, symbol, data, meta=None) -> str:
            persisted_symbol = self._symbol(symbol)
            writes.append(persisted_symbol)
            if len(writes) == 3:
                writes_complete.set()
            return persisted_symbol

    async def inline_to_thread(func, *args):
        return func(*args)

    def alpha_namer(contract: ibi.Contract) -> str:
        return f"alpha_{contract.localSymbol}"

    def beta_namer(contract: ibi.Contract) -> str:
        return f"beta_{contract.localSymbol}"

    shared_queue = SyncQueueRunner(
        "shared-default-store-queue",
        shutdown_policy=QueueShutdownPolicy.DISCARD,
    )
    monkeypatch.setattr(AsyncArcticStore, "_sync_class", RecordingArcticStore)
    monkeypatch.setattr(AsyncArcticStore, "_queue", shared_queue)
    monkeypatch.setattr("haymaker.async_wrappers.asyncio.to_thread", inline_to_thread)

    alpha_store = AsyncArcticStore("block_data", symbol_namer=alpha_namer)
    beta_store = AsyncArcticStore("block_data", symbol_namer=beta_namer)
    contract = ibi.Future(localSymbol="NQH6")
    data = pd.DataFrame({"close": [1]})

    assert alpha_store._queue is beta_store._queue is shared_queue
    assert alpha_store.shutdown_policy is QueueShutdownPolicy.DISCARD
    assert beta_store.shutdown_policy is QueueShutdownPolicy.DISCARD

    alpha_store.enqueue_write(contract, data)
    beta_store.enqueue_write(contract, data)
    alpha_store.enqueue_write(contract, data)
    await asyncio.wait_for(writes_complete.wait(), timeout=1)
    await shared_queue.close()

    assert writes == ["alpha_NQH6", "beta_NQH6", "alpha_NQH6"]
