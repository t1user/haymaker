"""Process-owned Mongo and datastore construction services."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from pymongo import MongoClient  # type: ignore
from pymongo.errors import ConfigurationError  # type: ignore

from .async_wrappers import QueueShutdownPolicy
from .config.settings import StorageSettings
from .datastore.collection_namer import SymbolNamer

if TYPE_CHECKING:
    from .datastore import (
        AsyncArcticStore,
        AsyncDataStore,
        FrameStoreProvider,
        QueuedDataSink,
    )

log = logging.getLogger(__name__)


@dataclass
class StoreFactory:
    """Own lazy persistence services for one application runtime.

    Args:
        settings: Validated filesystem, MongoDB, and datastore settings.
        health_checks: Runtime health checks populated as services are created.
    """

    settings: StorageSettings
    health_checks: list[Callable[[], bool]] = field(default_factory=list)
    _mongo_client: MongoClient | None = field(default=None, init=False, repr=False)

    def mongo_client(self) -> MongoClient:
        """Return the runtime Mongo client, creating and probing it once."""

        if self._mongo_client is not None:
            return self._mongo_client
        try:
            client = MongoClient(**self.settings.mongodb.client)
            client.admin.command("ping")
        except ConfigurationError:
            log.critical("Invalid MongoDB client configuration.")
            raise
        except Exception:
            log.exception("Failed to initialize MongoDB client.")
            raise
        self._mongo_client = client
        self.health_checks.append(self.mongodb_health_check)
        log.debug(
            "Started MongoDB with configured keys: %s",
            sorted(self.settings.mongodb.client),
        )
        return client

    def mongodb_health_check(self) -> bool:
        """Return whether the runtime Mongo client responds to a ping."""

        try:
            self.mongo_client().admin.command("ping")
            return True
        except Exception:
            log.exception("MongoDB health check failed.")
            return False

    def path(self, *parts: str) -> Path:
        """Return a created path below the configured home data directory."""

        path = Path.home() / self.settings.base_directory
        for part in parts:
            path /= part
        path.mkdir(exist_ok=True, parents=True)
        return path

    def arctic_store(
        self,
        library: str,
        collection_namer: SymbolNamer | None = None,
        shutdown_policy: QueueShutdownPolicy = QueueShutdownPolicy.DISCARD,
    ) -> AsyncArcticStore:
        """Construct an asynchronous Arctic store using the runtime client."""

        from .datastore import AsyncArcticStore

        return AsyncArcticStore(
            lib=library,
            host=self.mongo_client(),
            collection_namer=collection_namer,
            shutdown_policy=shutdown_policy,
        )

    def frame_store_provider(self) -> FrameStoreProvider:
        """Return the narrow strategy-composition view of this factory.

        Returns:
            Provider exposing only configured dataframe dependency creation.
        """

        return _StoreFactoryFrameStoreProvider(self)

    @property
    def database(self) -> str:
        """Return the configured application database name."""

        database = self.settings.mongodb.database
        if not database:
            raise ValueError("storage.mongodb.database is required for Mongo savers")
        return database


@dataclass(frozen=True)
class _StoreFactoryFrameStoreProvider:
    """Adapt ``StoreFactory`` to the strategy frame-store contract.

    Args:
        _store_factory: Temporary backing factory retained until DS-005.
    """

    _store_factory: StoreFactory = field(repr=False)

    def datastore(
        self, library: str, *, collection_namer: SymbolNamer
    ) -> AsyncDataStore:
        """Return an awaited dataframe datastore.

        Args:
            library: Arctic library name.
            collection_namer: Immutable contract-to-symbol naming policy.

        Returns:
            Awaited dataframe datastore backed by Arctic.
        """

        return self._store_factory.arctic_store(
            library,
            collection_namer=collection_namer,
        )

    def queued_sink(
        self, library: str, *, collection_namer: SymbolNamer
    ) -> QueuedDataSink:
        """Return a best-effort queued dataframe sink.

        Args:
            library: Arctic library name.
            collection_namer: Immutable contract-to-symbol naming policy.

        Returns:
            Queued dataframe sink backed by Arctic.
        """

        return self._store_factory.arctic_store(
            library,
            collection_namer=collection_namer,
        )
