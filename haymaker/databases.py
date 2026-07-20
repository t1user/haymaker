"""Process-owned Mongo services and dataframe-store composition."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pymongo import MongoClient  # type: ignore
from pymongo.errors import ConfigurationError  # type: ignore

from .datastore.collection_namer import SymbolNamer

if TYPE_CHECKING:
    from .datastore import (
        AsyncDataStore,
        FrameStoreProvider,
        QueuedDataSink,
    )

log = logging.getLogger(__name__)


@dataclass
class MongoService:
    """Own one lazy Mongo client and its runtime health check.

    Args:
        client_options: Keyword arguments used to construct ``MongoClient``.
        health_checks: Runtime health checks populated as services are created.
    """

    client_options: Mapping[str, Any] = field(default_factory=dict, repr=False)
    health_checks: list[Callable[[], bool]] = field(default_factory=list)
    _mongo_client: MongoClient | None = field(default=None, init=False, repr=False)

    def mongo_client(self) -> MongoClient:
        """Return the runtime Mongo client, creating and probing it once."""

        if self._mongo_client is not None:
            return self._mongo_client
        try:
            client = MongoClient(**self.client_options)
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
            sorted(self.client_options),
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


@dataclass(frozen=True)
class _ArcticFrameStoreProvider:
    """Construct Arctic stores for the strategy-composition contract.

    Args:
        _mongo_client: Lazy accessor for the process-owned Mongo client.
    """

    _mongo_client: Callable[[], MongoClient] = field(repr=False)

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

        from .datastore import AsyncArcticStore

        return AsyncArcticStore(
            lib=library,
            host=self._mongo_client(),
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

        from .datastore import AsyncArcticStore

        return AsyncArcticStore(
            lib=library,
            host=self._mongo_client(),
            collection_namer=collection_namer,
        )


def create_frame_store_provider(
    mongo_client: Callable[[], MongoClient],
) -> FrameStoreProvider:
    """Return the narrow Arctic provider used during strategy composition.

    Args:
        mongo_client: Lazy accessor for the process-owned Mongo client.

    Returns:
        Provider that constructs separately configured Arctic wrappers.
    """

    return _ArcticFrameStoreProvider(mongo_client)
