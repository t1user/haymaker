"""Strategy-composition contracts for dataframe persistence."""

from __future__ import annotations

from typing import Protocol

from .async_datastore import AsyncDataStore, QueuedDataSink
from .symbol_namer import SymbolNamer


class FrameStoreProvider(Protocol):
    """Construct fully configured dataframe persistence dependencies."""

    def datastore(self, library: str, *, symbol_namer: SymbolNamer) -> AsyncDataStore:
        """Return an awaited datastore for strategy composition.

        Args:
            library: Backend dataframe library name.
            symbol_namer: Immutable contract-to-symbol naming policy.

        Returns:
            Datastore whose operations complete at their await sites.
        """

        ...

    def queued_sink(self, library: str, *, symbol_namer: SymbolNamer) -> QueuedDataSink:
        """Return a best-effort queued sink for strategy composition.

        Args:
            library: Backend dataframe library name.
            symbol_namer: Immutable contract-to-symbol naming policy.

        Returns:
            Sink whose explicit enqueue methods use best-effort shutdown.
        """

        ...
