"""Strategy-composition contracts for dataframe persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from haymaker.async_wrappers import QueueShutdownPolicy

from .async_datastore import AsyncDataStore, QueuedDataSink
from .symbol_namer import MarketDataSymbolNamer, SymbolNamer


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

    def queued_sink(
        self,
        library: str,
        *,
        symbol_namer: SymbolNamer,
        shutdown_policy: QueueShutdownPolicy = QueueShutdownPolicy.DISCARD,
    ) -> QueuedDataSink:
        """Return a best-effort queued sink for strategy composition.

        Args:
            library: Backend dataframe library name.
            symbol_namer: Immutable contract-to-symbol naming policy.

        Returns:
            Sink whose explicit enqueue methods use ``shutdown_policy``.
        """

        ...


@dataclass
class MarketDataStoreFactory:
    """Create and cache runtime-default stores for historical bar series.

    Args:
        provider: Runtime-owned dataframe store provider.
        library: Backend dataframe library used for market-data history.

    One store is reused for each normalized combination of bar size, market
    data type, and regular-hours policy. Custom stores supplied directly to a
    component do not pass through this factory.
    """

    provider: FrameStoreProvider = field(repr=False)
    library: str
    _stores: dict[tuple[str, str, bool], AsyncDataStore] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Validate the configured market-data library name."""

        if not isinstance(self.library, str):
            raise TypeError("market_data_store.library must be a string")
        if not self.library:
            raise ValueError("market_data_store.library must not be empty")

    def __call__(
        self,
        *,
        bar_size_setting: str,
        what_to_show: str,
        use_rth: bool,
    ) -> AsyncDataStore:
        """Return the shared default store for one historical-bar identity.

        Args:
            bar_size_setting: Interactive Brokers bar-size value.
            what_to_show: Interactive Brokers market-data type.
            use_rth: Whether the series contains regular-hours bars only.

        Returns:
            Awaited datastore configured with collision-safe symbol naming.
        """

        symbol_namer = MarketDataSymbolNamer(
            barSizeSetting=bar_size_setting,
            whatToShow=what_to_show,
            useRTH=use_rth,
        )
        key = symbol_namer.identity
        if key not in self._stores:
            self._stores[key] = self.provider.datastore(
                self.library,
                symbol_namer=symbol_namer,
            )
        return self._stores[key]
