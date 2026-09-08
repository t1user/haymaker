"""Persistence policy for calculated Signal dataframes."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

import ib_insync as ibi
import pandas as pd

from haymaker.async_wrappers import QueueShutdownPolicy
from haymaker.misc import tree

from .async_datastore import QueuedDataSink
from .provider import FrameStoreProvider
from .symbol_namer import simple_symbol_namer

log = logging.getLogger(__name__)


@runtime_checkable
class SignalFramePersistence(Protocol):
    """Queue calculated Signal dataframes and return their lookup reference.

    Implement this protocol to customize how
    :class:`haymaker.components.PandasSignalModel` calculation data is stored.
    ``save`` is called synchronously on the trading data path and must therefore
    only accept or queue work; it must not wait for storage I/O.

    One instance normally belongs to one SignalModel because implementations
    may retain generation and append state.
    """

    def save(
        self,
        frame: pd.DataFrame,
        *,
        source_key: str,
        active_contract: ibi.Contract,
        run_started_at: datetime,
    ) -> str | None:
        """Queue one calculated dataframe for persistence.

        Args:
            frame: Complete calculated dataframe. The caller must not mutate it
                after it has been accepted by this method.
            source_key: Stable identity of the producing SignalModel.
            active_contract: Selector's ACTIVE Contract for generation naming.
            run_started_at: Fixed process run timestamp.

        Returns:
            Storage reference after the work has been accepted, or ``None``
            when no reference should be attached to the emitted Signal.
        """

        ...


@dataclass(eq=False)
class QueuedSignalFramePersistence:
    """Persist calculated Signal dataframes through an ordered ``DRAIN`` sink.

    Use this implementation when supplying a custom queued sink to a
    :class:`haymaker.components.PandasSignalModel`. The first calculation in a
    process/ACTIVE-contract generation writes the complete dataframe. Later
    calculations append only rows newer than the last accepted index.

    Args:
        sink: Queue-only dataframe sink configured with
            ``QueueShutdownPolicy.DRAIN``.

    Raises:
        ValueError: If the sink does not use the required ``DRAIN`` policy.
    """

    sink: QueuedDataSink = field(repr=False)
    _symbol: str | None = field(default=None, init=False, repr=False)
    _active_con_id: int | None = field(default=None, init=False, repr=False)
    _last_index: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Reject a sink that could discard accepted audit work at shutdown."""

        if self.sink.shutdown_policy is not QueueShutdownPolicy.DRAIN:
            raise ValueError("Signal frame persistence sink must use DRAIN shutdown")

    def save(
        self,
        frame: pd.DataFrame,
        *,
        source_key: str,
        active_contract: ibi.Contract,
        run_started_at: datetime,
    ) -> str:
        """Queue a complete generation or its newly calculated rows.

        Args:
            frame: Complete calculated dataframe for the current observation.
            source_key: Stable identity of the producing SignalModel.
            active_contract: Selector's ACTIVE Contract.
            run_started_at: Fixed process run timestamp.

        Returns:
            Run-scoped dataframe symbol after queue acceptance.
        """

        symbol = self._generation_symbol(
            source_key,
            active_contract,
            run_started_at,
        )
        metadata = {
            "source_key": source_key,
            "run_started_at": run_started_at,
            "active_contract": tree(active_contract),
        }
        is_new_generation = (
            self._active_con_id != active_contract.conId or self._symbol != symbol
        )
        if is_new_generation:
            self.sink.enqueue_write(symbol, frame, metadata)
            self._symbol = symbol
            self._active_con_id = active_contract.conId
            log.info("Started Signal dataframe generation %s", symbol)
        else:
            new_rows = frame
            if self._last_index is not None:
                new_rows = frame.loc[frame.index > self._last_index]
            if not new_rows.empty:
                self.sink.enqueue_append(symbol, new_rows, metadata)
                log.debug("Appended Signal dataframe generation %s", symbol)
        self._last_index = frame.index[-1]
        return symbol

    @staticmethod
    def _generation_symbol(
        source_key: str,
        active_contract: ibi.Contract,
        run_started_at: datetime,
    ) -> str:
        """Return the stable symbol for one process/ACTIVE generation."""

        contract_name = active_contract.localSymbol or active_contract.symbol
        return f"{source_key}_{contract_name}_{run_started_at.isoformat()}"


@dataclass(frozen=True)
class SignalFramePersistenceFactory:
    """Create independent default Signal dataframe persistence objects.

    Args:
        provider: Runtime-owned dataframe store provider.
        library: Arctic library used for calculated Signal data.
    """

    provider: FrameStoreProvider = field(repr=False)
    library: str

    def __post_init__(self) -> None:
        """Validate the configured default library name."""

        if not isinstance(self.library, str):
            raise TypeError("signal persistence library must be a string")
        if not self.library:
            raise ValueError("signal persistence library must not be empty")

    def __call__(self) -> SignalFramePersistence:
        """Create one model-owned persistence object with a dedicated queue."""

        sink = self.provider.queued_sink(
            self.library,
            symbol_namer=simple_symbol_namer,
            shutdown_policy=QueueShutdownPolicy.DRAIN,
        )
        return QueuedSignalFramePersistence(sink)


__all__ = [
    "QueuedSignalFramePersistence",
    "SignalFramePersistence",
    "SignalFramePersistenceFactory",
]
