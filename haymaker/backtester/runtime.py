"""Assemble isolated framework services for one replay simulation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, cast

import ib_insync as ibi

from haymaker.async_wrappers import QueueShutdownPolicy
from haymaker.base import Atom
from haymaker.book import Book
from haymaker.config import TimeoutPolicy
from haymaker.contract_registry import ContractRegistry
from haymaker.datastore import (
    AsyncDataStore,
    FrameStoreProvider,
    MarketDataStoreFactory,
    QueuedDataSink,
    SignalFramePersistence,
)
from haymaker.order_defaults import OrderDefaults
from haymaker.runtime import RuntimeContext
from haymaker.trader import Trader

from .broker import SimulatedIB
from .controller import SimulationController
from .exceptions import UnsupportedBacktestFeatureError
from .results import InMemorySaver


class _UnavailableFrameStoreProvider:
    """Reject persistence construction from a read-only simulation graph."""

    @staticmethod
    def datastore(
        library: str, *, symbol_namer: Callable[[ibi.Contract], str]
    ) -> AsyncDataStore:
        """Reject an awaited persistence dependency.

        Args:
            library: Requested backend library.
            symbol_namer: Requested Contract naming policy.

        Raises:
            UnsupportedBacktestFeatureError: Always; a replay is read-only.
        """

        del library, symbol_namer
        raise UnsupportedBacktestFeatureError(
            "Strategy-created dataframe stores are not supported in backtests"
        )

    @staticmethod
    def queued_sink(
        library: str,
        *,
        symbol_namer: Callable[[ibi.Contract], str],
        shutdown_policy: QueueShutdownPolicy = QueueShutdownPolicy.DISCARD,
    ) -> QueuedDataSink:
        """Reject a queued persistence dependency.

        Args:
            library: Requested backend library.
            symbol_namer: Requested Contract naming policy.
            shutdown_policy: Requested queue shutdown policy.

        Raises:
            UnsupportedBacktestFeatureError: Always; a replay is read-only.
        """

        del library, symbol_namer, shutdown_policy
        raise UnsupportedBacktestFeatureError(
            "Strategy-created queued stores are not supported in backtests"
        )


class _UnavailableMarketDataStoreFactory:
    """Reject live component persistence against the replay input store."""

    def __call__(
        self,
        *,
        bar_size_setting: str,
        what_to_show: str,
        use_rth: bool,
    ) -> AsyncDataStore:
        """Reject runtime-default market-data persistence.

        Args:
            bar_size_setting: Requested IB bar size.
            what_to_show: Requested IB market-data type.
            use_rth: Requested session policy.

        Raises:
            UnsupportedBacktestFeatureError: Always; replay input is read-only.
        """

        del bar_size_setting, what_to_show, use_rth
        raise UnsupportedBacktestFeatureError(
            "HistoricalDataStreamer(datastore=True) and dataframe persistence "
            "are not supported in backtests"
        )


def _unavailable_signal_persistence() -> SignalFramePersistence:
    """Reject runtime-default Signal dataframe persistence.

    Raises:
        UnsupportedBacktestFeatureError: Always; replay output is in memory.
    """

    raise UnsupportedBacktestFeatureError(
        "PandasSignalModel(persistence=True) is not supported in backtests"
    )


@dataclass
class SimulationRuntime:
    """Own framework services adapted for one in-memory simulation.

    Args:
        ib: Simulated broker and execution venue.
        contract_registry: Framework Contract registry used by strategy Atoms.
        book: Real framework Book backed by in-memory savers.
        context: Passive runtime context installed on :class:`~haymaker.base.Atom`.
        controller: Controller adapter retaining the real order and fill path.
    """

    ib: SimulatedIB
    contract_registry: ContractRegistry
    book: Book
    context: RuntimeContext
    controller: SimulationController

    @classmethod
    def create(
        cls,
        *,
        initial_cash: float,
        slippage_ticks: float,
        order_defaults: OrderDefaults | None = None,
        futures_roll_bdays: int = 3,
        futures_roll_margin_bdays: int = 3,
    ) -> "SimulationRuntime":
        """Construct and install one complete simulation context.

        Args:
            initial_cash: Opening account value reported by the simulated venue.
            slippage_ticks: Adverse whole or fractional ticks applied per fill.
            order_defaults: Optional framework order defaults for execution models.
            futures_roll_bdays: ACTIVE futures selection lead time.
            futures_roll_margin_bdays: NEXT futures selection lead time.

        Returns:
            Ready runtime for replay strategy orders.
        """

        ib = SimulatedIB(
            initial_cash=initial_cash,
            slippage_ticks=slippage_ticks,
        )
        registry = ContractRegistry(
            futures_roll_bdays=futures_roll_bdays,
            futures_roll_margin_bdays=futures_roll_margin_bdays,
        )
        book = Book(
            order_saver=InMemorySaver("orderId"),
            state_saver=InMemorySaver("state_key"),
            save_async=False,
        )
        trader = Trader(cast(ibi.IB, ib))
        provider = cast(FrameStoreProvider, _UnavailableFrameStoreProvider())
        context = RuntimeContext(
            ib=cast(ibi.IB, ib),
            contract_registry=registry,
            book=book,
            trader=trader,
            frame_store_provider=provider,
            market_data_store_factory=cast(
                MarketDataStoreFactory, _UnavailableMarketDataStoreFactory()
            ),
            signal_persistence_factory=_unavailable_signal_persistence,
            order_defaults=order_defaults or OrderDefaults(),
            timeout_policy=TimeoutPolicy(seconds=0, action="log"),
            request_restart=lambda _reason: False,
            run_started_at=datetime.now(timezone.utc),
            workload_generation=1,
        )
        Atom.set_runtime_context(context)
        controller = SimulationController(trader=trader)
        context.controller = controller
        return cls(ib, registry, book, context, controller)

    async def close(self) -> None:
        """Close the real Book's in-memory mutation boundary."""

        await self.book.close()


__all__ = ["SimulationRuntime"]
