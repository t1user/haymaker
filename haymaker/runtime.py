from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Self

import ib_insync as ibi

from .base import Atom
from .blotter import blotter_factory
from .book import (
    DEFAULT_ORDER_COLLECTION_NAME,
    DEFAULT_STATE_COLLECTION_NAME,
    Book,
)
from .config.settings import (
    LiveConfig,
    MarketDataStoreSettings,
    SignalFramePersistenceSettings,
    TimeoutPolicy,
)
from .contract_registry import ContractRegistry
from .controller.controller import Controller, SyncOutcome
from .databases import MongoService, create_frame_store_provider
from .datastore import (
    FrameStoreProvider,
    MarketDataStoreFactory,
    SignalFramePersistence,
    SignalFramePersistenceFactory,
)
from .handlers import IBHandlers
from .order_defaults import OrderDefaults
from .saver import MongoSaver
from .components.messages import StandardOrderRole
from .components.streamers import Streamer
from .components.timeouts import MarketDataTimeout
from .trader import Trader

log = logging.getLogger(__name__)


class NotConnectedError(Exception):
    """Raised when runtime startup cannot collect broker data."""


@dataclass
class InitData:
    """Load broker contract details into the runtime contract registry."""

    ib: ibi.IB
    contract_registry: ContractRegistry

    async def __call__(self) -> Self:
        """Refresh contract details for all registered contract blueprints."""

        log.debug(
            f"---------- INIT START -----> "
            f"{len(self.contract_registry.blueprints)} contracts."
        )
        blueprints = self.contract_registry.blueprints.copy()
        details = await self.acquire_contract_details(blueprints)
        log.debug(f"Acquired details for {len(details)} contracts.")
        self.contract_registry.reset_data(details)
        log.debug(
            f"Active contracts: {self.contract_registry.active_contracts_for_logs()}"
        )
        return self

    async def acquire_contract_details(
        self, contracts: list[ibi.Contract]
    ) -> list[list[ibi.ContractDetails]]:
        """Return contract details for all provided contract blueprints."""

        details: list[list[ibi.ContractDetails]] = []
        while len(details) != len(contracts):
            if not self.ib.isConnected():
                raise NotConnectedError()

            try:
                details = await asyncio.gather(
                    *(
                        self.ib.reqContractDetailsAsync(self._include_expired(contract))
                        for contract in contracts
                    ),
                    return_exceptions=False,
                )
            except Exception as exc:
                log.debug(f"Failed to get contract details {exc}")
                raise
        return details

    @staticmethod
    def _include_expired(contract: ibi.Contract) -> ibi.Contract:
        contract_ = copy(contract)
        contract_.includeExpired = True
        return contract_


class StartupJobs:
    """Hold contract initialization and run streamers after Controller recovery.

    LiveRuntime awaits ``init_data`` before Controller.run(), so recovery can
    price protection using qualified Contract details. ``run`` starts only the
    market-data jobs, after Controller's reconciliation outcome is available.
    """

    def __init__(
        self, init_data: InitData, ib: ibi.IB, streamers: Sequence[Streamer]
    ) -> None:
        self.init_data = init_data
        self.ib = ib
        self.streamers = streamers

    async def run(self) -> None:
        """Run streamers after runtime contract initialization and recovery."""

        log.info(
            f"Open positions on restart: "
            f"{ {p.contract.symbol: p.position for p in self.ib.positions()} }"
        )
        order_dict = defaultdict(list)
        for trade in self.ib.openTrades():
            order_dict[trade.contract.symbol].append(
                (
                    trade.order.orderId,
                    trade.order.orderType,
                    trade.order.action,
                    trade.order.totalQuantity,
                )
            )
        log.info(f"Orders on restart: {dict(order_dict)}")
        log.debug("Run streamers --->")
        await asyncio.gather(
            *[
                asyncio.create_task(streamer.run(), name=f"{streamer!s}, ")
                for streamer in self.streamers
            ]
        )

    def __str__(self) -> str:
        """Return a compact description of registered streamer jobs."""

        return (
            f"{self.__class__.__qualname__}"
            f"({'| '.join([str(streamer) for streamer in self.streamers])})"
        )

    def __repr__(self) -> str:
        """Return a diagnostic representation of startup job dependencies."""

        return (
            f"{self.__class__.__qualname__}(init_data={self.init_data!r}, "
            f"ib={self.ib!r}, streamers={self.streamers!r})"
        )


@dataclass
class RuntimeContext:
    """Ready live runtime services and metadata shared by Haymaker atoms.

    Attributes:
        ib: Broker client owned by the live process.
        contract_registry: Qualified contract and contract-details registry.
        book: Persistent typed accounting and order state.
        trader: Thin broker order gateway.
        frame_store_provider: Narrow dataframe persistence composition service.
        market_data_store_factory: Return shared runtime-default broker-bar
            datastores by request identity.
        signal_persistence_factory: Create one runtime-default Signal dataframe
            persistence object per requesting model.
        order_defaults: Validated default order fields for execution models.
        timeout_policy: Default timeout interval and action.
        controller: Live controller installed by ``LiveRuntime``.
        request_restart: Supervisor restart callback, bound before startup.
        future_roll_policies: Per-source automatic futures-roll policy.
        run_started_at: Fixed process/component-graph creation timestamp.
        workload_generation: Supervised workload-start generation counter.
    """

    ib: ibi.IB
    contract_registry: ContractRegistry = field(repr=False)
    book: Book = field(repr=False)
    trader: Trader = field(repr=False)
    frame_store_provider: FrameStoreProvider = field(repr=False)
    market_data_store_factory: MarketDataStoreFactory = field(repr=False)
    signal_persistence_factory: Callable[[], SignalFramePersistence] = field(repr=False)
    order_defaults: OrderDefaults = field(repr=False)
    timeout_policy: TimeoutPolicy = field(repr=False)
    controller: Controller = field(init=False, repr=False)
    request_restart: Callable[[str], bool | None] | None = field(
        default=None, repr=False
    )
    future_roll_policies: dict[str, bool] = field(default_factory=dict, repr=False)
    run_started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    workload_generation: int = 0

    def __str__(self) -> str:
        """Return a compact runtime summary suitable for logs."""
        return f"RuntimeContext<contracts={len(self.contract_registry.blueprints)}>"


@dataclass
class LiveRuntime:
    """Construct and coordinate one process-owned live runtime.

    Args:
        config: Merged framework configuration for this live process.
        ib: Optional broker client owned by the runtime.
        mongo_service: Optional Mongo client service for focused callers and tests.
        frame_store_provider: Optional strategy-composition persistence provider.
        contract_registry: Optional preconfigured contract registry.
        book: Optional preconfigured accounting book.
    """

    config: LiveConfig = field(repr=False)
    ib: ibi.IB = field(default_factory=ibi.IB)
    mongo_service: MongoService | None = field(default=None, repr=False)
    contract_registry: ContractRegistry | None = field(default=None, repr=False)
    book: Book | None = field(default=None, repr=False)
    frame_store_provider: FrameStoreProvider | None = field(default=None, repr=False)
    context: RuntimeContext = field(init=False, repr=False)
    startup_jobs: StartupJobs = field(init=False, repr=False)
    _broker_logger: IBHandlers | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Build services and install a complete Atom context before returning."""

        if self.mongo_service is None:
            self.mongo_service = MongoService(self.config.storage.mongodb.client)
        if self.frame_store_provider is None:
            self.frame_store_provider = create_frame_store_provider(
                self.mongo_service.mongo_client
            )
        if self.contract_registry is None:
            self.contract_registry = ContractRegistry(**dict(self.config.futures))
        blotter = blotter_factory(
            self.config.blotter,
            base_directory=self.config.storage.base_directory,
            mongo_client=self.mongo_service.mongo_client,
            database=self.config.storage.mongodb.database,
        )
        if self.book is None:
            self.book = self._create_book(self.config.book, blotter=blotter)
        trader = Trader(self.ib)
        self.context = RuntimeContext(
            ib=self.ib,
            contract_registry=self.contract_registry,
            book=self.book,
            trader=trader,
            frame_store_provider=self.frame_store_provider,
            market_data_store_factory=MarketDataStoreFactory(
                self.frame_store_provider,
                MarketDataStoreSettings.from_mapping(
                    self.config.market_data_store
                ).library,
            ),
            signal_persistence_factory=SignalFramePersistenceFactory(
                self.frame_store_provider,
                SignalFramePersistenceSettings.from_mapping(
                    self.config.signal_persistence
                ).library,
            ),
            order_defaults=OrderDefaults.from_mapping(self.config.orders),
            timeout_policy=TimeoutPolicy.from_mapping(self.config.timeout),
        )
        Atom.set_runtime_context(self.context)
        self.context.controller = Controller.from_mapping(
            self.config.controller,
            trader=trader,
            health_check_observables=[self.mongo_service.health_checks],
        )
        self.startup_jobs = StartupJobs(
            InitData(self.ib, self.contract_registry),
            self.ib,
            Streamer.instances,
        )
        if self.config.logging.get("log_broker", False):
            self._broker_logger = IBHandlers(self.ib)

    def _create_book(self, settings: Mapping[str, Any], *, blotter) -> Book:
        """Construct Book persistence from configuration and runtime storage.

        Args:
            settings: Merged ``book`` configuration section.
            blotter: Runtime blotter owned by the constructed Book.

        Returns:
            Book using runtime-owned Mongo savers.
        """

        assert self.mongo_service is not None
        options = dict(settings)
        order_collection_name = options.pop(
            "order_collection_name", DEFAULT_ORDER_COLLECTION_NAME
        )
        state_collection_name = options.pop(
            "state_collection_name", DEFAULT_STATE_COLLECTION_NAME
        )
        mongo_client = self.mongo_service.mongo_client()
        database = self._framework_database()
        return Book(
            order_saver=MongoSaver(
                order_collection_name,
                query_key="orderId",
                client=mongo_client,
                database=database,
            ),
            state_saver=MongoSaver(
                state_collection_name,
                query_key="state_key",
                client=mongo_client,
                database=database,
            ),
            blotter=blotter,
            **options,
        )

    def _framework_database(self) -> str:
        """Return the configured database required by framework Mongo savers."""

        database = self.config.storage.mongodb.database
        if not database:
            raise ValueError("storage.mongodb.database is required for Mongo savers")
        return database

    def bind_supervisor(
        self,
        request_restart: Callable[[str], bool | None],
        connection_unavailable: asyncio.Event,
    ) -> None:
        """Bind supervisor controls used by live runtime components."""

        self.context.request_restart = request_restart
        self.context.controller.set_sync_abort_event(connection_unavailable)

    async def start(self) -> None:
        """Start controller and strategy jobs after connectivity is verified."""

        self.context.workload_generation += 1
        try:
            log.debug("Will run controller...")
            self.context.controller.set_future_roll_policies(
                self.context.future_roll_policies
            )
            # Recovery builds price-sensitive protective orders before any
            # streamer starts; it needs actual Contract ticks, not fallbacks.
            await self.startup_jobs.init_data()
            controller_outcome = await self.context.controller.run()
            if controller_outcome is SyncOutcome.ABORTED:
                return
            await self.startup_jobs.run()
        finally:
            MarketDataTimeout._cancel_all()

    async def stop(self, reason: str) -> None:
        """Put the controller on hold while supervised work stops."""

        MarketDataTimeout._cancel_all()
        self.context.controller.set_hold()
        log.debug("Stopping live runtime: %s", reason)

    async def close(self) -> None:
        """Flush final live-runtime state before process shutdown."""

        self._warn_active_target_adjustments()
        self._warn_active_futures_rolls()
        await self.context.book.close()

    def _warn_active_target_adjustments(self) -> None:
        """Warn when a later deployment must preserve routed recovery."""

        active = self.context.book.orders.active(
            role=StandardOrderRole.TARGET_ADJUSTMENT
        )
        if not active:
            return
        orders = [
            {
                "orderId": info.orderId,
                "permId": info.permId,
                "execution_model_name": info.execution_model_name,
                "contract": (
                    info.trade.contract.localSymbol or info.trade.contract.symbol
                ),
                "conId": info.trade.contract.conId,
                "working_quantity": info.signed_working_quantity,
            }
            for info in active
        ]
        log.warning(
            "Process is exiting with active TARGET_ADJUSTMENT orders. Restart "
            "with unchanged routing rules and recovery-compatible execution "
            "model names and configuration, or finish/cancel these orders "
            "before deploying execution changes: %s",
            orders,
        )

    def _warn_active_futures_rolls(self) -> None:
        """Warn that incomplete roll state must remain recovery-compatible."""

        states = self.context.book.rolls.all(active_only=True)
        orders = self.context.book.orders.active(role=StandardOrderRole.ROLL)
        if not states and not orders:
            return
        log.warning(
            "Process is exiting with incomplete futures-roll work. Preserve "
            "the registered roll mode and executor name on restart: states=%s "
            "active_order_ids=%s",
            [
                {
                    "series_key": state.series_key,
                    "mode": state.mode.value,
                    "executor_name": state.executor_name,
                    "stage": state.stage.value,
                }
                for state in states
            ],
            [info.orderId for info in orders],
        )

    def __str__(self) -> str:
        """Return a compact live-runtime description suitable for logs."""

        return (
            f"LiveRuntime<contracts={len(self.context.contract_registry.blueprints)}, "
            f"streamers={len(self.startup_jobs.streamers)}>"
        )
