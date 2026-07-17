from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import copy
from dataclasses import dataclass, field
from typing import Any, Self

import ib_insync as ibi

from .base import Atom
from .blotter import blotter_factory
from .config.settings import LiveConfig
from .contract_registry import ContractRegistry
from .controller import Controller
from .databases import StoreFactory
from .handlers import IBHandlers
from .order_defaults import OrderDefaults
from .saver import MongoLatestSaver, MongoSaver
from .state_machine import (
    DEFAULT_ORDER_COLLECTION_NAME,
    DEFAULT_STRATEGY_COLLECTION_NAME,
    StateMachine,
)
from .streamers import Streamer
from .timeout import Timeout, TimeoutPolicy
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
    """Run startup data collection and all streamers for a runtime context."""

    def __init__(
        self, init_data: InitData, ib: ibi.IB, streamers: Sequence[Streamer]
    ) -> None:
        self.init_data = init_data
        self.ib = ib
        self.streamers = streamers

    async def run(self) -> None:
        """Initialize contract data and run all registered streamers."""

        await self.init_data()

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
        Timeout.reset()
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
        sm: Persistent strategy and order state.
        trader: Thin broker order gateway.
        store_factory: Runtime-owned persistence service factory.
        order_defaults: Validated default order fields for execution models.
        timeout_policy: Default timeout interval and action.
        dataframe_save_frequency: Default aggregation save interval in seconds.
        controller: Live controller installed by ``LiveRuntime``.
        request_restart: Supervisor restart callback, bound before startup.
        future_roll_policies: Per-strategy automatic futures-roll policy.
    """

    ib: ibi.IB
    contract_registry: ContractRegistry = field(repr=False)
    sm: StateMachine = field(repr=False)
    trader: Trader = field(repr=False)
    store_factory: StoreFactory = field(repr=False)
    order_defaults: OrderDefaults = field(repr=False)
    timeout_policy: TimeoutPolicy = field(repr=False)
    dataframe_save_frequency: int = field(repr=False)
    controller: Controller = field(init=False, repr=False)
    request_restart: Callable[[str], bool | None] | None = field(
        default=None, repr=False
    )
    future_roll_policies: dict[str, bool] = field(default_factory=dict, repr=False)

    def __str__(self) -> str:
        """Return a compact runtime summary suitable for logs."""
        return f"RuntimeContext<contracts={len(self.contract_registry.blueprints)}>"


@dataclass
class LiveRuntime:
    """Construct and coordinate one process-owned live runtime.

    Args:
        config: Merged framework configuration for this live process.
        ib: Optional broker client owned by the runtime.
        store_factory: Optional persistence factory for focused callers and tests.
        contract_registry: Optional preconfigured contract registry.
        sm: Optional preconfigured state machine.
    """

    config: LiveConfig = field(repr=False)
    ib: ibi.IB = field(default_factory=ibi.IB)
    store_factory: StoreFactory | None = field(default=None, repr=False)
    contract_registry: ContractRegistry | None = field(default=None, repr=False)
    sm: StateMachine | None = field(default=None, repr=False)
    context: RuntimeContext = field(init=False, repr=False)
    startup_jobs: StartupJobs = field(init=False, repr=False)
    _broker_logger: IBHandlers | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Build services and install a complete Atom context before returning."""

        if self.store_factory is None:
            self.store_factory = StoreFactory(self.config.storage)
        if self.contract_registry is None:
            self.contract_registry = ContractRegistry(**dict(self.config.futures))
        if self.sm is None:
            self.sm = self._create_state_machine(self.config.state_machine)
        trader = Trader(self.ib)
        self.context = RuntimeContext(
            self.ib,
            self.contract_registry,
            self.sm,
            trader,
            self.store_factory,
            OrderDefaults.from_mapping(self.config.orders),
            TimeoutPolicy.from_mapping(self.config.timeout),
            self.config.storage.dataframe_save_frequency,
        )
        Atom.set_runtime_context(self.context)
        self.context.controller = Controller.from_mapping(
            self.config.controller,
            trader=trader,
            blotter=blotter_factory(self.config.blotter, self.store_factory),
            health_check_observables=[self.store_factory.health_checks],
        )
        self.startup_jobs = StartupJobs(
            InitData(self.ib, self.contract_registry),
            self.ib,
            Streamer.instances,
        )
        if self.config.logging.get("log_broker", False):
            self._broker_logger = IBHandlers(self.ib)

    def _create_state_machine(self, settings: Mapping[str, Any]) -> StateMachine:
        """Construct state services from configuration and injected storage.

        Args:
            settings: Merged ``state_machine`` configuration section.

        Returns:
            State machine using runtime-owned Mongo savers.
        """

        assert self.store_factory is not None
        options = dict(settings)
        order_collection_name = options.pop(
            "order_collection_name", DEFAULT_ORDER_COLLECTION_NAME
        )
        strategy_collection_name = options.pop(
            "strategy_collection_name", DEFAULT_STRATEGY_COLLECTION_NAME
        )
        mongo_client = self.store_factory.mongo_client()
        return StateMachine(
            order_saver=MongoSaver(
                order_collection_name,
                query_key="orderId",
                client=mongo_client,
                database=self.store_factory.database,
            ),
            strategy_saver=MongoLatestSaver(
                strategy_collection_name,
                client=mongo_client,
                database=self.store_factory.database,
            ),
            **options,
        )

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

        log.debug("Will run controller...")
        self.context.controller.set_future_roll_policies(
            self.context.future_roll_policies
        )
        await self.context.controller.run()
        await self.startup_jobs.run()

    async def stop(self, reason: str) -> None:
        """Put the controller on hold while supervised work stops."""

        self.context.controller.set_hold()
        log.debug("Stopping live runtime: %s", reason)

    async def close(self) -> None:
        """Flush final live-runtime state before process shutdown."""

        self.context.sm.flush_pending_save()

    def __str__(self) -> str:
        """Return a compact live-runtime description suitable for logs."""

        return (
            f"LiveRuntime<contracts={len(self.context.contract_registry.blueprints)}, "
            f"streamers={len(self.startup_jobs.streamers)}>"
        )
