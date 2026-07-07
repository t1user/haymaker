from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from copy import copy
from dataclasses import dataclass, field
from types import ModuleType
from typing import Self

import ib_insync as ibi

from .base import Atom
from .blotter import blotter_factory
from .contract_registry import ContractRegistry
from .controller import Controller
from .databases import HEALTH_CHECK_OBSERVABLES
from .handlers import IBHandlers
from .state_machine import StateMachine
from .streamers import Streamer
from .timeout import Timeout
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
                log.error("No connection, no contract details. Waiting for restart...")
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


class Jobs:
    """Run startup data collection and all streamers for a runtime context."""

    def __init__(
        self, init_data: InitData, ib: ibi.IB, streamers: Sequence[Streamer]
    ) -> None:
        self.init_data = init_data
        self.ib = ib
        self.streamers = list(streamers)

    async def __call__(self) -> None:
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

    def __repr__(self) -> str:
        """Return a compact representation of registered streamer jobs."""

        return (
            f"{self.__class__.__qualname__}"
            f"({'| '.join([str(streamer) for streamer in self.streamers])})"
        )


@dataclass
class RuntimeContext:
    """Process-owned live runtime services shared by Haymaker atoms."""

    config: MutableMapping
    ib: ibi.IB
    contract_registry: ContractRegistry
    state_machine: StateMachine
    trader: Trader
    _controller: Controller | None = field(default=None, init=False, repr=False)
    jobs: Jobs | None = field(default=None, init=False)
    no_future_roll_strategies: list[str] = field(default_factory=list)
    _restart_handler: Callable[[str], bool | None] | None = field(
        default=None, init=False, repr=False
    )
    _broker_logger: IBHandlers | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_config(cls, config: MutableMapping) -> RuntimeContext:
        """Create runtime services from Haymaker configuration."""

        ib = ibi.IB()
        contract_registry = ContractRegistry()
        state_machine = StateMachine()
        trader = Trader(ib)
        context = cls(config, ib, contract_registry, state_machine, trader)
        Atom.set_runtime_context(context)
        context.controller = Controller.from_config(
            trader,
            blotter_factory(config["use_blotter"]),
            config,
            [HEALTH_CHECK_OBSERVABLES],
        )
        if config.get("log_broker"):
            context._broker_logger = IBHandlers(ib)
        return context

    @property
    def controller(self) -> Controller:
        """Return the process controller after it has been initialized."""

        if self._controller is None:
            raise RuntimeError("Runtime controller has not been initialized.")
        return self._controller

    @controller.setter
    def controller(self, controller: Controller) -> None:
        self._controller = controller

    @property
    def sm(self) -> StateMachine:
        """Return the runtime state machine."""

        return self.state_machine

    def bind_restart_handler(
        self, restart_handler: Callable[[str], bool | None]
    ) -> None:
        """Bind the supervisor restart callback for runtime components."""

        self._restart_handler = restart_handler

    def restart_request(self, reason: str = "") -> bool | None:
        """Request a supervised restart when the supervisor is available."""

        if self._restart_handler is None:
            log.error("Cannot restart: no supervisor restart handler configured.")
            return False
        return self._restart_handler(reason)

    def bind_strategy_module(self, module: ModuleType) -> None:
        """Apply strategy-module metadata after module-level pipelines exist."""

        self.no_future_roll_strategies = self._read_no_future_roll_strategies(module)
        self.jobs = Jobs(
            InitData(self.ib, self.contract_registry),
            self.ib,
            Streamer.instances,
        )

    @staticmethod
    def _read_no_future_roll_strategies(module: ModuleType) -> list[str]:
        strategies = getattr(module, "no_future_roll_strategies", [])
        if strategies is None:
            return []
        if not isinstance(strategies, list) or not all(
            isinstance(strategy, str) for strategy in strategies
        ):
            raise TypeError(
                "Strategy module variable no_future_roll_strategies must be "
                "a list[str]."
            )
        return list(strategies)

    def require_jobs(self) -> Jobs:
        """Return live jobs after strategy module loading has completed."""

        if self.jobs is None:
            raise RuntimeError("Runtime jobs have not been initialized.")
        return self.jobs
