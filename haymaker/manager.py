from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from copy import copy
from dataclasses import dataclass
from typing import Final, Self

import ib_insync as ibi

from .base import Atom
from .blotter import blotter_factory
from .config import CONFIG
from .contract_registry import ContractRegistry
from .controller import Controller
from .databases import HEALTH_CHECK_OBSERVABLES
from .state_machine import StateMachine
from .streamers import Streamer
from .timeout import Timeout
from .trader import Trader

log = logging.getLogger(__name__)


class NotConnectedError(Exception):
    pass


USE_BLOTTER = CONFIG["use_blotter"]


@dataclass
class InitData:
    """
    Obtain certain information about traded contracts from Interactive
    Brokers and directly set parameters on `Atom` class to be
    available for all objects inheriting from `Atom`.
    """

    ib: ibi.IB
    contract_registry: ContractRegistry

    async def __call__(self) -> Self:
        log.debug(
            f"---------- INIT START -----> "
            f"{len(self.contract_registry.blueprints)} contracts."
        )

        # making tripple sure blueprints will not be modified
        blueprints = self.contract_registry.blueprints.copy()

        # details always from blueprint, not from qualified contracts
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

        details: list[list[ibi.ContractDetails]] = []

        while not (len(details) == len(contracts)):
            if not self.ib.isConnected():
                log.error("No connection, no contract details. Waiting for restart...")
                raise NotConnectedError()

            try:
                details = await asyncio.gather(
                    *(
                        IB.reqContractDetailsAsync(self._include_expired(contract))
                        for contract in contracts
                    ),
                    return_exceptions=False,
                )
            except Exception as e:
                log.debug(f"Failed to get contract details {e}")
                raise
        return details

    @staticmethod
    def _include_expired(contract):
        # make sure NOT to modify contracts because these objects are
        # used as keys used to lookup `contract` on all Atoms
        # Important as fuck!!!!
        contract_ = copy(contract)
        contract_.includeExpired = True
        return contract_


class Jobs:

    def __init__(self, init_data: InitData):
        self.init_data = init_data
        self.streamers = Streamer.instances

    def _handle_error(self, task: asyncio.Task):
        try:
            task.result()
        except asyncio.CancelledError:
            log.debug(f"task {task} has been cancelled.")
        except Exception as e:
            log.exception(e)

    async def __call__(self):

        await self.init_data()

        log.info(
            f"Open positions on restart: "
            f"{ {p.contract.symbol: p.position for p in IB.positions()} }"  # noqa
        )
        order_dict = defaultdict(list)
        for t in IB.openTrades():
            order_dict[t.contract.symbol].append(
                (
                    t.order.orderId,
                    t.order.orderType,
                    t.order.action,
                    t.order.totalQuantity,
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
        return (
            f"{self.__class__.__qualname__}"
            f"({'| '.join([str(s) for s in self.streamers])})"
        )


log.debug("--- INITIALIZATION ---")
IB: Final[ibi.IB] = ibi.IB()
# Atom passes empty contrianers so that INIT_DATA can supply them with data
# InitData knows nothing about Atom, just gets containers to fill-up
CONTRACT_REGISTRY = ContractRegistry()
INIT_DATA = InitData(IB, CONTRACT_REGISTRY)
JOBS = Jobs(INIT_DATA)
STATE_MACHINE = StateMachine()
Atom.set_init_data(IB, STATE_MACHINE, CONTRACT_REGISTRY)
Timeout.set_ib(IB)
log.debug("Will initialize Controller")
trader = Trader(IB)
blotter = blotter_factory(USE_BLOTTER)
CONTROLLER: Final[Controller] = Controller.from_config(
    trader, blotter, CONFIG, [HEALTH_CHECK_OBSERVABLES]
)
