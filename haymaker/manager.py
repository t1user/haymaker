from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Final, Self

import ib_insync as ibi

from .base import Atom, DetailsContainer
from .config import CONFIG
from .contract_selector import selector_factory
from .controller import Controller
from .state_machine import StateMachine
from .streamers import Streamer
from .trader import Trader

log = logging.getLogger(__name__)


class NotConnectedError(Exception):
    pass


@dataclass
class InitData:
    ib: ibi.IB
    contract_dict: dict[int, ibi.Contract]
    contract_details: DetailsContainer
    _contracts: dict[int, ibi.Contract] = field(default_factory=dict, repr=False)

    async def __call__(self) -> Self:
        log.debug(f"---------- INIT START -----> {len(self.contract_dict)} contracts.")
        if not self._contracts:
            self._contracts = self.contract_dict.copy()
            log.debug(
                f"contracts blueprint saved: {[c for c in self._contracts.values()]}"
            )
        details = await self.acquire_contract_details(list(self._contracts.values()))
        log.debug(f"Acquired details for {len(details)} contracts.")
        selectors = (selector_factory(details_list) for details_list in details)

        for contract_hash, selector in zip(self.contract_dict, selectors):
            self.contract_details[selector.active_contract] = selector.active_details
            self.contract_details[selector.next_contract] = selector.next_details
            self.contract_dict[contract_hash] = selector.next_contract
        log.debug("InitData done...")
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
                    *(IB.reqContractDetailsAsync(contract) for contract in contracts),
                    return_exceptions=False,
                )
            except Exception as e:
                log.debug(f"Failed to get contract details {e}")
                raise
        return details


class Jobs:
    _tasks: set = set()

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
        log.debug("Run streamers --->")
        for streamer in self.streamers:
            task = asyncio.create_task(streamer.run(), name=f"{streamer!s}, ")
            log.debug(f"Task created: {task.get_name()}")

            # Add task to the set. This creates a strong reference.
            self._tasks.add(task)

            # To prevent keeping references to finished tasks forever,
            # make each task remove its own reference from the set after
            # completion:
            task.add_done_callback(self._tasks.discard)
            # ensure errors are logged for debugging
            task.add_done_callback(self._handle_error)
        await asyncio.gather(*self._tasks, return_exceptions=True)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}"
            f"({'| '.join([str(s) for s in self.streamers])})"
        )


log.debug("--- INITIALIZATION ---")
IB: Final[ibi.IB] = ibi.IB()
# Atom passes empty contrianers so that INIT_DATA can supply them with data
# InitData knows nothing about Atom, just gets containers to fill-up
INIT_DATA = InitData(IB, Atom.contract_dict, Atom.contract_details)
JOBS = Jobs(INIT_DATA)
STATE_MACHINE = StateMachine()
Atom.set_init_data(IB, STATE_MACHINE)
log.debug("Will initialize Controller")
trader = Trader(IB)
CONTROLLER: Final[Controller] = Controller.from_config(trader, CONFIG)
