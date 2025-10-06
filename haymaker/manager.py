from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Final, Self

import ib_insync as ibi

from . import misc
from .base import ActiveNext, Atom, DetailsContainer
from .blotter import blotter_factory
from .config import CONFIG
from .contract_selector import selector_factory
from .controller import Controller
from .databases import HEALTH_CHECK_OBSERVABLES
from .state_machine import StateMachine
from .streamers import Streamer
from .timeout import Timeout
from .trader import Trader

log = logging.getLogger(__name__)


class NotConnectedError(Exception):
    pass


FUTURES_ROLL_BDAYS = CONFIG["futures_roll_bdays"]
FUTURES_ROLL_MARGIN_BDAYS = CONFIG["futures_roll_margin_bdays"]
USE_BLOTTER = CONFIG["use_blotter"]


@dataclass
class InitData:
    ib: ibi.IB
    # contract_dict and contract details are saved on Atom class
    contract_dict: dict[tuple[misc.ContractKey, ActiveNext], ibi.Contract]
    contract_details: DetailsContainer
    _contracts: dict[tuple[misc.ContractKey, ActiveNext], ibi.Contract] = field(
        default_factory=dict, repr=False
    )

    async def __call__(self) -> Self:
        log.debug(f"---------- INIT START -----> {len(self.contract_dict)} contracts.")
        if not self._contracts:
            self._contracts = self.contract_dict.copy()
            log.debug(
                f"contracts blueprint saved: {[c for c in self._contracts.values()]}"
            )
        # details always from blueprint
        details = await self.acquire_contract_details(list(self._contracts.values()))
        log.debug(f"Acquired details for {len(details)} contracts.")
        selectors = (
            selector_factory(
                details_list, FUTURES_ROLL_BDAYS, FUTURES_ROLL_MARGIN_BDAYS
            )
            for details_list in details
        )
        _details = {
            details.contract: details
            for details_list in details
            for details in details_list
        }

        for (contract_hash, _), selector in zip(self.contract_dict.copy(), selectors):
            for tag in ActiveNext:
                contract = misc.general_to_specific_contract_class(
                    getattr(selector, f"{tag.name.lower()}_contract")
                )
                self.contract_details[contract] = _details[contract]
                self.contract_dict[(contract_hash, tag)] = contract

        log.debug("InitData done...")
        contract_dict_str = " | ".join(
            [
                f"{str(k[0])}_{str(k[1])}: {v.localSymbol}"
                for k, v in self.contract_dict.items()
            ]
        )
        log.debug(f"contract_dict {len(self.contract_dict)} items: {contract_dict_str}")
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
        contract.includeExpired = True
        return contract


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
INIT_DATA = InitData(IB, Atom.contract_dict, Atom.contract_details)
JOBS = Jobs(INIT_DATA)
STATE_MACHINE = StateMachine()
Atom.set_init_data(IB, STATE_MACHINE)
Timeout.set_ib(IB)
log.debug("Will initialize Controller")
trader = Trader(IB)
blotter = blotter_factory(USE_BLOTTER)
CONTROLLER: Final[Controller] = Controller.from_config(
    trader, blotter, CONFIG, [HEALTH_CHECK_OBSERVABLES]
)
