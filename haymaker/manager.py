from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Final, cast

import ib_insync as ibi

from .base import Atom, DetailsContainer
from .controller import Controller
from .state_machine import StateMachine
from .streamers import Streamer

log = logging.getLogger(__name__)


@dataclass
class InitData:
    ib: ibi.IB
    contract_list: list[ibi.Contract]
    contract_details: DetailsContainer

    async def __call__(self) -> "InitData":
        try:
            await self.qualify_contracts()
            await self.acquire_contract_details()
        except Exception as e:
            log.exception(e)

        return self

    async def qualify_contracts(self) -> "InitData":
        # "reset" conId to make IB check what the current contract is
        for c in self.contract_list:
            if isinstance(c, ibi.ContFuture) and c.conId != 0:
                c.conId = 0
                log.debug(f"ContFuture reset: {c}")
        try:
            # qualify
            await self.ib.qualifyContractsAsync(*self.contract_list)

            # for every ContFuture add regular Future object, that will be actually
            # used by Atoms
            futures_for_cont_futures = [
                ibi.Future(conId=c.conId)
                for c in self.contract_list
                if isinstance(c, ibi.ContFuture)
            ]
            await self.ib.qualifyContractsAsync(*futures_for_cont_futures)
        except Exception as e:
            log.exception(e)

        self.contract_list.extend(futures_for_cont_futures)
        log.debug(f"contracts qualified {set([c.symbol for c in self.contract_list])}")
        return self

    async def acquire_contract_details(self) -> "InitData":
        for contract in set(self.contract_list):
            log.debug(f"Acquiring details for: {contract.symbol}")
            details_ = await IB.reqContractDetailsAsync(contract)
            try:
                assert len(details_) == 1
            except AssertionError:
                log.exception(f"Ambiguous contract: {contract}. Critical error.")

            details = details_[0]
            self.contract_details[cast(ibi.Contract, details.contract)] = details
        log.debug(
            f"Details acquired: {set([k.symbol for k in self.contract_details.keys()])}"
        )
        return self


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
            f"{ {p.contract.symbol: p.position for p in IB.positions()} }"
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
        await asyncio.gather(*self._tasks, return_exceptions=False)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}({self.init_data})"


IB: Final[ibi.IB] = ibi.IB()
# Atom passes empty contrianers so that INIT_DATA can supply them with data
INIT_DATA = InitData(IB, Atom.contracts, Atom.contract_details)
JOBS = Jobs(INIT_DATA)
Atom.set_init_data(INIT_DATA.ib, StateMachine())
log.debug("Will initialize Controller")
CONTROLLER: Final[Controller] = Controller()
