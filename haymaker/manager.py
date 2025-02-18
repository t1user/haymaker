from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Final, cast

import ib_insync as ibi

from .base import Atom, DetailsContainer
from .controller import Controller
from .state_machine import StateMachine
from .streamers import Streamer

log = logging.getLogger(__name__)


class NotConnectedError(Exception):
    pass


@dataclass
class InitData:
    ib: ibi.IB
    contract_list: list[ibi.Contract]
    contract_details: DetailsContainer
    _master_contfuture_dict: dict[int, dict[str, str]] = field(
        init=False, repr=False, default_factory=dict
    )
    _futures: set[ibi.Future] = field(init=False, repr=False, default_factory=set)

    def _build_contfuture_list(self) -> None:
        log.debug("building contfuture list...")
        self._master_contfuture_dict = {
            id(contract): ibi.util.dataclassNonDefaults(contract)
            for contract in self.contract_list
            if isinstance(contract, ibi.ContFuture)
        }
        log.debug(f"{self._master_contfuture_dict=}")

    async def __call__(self) -> "InitData":
        log.debug(f"Number of contracts: {len(self.contract_list)}")
        if not self._master_contfuture_dict:
            self._build_contfuture_list()
        try:
            self.replace_contfutures()
            await self.qualify_contracts()
            await self.acquire_contract_details()
        except Exception as e:
            log.exception(e)
            # raise
        return self

    def replace_contfutures(self) -> "InitData":
        # "reset" ContFuture to make IB check what the current contract is
        default_contfuture_kwargs = asdict(ibi.ContFuture())
        for c in self.contract_list:
            if isinstance(c, ibi.ContFuture) and c.conId != 0:
                # modify contract 'in place' making sure it stays the same object
                non_defaults = self._master_contfuture_dict[id(c)]
                c.__dict__.update(**default_contfuture_kwargs)
                c.__dict__.update(**non_defaults)
                log.debug(f"ContFuture reset: {c}")
        return self

    async def qualify_contracts(self) -> "InitData":
        while not all([f.conId != 0 for f in self.contract_list]):
            if not self.ib.isConnected():
                log.error(
                    "Will not qualify contracts, no connection. Waiting for restart..."
                )
                raise NotConnectedError()
                break
            log.debug("Will attempt to qualify contracts...")
            await self.ib.qualifyContractsAsync(*self.contract_list)
            log.debug(f"{len(self.contract_list)} Contract objects qualified")
            # for every ContFuture add regular Future object, that will be actually
            # used by Atoms
            futures_for_cont_futures = [
                ibi.Future(conId=c.conId)
                for c in self.contract_list
                if isinstance(c, ibi.ContFuture)
            ]
            await self.ib.qualifyContractsAsync(*futures_for_cont_futures)

            dropped_futures = self._futures - set(futures_for_cont_futures)
            new_futures = set(futures_for_cont_futures) - self._futures
            if new_futures and dropped_futures:
                log.warning(
                    f"Dropping futures: "
                    f"{[(f.localSymbol, type(f)) for f in dropped_futures]} "
                    f"{len(new_futures)} new future objects found."
                )
            self.contract_list.extend(list(new_futures))
            for f in dropped_futures:
                # TODO: check why f might be more than once in the list
                while f in self.contract_list:
                    log.debug(f"removing from contract list: {f}")
                    self.contract_list.remove(f)
            self._futures = set(futures_for_cont_futures)

            contract_string = ", ".join(
                (f"{c.__class__.__name__}({c.localSymbol})" for c in self.contract_list)
            )
            log.debug(f"All qualified contracts: {contract_string}")

        return self

    async def acquire_contract_details(self) -> "InitData":
        for contract in set(self.contract_list):
            log.debug(
                f"Acquiring details for: {contract.__class__.__name__}"
                f"(symbol={contract.symbol}, localSymbol={contract.localSymbol})"
            )
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
        log.debug("Will initialize...")
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
        return f"{self.__class__.__qualname__}({self.init_data})"


IB: Final[ibi.IB] = ibi.IB()
# Atom passes empty contrianers so that INIT_DATA can supply them with data
# InitData knows nothing about Atom, just gets containers to fill-up
INIT_DATA = InitData(
    IB,
    Atom.contracts,
    Atom.contract_details,
)
JOBS = Jobs(INIT_DATA)
Atom.set_init_data(INIT_DATA.ib, StateMachine())
log.debug("Will initialize Controller")
CONTROLLER: Final[Controller] = Controller()
