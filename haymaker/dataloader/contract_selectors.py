from __future__ import annotations

import logging
from dataclasses import fields
from datetime import date
from typing import AsyncGenerator, Self, Type

import ib_insync as ibi

from haymaker.config import CONFIG
from haymaker.misc import async_cached_property

log = logging.getLogger(__name__)


FUTURES_SELECTOR = CONFIG.get("futures_selector", "current")
FUTURES_FULLCHAIN_SPEC = CONFIG.get("futures_fullchain_spec", "full")
DESIRED_INDEX = CONFIG.get("futures_current_index", 0)


class ContractSelector:
    """
    Based on passed contract attributes and config parameters
    determine instrument(s) that should be added to data download
    queue.  The output of any computations is accessible via
    :meth:`objects`

    :meth:`from_kwargs` will return instance of correct
    (sub)class based on instrument type and config

    :meth:`objects` contains a list of :class:`ibi.Contract` objects
    for which historical data will be loaded; those objects are not
    guaranteed to be qualified.
    """

    ib: ibi.IB
    sec_types = {
        "STK",  # Stock
        "OPT",  # Option
        "FUT",  # Future
        "CONTFUT",  # ContFuture
        "CASH",  # Forex
        "IND",  # Index
        "CFD",  # CFD
        "BOND",  # Bond
        "CMDTY",  # Commodity
        "FOP",  # FuturesOption
        "FUND",  # MutualFund
        "WAR",  # Warrant
        "IOPT",  # Warrant
        "BAG",  # Bag
        "CRYPTO",  # Crypto
    }
    contract_fields = {i.name for i in fields(ibi.Contract)}

    @classmethod
    def set_ib(cls, ib: ibi.IB) -> Type[Self]:
        cls.ib = ib
        return cls

    @classmethod
    def from_kwargs(cls, **kwargs) -> Self:
        secType = kwargs.get("secType")
        if secType not in cls.sec_types:
            raise TypeError(f"secType must be one of {cls.sec_types} not: {secType}")
        elif secType in {"FUT", "CONTFUT"}:
            return cls.from_future_kwargs(**kwargs)
        # TODO: specific cases for other asset classes
        else:
            return cls(**kwargs)

    @classmethod
    def from_future_kwargs(cls, **kwargs):
        return FutureContractSelector.create(**kwargs)

    def __init__(self, **kwargs) -> None:
        try:
            self.ib
        except AttributeError:
            raise AttributeError(
                f"ib attribute must be set on {self.__class__.__name__} "
                f"before class is instantiated."
            )
        self.kwargs = self.clean_fields(**kwargs)

    def clean_fields(self, **kwargs) -> dict:
        if diff := (set(kwargs.keys()) - self.contract_fields):
            for k in diff:
                del kwargs[k]
            log.warning(
                f"Removed incorrect contract parameters: {diff}, "
                f"will attemp to get Contract anyway"
            )
        return kwargs

    async def objects(self) -> AsyncGenerator[ibi.Contract, None]:
        if hasattr(self, "_objects"):
            # this is present only in subclasses
            async for ob in self._objects():
                yield ob
        else:
            contract = ibi.Contract.create(**self.kwargs)
            await self.ib.qualifyContractsAsync(contract)
            yield contract

    def repr(self) -> str:
        kwargs_str = ", ".join([f"{k}={v}" for k, v in self.kwargs.items()])
        return f"{self.__class__.__qualname__}({kwargs_str})"


class FutureContractSelector(ContractSelector):

    def __init__(self, **kwargs):
        kwargs.update(includeExpired=True)
        super().__init__(**kwargs)

    @classmethod
    def create(cls, **kwargs):
        # in case of any ambiguities just go for contfuture
        klass = {
            "contfuture": ContfutureFutureContractSelector,
            "fullchain": FullchainFutureContractSelector,
            "current": CurrentFutureContractSelector,
            "exact": ExactFutureContractSelector,
            "current_and_contfuture": CurrentContfutureFutureContractSelector,
            "current_and_expired": CurrentExpiredFutureContractSelector,
        }.get(FUTURES_SELECTOR, CurrentExpiredFutureContractSelector)
        return klass(**kwargs)

    @async_cached_property
    async def _fullchain(self) -> list[ibi.Contract]:
        kwargs = self.kwargs.copy()
        kwargs["secType"] = "FUT"  # it might have been CONTFUT at this point
        details = await self.ib.reqContractDetailsAsync(ibi.Contract.create(**kwargs))
        return sorted(
            [c.contract for c in details if c.contract is not None],
            key=lambda x: x.lastTradeDateOrContractMonth,
        )

    @async_cached_property
    async def _current_contract_index(self) -> int:
        full_chain = await self._fullchain
        return full_chain.index(await self._current_contract)

    @async_cached_property
    async def _current_contract(self) -> ibi.Contract:
        future_kwargs = ibi.util.dataclassNonDefaults(await self._contfuture)
        future_kwargs["secType"] = "FUT"
        new_contract = ibi.Contract.create(**future_kwargs)
        await self.ib.qualifyContractsAsync(new_contract)
        return new_contract

    @async_cached_property
    async def _contfuture(self) -> ibi.Contract:
        kwargs = self.kwargs.copy()
        kwargs["secType"] = "CONTFUT"
        contract = ibi.Contract.create(**kwargs)
        await self.ib.qualifyContractsAsync(contract)
        return contract


class ContfutureFutureContractSelector(FutureContractSelector):

    async def objects(self) -> AsyncGenerator[ibi.Contract, None]:
        yield await self._contfuture


class FullchainFutureContractSelector(FutureContractSelector):
    spec = FUTURES_FULLCHAIN_SPEC

    @classmethod
    def with_spec(cls, spec: str, **kwargs) -> Self:
        cls.spec = spec
        return cls(**kwargs)

    async def _objects(self) -> AsyncGenerator[ibi.Contract, None]:

        today = date.today()
        contracts = await self._fullchain

        if not contracts:
            log.warning(f"No contracts found for {self.kwargs}")
            return

        if self.spec == "full":
            for contract in contracts:
                yield contract
        elif self.spec == "active":
            for contract in contracts:
                if date.fromisoformat(contract.lastTradeDateOrContractMonth) > today:
                    yield contract
        elif self.spec == "expired":
            for contract in contracts:
                if date.fromisoformat(contract.lastTradeDateOrContractMonth) <= today:
                    yield contract
        else:
            raise ValueError(
                f"futures_fullchain_spec must be one of: `full`, `active`, `expired`, "
                f"not {self.spec}"
            )


class CurrentFutureContractSelector(FutureContractSelector):
    async def _objects(self) -> AsyncGenerator[ibi.Contract, None]:
        desired_index = DESIRED_INDEX
        if desired_index == 0:
            yield await self._current_contract
        else:
            full_chain = await self._fullchain
            yield full_chain[await self._current_contract_index + int(desired_index)]


class CurrentContfutureFutureContractSelector(FutureContractSelector):
    """Current and ContFuture"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.current = CurrentFutureContractSelector(**kwargs)
        self.contfuture = ContfutureFutureContractSelector(**kwargs)

    async def objects(self) -> AsyncGenerator[ibi.Contract, None]:
        for gen in (self.current, self.contfuture):
            async for contract in gen.objects():
                yield contract


class CurrentExpiredFutureContractSelector(FutureContractSelector):
    """Current and Expired"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.current = CurrentFutureContractSelector(**kwargs)
        self.expired = FullchainFutureContractSelector.with_spec("expired", **kwargs)

    async def objects(self) -> AsyncGenerator[ibi.Contract, None]:
        for gen in (self.current, self.expired):
            async for contract in gen.objects():
                yield contract


class ExactFutureContractSelector(FutureContractSelector):
    """Just pick the future contract without messing with it."""

    pass
