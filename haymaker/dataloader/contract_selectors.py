"""Build dataloader contract streams from YAML contract specifications.

Contract selectors translate config dictionaries into one or more qualified
``ib_insync.Contract`` objects for historical downloads. Selectors use the
session-scoped :class:`haymaker.dataloader.pacer.RequestPacing` instance for all
contract-details lookups so broker requests share one pacing policy.
"""

from __future__ import annotations

import logging
from dataclasses import fields
from datetime import date
from typing import AsyncGenerator, Self

import ib_insync as ibi

from haymaker.config import CONFIG
from haymaker.misc import async_cached_property

from .pacer import RequestPacing

log = logging.getLogger(__name__)


FUTURES_SELECTOR = CONFIG.get("futures_selector", "current")
FUTURES_FULLCHAIN_SPEC = CONFIG.get("futures_fullchain_spec", "full")
DESIRED_INDEX = CONFIG.get("futures_current_index", 0)


class ContractQualificationError(ValueError):
    """Raised when an IB contract specification cannot be resolved uniquely."""


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
    def from_kwargs(cls, pacing: RequestPacing, **kwargs) -> Self:
        """Create the selector appropriate for a contract specification."""

        secType = kwargs.get("secType")
        if secType not in cls.sec_types:
            raise TypeError(f"secType must be one of {cls.sec_types} not: {secType}")
        elif secType in {"FUT", "CONTFUT"}:
            return cls.from_future_kwargs(pacing=pacing, **kwargs)
        # TODO: specific cases for other asset classes
        else:
            return cls(pacing=pacing, **kwargs)

    @classmethod
    def from_future_kwargs(cls, pacing: RequestPacing, **kwargs):
        """Create the configured futures selector."""

        return FutureContractSelector.create(pacing=pacing, **kwargs)

    def __init__(self, pacing: RequestPacing, **kwargs) -> None:
        """Initialize a selector for one contract specification."""

        self.pacing = pacing
        self.kwargs = self.clean_fields(**kwargs)

    def clean_fields(self, **kwargs) -> dict:
        """Remove keys that are not accepted by ``ib_insync.Contract``."""

        if diff := (set(kwargs.keys()) - self.contract_fields):
            for k in diff:
                del kwargs[k]
            log.warning(
                f"Removed incorrect contract parameters: {diff}, "
                f"will attempt to get Contract anyway"
            )
        return kwargs

    async def objects(self) -> AsyncGenerator[ibi.Contract, None]:
        """Yield contracts selected for historical download."""

        contract = ibi.Contract.create(**self.kwargs)
        if contract.conId == 0:
            await self.qualify(contract)
        yield contract

    async def qualify(self, contract: ibi.Contract) -> ibi.Contract:
        """Qualify a contract using paced contract details."""

        details = await self.pacing.contract_details(contract)
        if not details:
            raise ContractQualificationError(f"Unknown contract: {contract}")
        if len(details) > 1:
            possibles = [detail.contract for detail in details]
            raise ContractQualificationError(
                f"Ambiguous contract: {contract}, possibles are {possibles}"
            )

        qualified = details[0].contract
        if qualified is None:
            raise ContractQualificationError(
                f"Missing contract details for: {contract}"
            )
        expiry = qualified.lastTradeDateOrContractMonth
        if expiry:
            qualified.lastTradeDateOrContractMonth = expiry.split()[0]
        if contract.exchange == "SMART":
            qualified.exchange = contract.exchange
        ibi.util.dataclassUpdate(contract, qualified)
        return contract

    def __repr__(self) -> str:
        """Return a stable human-readable selector representation."""

        kwargs_str = ", ".join([f"{k}={v}" for k, v in self.kwargs.items()])
        return f"{self.__class__.__qualname__}({kwargs_str})"


class FutureContractSelector(ContractSelector):

    def __init__(self, pacing: RequestPacing, **kwargs):
        kwargs.update(includeExpired=True)
        super().__init__(pacing=pacing, **kwargs)

    @classmethod
    def create(cls, pacing: RequestPacing, **kwargs):
        # in case of any ambiguities just go for contfuture
        klass = {
            "contfuture": ContfutureFutureContractSelector,
            "fullchain": FullchainFutureContractSelector,
            "current": CurrentFutureContractSelector,
            "exact": FutureContractSelector,
            "current_and_contfuture": CurrentContfutureFutureContractSelector,
            "current_and_expired": CurrentExpiredFutureContractSelector,
        }.get(FUTURES_SELECTOR, CurrentExpiredFutureContractSelector)
        return klass(pacing=pacing, **kwargs)

    @async_cached_property
    async def _fullchain(self) -> list[ibi.Contract]:
        kwargs = self.kwargs.copy()
        kwargs["secType"] = "FUT"  # it might have been CONTFUT at this point
        details = await self.pacing.contract_details(ibi.Contract.create(**kwargs))
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
        return await self.qualify(new_contract)

    @async_cached_property
    async def _contfuture(self) -> ibi.Contract:
        kwargs = self.kwargs.copy()
        kwargs["secType"] = "CONTFUT"
        contract = ibi.Contract.create(**kwargs)
        await self.qualify(contract)
        return contract


class ContfutureFutureContractSelector(FutureContractSelector):

    async def objects(self) -> AsyncGenerator[ibi.Contract, None]:
        yield await self._contfuture


class FullchainFutureContractSelector(FutureContractSelector):

    def __init__(
        self, pacing: RequestPacing, spec: str = FUTURES_FULLCHAIN_SPEC, **kwargs
    ):
        self.spec = spec
        super().__init__(pacing=pacing, **kwargs)

    @classmethod
    def with_spec(cls, spec: str, pacing: RequestPacing, **kwargs) -> Self:
        return cls(pacing=pacing, spec=spec, **kwargs)

    async def objects(self) -> AsyncGenerator[ibi.Contract, None]:

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
    async def objects(self) -> AsyncGenerator[ibi.Contract, None]:
        desired_index = DESIRED_INDEX
        if desired_index == 0:
            yield await self._current_contract
        else:
            full_chain = await self._fullchain
            yield full_chain[await self._current_contract_index + int(desired_index)]


class CurrentContfutureFutureContractSelector(FutureContractSelector):
    """Current and ContFuture"""

    def __init__(self, pacing: RequestPacing, **kwargs) -> None:
        super().__init__(pacing=pacing, **kwargs)
        self.current = CurrentFutureContractSelector(pacing=self.pacing, **kwargs)
        self.contfuture = ContfutureFutureContractSelector(pacing=self.pacing, **kwargs)

    async def objects(self) -> AsyncGenerator[ibi.Contract, None]:
        for gen in (self.current, self.contfuture):
            async for contract in gen.objects():
                yield contract


class CurrentExpiredFutureContractSelector(FutureContractSelector):
    """Current and Expired"""

    def __init__(self, pacing: RequestPacing, **kwargs) -> None:
        super().__init__(pacing=pacing, **kwargs)
        self.current = CurrentFutureContractSelector(pacing=self.pacing, **kwargs)
        self.expired = FullchainFutureContractSelector.with_spec(
            "expired", pacing=self.pacing, **kwargs
        )

    async def objects(self) -> AsyncGenerator[ibi.Contract, None]:
        for gen in (self.current, self.expired):
            async for contract in gen.objects():
                yield contract
