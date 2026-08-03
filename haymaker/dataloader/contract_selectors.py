"""Build dataloader contract streams from YAML contract specifications.

Contract selectors translate config dictionaries into one or more qualified
``ib_insync.Contract`` objects for historical downloads. Selectors use the
session-scoped :class:`haymaker.dataloader.pacer.RequestPacing` instance for all
contract-details lookups so broker requests share one pacing policy.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import date
from typing import Any, AsyncGenerator, Self

import ib_insync as ibi

from haymaker.misc import async_cached_property

from .pacer import RequestPacing

log = logging.getLogger(__name__)


class ContractQualificationError(ValueError):
    """Raised when an IB contract specification cannot be resolved uniquely."""


@dataclass(frozen=True)
class FuturesSelectionPolicy:
    """Policy shared by dataloader futures contract selectors.

    Attributes:
        selector: Selector implementation used for futures rows.
        full_chain_spec: Full-chain subset selected by full-chain expansion.
        current_index: Offset from the contract IB identifies as current.
    """

    selector: str = "current_and_expired"
    full_chain_spec: str = "full"
    current_index: int = 0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> Self:
        """Construct futures selection policy from plain configuration.

        Args:
            values: Merged dataloader ``futures`` section.

        Returns:
            Validated futures selection policy.
        """

        return cls(**dict(values))

    def __post_init__(self) -> None:
        """Reject policy values that would otherwise select silently."""

        if not isinstance(self.selector, str):
            raise TypeError("futures.selector must be a string")
        if self.selector not in {
            "contfuture",
            "fullchain",
            "current",
            "exact",
            "current_and_contfuture",
            "current_and_expired",
        }:
            raise ValueError(f"Unknown futures selector: {self.selector!r}")
        if not isinstance(self.full_chain_spec, str):
            raise TypeError("futures.full_chain_spec must be a string")
        if self.full_chain_spec not in {"full", "active", "expired"}:
            raise ValueError(
                f"Unknown futures full-chain policy: {self.full_chain_spec!r}"
            )
        if not isinstance(self.current_index, int) or isinstance(
            self.current_index, bool
        ):
            raise TypeError("futures.current_index must be an integer")


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
    def from_kwargs(
        cls,
        pacing: RequestPacing,
        futures: FuturesSelectionPolicy | None = None,
        **kwargs,
    ) -> Self:
        """Create the selector appropriate for a contract specification."""

        futures = futures or FuturesSelectionPolicy()
        secType = kwargs.get("secType")
        if secType not in cls.sec_types:
            raise TypeError(f"secType must be one of {cls.sec_types} not: {secType}")
        elif secType in {"FUT", "CONTFUT"}:
            return cls.from_future_kwargs(pacing=pacing, futures=futures, **kwargs)
        # TODO: specific cases for other asset classes
        else:
            return cls(pacing=pacing, futures=futures, **kwargs)

    @classmethod
    def from_future_kwargs(
        cls, pacing: RequestPacing, futures: FuturesSelectionPolicy, **kwargs
    ):
        """Create the configured futures selector."""

        return FutureContractSelector.create(pacing=pacing, futures=futures, **kwargs)

    def __init__(
        self,
        pacing: RequestPacing,
        futures: FuturesSelectionPolicy | None = None,
        **kwargs,
    ) -> None:
        """Initialize a selector for one contract specification."""

        self.pacing = pacing
        self.futures = futures or FuturesSelectionPolicy()
        self.kwargs = self.clean_fields(**kwargs)

    def clean_fields(self, **kwargs) -> dict:
        """Return contract fields after rejecting unsupported names."""

        if diff := (set(kwargs.keys()) - self.contract_fields):
            names = ", ".join(sorted(diff))
            raise ContractQualificationError(f"Unknown contract field(s): {names}")
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

    def __init__(
        self,
        pacing: RequestPacing,
        futures: FuturesSelectionPolicy | None = None,
        **kwargs,
    ):
        kwargs.update(includeExpired=True)
        super().__init__(pacing=pacing, futures=futures, **kwargs)

    @classmethod
    def create(cls, pacing: RequestPacing, futures: FuturesSelectionPolicy, **kwargs):
        # in case of any ambiguities just go for contfuture
        klass = {
            "contfuture": ContfutureFutureContractSelector,
            "fullchain": FullchainFutureContractSelector,
            "current": CurrentFutureContractSelector,
            "exact": FutureContractSelector,
            "current_and_contfuture": CurrentContfutureFutureContractSelector,
            "current_and_expired": CurrentExpiredFutureContractSelector,
        }.get(futures.selector, CurrentExpiredFutureContractSelector)
        return klass(pacing=pacing, futures=futures, **kwargs)

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
        self,
        pacing: RequestPacing,
        futures: FuturesSelectionPolicy | None = None,
        spec: str | None = None,
        **kwargs,
    ):
        futures = futures or FuturesSelectionPolicy()
        self.spec = spec or futures.full_chain_spec
        super().__init__(pacing=pacing, futures=futures, **kwargs)

    @classmethod
    def with_spec(
        cls,
        spec: str,
        pacing: RequestPacing,
        futures: FuturesSelectionPolicy | None = None,
        **kwargs,
    ) -> Self:
        return cls(pacing=pacing, futures=futures, spec=spec, **kwargs)

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
        desired_index = self.futures.current_index
        if desired_index == 0:
            yield await self._current_contract
        else:
            full_chain = await self._fullchain
            yield full_chain[await self._current_contract_index + int(desired_index)]


class CurrentContfutureFutureContractSelector(FutureContractSelector):
    """Current and ContFuture"""

    def __init__(
        self,
        pacing: RequestPacing,
        futures: FuturesSelectionPolicy | None = None,
        **kwargs,
    ) -> None:
        super().__init__(pacing=pacing, futures=futures, **kwargs)
        self.current = CurrentFutureContractSelector(
            pacing=self.pacing, futures=self.futures, **kwargs
        )
        self.contfuture = ContfutureFutureContractSelector(
            pacing=self.pacing, futures=self.futures, **kwargs
        )

    async def objects(self) -> AsyncGenerator[ibi.Contract, None]:
        for gen in (self.current, self.contfuture):
            async for contract in gen.objects():
                yield contract


class CurrentExpiredFutureContractSelector(FutureContractSelector):
    """Current and Expired"""

    def __init__(
        self,
        pacing: RequestPacing,
        futures: FuturesSelectionPolicy | None = None,
        **kwargs,
    ) -> None:
        super().__init__(pacing=pacing, futures=futures, **kwargs)
        self.current = CurrentFutureContractSelector(
            pacing=self.pacing, futures=self.futures, **kwargs
        )
        self.expired = FullchainFutureContractSelector.with_spec(
            "expired", pacing=self.pacing, futures=self.futures, **kwargs
        )

    async def objects(self) -> AsyncGenerator[ibi.Contract, None]:
        for gen in (self.current, self.expired):
            async for contract in gen.objects():
                yield contract
