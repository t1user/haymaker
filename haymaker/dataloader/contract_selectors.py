from __future__ import annotations

import functools
import logging
from dataclasses import fields
from datetime import date, datetime
from typing import Self, Type

import ib_insync as ibi

from haymaker.config import CONFIG

log = logging.getLogger(__name__)


class ContractSelector:
    """
    Based on passes contract attributes and config parameters
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

    def objects(self) -> list[ibi.Contract]:
        try:
            return ibi.util.run(self._objects())  # type: ignore
        except AttributeError:
            return [ibi.Contract.create(**self.kwargs)]

    def repr(self) -> str:
        kwargs_str = ", ".join([f"{k}={v}" for k, v in self.kwargs.items()])
        return f"{self.__class__.__qualname__}({kwargs_str})"


class FutureContractSelector(ContractSelector):

    @classmethod
    def create(cls, **kwargs):
        # in case of any ambiguities just go for contfuture
        klass = {
            "contfuture": ContfutureFutureContractSelector,
            "fullchain": FullchainFutureContractSelector,
            "current": CurrentFutureContractSelector,
            "exact": ExactFutureContractSelector,
        }.get(
            CONFIG.get("futures_selector", "contfuture"),
            ContfutureFutureContractSelector,
        )
        return klass(**kwargs)

    @functools.cached_property
    def _fullchain(self) -> list[ibi.Contract]:
        kwargs = self.kwargs.copy()
        kwargs["secType"] = "FUT"
        kwargs["includeExpired"] = True
        details = ibi.util.run(self.ib.reqContractDetailsAsync(ibi.Contract(**kwargs)))
        return sorted(
            [c.contract for c in details if c.contract is not None],
            key=lambda x: x.lastTradeDateOrContractMonth,
        )

    @functools.cached_property
    def _contfuture_index(self) -> int:
        return self._fullchain.index(self._contfuture_qualified)

    @functools.cached_property
    def _contfuture_qualified(self) -> ibi.Contract:
        contfuture = self._contfuture
        ibi.util.run(self.ib.qualifyContractsAsync(contfuture))
        future_kwargs = contfuture.nonDefaults()  # type: ignore
        del future_kwargs["secType"]
        return ibi.Contract.create(**future_kwargs)

    @functools.cached_property
    def _contfuture(self) -> ibi.Contract:
        kwargs = self.kwargs.copy()
        kwargs["secType"] = "CONTFUT"
        return ibi.Contract.create(**kwargs)


class ContfutureFutureContractSelector(FutureContractSelector):

    def objects(self) -> list[ibi.Contract]:
        return [self._contfuture]


class FullchainFutureContractSelector(FutureContractSelector):

    async def _objects(self) -> list[ibi.Contract]:  # type: ignore
        spec = CONFIG.get("futures_fullchain_spec", "full")
        today = date.today()
        if spec == "full":
            return self._fullchain
        elif spec == "active":
            return [
                c
                for c in self._fullchain
                if datetime.fromisoformat(c.lastTradeDateOrContractMonth) > today
            ]
        elif spec == "expired":
            return [
                c
                for c in self._fullchain
                if datetime.fromisoformat(c.lastTradeDateOrContractMonth) <= today
            ]
        else:
            raise ValueError(
                f"futures_fullchain_spec must be one of: `full`, `active`, `expired`, "
                f"not {spec}"
            )


class CurrentFutureContractSelector(FutureContractSelector):
    async def _objects(self) -> list[ibi.Contract]:
        desired_index = CONFIG.get("futures_current_index", 0)
        if desired_index == 0:
            return [self._contfuture_qualified]
        else:
            return [self._fullchain[self._contfuture_index + int(desired_index)]]


class ExactFutureContractSelector(FutureContractSelector):
    pass
