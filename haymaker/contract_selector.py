from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from logging import getLogger
from typing import ClassVar, Self, Type

import ib_insync as ibi
import pandas as pd
import pandas_market_calendars as mcal  # type: ignore
from pandas.tseries.offsets import CustomBusinessDay

log = getLogger(__name__)


nyse_calendar = mcal.get_calendar("NYSE")
nyse_holidays = nyse_calendar.holidays().holidays  # type: ignore
custom_bday = CustomBusinessDay(holidays=nyse_holidays)


class AmbiguousContractError(Exception):
    pass


class NoContractFound(Exception):
    pass


def customize_month_end(month_end: datetime) -> datetime:
    """
    Take :class:`pd.offsets.BusinessMonthEnd` and shift it back if it's a NYSE holiday
    """

    if month_end in nyse_holidays:
        return month_end - custom_bday
    else:
        return month_end


# ################################################################################
# These are used with FutureContractSelector to allow flexibility in which
# future contract will be used
# ################################################################################


@dataclass
class AbstractBaseSingleFuture(ABC):
    # number of business days before last trading day that contract will be rolled
    bdays_roll: ClassVar[int] = 4
    # if empty ignored; otherwise only months listed will be included in chain
    active_months: ClassVar[list[int]] = []

    contract: ibi.Future
    details: ibi.ContractDetails

    @property
    @abstractmethod
    def last_trading_day(self) -> datetime: ...

    @classmethod
    def from_details(cls, details: ibi.ContractDetails) -> Self:
        try:
            contract = details.contract
        except AttributeError as e:
            log.exception(e)
            raise
        assert contract
        future_contract = ibi.Contract.create(**ibi.util.dataclassAsDict(contract))
        try:
            assert isinstance(future_contract, ibi.Future)
        except AssertionError:
            log.error(f"Expected future, got {type(contract)}: {contract=}")
            raise
        details.contract = contract
        return cls(future_contract, details)

    @property
    def lastTradeDateOrContractMonth(self) -> datetime:
        return datetime.strptime(self.contract.lastTradeDateOrContractMonth, "%Y%m%d")

    @property
    def expiry_month(self) -> int:
        return self.lastTradeDateOrContractMonth.month

    @property
    def isActiveMonth(self) -> bool:
        if not self.active_months:
            return True
        elif self.expiry_month in self.active_months:
            return True
        else:
            return False

    @property
    def roll_day(self) -> datetime:
        return self.last_trading_day - self.bdays_roll * custom_bday

    def __str__(self) -> str:
        return (
            f"<{self.__class__.__name__} {self.contract.localSymbol} {self.roll_day=}>"
        )


@dataclass
class GoldComex(AbstractBaseSingleFuture):

    active_months: ClassVar[list[int]] = [2, 4, 6, 8, 10, 12]

    @property
    def last_trading_day(self) -> datetime:
        # first notice is the last business day of the month preceding
        # the delivery month, so our last trading day is one day prior
        return (
            customize_month_end(
                self.lastTradeDateOrContractMonth - pd.offsets.BusinessMonthEnd()
            )
            - custom_bday
        )


@dataclass
class NG(AbstractBaseSingleFuture):

    @property
    def last_trading_day(self) -> datetime:
        return (
            self.lastTradeDateOrContractMonth
            - pd.offsets.BusinessMonthEnd()
            - 3 * custom_bday
        )


@dataclass
class EnergyNymex(AbstractBaseSingleFuture):
    # NOT IN USE
    # CL doesn't need offset

    @property
    def last_trading_day(self) -> datetime:
        # The third business day prior to the 25th calendar day of the
        # month preceding the delivery month (or the third business
        # day prior to the first business day before the 25th, if the
        # 25th is a non-business day).
        previous_month = pd.offsets.MonthEnd().rollback(
            self.lastTradeDateOrContractMonth
        )
        twenty_fifth = pd.offsets.BusinessDay().rollback(
            datetime(previous_month.year, previous_month.month, 25)
        )
        return twenty_fifth - pd.offsets.BusinessDay(3)


@dataclass
class NoOffset(AbstractBaseSingleFuture):
    # default unless otherwise specified

    @property
    def last_trading_day(self) -> datetime:
        return self.lastTradeDateOrContractMonth


# ######################################################################
# Contract Selectors below
# ######################################################################


@dataclass
class AbstractBaseContractSelector(ABC):
    detailsChain: list[ibi.ContractDetails]

    @property
    @abstractmethod
    def active_contract(self) -> ibi.Contract: ...

    @property
    @abstractmethod
    def next_contract(self) -> ibi.Contract: ...

    @property
    @abstractmethod
    def active_details(self) -> ibi.ContractDetails: ...

    @property
    @abstractmethod
    def next_details(self) -> ibi.ContractDetails: ...

    def __str__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"active_contract={self.active_contract.localSymbol} "
        )


@dataclass
class DefaultSelector(AbstractBaseContractSelector):
    detailsChain: list[ibi.ContractDetails]

    def __post_init__(self):
        if not self.detailsChain:
            raise NoContractFound
        elif len(self.detailsChain) > 1:
            raise AmbiguousContractError([d.contract for d in self.detailsChain])

    @property
    def active_contract(self) -> ibi.Contract:
        assert len(self.detailsChain) == 1
        assert self.detailsChain[0].contract
        return self.detailsChain[0].contract

    @property
    def active_details(self) -> ibi.ContractDetails:
        return self.detailsChain[0]

    @property
    def next_contract(self) -> ibi.Contract:
        return self.active_contract

    @property
    def next_details(self) -> ibi.ContractDetails:
        return self.active_details


@dataclass
class ContFutureSelector(DefaultSelector):

    def __post_init__(self):
        """Replace ContFuture with Future."""
        super().__post_init__()
        contract = self.detailsChain[0].contract
        contfuture_contract = ibi.Contract.create(**ibi.util.dataclassAsDict(contract))
        assert isinstance(contfuture_contract, ibi.ContFuture)
        contract_kwargs = ibi.util.dataclassNonDefaults(contfuture_contract)
        del contract_kwargs["secType"]
        self.detailsChain[0].contract = ibi.Future(**contract_kwargs)


@dataclass
class FutureSelector(AbstractBaseContractSelector):
    detailsChain: list[ibi.ContractDetails]
    selector: Type[AbstractBaseSingleFuture]
    roll_margin_bdays: int = 5
    today: datetime = field(default_factory=datetime.now, repr=False)

    @cached_property
    def contracts(self) -> list[AbstractBaseSingleFuture]:
        """Contracts eligible for trading sorted by roll_date."""
        _contracts = (
            self.selector.from_details(details)
            for details in self.detailsChain
            if details.contract.exchange != "QBALGO"  # type: ignore
        )
        return sorted(
            [c for c in _contracts if (c.isActiveMonth and c.roll_day > self.today)],
            key=lambda x: x.roll_day,
        )

    @property
    def bdays_till_roll(self) -> int:
        return len(pd.bdate_range(self.today, self._active_contract().roll_day))

    def nth_contract(self, index: int) -> AbstractBaseSingleFuture:
        index = index if index < len(self.contracts) - 1 else len(self.contracts) - 1
        return self.contracts[index]

    def _active_contract(self) -> AbstractBaseSingleFuture:
        return self.nth_contract(0)

    def _next_contract(self) -> AbstractBaseSingleFuture:
        return (
            self._active_contract()
            if self.bdays_till_roll > self.roll_margin_bdays
            else self.nth_contract(1)
        )

    @property
    def active_contract(self) -> ibi.Future:
        return self._active_contract().contract

    @property
    def next_contract(self) -> ibi.Future:
        return self._next_contract().contract

    @property
    def active_details(self) -> ibi.ContractDetails:
        return self._active_contract().details

    @property
    def next_details(self) -> ibi.ContractDetails:
        return self._next_contract().details

    def __str__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"active_contract={self.active_contract.localSymbol} "
            f"next_contract={self.next_contract.localSymbol}>"
        )


def single_future_factory(contract: ibi.Contract) -> Type[AbstractBaseSingleFuture]:
    return {
        "GC": GoldComex,
        "MGC": GoldComex,
    }.get(contract.symbol, NoOffset)


def selector_factory(
    details_list: list[ibi.ContractDetails],
) -> AbstractBaseContractSelector:
    first_details_instance = details_list[0]
    contract = first_details_instance.contract
    assert contract
    selector_class, args = {
        "CONTFUT": (ContFutureSelector, ()),
        "FUT": (FutureSelector, (single_future_factory(contract),)),
    }.get(contract.secType, (DefaultSelector, ()))
    return selector_class(details_list, *args)
