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

from . import misc

log = getLogger(__name__)

# THESE ARE DEFAULTS THAT CAN BE OVERRIDEN IN CLASS INITIALIZATION
# number of business days before last trading day that contract will be rolled
FUTURES_ROLL_BDAYS = 3
# number of business days before roll day when new positions will use next contract
FUTURES_ROLL_MARGIN_BDAYS = 3

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
# These are used with FutureContractSelector to allow flexibility in selecting
# future contract
# ################################################################################


@dataclass
class AbstractBaseFutureWrapper(ABC):
    # if empty ignored; otherwise only months listed will be included in chain
    active_months: ClassVar[list[int]] = []

    contract: ibi.Future
    # number of business days before last trading day that contract will be rolled
    roll_bdays: int = FUTURES_ROLL_BDAYS

    @property
    @abstractmethod
    def last_trading_day(self) -> datetime: ...

    @classmethod
    def from_details(
        cls, details: ibi.ContractDetails, roll_bdays: int = FUTURES_ROLL_BDAYS
    ) -> Self:
        try:
            contract = details.contract
        except AttributeError as e:
            log.exception(e)
            raise
        assert contract
        future_contract = misc.general_to_specific_contract_class(contract)
        try:
            assert isinstance(future_contract, ibi.Future)
        except AssertionError:
            log.error(f"Expected future, got {type(contract)}: {contract=}")
            raise
        return cls(future_contract, roll_bdays)

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
        return self.last_trading_day - self.roll_bdays * custom_bday

    def __str__(self) -> str:
        return (
            f"<{self.__class__.__name__} {self.contract.localSymbol} {self.roll_day=}>"
        )


@dataclass
class GoldComex(AbstractBaseFutureWrapper):

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
class NG(AbstractBaseFutureWrapper):
    # NOT IN USE
    # should it be used?
    @property
    def last_trading_day(self) -> datetime:
        return (
            self.lastTradeDateOrContractMonth
            - pd.offsets.BusinessMonthEnd()
            - 3 * custom_bday
        )


@dataclass
class EnergyNymex(AbstractBaseFutureWrapper):
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
class NoOffset(AbstractBaseFutureWrapper):
    # default unless otherwise specified

    @property
    def last_trading_day(self) -> datetime:
        return self.lastTradeDateOrContractMonth


def future_wrapper_factory(contract: ibi.Contract) -> Type[AbstractBaseFutureWrapper]:
    return {
        "GC": GoldComex,
        "MGC": GoldComex,
    }.get(contract.symbol, NoOffset)


# ######################################################################
# Contract Selectors below
# ######################################################################


class AbstractBaseContractSelector(ABC):

    @classmethod
    @abstractmethod
    def from_details(
        cls, detailsChain: list[ibi.ContractDetails], *args, **kwargs
    ) -> Self: ...

    @property
    @abstractmethod
    def active_contract(self) -> ibi.Contract: ...

    @property
    @abstractmethod
    def next_contract(self) -> ibi.Contract: ...

    def __str__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"active_contract={self.active_contract.localSymbol} "
        )


@dataclass
class DefaultSelector(AbstractBaseContractSelector):
    contractChain: list[ibi.Contract]

    @classmethod
    def from_details(
        cls, detailsChain: list[ibi.ContractDetails], *args, **kwargs
    ) -> Self:
        return cls(
            [
                misc.general_to_specific_contract_class(details.contract)
                for details in detailsChain
                if (details.contract and details.contract.exchange != "QBALGO")
            ]
        )

    def __post_init__(self):
        if not self.contractChain:
            raise NoContractFound
        elif len(self.contractChain) > 1:
            raise AmbiguousContractError([d for d in self.contractChain])

    @property
    def active_contract(self) -> ibi.Contract:
        return self.contractChain[0]

    @property
    def next_contract(self) -> ibi.Contract:
        return self.active_contract


@dataclass
class ContFutureSelector(DefaultSelector):
    pass


@dataclass
class FutureSelector(AbstractBaseContractSelector):
    futuresChain: list[AbstractBaseFutureWrapper]
    roll_bdays: int = FUTURES_ROLL_BDAYS
    roll_margin_bdays: int = FUTURES_ROLL_MARGIN_BDAYS
    today: datetime = field(default_factory=datetime.now, repr=False)

    def __post_init__(self):
        if not self.futuresChain:
            raise NoContractFound("No Future contracts found.")
        for future_wrapper in self.futuresChain:
            assert isinstance(future_wrapper, AbstractBaseFutureWrapper)

    @classmethod
    def from_details(
        cls,
        detailsChain: list[ibi.ContractDetails],
        roll_bdays: int = FUTURES_ROLL_BDAYS,
        roll_margin_bdays: int = FUTURES_ROLL_MARGIN_BDAYS,
        today: datetime = datetime.now(),
        _future_wrapper_factory=future_wrapper_factory,  # for testing
        *args,
        **kwargs,
    ) -> Self:

        contract = detailsChain[0].contract
        assert contract
        future_wrapper = _future_wrapper_factory(contract)
        return cls(
            [
                future_wrapper.from_details(details, roll_bdays)
                for details in detailsChain
                if details.contract.exchange != "QBALGO"  # type: ignore
            ],
            roll_bdays,
            roll_margin_bdays,
            today,
        )

    @classmethod
    def from_contracts(
        cls,
        futuresChain: list[ibi.Future],
        roll_bdays: int = FUTURES_ROLL_BDAYS,
        roll_margin_bdays: int = FUTURES_ROLL_MARGIN_BDAYS,
        today: datetime = datetime.now(),
        _future_wrapper_factory=future_wrapper_factory,  # for testing
    ) -> Self:

        future_wrapper = _future_wrapper_factory(futuresChain[0])
        return cls(
            [future_wrapper(future) for future in futuresChain],
            roll_bdays,
            roll_margin_bdays,
            today,
        )

    @cached_property
    def contracts(self) -> list[AbstractBaseFutureWrapper]:
        """Contracts eligible for trading sorted by roll_date."""
        return sorted(
            [
                c
                for c in self.futuresChain
                if (c.isActiveMonth and self._bdays(self.today, c.roll_day)) > 0
            ],
            key=lambda x: x.roll_day,
        )

    @cached_property
    def back_contracts(self) -> list[AbstractBaseFutureWrapper]:
        """Historical contracts sorted by roll_date."""
        return sorted(
            [
                c
                for c in self.futuresChain
                if (c.isActiveMonth and self._bdays(self.today, c.roll_day)) <= 0
            ],
            key=lambda x: x.roll_day,
        )

    @cached_property
    def back_roll_days(self) -> dict[ibi.Future, datetime]:
        return {c.contract: c.roll_day for c in self.back_contracts}

    @staticmethod
    def _bdays(from_: datetime, to_: datetime) -> int:
        return len(pd.bdate_range(from_, to_)) - 1

    @property
    def bdays_till_roll(self) -> int:
        return self._bdays(self.today, self._active_contract().roll_day)

    def nth_contract(self, index: int) -> AbstractBaseFutureWrapper:
        """Return active contract at given index."""
        assert index >= 0
        # if index to large, return last element
        index = index if index < len(self.contracts) - 1 else len(self.contracts) - 1
        return self.contracts[index]

    def _active_contract(self) -> AbstractBaseFutureWrapper:
        return self.nth_contract(0)

    def _next_contract(self) -> AbstractBaseFutureWrapper:
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

    def __str__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"active_contract={self.active_contract.localSymbol} "
            f"next_contract={self.next_contract.localSymbol}>"
        )


def selector_factory(
    details_list: list[ibi.ContractDetails],
    futures_roll_bdays: int = FUTURES_ROLL_BDAYS,
    futures_roll_margin_bdays: int = FUTURES_ROLL_MARGIN_BDAYS,
) -> AbstractBaseContractSelector:
    first_details_instance = details_list[0]
    contract = first_details_instance.contract
    assert contract
    selector_class, args = {
        "CONTFUT": (ContFutureSelector, ()),
        "FUT": (
            FutureSelector,
            (
                futures_roll_bdays,
                futures_roll_margin_bdays,
            ),
        ),
    }.get(contract.secType, (DefaultSelector, ()))
    return selector_class.from_details(details_list, *args)  # type: ignore
