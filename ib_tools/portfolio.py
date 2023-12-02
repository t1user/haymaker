from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import ClassVar, Literal, Optional, Type

from ib_tools.base import Atom

log = logging.getLogger(__name__)

note = Literal[-1, 0, 1]


# class _PortfolioWrapper(Atom):
#     _master_class = None
#     _master_instance = None

#     @classmethod
#     def setup(cls, master_class, master_instance):
#         cls._master_instance = master_instance
#         return cls

#     def onStart(self, data, *args):
#         super().onStart(data, *args)

#     def onData(self, data, *args):
#         print("Called!!")
#         super().onData(data, *args)
#         print("Success")
#         self.dataEvent.emit(data)

#     def __getattr__(self, name):
#         return getattr(self._master_instance, name)

#     def __setattr__(self, name, value):
#         setattr(self._master_instance, name, value)


class AbstractBasePortfolio(ABC):
    """
    Decides what, if and how much to trade based on received signals
    and queries to [SM?].

    Each strategy should have its own instance of portfolio to ensure
    that signals form various strategies should not be mixed-up.
    Actual singleton `porfolio` object should be passed to those
    instances, which should delegate allocation to this object.
    """

    _instance: Optional[AbstractBasePortfolio] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @abstractmethod
    def allocate(self, data: dict) -> float:
        """
        Return desired position size in contracts.  Interpretation of
        this number is up to execution model.
        """
        ...


class PortfolioWrapper(Atom):
    portfolio: ClassVar[AbstractBasePortfolio]

    @classmethod
    def setup(
        cls, portfolio_class: Type[AbstractBasePortfolio], *args, **kwargs
    ) -> Type[PortfolioWrapper]:
        cls.portfolio = portfolio_class(*args, **kwargs)
        return cls

    def __init__(self) -> None:
        super().__init__()
        self.strategy: str = ""

    def onData(self, data: dict, *args) -> None:
        amount = self.allocate(data)
        data.update({"amount": amount})
        log.debug(f"Portfolio processed data: {data}")
        super().onData(data)  # timestamp on departure
        self.dataEvent.emit(data)

    def allocate(self, data: dict) -> float:
        return self.portfolio.allocate(data)


def wrap(cls, *args, **kwargs):
    return PortfolioWrapper.setup(cls, *args, **kwargs)


class FixedPortfolio(AbstractBasePortfolio, PortfolioWrapper):
    def __init__(self, amount: float = 1) -> None:
        super().__init__()
        self.amount = amount

    def allocate(self, data) -> float:
        if data["signal"] == "CLOSE":
            return 0
        else:
            return self.amount
