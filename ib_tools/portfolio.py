from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

from ib_tools.base import Atom

log = logging.getLogger(__name__)


class PortfolioWrapper(Atom):
    def __init__(self, portfolio: AbstractBasePortfolio) -> None:
        super().__init__()
        self.strategy: str = ""
        self._portfolio = portfolio

    def onData(self, data: dict, *args) -> None:
        amount = self.allocate(data)
        data.update({"amount": amount})
        log.debug(f"Portfolio processed data: {data}")
        super().onData(data)  # timestamp on departure
        self.dataEvent.emit(data)

    def allocate(self, data: dict) -> float:
        return self._portfolio.allocate(data)


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
        if Atom in cls.__mro__:
            raise TypeError("Portfolio cannot be an Atom.")

        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__init__(*args, **kwargs)
        wrapper = PortfolioWrapper(cls._instance)
        return wrapper

    def __setattr__(self, key, value):
        if key in ("onStart", "onData", "startEvent", "dataEvent"):
            raise AttributeError(
                f"Attribute {key} must be implemented on `PortfolioWrapper`, "
                f"not on `Portfolio`"
            )
        super().__setattr__(key, value)

    @abstractmethod
    def allocate(self, data: dict) -> float:
        """
        Return desired position size in contracts.  Interpretation of
        this number is up to execution model.
        """
        ...


class FixedPortfolio(AbstractBasePortfolio):
    def __init__(self, amount: float = 1) -> None:
        self.amount = amount
        super().__init__()

    def allocate(self, data) -> float:
        if data["signal"] == "CLOSE":
            return 0
        else:
            return self.amount
