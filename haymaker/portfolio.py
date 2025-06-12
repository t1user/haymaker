from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from .base import Atom

log = logging.getLogger(__name__)


class AbstractBasePortfolio(ABC):
    """
    Decides what, if and how much to trade based on received signals
    and queries to [SM?].

    Each strategy should have its own instance of portfolio to ensure
    that signals form various strategies should not be mixed-up.
    Actual singleton `porfolio` object should be passed to those
    instances, which should delegate allocation to this object.
    """

    instance: AbstractBasePortfolio | None = None

    def __new__(cls, *args, **kwargs):
        if Atom in cls.__mro__:
            raise TypeError("Portfolio cannot be an Atom.")

        if cls.instance is None:
            AbstractBasePortfolio.instance = super().__new__(cls)
        return cls.instance

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
        if data.get("action") == "CLOSE":
            return 0
        else:
            return self.amount


class PortfolioWrapper(Atom):
    def __init__(self) -> None:
        super().__init__()
        self.strategy: str = ""
        if AbstractBasePortfolio.instance:
            self._portfolio = AbstractBasePortfolio.instance
        else:
            raise TypeError("Portfolio must be instantiated before PortfolioWrapper.")

    def onData(self, data: dict, *args) -> None:
        if data.get("signal") not in (-1, 0, 1):
            log.error(f"Ignoring invalid signal: {data.get('signal')} in {data}")
        else:
            amount = self.allocate(data)
            data.update({"amount": amount})
            log.debug(f"Portfolio processed data: {data}")
            super().onData(data)  # timestamp on departure
            self.dataEvent.emit(data)

    def allocate(self, data: dict) -> float:
        return self._portfolio.allocate(data)
