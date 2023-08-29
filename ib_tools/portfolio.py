from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Literal

from ib_tools.base import Atom

log = logging.getLogger(__name__)

note = Literal[-1, 0, 1]


class AbstractBasePortfolio(Atom, ABC):
    """
    Decides what, if and how much to trade based on received signals
    and queries to [SM?].
    """

    def onData(self, data: dict, *args) -> None:
        amount = self.allocate(data)
        data.update({"amount": amount})
        self.dataEvent.emit(data)

    @abstractmethod
    def allocate(self, data: dict) -> float:
        """
        Return desired position size in contracts.  Interpretation of
        this number is up to execution model.
        """
        ...


class FixedPortfolio(AbstractBasePortfolio):
    def __init__(self, amount: float) -> None:
        self.amount = amount
        super().__init__()

    def allocate(self, data) -> float:
        if data["signal"] == "CLOSE":
            return 0
        else:
            return self.amount
