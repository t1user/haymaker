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

    Each strategy should have its own instance of portfolio to ensure
    that signals form various strategies should not be mixed-up.
    Actual singleton `porfolio` object should be passed to those
    instances, which should delegate allocation to this object.
    """

    def __init__(self) -> None:
        super().__init__()
        self.strategy: str = ""
        log.debug(f"Porfolio initiated: {self}")

    def onData(self, data: dict, *args) -> None:
        amount = self.allocate(data)
        data.update({"amount": amount})
        log.debug(f"Portfolio processed data: {data}")
        #     "{data['date']}, "
        #     f"action: {data['action']}, "
        #     f"signal: {data['signal']}, target_position: {data['target_position']}",
        super().onData(data)  # timestamp on departure
        self.dataEvent.emit(data)

    @abstractmethod
    def allocate(self, data: dict) -> float:
        """
        Return desired position size in contracts.  Interpretation of
        this number is up to execution model.
        """
        ...


class FixedPortfolio(AbstractBasePortfolio):
    def __init__(self, amount: float = 1) -> None:
        super().__init__()
        self.amount = amount

    def allocate(self, data) -> float:
        if data["signal"] == "CLOSE":
            return 0
        else:
            return self.amount
