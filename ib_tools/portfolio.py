from __future__ import annotations

from collections import defaultdict
from typing import Literal

from ib_tools.base import Atom
from ib_tools.logger import Logger
from ib_tools.state_machine import StateMachine

log = Logger(__name__)

note = Literal[-1, 0, 1]


class Holder(defaultdict):
    __slots__ = ()

    _allowed_values = [-1, 1]

    def __init__(self):
        super().__init__(int)

    def __call__(self, k, v):
        if v not in self._allowed_values:
            raise ValueError
        else:
            self[k] += v


class SignalHolder(Holder):
    def contracts(self, key: str):
        """Sum up holdings by contract."""

        out: defaultdict[str, int] = defaultdict(int)

        for (contract, _strategy), amount in self.items():
            out[contract] += amount
        return out

    def strategies(self, key: str):
        """Sum up holdings by strategy."""

        out: defaultdict[str, int] = defaultdict(int)

        for (_contract, strategy), amount in self.items():
            out[strategy] += amount
        return out


class Portfolio(Atom):
    """
    Decides what to trade and how much based on received signals and queries to [SM?].
    """

    _signals = SignalHolder()
    _positions = Holder()
    _alocations: dict[
        str, int
    ]  # this should contain number of contracts for every strategy

    def __init__(self, state_machine: StateMachine):
        super().__init__()
        self.sm = state_machine

    def __onData__(self, data, *args) -> None:
        self.dataEvent.emit(self.allocate(data))

    def process_signal(self, data):
        self._signals(data)

    def allocate(self, data):
        pass

    def preallocate(self):
        self._alcocations = {}
