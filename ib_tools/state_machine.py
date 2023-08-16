from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import ib_insync as ibi

from .base import Atom
from .execution_models import AbstractExecModel
from .logger import Logger  # type: ignore

log = Logger(__name__)


@dataclass
class StrategyData:
    pass


class StateMachine(Atom):
    """
    This class provides information.  It should be other classes that
    act on this information.

    Answers questions:

        1. Is a strategy in the market?

        2. Is strategy locked?

        3. After (re)start:

            - do all positions have stop-losses?

            - has anything happened during blackout that needs to be
              reported in blotter/logs?

            - are all active orders accounted for (i.e. linked to
              strategies)?

            - compare position records with live update (do we hold
              what we think we hold?)

            - attach events (or check if events attached after another
              object attaches them) for:

                - reporting

                - cancellation of redundand orders (tp after stop hit,
                  etc)

        4. Was execution successful (i.e. record desired position
           after signal sent to execution model, then record actual
           live transactions)?

        5. Are holdings linked to strategies?

        6. Are orders linked to strategies?

        7. Make sure order events connected to call-backs
    """

    _positions: dict[tuple[str, str], StrategyData] = {}

    def position(self, key: str) -> int:
        return 1

    def locked(self, key: str) -> bool:
        return True

    def onData(self, data, *args) -> None:
        pass

    def book_trade(
        self, trade: ibi.Trade, exec_model: AbstractExecModel, note: str
    ) -> None:
        pass

    def book_cancel(self, trade):
        pass
