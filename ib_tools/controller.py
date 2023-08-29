from __future__ import annotations

import ib_insync as ibi

from .base import Atom
from .handlers import IBHandlers
from .logger import Logger
from .state_machine import StateMachine

log = Logger(__name__)


class Controller(Atom):
    """
    Code here should work for every kind of `brick`, `portfolio` and
    `execution model`.  Those elements can be specific to each other
    (i.e. require that they're complementary), it's user's
    responsibility to use matching elements.  However, :class:`.Controller`
    should be agnostic to type of other modules.  It shouldn't be
    neccessary for user to modify or subclass it.
    """

    def __init__(self, state_machine: StateMachine):
        super().__init__()
        self.sm = state_machine

    def onData(self, data: dict, *args) -> None:
        """
        Pass control to exec_model, get ``ibi.Trade`` object and pass
        it to ``state_machine``.

        Args:
        -----

        data (dict) : must have keys:
            * exec_model
            * contract
            * signal
            * amount

            otherwise an error will be logged and transaction will not be processed.

            * and any params required by the execution model
        """
        try:
            exec_model = data["exec_model"]
            contract = data["contract"]
            signal = data["signal"]
            amount = data["amount"]
        except KeyError as e:
            log.error("Missing data!", e)
            return
        trade_object, blotter_note = exec_model.execute(contract, signal, amount)
        # what about info for trade blotter?
        # self.sm.book_trade(
        #     trade_object, exec_model, blotter_note
        # )  # figure out how you wanna do it
        self.dataEvent.emit(data)

    def contract(self, key: tuple[str, str]) -> ibi.Contract:
        """
        Get correct ``Contract`` object for given strategy.  Not clear
        HOW?  yet.
        """
        return ibi.Contract()


class Handlers(IBHandlers):
    def __init__(self, ib):
        IBHandlers.__init__(self, ib)
        # self.ib.newOrderEvent += self.onNewOrder
        # self.ib.openOrderEvent += self.onOpenOrder
        # self.ib.cancelOrderEvent += self.onCancelOrder
        # self.ib.orderModifyEvent += self.onModifyOrder
        # self.ib.orderStatusEvent += self.onOrderStatus
        # ib.execDetailsEvent += self.onExecDetails
        # ib.commissionReportEvent += self.onCommissionReport
        # self.ib.positionEvent += self.onPosition
        # self.ib.accountValueEvent += self.onAccountValue
        # self.ib.accountSummaryEvent += self.onAccountSummary

    def onNewOrder(self, trade: ibi.Trade) -> None:
        pass

    def onOpenOrder(self, trade: ibi.Trade) -> None:
        pass

    def onCancelOrder(self, trade: ibi.Trade) -> None:
        pass

    def onModifyOrder(self, trade: ibi.Trade) -> None:
        pass

    def onOrderStatus(self, trade: ibi.Trade) -> None:
        pass

    def onExecDetails(self, trade: ibi.Trade, fill: ibi.Fill) -> None:
        pass

    def onCommissionReport(
        self, trade: ibi.Trade, fill: ibi.Fill, report: ibi.CommissionReport
    ) -> None:
        pass

    def onPosition(self, trade: ibi.Position) -> None:
        pass

    def onAccountValue(self, value: ibi.AccountValue) -> None:
        pass

    def onAccountSummary(self, value: ibi.AccountValue) -> None:
        pass
