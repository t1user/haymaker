from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import ib_insync as ibi

from . import misc
from .handlers import IBHandlers
from .logger import Logger
from .state_machine import StateMachine
from .trader import Trader

if TYPE_CHECKING:
    from .execution_models import AbstractExecModel

log = Logger(__name__)


class Controller:
    """
    Intermediary between execution models (which are off ramps for
    strategies) and `trader` and `state_machine`.  Use information
    provided by `state_machine` to make sure that positions held in
    the market reflect what is requested by strategies.

    It shouldn't be neccessary for user to modify or subclass
    :class:`.Controller`.
    """

    def __init__(self, state_machine: StateMachine, ib: ibi.IB):
        super().__init__()
        self.sm = state_machine
        self.ib = ib
        self.trader = Trader(self.ib)

    def trade(
        self,
        contract: ibi.Contract,
        order: ibi.Order,
        label: str,
        exec_model: AbstractExecModel,
        callback: Optional[misc.Callback] = None,
    ) -> None:
        trade = self.trader.trade(contract, order)
        if callback is not None:
            callback(trade)
        self.sm.register_order(exec_model.strategy, label, trade)

    def cancel(
        self,
        trade: ibi.Trade,
        exec_model: AbstractExecModel,
        callback: Optional[Callable[[ibi.Trade], None]] = None,
    ) -> None:
        trade = self.trader.cancel(trade)
        if callback is not None:
            callback(trade)
        self.sm.register_cancel(trade, exec_model)


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
