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

    @staticmethod
    def check_for_orphan_positions(
        trades: list[ibi.Trade], positions: list[ibi.Position]
    ) -> list[ibi.Position]:
        """
        Positions that don't have associated stop-loss orders.
        """

        trade_contracts = set(
            [t.contract for t in trades if t.order.orderType in ("STP", "TRAIL")]
        )
        position_contracts = set([p.contract for p in positions])
        orphan_contracts = position_contracts - trade_contracts
        orphan_positions = [
            position for position in positions if position.contract in orphan_contracts
        ]
        return orphan_positions

    @staticmethod
    def check_for_orphan_trades(
        trades: list[ibi.Trade], positions: list[ibi.Position]
    ) -> list[ibi.Trade]:
        """
        Trades for stop-loss orders without associated positions.
        """

        trade_contracts = set(
            [t.contract for t in trades if t.order.orderType in ("STP", "TRAIL")]
        )
        position_contracts = set([p.contract for p in positions])
        orphan_contracts = trade_contracts - position_contracts
        orphan_trades = [
            trade for trade in trades if trade.contract in orphan_contracts
        ]
        return orphan_trades

    @staticmethod
    def positions_and_stop_losses(
        trades: list[ibi.Trade], positions: list[ibi.Position]
    ) -> dict[ibi.Contract, tuple[float, float]]:
        positions_by_contract = {
            position.contract: position.position for position in positions
        }

        trades_by_contract = {
            trade.contract: (
                trade.order.totalQuantity * -misc.action_to_signal(trade.order.action)
            )
            for trade in trades
            if trade.order.orderType in ("STP", "TRAIL")
        }

        contracts = set([*positions_by_contract.keys(), *trades_by_contract.keys()])
        contracts_dict = {
            contract: (
                positions_by_contract.get(contract, 0),
                trades_by_contract.get(contract, 0),
            )
            for contract in contracts
        }
        diff = {}
        for contract, amounts in contracts_dict.items():
            if amounts[0] != amounts[1]:
                diff[contract] = amounts
        return diff


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
