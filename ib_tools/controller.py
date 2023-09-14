from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Final, Optional

import ib_insync as ibi

from ib_tools.blotter import CsvBlotter

from . import misc
from .blotter import AbstractBaseBlotter
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

    def __init__(
        self,
        state_machine: StateMachine,
        ib: ibi.IB,
        blotter: Optional[AbstractBaseBlotter] = None,
    ):
        super().__init__()
        self.sm = state_machine
        self.ib = ib
        self.trader = Trader(self.ib)
        self._attach_logging_events()
        if blotter:
            self.blotter = blotter
            self.ib.commissionReportEvent += self.onCommissionReport

    def _attach_logging_events(self):
        self.ib.newOrderEvent += self.log_trade
        self.ib.cancelOrderEvent += self.log_cancel
        self.ib.orderModifyEvent += self.log_modification

    def trade(
        self,
        contract: ibi.Contract,
        order: ibi.Order,
        action: str,
        exec_model: AbstractExecModel,
        callback: Optional[misc.Callback] = None,
    ) -> None:
        trade = self.trader.trade(contract, order)
        if callback is not None:
            callback(trade)
        self.sm.register_order(exec_model.strategy, action, trade)

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
        Amounts are not compared, just the fact whether number of all
        posiitons have stop orders for the same contract.
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
        Amounts are not compared, just the fact whether there exists a
        position for every contract that has a stop order.
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
        """
        Check if all positions are associated with stop-losses.

        Args:
        =====

        trades : from :meth:`ibi.ib.positions()`

        positions: from :meth:`ibi.ib.open_trades()`

        Returns:
        ========

        `tuple` of (positions, orders) for every :class:`ibi.Contract'
        where those two values are not equal;

        * positions means amount of contracts currently held in the
        market

        * orders means minus amount of stop orders (or trailing stops)
        in the market associated with this contract

        Opposite numbers mean stops are on the wrong side of the
        market ('buy' instead of 'sell' and vice versa)

        key means total amount in stop orders not associated with
        positions for that contract (negative value means positions
        not covered by stops), i.e.:

        * empty dict means no orphan trades/positions, i.e. all active
        positions are covered by stop losses and there are no
        stop-losses without active positions

        Surplus of stop orders over held positions doesn't neccessary
        mean an error, as a strategy might use stops to open position;

        Positions not being associated with stop orders might also be
        strategy dependent and not necessarily an error.
        """

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
            if amounts[1] - amounts[0]:
                diff[contract] = (amounts[0], amounts[1])
        return diff

    def onCommissionReport(
        self, trade: ibi.Trade, fill: ibi.Fill, report: ibi.CommissionReport
    ):
        if trade.isDone():
            strategy, action, trade = self.sm.orders[trade.order.orderId]
            try:
                exec_model = self.sm.data["strategy"]
            except KeyError:
                log.error(
                    "Missing strategy records in `state machine`. "
                    "Unable to write full transaction data to blotter."
                )
            if exec_model:
                position_id = exec_model.position_id
                params = exec_model.params.get(action.lower(), {})
            else:
                position_id = ""
                params = {}
            kwargs = {
                "strategy": strategy,
                "action": action,
                "position_id": position_id,
                "params": params,
            }
            self.blotter.log_commission(trade, fill, report, **kwargs)

    def log_trade(self, trade: ibi.Trade, reason: str = "") -> None:
        log.info(
            f"{reason} trade filled: {trade.contract.localSymbol} "
            f"{trade.order.action} {trade.orderStatus.filled}"
            f"@{trade.orderStatus.avgFillPrice}"
        )

    def log_cancel(self, trade: ibi.Trade) -> None:
        log.info(
            f"{trade.order.orderType} order {trade.order.action} "
            f"{trade.orderStatus.remaining} (of "
            f"{trade.order.totalQuantity}) for "
            f"{trade.contract.localSymbol} cancelled"
        )

    def log_modification(self, trade: ibi.Trade) -> None:
        log.debug(f"Order modified: {trade.order}")

    # def trace_manual_orders(self, trade: ibi.Trade) -> None:
    #     """
    #     Attempt to attach reporting events for orders entered
    #     outside of the framework. This will not work if framework is not
    #     connected with clientId == 0.
    #     """
    #     if trade.order.orderId <= 0:
    #         log.debug("manual trade reporting event attached")
    #         self.trade_handler.attach_events(trade, "MANUAL TRADE")

    # def trade_(self, contract, reason):
    #     if reason:
    #         trade = self.trade(contract)
    #         self.trade_handler.attach_events(trade, reason)
    #         log.debug(f"{contract.localSymbol} order placed: {order}")
    #     else:
    #         log.debug(f"{contract.localSymbol} order updated: {order}")
