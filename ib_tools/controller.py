from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import TYPE_CHECKING, Callable, Final, Optional

import ib_insync as ibi

from . import misc
from .blotter import Blotter
from .config import CONFIG
from .manager import IB, STATE_MACHINE
from .trader import Trader

if TYPE_CHECKING:
    from .execution_models import AbstractExecModel
    from .state_machine import Model, StateMachine

log = logging.getLogger(__name__)


class OrderSyncStrategy:
    def __init__(self) -> None:
        # IB has trades that we don't know about
        self.unknown_trades: list[ibi.Trade] = []  # <-We're fucked
        # Our active trades that IB doesn't report as active
        self.question_trades: list[ibi.Trade] = []
        # Question trades that we managed to match to IB
        self.matched_trades: list[ibi.Trade] = []
        # Trades on record that we cannot resolve with IB
        self.unmatched_trades: list[ibi.Trade] = []  # <- We're fucked

    @classmethod
    def run(cls):
        return cls().update_trades().review_trades().handle_question_trades().report()

    def update_trades(self):
        # update order records with current Trade objects
        self.unknown_trades = STATE_MACHINE.update_trades(*IB.openTrades())
        return self

    def review_trades(self):
        # Report trades on which we can find information
        self.question_trades = STATE_MACHINE.review_trades(*IB.openTrades())
        return self

    def handle_question_trades(self):
        # these are active trades on record that haven't been matched to
        # open trades in IB
        # try finding in closed trades from the session
        log.debug(
            f"ib trades: "
            f"{[(trade.order.orderId, trade.order.permId) for trade in IB.trades()]}"
        )
        log.debug(
            f"ib open trades: "
            f"{[(trade.order.orderId, trade.order.permId) for trade in IB.openTrades()]}"
        )
        known_trades = {trade.order.orderId: trade for trade in IB.trades()}
        log.debug(
            f"Known trades: "
            f"{known_trades.keys()} {[(v.order.orderId, v.order.permId) for k, v in known_trades.items()]}"
        )
        log.debug(f"Question trades: {[t.order.orderId for t in self.question_trades]}")
        matched_trades = {
            trade.order.orderId: trade
            for trade in self.question_trades
            if trade.order.orderId in known_trades
        }
        self.unmatched_trades = [
            trade
            for trade in self.question_trades
            if trade.order.orderId not in matched_trades
        ]
        self.matched_trades = list(matched_trades.values())
        log.debug(f"matched trades: {self.matched_trades}")
        return self

    def report(self):
        return self.unknown_trades, self.matched_trades, self.unmatched_trades


class PositionSyncStrategy:
    """
    Must be called after :class:`.OrderSyncStrategy`
    """

    def __init__(self):
        self.errors = []

    @classmethod
    def run(cls):
        return cls().verify_positions().report()

    def verify_positions(self):
        self.errors = STATE_MACHINE.verify_positions()
        return self

    def report(self):
        return self.errors


class Controller:
    """
    Intermediary between execution models (which are off ramps for
    strategies) and `trader` and `state_machine`.  Use information
    provided by `state_machine` to make sure that positions held in
    the market reflect what is requested by strategies.

    """

    def __init__(
        self,
        state_machine: Optional[StateMachine] = None,
        ib: Optional[ibi.IB] = None,
        trader: Optional[Trader] = None,
    ):
        self.sm = state_machine or STATE_MACHINE
        self.ib = ib or IB
        self.trader = trader or Trader(self.ib)

        # these are essential events
        self.ib.execDetailsEvent.connect(self.onExecDetailsEvent)
        self.ib.newOrderEvent.connect(self.onNewOrderEvent)
        self.ib.orderStatusEvent.connect(self.onOrderStatusEvent)

        log.debug("hold set")
        self.hold = True

        if CONFIG.get("use_blotter"):
            self.blotter = Blotter()
            self.ib.commissionReportEvent += self.onCommissionReport

        if CONFIG.get("log_IB_events"):
            self._attach_logging_events()

        self.cold_start = CONFIG.get("coldstart")

        log.debug(f"Controller initiated: {self}")

    def _attach_logging_events(self):
        # these are non-essential events
        self.ib.newOrderEvent += self.log_new_order
        self.ib.cancelOrderEvent += self.log_cancel
        self.ib.orderModifyEvent += self.log_modification
        self.ib.errorEvent += self.log_error

    async def sync(self, *args, **kwargs) -> None:
        log.debug("Sync...")

        if self.cold_start:
            log.debug("Starting cold...")
            self.cold_start = False
        else:
            try:
                log.debug("Reading from store...")
                await self.sm.read_from_store()
            except Exception as e:
                log.exception(e)

        try:
            unknown, matched, unmatched = OrderSyncStrategy.run()
            log.debug(
                f"Trades on re-start - unknown: {len(unknown)}, "
                f"matched: {len(matched)}, unmatched: {len(unmatched)}"
            )
        except Exception as e:
            log.exception(e)
            raise

        # update order records with current Trade objects
        for error_trade in unknown:
            log.critical(f"Unknow trade in the system: {error_trade}.")

        # From here IB events will be handled...
        self.hold = False
        log.debug("hold released")
        # ...so that matched trades can be sent to blotter

        try:
            for trade in matched:
                self.report_unresolved_trade(trade)
        except Exception as e:
            log.exception(f"Error with unresolved trade: {e}")

        try:
            # don't know what to do with that yet:
            self.sm.override_inactive_trades(*unmatched)
            # if self.cold_start:
            #     self.sm.override_inactive_trades(*unmatched)
            # else:
            #     for trade in unmatched:
            #         log.error(f"We have trade that IB doesn't know about: {trade}")

        except Exception as e:
            log.exception(f"Error handling unmatched trades: {e}")

        try:
            if error_positions := PositionSyncStrategy.run():
                log.critical(f"Wrong positions on restart: {error_positions}")
            log.debug("Sync completed.")
        except Exception as e:
            log.exception(f"Error handling wrong position on restart: {e}")

    def report_unresolved_trade(self, trade: ibi.Trade):
        log.debug(
            f"Back-reporting trade: {trade.contract.symbol} "
            f"{trade.order.action} {trade.order.totalQuantity} "
            f"order id: {trade.order.orderId}"
        )

        self.ib.orderStatusEvent.emit(trade)
        self.ib.commissionReportEvent.emit(
            trade, trade.fills[-1], trade.fills[-1].commissionReport
        )

    def trade(
        self,
        strategy: str,
        contract: ibi.Contract,
        order: ibi.Order,
        action: str,
        data: Model,
    ) -> ibi.Trade:
        trade = self.trader.trade(contract, order)
        self.sm.register_order(strategy, action, trade, data)
        trade.filledEvent += partial(self.log_trade, reason=action, strategy=strategy)
        return trade

    def cancel(
        self,
        trade: ibi.Trade,
        exec_model: AbstractExecModel,
        callback: Optional[Callable[[ibi.Trade], None]] = None,
    ) -> None:
        trade = self.trader.cancel(trade)
        if callback is not None:
            callback(trade)
        # orders are cancelled by callbacks so this is duplicating
        # self.sm.register_cancel(trade, exec_model)

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

    async def onNewOrderEvent(self, trade: ibi.Trade) -> None:
        """
        Check if the system knows about the order that was just posted
        to the broker.

        This is an event handler (callback).  Connected (subscribed)
        to :meth:`ibi.IB.newOrderEvent` in :meth:`__init__`
        """

        await asyncio.sleep(0.1)
        existing_order_record = self.sm.get_order(trade.order.orderId)
        if not existing_order_record:
            log.critical(f"Unknown trade in the system {trade.order}")

        # Give exec_model a chance to save bracket
        await asyncio.sleep(0.5)
        self.sm.report_new_order(trade)

    def onOrderStatusEvent(self, trade: ibi.Trade) -> None:
        if trade.order.orderId < 0:
            log.error(f"Manual trade: {trade.order} status update: {trade.orderStatus}")
        elif trade.orderStatus.status == ibi.OrderStatus.Inactive:
            messages = ";".join([m.message for m in trade.log])
            log.error(f"Rejected order: {trade.order}, messages: {messages}")
        elif trade.isDone():
            # TODO: FIX THIS
            if self.sm.report_done_order(trade):
                log.debug(
                    f"{trade.contract.symbol}: order {trade.order.orderId} "
                    f"{trade.order.orderType} done {trade.orderStatus.status}."
                )
            else:
                log.debug(
                    f"Unknown order cancelled id: {trade.order.orderId} "
                    f"{trade.order.action} {trade.contract.symbol}"
                )

        else:
            log.debug(
                f"{trade.contract.symbol}: OrderStatus ->{trade.orderStatus.status}<-"
                f" for order: {trade.order},"
            )

    def register_position(
        self, strategy: str, model: Model, trade: ibi.Trade, fill: ibi.Fill
    ) -> None:
        if fill.execution.side == "BOT":
            model.position += fill.execution.shares
            log.debug(
                f"Registered {trade.order.orderId} BUY {trade.order.orderType} "
                f"for {strategy} --> position: {model.position}"
            )
        elif fill.execution.side == "SLD":
            model.position -= fill.execution.shares
            log.debug(
                f"Registered {trade.order.orderId} SELL {trade.order.orderType} "
                f"for {strategy} --> position: {model.position}"
            )
        else:
            log.critical(
                f"Abiguous fill: {fill} for order: {trade.order} for "
                f"{trade.contract.localSymbol} strategy: {strategy}"
            )

    def onExecDetailsEvent(self, trade: ibi.Trade, fill: ibi.Fill) -> None:
        order_info = self.sm.get_order(trade.order.orderId)
        if order_info:
            strategy = order_info.strategy
        model = self.sm.get_strategy(strategy)
        if model:
            self.register_position(strategy, model, trade, fill)
        else:
            log.critical(f"Unknow trade: {trade}")

    def onCommissionReport(
        self, trade: ibi.Trade, fill: ibi.Fill, report: ibi.CommissionReport
    ) -> None:
        """
        Writing commission on :class:`ibi.Trade` is the final stage of order
        execution.  After that trade object is ready for stroring in
        blotter.
        """
        # prevent writing all orders from session on startup
        if self.hold:
            return
        data = self.sm.get_order(trade.order.orderId)
        if data:
            strategy, action, _, params, _ = data
            position_id = params.get("position_id")

            kwargs = {
                "strategy": strategy,
                "action": action,
                "position_id": position_id,
                "params": ibi.util.tree(params),
            }

            if arrival_price := params.get("arrival_price"):
                kwargs.update(
                    {
                        "price_time": arrival_price["time"],
                        "bid": arrival_price["bid"],
                        "ask": arrival_price["ask"],
                    }
                )

        elif trade.order.totalQuantity == 0:
            return
        elif trade.order.orderId < 0:
            kwargs = {"action": "MANUAL TRADE"}
        else:
            kwargs = {"action": "UNKNOW"}
            log.debug(
                f"Missing strategy records in `state machine`. "
                f"Incomplete data for blotter."
                f"orderId: {trade.order.orderId} symbol: {trade.contract.symbol} "
                f"orderType: {trade.order.orderType}"
            )

        assert self.blotter is not None
        log.debug(f"Order will be logged: {trade.order.nonDefaults()}")  # type: ignore
        self.blotter.log_commission(trade, fill, report, **kwargs)

    def log_new_order(self, trade: ibi.Trade) -> None:
        log.debug(f"New order {trade.order} for {trade.contract.localSymbol}")

    def log_trade(self, trade: ibi.Trade, reason: str = "", strategy: str = "") -> None:
        log.info(
            f"{reason} trade filled: {trade.contract.localSymbol} "
            f"{trade.order.action} {trade.orderStatus.filled}"
            f"@{trade.orderStatus.avgFillPrice} --> {strategy}"
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

    def log_error(
        self, reqId: int, errorCode: int, errorString: str, contract: ibi.Contract
    ) -> None:
        pass
        if errorCode < 400:
            # reqId is most likely orderId
            # order rejected is errorCode = 201
            order_info = self.sm.get_order(reqId)
            if order_info:
                strategy, action, trade, _ = order_info
                order = trade.order
            else:
                strategy, action, trade, order = "", "", "", ""  # type: ignore

            log.error(
                f"Error {errorCode}: {errorString} {contract}, "
                f"{strategy} | {action} | {order}"
            )

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

    def __repr__(self):
        return f"{__class__.__name__}({self.sm, self.ib, self.blotter})"


CONTROLLER: Final[Controller] = Controller()
