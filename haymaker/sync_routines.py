from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import TYPE_CHECKING

import ib_insync as ibi

from . import misc
from .state_machine import OrderInfo, StateMachine, Strategy

if TYPE_CHECKING:
    from .controller import Controller


log = logging.getLogger(__name__)


class OrderSyncStrategy:
    def __init__(self, ib: ibi.IB, sm: StateMachine) -> None:
        self.ib = ib
        self.sm = sm
        # IB has trades that we don't know about
        self.unknown: list[ibi.Trade] = []  # <-We're fucked
        # Our active trades that IB doesn't report as active
        self.inactive: list[ibi.Trade] = []
        # Inactive trades that we managed to match to IB
        self.done: list[ibi.Trade] = []
        # Trades on record that we cannot resolve with IB
        self.errors: list[ibi.Trade] = []  # <- We're potentially fucked
        self._issues: list[int] = []  # done orders for double checking
        self.is_ok: bool = True

    @classmethod
    def run(cls, ib: ibi.IB, sm: StateMachine) -> OrderSyncStrategy:
        return (
            cls(ib, sm)
            .update_trades()
            .review_trades()
            .handle_inactive_trades()
            .report()
        )

    @property
    def lists(self) -> tuple[list[ibi.Trade], list[ibi.Trade], list[ibi.Trade]]:
        return self.unknown, self.done, self.errors

    def update_trades(self) -> OrderSyncStrategy:
        """
        Update order records with current Trade objects.
        `unknown_trades` are IB trades without records in SM.
        """
        for trade in self.ib.openTrades():
            if ut := self.sm.update_trade(trade):  # <- CHANGING RECORDS
                self.unknown.append(ut)
        return self

    def review_trades(self) -> OrderSyncStrategy:
        """
        Review all trades on record and compare their status with IB.

        Produce a list of trades that we have as open, while IB has
        them as done.  We have to reconcile those trades' status and
        report them as appropriate.
        """
        ib_open_trades = {trade.order.orderId: trade for trade in self.ib.openTrades()}
        # log.debug(f"ib open trades: {list(ib_open_trades.keys())}")
        for orderId, oi in self.sm._orders.copy().items():
            # log.debug(f"verifying {orderId}")
            if orderId not in ib_open_trades:
                # if inactive it's already been dealt with before sync
                if oi.active:
                    log.debug(f"reporting inactive: {orderId}")
                    # this is a trade that we have as active in self.orders
                    # but IB doesn't have it in open orders
                    # we have to figure out what happened to this trade
                    # while we were disconnected and report it as appropriate
                    self.inactive.append(oi.trade)
                else:
                    log.debug(f"Will prune order: {orderId}")
                    # this order is no longer of interest
                    # it's inactive in our records and inactive in IB
                    self.sm.prune_order(orderId)  # <- CHANGING RECORDS
        return self

    def handle_inactive_trades(self) -> OrderSyncStrategy:
        """
        Handle trades that are we have as active (in state machine),
        but they haven't been matched to open trades in IB.  Try
        finding them in closed trades from the session.
        """
        ib_known_trades = {
            trade.order.orderId or trade.order.permId: trade
            for trade in self.ib.trades()
        }

        # log.debug(f"ib known trades: {list(ib_known_trades.keys())}")
        if self.inactive:
            log.debug(
                f"inactive trades: "
                f"{[(i.order.orderId, i.order.permId) for i in self.inactive]}"
            )

        for old_trade in self.inactive:
            new_trade = ib_known_trades.get(old_trade.order.permId)
            if new_trade:
                new_trade.order.orderId = old_trade.order.orderId
                self.done.append(new_trade)
                self.sm.update_trade(new_trade)  # <- CHANGING RECORDS
            else:
                self.errors.append(old_trade)

        if self.done:
            log.debug(f"done: {[(t.order.orderId, t.order.permId) for t in self.done]}")

        return self

    def verify(self) -> OrderSyncStrategy:
        completed_trades = self.ib.reqCompletedOrders(True)
        completed_trades_list = [trade.order.permId for trade in completed_trades]
        my_orders = [oi.permId for oi in self.sm._orders.values()]
        self._issues = [
            permId for permId in my_orders if permId in completed_trades_list
        ]
        log.debug(f"{self._issues=}")
        return self

    def report(self) -> OrderSyncStrategy:
        if any(self.lists):
            log.debug(
                f"Trades on sync -> "
                f"unknown: {len(self.unknown)}, "
                f"done: {len(self.done)}, "
                f"unmatched: {len(self.errors)}"
            )
        # elif self._issues and not self.done:
        #     log.debug(f"{self._issues}")
        #     log.debug(f"{self.done}")
        #     log.critical("We are fucked and shouldn't proceed!!!")
        #     self.is_ok = False
        else:
            log.debug("Orders sync OK.")

        for unknown_trade in self.unknown:
            log.critical(f"Unknow trade in the system: {unknown_trade}.")

        return self


class PositionSyncStrategy:
    """
    Must be called after :class:`.OrderSyncStrategy`
    """

    def __init__(self, ib: ibi.IB, sm: StateMachine):
        self.ib = ib
        self.sm = sm
        self.errors: dict[ibi.Contract, float] = {}

    @classmethod
    def run(cls, ib: ibi.IB, sm: StateMachine):
        return cls(ib, sm).verify_positions().report()

    def verify_positions(self) -> PositionSyncStrategy:
        """
        Compare positions actually held with position records in state
        machine.  Return differences if any and log an error.
        """

        broker_positions_dict = {i.contract: i.position for i in self.ib.positions()}
        # log.debug(
        #     f"broker positions: "
        #     f"{ {k.symbol: v for k,v in broker_positions_dict.items()} }"
        # )
        my_positions_dict = self.sm.strategy.total_positions()
        # log.debug(
        #     f"my positions: "
        #     f"{ {k.symbol: v for k, v in my_positions_dict.items() if v} }"
        # )
        diff = {
            i: (
                (my_positions_dict.get(i) or 0.0)
                - (broker_positions_dict.get(i) or 0.0)
            )
            for i in set([*broker_positions_dict.keys(), *my_positions_dict.keys()])
        }
        self.errors = {k: v for k, v in diff.items() if v}
        if self.errors:
            log.error(f"errors: { {k.symbol: v for k, v in self.errors.items()} }")

        return self

    def report(self) -> PositionSyncStrategy:
        if self.errors:
            log.critical(f"Failed to match positions to broker: {self.errors}")
        else:
            log.debug("Positions matched to broker OK.")
        return self


class ErrorHandlers:
    def __init__(self, ib: ibi.IB, sm: StateMachine, ct: Controller) -> None:
        self.ib = ib
        self.sm = sm
        self.controller = ct
        self.faulty_trades: list[OrderInfo] = []

    async def handle_orders(self, report: OrderSyncStrategy) -> None:
        try:
            for done_trade in report.done:
                self.report_done_trade(done_trade)
        except Exception as e:
            log.exception(f"Error with done trade: {e}")

        await asyncio.sleep(0)
        try:
            # TODO: don't know what to do with that yet
            self.clear_error_trades(report.errors)

        except Exception as e:
            log.exception(f"Error handling inactive trades: {e}")

        await asyncio.sleep(0)

    async def handle_positions(self, report: PositionSyncStrategy) -> None:
        errors: dict[ibi.Contract, float] = report.errors
        if not errors:
            return

        # too many externalities... refactor
        log.error("Will attempt to fix position records")
        for contract, diff in errors.items():
            strategies = self.sm.for_contract.get(contract)
            log.debug(f"{strategies=}")
            if strategies and len(strategies) == 1:
                self.sm.strategy[strategies[0]].position -= diff
                log.error(
                    f"Corrected position records for strategy "
                    f"{strategies[0]} by {-diff}"
                )
                self.sm.save_models()

            elif (
                strategies
                and self.controller.trader.position_for_contract(contract) == 0
            ):
                for strategy in strategies:
                    self.sm.strategy[strategy].position = 0
                self.sm.save_models()
                log.error(
                    f"Position records zeroed for {strategies} "
                    f"to reflect zero position for {contract.symbol}."
                )
            elif strategies:
                strategy_faults = [
                    order_info.strategy for order_info in self.faulty_trades
                ]
                for strategy in strategies:
                    if strategy in strategy_faults:
                        self.sm.strategy[strategy].position = 0
                        log.error(
                            f"Position records zeroed for {strategy} "
                            f"to reflect faulty trade previously removed."
                        )

            else:
                # TODO: ASSIGN POSITION TO UNKNOWN or random or close position
                log.critical(
                    f"More than 1 strategy for contract {contract.symbol}, "
                    f"cannot fix position records."
                )
            self.faulty_trades.clear()

    def clear_error_trades(self, trades: list[ibi.Trade]) -> None:
        """
        Trades that we have as active but IB doesn't know about them.
        """
        for trade in trades:
            log.error(
                f"Will delete record for trade that IB doesn't known about: "
                f"{trade.order.orderId}"
            )
            self.faulty_trades.append(self.sm._orders[trade.order.orderId])
            self.sm.delete_trade_record(trade)

    def report_done_trade(self, trade: ibi.Trade) -> None:
        """
        IB events artificially emitted here will trigger saving trade
        to blotter.
        """
        log.debug(
            f"Back-reporting trade: {trade.contract.symbol} "
            f"{trade.order.action} {misc.trade_fill_price(trade)} "
            f"order id: {trade.order.orderId} {trade.order.permId} "
            f"active?: {trade.isActive()}"
        )
        self.ib.orderStatusEvent.emit(trade)
        for fill in trade.fills:
            self.ib.execDetailsEvent.emit(trade, fill)
        if trade.orderStatus.status == "Filled":
            self.ib.commissionReportEvent.emit(
                trade, trade.fills[-1], trade.fills[-1].commissionReport
            )

    def __repr__(self):
        return "ErrorHandlers()"


# -------------------------------------------
# Everything below is not in use at the moment


class StoplossesInPlacePlaceholder:
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


class Terminator:
    """
    Class organizing the process of cancelling all resting orders and
    closing all positins.

    First, an attempt is made to identify resting stop-losses that are
    associated with strategies.  Those orders are cancelled but
    corresponding identical market orders are issued associated with
    those strategies.  The purpose of this operation is to create
    blotter records allowing to match those closing positions with
    their openning counterparts.

    All other orders are cancelled and remaining positions closed.
    Throughout the process records are kept to make sure no double
    orders are issued for the same position or stop losses are not
    executed for non-existing positions.
    """

    def __init__(self, controller: Controller) -> None:
        log.warning("RESET initiated.")
        self.controller = controller
        self.in_progress_trades: list[ibi.Trade] = []
        self.read_broker_state()

    def read_broker_state(self):
        self.trades = self.controller.ib.openTrades()
        self.positions = self.controller.trader.positions()

    async def run(self):
        log.debug("Reset S T A G E 1: Cancel stops and send close market orders")
        await self.cancel_orders()

        log.debug("Reset S T A G E 2: Close remaining positions...")
        await self.controller.ib.qualifyContractsAsync(*self.positions.keys())
        await self.close_positions()

        log.debug("Closing positions finished, will verify now.")
        if unsuccessful_trades := await self.report():
            log.debug(
                f"These trades failed to execute: "
                f"{[self.describe_trade(t) for t in unsuccessful_trades]}"
            )
            log.debug("Going nuclear...")
            self.controller.ib.reqGlobalCancel()
            await asyncio.sleep(3)
            log.debug("All orders should be cancelled now.")
            log.debug(f"Opend trades: {self.controller.ib.openTrades()}")
            log.debug("Will run close positions.")
            self.controller.close_positions()
        # neccessary to make sure order cancelletions completed
        await asyncio.sleep(1)
        if self.verify_zero():
            log.debug(
                f"Closed all positions (trades in progress: {self.in_progress_trades}, "
                f"all open trades: {self.trades})."
            )
        else:
            log.critical("We are F U C K E D !")
            # Nuke here ?

    def verify_zero(self):
        self.read_broker_state()
        if positions := self.positions:
            log.critical(f"Failed to close positions: {positions}")
        if trades := self.trades:
            log.critical(f"Failed to cancel orders: {trades}")
        return positions or trades or True

    def register_trade(self, trade: ibi.Trade) -> None:
        self.in_progress_trades.append(trade)
        trade.filledEvent += self.in_progress_trades.remove
        trade.cancelledEvent += self.in_progress_trades.remove
        trade.cancelledEvent += lambda x: log.error(f"Rejected trade: {x}")

    @staticmethod
    def describe_trade(t: ibi.Trade) -> tuple[str, str, float, int]:
        return (
            t.contract.symbol,
            t.order.action,
            t.order.totalQuantity,
            t.order.orderId,
        )

    async def _strategy_trade(
        self,
        trade: ibi.Trade,
        strategy: Strategy,
        contract: ibi.Contract,
        action: str,
        amount: float,
    ) -> ibi.Trade:
        log.debug(
            f"Will attempt to execute closing trade for "
            f"{strategy.strategy} {contract} {action} {amount}"
        )
        new_trade = self.controller.trade(
            strategy.strategy,
            contract,
            ibi.MarketOrder(action, amount),
            "RESET",
            strategy,
        )
        self.register_trade(new_trade)
        return new_trade

    async def cancel_orders(self):
        log.debug(
            f"Will attempt to cancel orders: "
            f"{[self.describe_trade(t) for t in self.trades]}"
        )

        for trade in self.trades:
            if (
                trade.order.orderType in ("STP", "TRAIL")
                and (strategy := self.controller.sm.strategy_for_trade(trade))
                and (strategy != "UNKNOWN")
            ):
                # Make sure no orders executed if there's no corresponding position
                try:
                    stop_amount = (
                        trade.remaining()
                        if trade.order.action == "BUY"
                        else -trade.remaining()
                    )
                    existing_amount = self.positions.get(trade.contract, 0)
                except Exception as e:
                    log.exception(e)
                if existing_amount and (
                    misc.sign(stop_amount) == -misc.sign(existing_amount)
                ):
                    trade_amount = min(
                        abs(existing_amount), abs(stop_amount)
                    ) * misc.sign(stop_amount)

                    log.debug(f"Will cancel {trade_amount} for {strategy.strategy}")
                    trade.cancelledEvent.connect(
                        partial(
                            self._strategy_trade,
                            strategy=strategy,
                            contract=trade.contract,
                            action=trade.order.action,
                            amount=abs(trade_amount),
                        ),
                        self.controller._log_event_error,
                    )
                    self.controller.cancel(trade)
                    self.positions[trade.contract] += trade_amount
                    await asyncio.sleep(0)
                    continue
            self.controller.cancel(trade)

    async def close_positions(self):
        log.debug(
            f"Trades in progress: "
            f"{[self.describe_trade(t) for t in self.in_progress_trades]}."
        )
        log.debug(
            f"Will attempt to close remaining positions: "
            f"{ {c.symbol: p for c, p in self.positions.items() if p} }"
        )

        for contract, position in self.positions.items():
            if position:
                log.debug(f"Closing {contract.symbol}, {position}")
                # Again make sure total orders issued match existing positions
                self.register_trade(
                    self.controller.trader.trade(
                        contract,
                        ibi.MarketOrder(
                            "BUY" if position < 0 else "SELL",
                            abs(position),
                            tif="DAY",
                            outsideRth=True,
                        ),
                    )
                )
                await asyncio.sleep(0)

    async def report(self):
        n = 0
        log.debug(
            f"Reset in progress: {[t.order.orderId for t in self.in_progress_trades]}"
        )
        while not all([t.isDone() for t in self.in_progress_trades]):
            log.warning(
                f"Trades in progress: "
                f"{[self.describe_trade(t) for t in self.trades]}"
            )
            await asyncio.sleep(1)
            if (n := n + 1) > 10:
                break
        return [t for t in self.in_progress_trades if not t.isDone()]


# class StoplossesInPlace:
#     def __init__(
#         self,
#         trader: Trader,
#         state_machine: StateMachine,
#         exec_model: EventDrivenExecModel,
#     ):
#         self._trader = trader
#         self.sm = state_machine
#         self.exec_model = exec_model

#     def onStarted(self) -> None:
#         """
#         Make sure all positions have corresponding stop-losses or
#         emergency-close them. Check for stray orders (take profit or stop loss
#         without associated position) and cancel them if any. Attach events
#         canceling take-profit/stop-loss after one of them is hit.
#         """
#         # check for orphan positions
#         trades = self._trader.trades()
#         positions = self._trader.positions()
#         positions_for_log = [
#             (position.contract.symbol, position.position) for position in positions
#         ]
#         log.debug(f"positions on re-connect: {positions_for_log}")

#         orphan_positions = self.check_for_orphan_positions(trades, positions)
#         if orphan_positions:
#             log.warning(f"orphan positions: {orphan_positions}")
#             log.debug(f"number of orphan positions: {len(orphan_positions)}")
#             self.handle_orphan_positions(orphan_positions)

#         orphan_trades = self.check_for_orphan_trades(trades, positions)
#         if orphan_trades:
#             log.warning(f"orphan trades: {orphan_trades}")
#             self.handle_orphan_trades(orphan_trades)

#         for trade in trades:
#             trade.filledEvent += self.exec_model.bracket_filled_callback

#     def handle_orphan_positions(self, orphan_positions: list[ibi.Position]) -> None:
#         self._trader.ib.qualifyContracts(*[p.contract for p in orphan_positions])
#         for p in orphan_positions:
#             self.emergencyClose(
#                 p.contract, -np.sign(p.position), int(np.abs(p.position))
#             )
#             log.error(
#                 f"emergencyClose position: {p.contract}, "
#                 f"{-np.sign(p.position)}, {int(np.abs(p.position))}"
#             )
#             t = self._trader.ib.reqAllOpenOrders()
#             log.debug(f"reqAllOpenOrders: {t}")
#             trades_for_log = [
#                 (
#                     trade.contract.symbol,
#                     trade.order.action,
#                     trade.order.orderType,
#                     trade.order.totalQuantity,
#                     trade.order.lmtPrice,
#                     trade.order.auxPrice,
#                     trade.orderStatus.status,
#                     trade.order.orderId,
#                 )
#                 for trade in self._trader.ib.openTrades()
#             ]
#             log.debug(f"openTrades: {trades_for_log}")

#     def handle_orphan_trades(self, trades: list[ibi.Trade]) -> None:
#         for trade in trades:
#             self._trader.cancel(trade)

#     def emergencyClose(self, contract: ibi.Contract, signal: int, amount: int) -> None:
#         """
#         Used if on (re)start positions without associated stop-loss orders
#         detected.
#         """
#         log.warning(
#             f"Emergency close on restart: {contract.localSymbol} "
#             f"side: {signal} amount: {amount}"
#         )
#         trade = self._trader.trade(
#             contract,
#             ibi.MarketOrder(misc.action(signal), amount, tif="GTC"),
#             "EMERGENCY CLOSE",
#         )
#         trade.filledEvent += self.exec_model.bracket_filled_callback


# class ReportEvents:
#     def onStarted(self):
#         """
#         Attach reporting events to all open trades.
#         """
#         trades = self.trades()
#         log.info(
#             f"open trades on re-connect: {len(trades)} "
#             f"{[t.contract.localSymbol for t in trades]}"
#         )
#         # attach reporting events
#         for trade in trades:
#             if trade.orderStatus.remaining != 0:
#                 if trade.order.orderType in ("STP", "TRAIL"):
#                     self.attach_events(trade, "STOP-LOSS")
#                 else:
#                     self.attach_events(trade, "TAKE-PROFIT")
