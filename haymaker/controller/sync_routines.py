from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self

import ib_insync as ibi

from haymaker import misc
from haymaker.state_machine import OrderInfo, StateMachine, Strategy
from haymaker.config import CONFIG

config = CONFIG.get("sync", {})

CANCEL_UNKNOWN_TRADES: bool = config.get("cancel_unknown_trades", False)
CANCEL_STRAY_ORDERS: bool = config.get("cancel_stray_orders", False)
HANDLE_MISSING_BRACKETS: str = config.get("handle_missing_brackets", "remove")


if TYPE_CHECKING:
    from .controller import Controller


log = logging.getLogger(__name__)


ERROR_STRATEGIES: set[str] = set()


@dataclass
class OrderSyncActions:
    """Actions taken during order sync that later checks must know about."""

    faulty_trades: list[OrderInfo] = field(default_factory=list)


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

    @classmethod
    def run(cls, ib: ibi.IB, sm: StateMachine) -> Self:
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

    def update_trades(self) -> Self:
        """
        Update order records with current Trade objects.
        `unknown_trades` are IB trades without records in SM.
        """
        for trade in self.ib.openTrades():
            if ut := self.sm.update_trade(trade):  # <- CHANGING RECORDS
                self.unknown.append(ut)
        return self

    def review_trades(self) -> Self:
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

    def handle_inactive_trades(self) -> Self:
        """
        Handle trades that we have as active (in state machine),
        but they haven't been matched to open trades in IB.  Try
        finding them in closed trades from the session.
        """
        ib_known_trades = {
            trade.order.orderId or trade.order.permId: trade
            for trade in self.ib.trades()
        }

        if self.inactive:
            log.debug(
                f"inactive trades: "
                f"{[(i.order.orderId, i.order.permId) for i in self.inactive]}"
            )
            log.debug(f"ib_known_trades: {list(ib_known_trades)}")
        # if cannot match inactive trade by orderId try by permId
        # (permId persists in between restarts)
        # if succcessful modify its orderId
        # orders have orderId == 0 if they were filled while system was off
        # we're assigning to them our last known orderId
        # (from before system went off) so that they can be matched correctly
        # to db records and their status updated (to done)
        for old_trade in self.inactive:
            if new_trade := ib_known_trades.get(old_trade.order.permId):
                log.warning(
                    f"Will change orderId: {new_trade.order.orderId} "
                    f"to: {old_trade.order.orderId}"
                )
                new_trade.order.orderId = old_trade.order.orderId
                self.done.append(new_trade)
                self.sm.update_trade(new_trade)  # <- CHANGING RECORDS
            # trying to manually update trade from fills
            elif fills := [
                f
                for f in self.ib.fills()
                if f.execution.orderId == old_trade.order.orderId
            ]:
                old_trade.fills = fills
                filled = sum([fill.execution.shares for fill in fills])
                remaining = old_trade.order.totalQuantity - filled
                old_trade.log.append(
                    ibi.objects.TradeLogEntry(
                        time=fills[-1].execution.time,
                        status="Filled" if remaining == 0 else "Submitted",
                        message="composed by sync_routines",
                    )
                )
                old_trade.orderStatus = ibi.OrderStatus(
                    orderId=old_trade.order.orderId,
                    status=(
                        ibi.OrderStatus.Filled
                        if remaining == 0
                        else ibi.OrderStatus.Submitted
                    ),
                    filled=filled,
                    remaining=remaining,
                )

                self.done.append(old_trade)
                self.sm.update_trade(old_trade)
            else:
                self.errors.append(old_trade)

        if self.done:
            log.debug(f"done: {[(t.order.orderId, t.order.permId) for t in self.done]}")

        return self

    def verify(self) -> Self:
        completed_trades = self.ib.reqCompletedOrders(True)
        completed_trades_list = [trade.order.permId for trade in completed_trades]
        my_orders = [oi.permId for oi in self.sm._orders.values()]
        self._issues = [
            permId for permId in my_orders if permId in completed_trades_list
        ]
        log.debug(f"{self._issues=}")
        return self

    def report(self) -> Self:
        if any(self.lists):
            log.debug(
                f"Trades on sync -> "
                f"unknown: {len(self.unknown)}, "
                f"done: {len(self.done)}, "
                f"unmatched: {len(self.errors)}"
            )
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

    def verify_positions(self) -> Self:
        """
        Compare positions actually held with position records in state
        machine.  Return differences if any and log an error.
        """

        broker_positions_dict = {i.contract: i.position for i in self.ib.positions()}
        my_positions_dict = self.sm.strategy.total_positions()
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

    def report(self) -> Self:
        if self.errors:
            log.critical(f"Failed to match positions to broker: {self.errors}")
        else:
            log.debug("Positions matched to broker OK.")
        return self


class OrderErrorHandlers:
    def __init__(
        self,
        ib: ibi.IB,
        sm: StateMachine,
        ct: Controller,
    ) -> None:
        self.ib = ib
        self.sm = sm
        self.controller = ct

    async def handle_report(self, report: OrderSyncStrategy) -> OrderSyncActions:
        actions = OrderSyncActions()

        self.handle_unknown_broker_orders(report.unknown)

        try:
            for done_trade in report.done:
                self.report_done_trade(done_trade)
        except Exception as e:
            log.exception(f"Error with done trade: {e}")

        await asyncio.sleep(0)
        try:
            self.clear_error_trades(report.errors, actions)

        except Exception as e:
            log.exception(f"Error handling inactive trades: {e}")

        await asyncio.sleep(0)
        return actions

    def handle_unknown_broker_orders(self, trades: list[ibi.Trade]) -> None:
        if not trades:
            return

        log.critical(f"Unknown broker orders during sync: {trades}.")
        if not CANCEL_UNKNOWN_TRADES:
            log.critical(
                "Unknown broker orders left active because "
                "cancel_unknown_trades is False."
            )
            return

        for trade in trades:
            log.debug(f"Cancelling unknown broker order: {trade.order.orderId}")
            self.controller.cancel(trade)

    def clear_error_trades(
        self, trades: list[ibi.Trade], actions: OrderSyncActions
    ) -> None:
        """
        Trades that we have as active but IB doesn't know about them.
        """
        for trade in trades:
            log.error(
                f"Will delete record for trade that IB doesn't known about: "
                f"{trade.order.orderId}"
            )
            actions.faulty_trades.append(self.sm._orders[trade.order.orderId])
            self.sm.prune_order(trade.order.orderId)

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


class PositionErrorHandlers:
    def __init__(
        self,
        ib: ibi.IB,
        sm: StateMachine,
        ct: Controller,
    ) -> None:
        self.ib = ib
        self.sm = sm
        self.controller = ct

    async def handle_report(
        self, report: PositionSyncStrategy, order_actions: OrderSyncActions
    ) -> None:
        errors: dict[ibi.Contract, float] = report.errors
        if not errors:
            return

        log.error("Will attempt to fix position records")
        for contract, diff in errors.items():
            strategies = self.sm.for_contract.get(contract)
            log.debug(f"Strategies for contract {contract.localSymbol}: {strategies}")
            if strategies and len(strategies) == 1:
                self.sm.strategy[strategies[0]].position -= diff
                log.error(
                    f"Corrected position records for strategy "
                    f"{strategies[0]} by {-diff}"
                )
                self.sm.save_strategies()

            elif (
                strategies
                and self.controller.trader.position_for_contract(contract) == 0
            ):
                for strategy in strategies:
                    self.sm.strategy[strategy].position = 0
                self.sm.save_strategies()
                log.error(
                    f"Position records zeroed for {strategies} "
                    f"to reflect zero position for {contract.symbol}."
                )
            elif strategies:
                strategy_faults = [
                    order_info.strategy for order_info in order_actions.faulty_trades
                ]
                for strategy in strategies:
                    if strategy in strategy_faults:
                        self.sm.strategy[strategy].position = 0
                        log.error(
                            f"Position records zeroed for {strategy} "
                            f"to reflect faulty trade previously removed."
                        )

            else:
                # too risky to make assumptions about strategy (what about sl?)
                log.critical(
                    f"Cannot fix position records for {contract.localSymbol}, "
                    f"{strategies=}."
                )

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class OrderReconciliationSync:
    """
    Check if there are closing orders for a strategy that doesn't have a position.
    Cancel or warn if there is.
    """

    def __init__(
        self,
        ib: ibi.IB,
        sm: StateMachine,
        ct: Controller,
    ) -> None:
        self.ib = ib
        self.sm = sm
        self.ct = ct

    @classmethod
    def run(cls, ct: Controller) -> Self:
        return cls(ct.ib, ct.sm, ct).run_strategies()

    def run_strategies(self) -> Self:
        for strategy_str, strategy in self.sm.strategy.items():
            order_infos = self.sm.orders_for_strategy(strategy.strategy)
            if strategy.position == 0:
                for oi in self._find_obsolete_orders(order_infos):
                    self._handle_obsolete_order(strategy_str, oi)
            else:
                self._check_brackets(strategy, order_infos)
        log.debug("Orders synced to positions.")
        return self

    def _find_obsolete_orders(self, order_infos: list[OrderInfo]) -> list[OrderInfo]:
        """Find stop/take-profit/close orders for a strategy that has no position."""
        # there is no position and we're not trying to open a new one
        return [oi for oi in order_infos if (oi.action != "OPEN") and oi.active]

    def _handle_obsolete_order(self, strategy_str: str, oi: OrderInfo) -> None:
        if CANCEL_STRAY_ORDERS:
            log.error(
                f"Cancelling obsolete order: "
                f"{strategy_str, oi.action, oi.trade.order.orderId}"
            )
            self.ct.trader.cancel(oi.trade)
        else:
            log.critical(
                f"Obsolete order for "
                f"{strategy_str}: {oi.action, oi.trade.order.orderId}"
            )

    def _check_brackets(self, strategy: Strategy, order_infos: list[OrderInfo]) -> None:
        if HANDLE_MISSING_BRACKETS not in ["remove", "warn"]:
            return

        params = strategy.get("params")
        if params and (
            brackets := [
                b_ for b in ("stop-loss", "take-profit") if (b_ := params.get(b))
            ]
        ):
            existing_orders = [
                oi
                for oi in order_infos
                if (oi.action in ("STOP-LOSS", "TAKE-PROFIT") and oi.active)
            ]
            if len(brackets) != len(existing_orders):
                _log = (
                    log.error
                    if strategy.strategy not in ERROR_STRATEGIES
                    else log.debug
                )
                _log(
                    f"Bracket error for {strategy.strategy}, "
                    f"position: {strategy.position} "
                    f"we have: {len(existing_orders)} orders, "
                    f"we should have: {len(brackets)} orders."
                )
                ERROR_STRATEGIES.add(strategy.strategy)

        if ERROR_STRATEGIES and HANDLE_MISSING_BRACKETS == "remove":

            # close positions for strategies without brackets, but
            # only if they don't cancel each other (skip cancelling strategies)
            non_cancelling_positions: defaultdict[ibi.Contract, int] = defaultdict(int)
            for strategy_str in ERROR_STRATEGIES:
                strategy = self.ct.sm.strategy[strategy_str]
                non_cancelling_positions[strategy.active_contract] += strategy.position
                self.ct.lock_new_positions()

            for strategy_str in ERROR_STRATEGIES:
                strategy = self.ct.sm.strategy[strategy_str]
                if non_cancelling_positions.get(strategy.active_contract):
                    if not self.ct.can_emergency_close_strategy(strategy):
                        continue
                    log.error(
                        f"Closing positions for strategy with missing bracket: "
                        f"{strategy.strategy}"
                    )
                    non_cancelling_positions[
                        strategy.active_contract
                    ] -= strategy.position
                    self.ct.close_positions_for_strategy(
                        strategy.strategy, "MISSING BRACKET EMERGENCY CLOSE"
                    )
            ERROR_STRATEGIES.clear()
