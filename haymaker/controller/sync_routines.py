from __future__ import annotations

import logging
from typing import Self

import ib_insync as ibi

from haymaker.state_machine import StateMachine

log = logging.getLogger(__name__)


class OrderSync:
    def __init__(self, ib: ibi.IB, sm: StateMachine) -> None:
        self.ib = ib
        self.sm = sm
        # broker has trades that local doesn't know about
        self.unknown: list[ibi.Trade] = []  # <-We're fucked
        # local active trades that broker doesn't report as active
        self.inactive: list[ibi.Trade] = []
        # inactive trades that were matched between local and broker
        self.done: list[ibi.Trade] = []
        # local trades that cannot be resolved with broker
        self.errors: list[ibi.Trade] = []  # <- We're potentially fucked
        self._issues: list[int] = []  # done orders for double checking

        self.update_trades().review_trades().handle_inactive_trades().report()

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
        for orderId, oi in self.sm._orders.copy().items():
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
        for old_trade in self.inactive:
            if (
                trade_ := self._match_trade(
                    old_trade, ib_known_trades.get(old_trade.order.permId)
                )
            ) or (trade_ := self._reconstruct_from_fills(old_trade)):
                self.done.append(trade_)
                self.sm.update_trade(trade_)  # <- CHANGING RECORDS
            else:
                self.errors.append(old_trade)

        if self.done:
            log.debug(f"done: {[(t.order.orderId, t.order.permId) for t in self.done]}")

        return self

    def _match_trade(
        self, old_trade: ibi.Trade, new_trade: ibi.Trade | None
    ) -> ibi.Trade | None:
        # if cannot match inactive trade by orderId try by permId
        # (permId persists in between restarts)
        # if succcessful modify its orderId
        # orders have orderId == 0 if they were filled while system was off
        # we're assigning to them our last known orderId
        # (from before system went off) so that they can be matched correctly
        # to db records and their status updated (to done)

        if new_trade is not None:
            log.warning(
                f"Will change orderId: {new_trade.order.orderId} "
                f"to: {old_trade.order.orderId}"
            )
            new_trade.order.orderId = old_trade.order.orderId
            return new_trade

    def _reconstruct_from_fills(self, trade: ibi.Trade) -> ibi.Trade | None:
        # trying to manually update trade from fills
        if fills := [
            f for f in self.ib.fills() if f.execution.orderId == trade.order.orderId
        ]:
            trade.fills = fills
            filled = sum([fill.execution.shares for fill in fills])
            remaining = trade.order.totalQuantity - filled
            trade.log.append(
                ibi.objects.TradeLogEntry(
                    time=fills[-1].execution.time,
                    status="Filled" if remaining == 0 else "Submitted",
                    message="composed by sync_routines",
                )
            )
            trade.orderStatus = ibi.OrderStatus(
                orderId=trade.order.orderId,
                status=(
                    ibi.OrderStatus.Filled
                    if remaining == 0
                    else ibi.OrderStatus.Submitted
                ),
                filled=filled,
                remaining=remaining,
            )
            return trade

    def verify(self) -> Self:
        # This is dead code now; use or delete
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

    @property
    def is_ok(self) -> bool:
        return not any(self.lists)


class PositionSync:
    """
    Must be called after :class:`.OrderSync`
    """

    def __init__(self, ib: ibi.IB, sm: StateMachine):
        self.ib = ib
        self.sm = sm
        self.errors: dict[ibi.Contract, float] = {}

        self.verify_positions().report()

    def verify_positions(self) -> Self:
        """
        Compare broker state and local state for positions.
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
        self._has_run = True
        if self.errors:
            log.critical(f"Failed to match positions to broker: {self.errors}")
        else:
            log.debug("Positions matched to broker OK.")
        return self

    @property
    def is_ok(self) -> bool:
        return not bool(self.errors)
