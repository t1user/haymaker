from __future__ import annotations

import logging

import ib_insync as ibi

from ib_tools import misc
from ib_tools.state_machine import StateMachine

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
        self.errors: list[ibi.Trade] = []  # <- We're fucked
        # self.report = {
        #     "unknown": self.unknown,
        #     "done": self.done,
        #     "errors": self.errors,
        # }

    @classmethod
    def run(cls, ib: ibi.IB, sm: StateMachine):
        return cls(ib, sm).update_trades().review_trades().handle_inactive_trades()

    def update_trades(self):
        """
        Update order records with current Trade objects.
        `unknown_trades` are IB trades without records in SM.
        """
        self.unknown = []
        for trade in self.ib.openTrades():
            if ut := self.sm.update_trade(trade):  # <- CHANGING RECORDS
                self.unknown_trades.append(ut)
        return self

    def review_trades(self):
        """
        Review all trades on record and compare their status with IB.

        Produce a list of trades that we have as open, while IB has
        them as done.  We have to reconcile those trades' status and
        report them as appropriate.
        """
        self.inactive = []
        ib_open_trades = {trade.order.orderId: trade for trade in self.ib.openTrades()}
        log.debug(f"ib open trades: {list(ib_open_trades.keys())}")
        for orderId, oi in self.sm._orders.copy().items():
            log.debug(f"verifying {orderId}")
            if orderId not in ib_open_trades:
                # if inactive it's already been dealt with before restart
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

    def handle_inactive_trades(self):
        # these are active trades on record that haven't been matched to
        # open trades in IB
        # try finding them in closed trades from the session

        ib_known_trades = {
            trade.order.orderId or trade.order.permId: trade
            for trade in self.ib.trades()
        }

        log.debug(f"ib known trades: {list(ib_known_trades.keys())}")
        log.debug(f"inactive trades: {self.inactive}")

        for old_trade in self.inactive:
            new_trade = ib_known_trades.get(old_trade.order.permId)
            if new_trade:
                new_trade.order.orderId = old_trade.order.orderId
                self.done.append(new_trade)
                self.sm.update_trade(new_trade)  # <- CHANGING RECORDS
            else:
                self.errors.append(old_trade)

        # done = {
        #     trade.order.permId: trade
        #     for trade in self.inactive
        #     if trade.order.permId in ib_known_trades
        # }

        log.debug(f"done: {[(t.order.orderId, t.order.permId) for t in self.done]}")

        return self


class PositionSyncStrategy:
    """
    Must be called after :class:`.OrderSyncStrategy`
    """

    def __init__(self, ib: ibi.IB, sm: StateMachine):
        self.ib = ib
        self.sm = sm
        self.errors: dict[ibi.Contract, float] = {}
        # self.report = {"errors": self.errors}

    @classmethod
    def run(cls, ib: ibi.IB, sm: StateMachine):
        return cls(ib, sm).verify_positions()

    def verify_positions(self) -> PositionSyncStrategy:
        """
        Compare positions actually held with broker with position
        records.  Return differences if any and log an error.
        """

        broker_positions_dict = {i.contract: i.position for i in self.ib.positions()}
        log.debug(f"broker_positions: {broker_positions_dict}")
        my_positions_dict = self.sm.strategy.total_positions()
        log.debug(f"my positions: {my_positions_dict}")
        # log.debug(f"Broker positions: {broker_positions_dict}")
        # log.debug(f"My positions: {my_positions_dict}")
        diff = {
            i: (
                (my_positions_dict.get(i) or 0.0)
                - (broker_positions_dict.get(i) or 0.0)
            )
            for i in set([*broker_positions_dict.keys(), *my_positions_dict.keys()])
        }
        log.debug(f"diff: {diff}")
        self.errors = {k: v for k, v in diff.items() if v != 0}
        log.debug(f"errors: {self.errors}")

        return self


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


class ReportEvents:
    def onStarted(self):
        """
        Attach reporting events to all open trades.
        """
        trades = self.trades()
        log.info(
            f"open trades on re-connect: {len(trades)} "
            f"{[t.contract.localSymbol for t in trades]}"
        )
        # attach reporting events
        for trade in trades:
            if trade.orderStatus.remaining != 0:
                if trade.order.orderType in ("STP", "TRAIL"):
                    self.attach_events(trade, "STOP-LOSS")
                else:
                    self.attach_events(trade, "TAKE-PROFIT")
