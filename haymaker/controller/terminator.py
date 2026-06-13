from __future__ import annotations

import ib_insync as ibi
import logging
import asyncio
from functools import partial
from typing import TYPE_CHECKING

from haymaker import misc
from haymaker.state_machine import Strategy

if TYPE_CHECKING:
    from . import Controller


log = logging.getLogger(__name__)


class Terminator:
    """
    Class organizing the process of cancelling all resting orders and
    closing all positions.

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
            await self.controller.close_positions()
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
        _trade: ibi.Trade,
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
            strategy.strategy, contract, ibi.MarketOrder(action, amount), "RESET", {}
        )
        assert new_trade
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
                    existing_amount = 0
                    stop_amount = 0
                if existing_amount and (
                    misc.sign(stop_amount) == -misc.sign(existing_amount)
                ):
                    trade_amount = min(
                        abs(existing_amount), abs(stop_amount)
                    ) * misc.sign(stop_amount)

                    log.debug(f"Will cancel {trade_amount} for {strategy.strategy}")
                    assert trade
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
