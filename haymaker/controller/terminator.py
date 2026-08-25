"""Explicit account reset orchestration for Controller startup actions."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import ib_insync as ibi

from haymaker.components.messages import StandardOrderRole

if TYPE_CHECKING:
    from .controller import Controller

log = logging.getLogger(__name__)


class Terminator:
    """Cancel working orders briefly, then flatten during explicit reset."""

    cancellation_timeout = 10.0
    cancellation_poll_interval = 0.1

    def __init__(self, controller: Controller) -> None:
        self.controller = controller
        self.in_progress_trades: list[ibi.Trade] = []

    async def run(self) -> bool:
        """Cancel and close the account, returning whether reset completed."""

        log.warning("Explicit account reset initiated.")
        open_trades = tuple(self.controller.ib.openTrades())
        for trade in open_trades:
            self.controller.cancel(trade)
        await self._wait_for_cancellations(open_trades)

        logical_contracts: set[int] = set()
        for source_key, state in self.controller.book.position_states().items():
            if state.quantity and state.contract is not None:
                logical_trade = self.controller.trade(
                    state.contract,
                    ibi.MarketOrder(
                        "BUY" if state.quantity < 0 else "SELL",
                        abs(state.quantity),
                    ),
                    role=StandardOrderRole.LIQUIDATION,
                    execution_model_name=state.execution_model_name,
                    source_key=source_key,
                    position_id=state.position_id,
                )
                if logical_trade is not None:
                    self.in_progress_trades.append(logical_trade)
                    logical_contracts.add(state.contract.conId)

        if not await self._wait_for_logical_closes():
            return False
        for position in self.controller.ib.positions():
            if position.position and position.contract.conId not in logical_contracts:
                residual_trade = self.controller.trade(
                    position.contract,
                    ibi.MarketOrder(
                        "BUY" if position.position < 0 else "SELL",
                        abs(position.position),
                        tif="DAY",
                        outsideRth=True,
                    ),
                    role=StandardOrderRole.LIQUIDATION,
                    execution_model_name=(
                        self.controller.book.active_order_model_for_contract(
                            position.contract
                        )
                        or "reset_liquidation"
                    ),
                )
                if residual_trade is not None:
                    self.in_progress_trades.append(residual_trade)
        return await self._wait_for_logical_closes()

    async def _wait_for_cancellations(self, trades: tuple[ibi.Trade, ...]) -> bool:
        """Give pre-reset orders bounded time to become terminal."""

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.cancellation_timeout
        while any(not trade.isDone() for trade in trades):
            remaining = deadline - loop.time()
            if remaining <= 0:
                pending = [
                    trade.order.orderId for trade in trades if not trade.isDone()
                ]
                log.critical(
                    "Order cancellations did not complete before reset "
                    "liquidation continued: %s",
                    pending,
                )
                return False
            await asyncio.sleep(min(self.cancellation_poll_interval, remaining))
        return True

    async def _wait_for_logical_closes(self) -> bool:
        """Wait briefly for reset orders without creating a shutdown framework."""

        for _ in range(10):
            if all(trade.isDone() for trade in self.in_progress_trades):
                return True
            await asyncio.sleep(1)
        if pending := [
            trade.order.orderId
            for trade in self.in_progress_trades
            if not trade.isDone()
        ]:
            log.critical("Reset trades did not complete: %s", pending)
            return False
        return True
