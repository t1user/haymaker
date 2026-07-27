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
    """Cancel working orders and close broker positions during explicit reset."""

    def __init__(self, controller: Controller) -> None:
        self.controller = controller
        self.in_progress_trades: list[ibi.Trade] = []

    async def run(self) -> None:
        """Cancel attributed orders, close logical sources, then broker residue."""

        log.warning("Explicit account reset initiated.")
        for trade in tuple(self.controller.ib.openTrades()):
            self.controller.cancel(trade)
        await asyncio.sleep(0)

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

        await self._wait_for_logical_closes()
        logical_contracts = {
            state.contract.conId
            for state in self.controller.book.position_states().values()
            if state.contract is not None
        }
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
                        self.controller.book.affinity_for_contract(
                            position.contract
                        )
                        or "reset_liquidation"
                    ),
                )
                if residual_trade is not None:
                    self.in_progress_trades.append(residual_trade)
        await self._wait_for_logical_closes()

    async def _wait_for_logical_closes(self) -> None:
        """Wait briefly for reset orders without creating a shutdown framework."""

        for _ in range(10):
            if all(trade.isDone() for trade in self.in_progress_trades):
                return
            await asyncio.sleep(1)
        if pending := [
            trade.order.orderId
            for trade in self.in_progress_trades
            if not trade.isDone()
        ]:
            log.critical("Reset trades did not complete: %s", pending)
