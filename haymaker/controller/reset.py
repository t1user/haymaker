"""Account reset orchestration for Controller startup actions."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import ib_insync as ibi

from haymaker.components.messages import StandardOrderRole

if TYPE_CHECKING:
    from .controller import Controller

log = logging.getLogger(__name__)


class Reset:
    """Reset the account: close all open positions and cancel pending orders.

    Wait briefly for cancellation confirmations, then submit liquidation orders
    even if some cancellations are still pending. Report success only when every
    required liquidation fills completely, all original orders have finished,
    and a fresh broker position request confirms that no positions remain.
    """

    cancellation_timeout = 10.0
    cancellation_poll_interval = 0.1

    def __init__(self, controller: Controller) -> None:
        self.controller = controller
        self.in_progress_trades: list[ibi.Trade] = []

    async def run(self) -> bool:
        """Execute the reset and report whether it completed."""

        log.warning("Explicit account reset initiated.")
        open_trades = tuple(self.controller.ib.openTrades())
        for trade in open_trades:
            self.controller.cancel(trade)
        await self._wait_for_cancellations(open_trades)

        submissions_accepted = True
        logical_contracts: set[int] = set()
        for source_key, state in self.controller.book.positions.source_states().items():
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
                else:
                    submissions_accepted = False
                    log.critical(
                        "Reset liquidation suppressed: "
                        "contract=%s source=%s quantity=%s",
                        state.contract,
                        source_key,
                        state.quantity,
                    )

        logical_filled = await self._wait_for_liquidations(self.in_progress_trades)
        positions = await self._request_positions("residual discovery")
        if positions is None:
            positions = tuple(self.controller.ib.positions())
        residual_trades: list[ibi.Trade] = []
        for position in positions:
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
                        self.controller.book.orders.owner_for_contract(
                            position.contract
                        )
                        or "reset_liquidation"
                    ),
                )
                if residual_trade is not None:
                    self.in_progress_trades.append(residual_trade)
                    residual_trades.append(residual_trade)
                else:
                    submissions_accepted = False
                    log.critical(
                        "Reset residual liquidation suppressed: "
                        "contract=%s quantity=%s",
                        position.contract,
                        position.position,
                    )
        residual_filled = await self._wait_for_liquidations(residual_trades)
        final_positions = await self._request_positions("final flat verification")
        non_flat = [position for position in final_positions or () if position.position]
        if non_flat:
            log.critical("Reset broker positions remain non-flat: %s", non_flat)
        unresolved = [trade for trade in open_trades if not trade.isDone()]
        for trade in unresolved:
            log.critical(
                "Reset pre-existing order remains unresolved: "
                "orderId=%s status=%s filled=%s remaining=%s",
                trade.order.orderId,
                trade.orderStatus.status,
                trade.orderStatus.filled,
                trade.orderStatus.remaining,
            )
        return (
            submissions_accepted
            and logical_filled
            and residual_filled
            and not unresolved
            and final_positions is not None
            and not non_flat
        )

    async def _request_positions(self, phase: str) -> tuple[ibi.Position, ...] | None:
        """Request broker authority; an unavailable snapshot cannot prove flatness."""

        timeout = self.controller.broker_request_timeout
        try:
            return tuple(
                await asyncio.wait_for(self.controller.ib.reqPositionsAsync(), timeout)
            )
        except asyncio.TimeoutError:
            log.critical(
                "Reset position request timed out during %s after %ss", phase, timeout
            )
        except Exception as exc:
            log.critical("Reset position request failed during %s: %r", phase, exc)
        return None

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

    async def _wait_for_liquidations(self, trades: list[ibi.Trade]) -> bool:
        """Bound settlement time, then require full fills for every liquidation."""

        for _ in range(10):
            if all(
                trade.isDone() or trade.orderStatus.status == ibi.OrderStatus.Inactive
                for trade in trades
            ):
                break
            await asyncio.sleep(1)
        incomplete = [
            trade
            for trade in trades
            if not (
                trade.orderStatus.status == ibi.OrderStatus.Filled
                and trade.orderStatus.filled == trade.order.totalQuantity
                and trade.orderStatus.remaining == 0
            )
        ]
        for trade in incomplete:
            log.critical(
                "Reset liquidation not fully filled: "
                "orderId=%s contract=%s status=%s filled=%s remaining=%s total=%s",
                trade.order.orderId,
                trade.contract,
                trade.orderStatus.status,
                trade.orderStatus.filled,
                trade.orderStatus.remaining,
                trade.order.totalQuantity,
            )
        return not incomplete
