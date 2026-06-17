"""Single-pass controller sync checks for broker and local state.

Sync starts by validating that ``ib.positions()`` agrees with
``await ib.reqPositionsAsync()``.  If broker state verification fails, the pass
returns ``False`` without attempting recovery or correction actions;
:meth:`Controller.sync` owns checking the connection, retrying, and deciding
whether repeated failures should disable trading.

After broker validation, each step reads current state directly from
``controller.ib`` or ``controller.sm`` instead of using stored broker/local
snapshots.  The ordered flow is:

1. Verify that requested broker order/execution data agrees with
   ``ib_insync``'s local order and fill registers.
2. Relink broker ``ibi.Trade`` objects to local order records and back-report
   fills for orders that completed while the process was disconnected.
3. Compare local aggregate strategy positions with broker positions and
   correct local position records when the existing recovery rules allow it.
4. Skip correction trades when unresolved unknown broker orders remain active.
5. Delegate bracket-record and broker stop-loss protection handling to
   :mod:`haymaker.controller.sync_brackets`.

The coordinator does not disable trading and does not retry.  Any recovery
action returns ``False`` so :meth:`Controller.sync` can start a fresh pass from
current broker/local state.  Non-retryable unsafe state raises
``SyncBrokenStateError``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import ib_insync as ibi

from haymaker import misc
from haymaker.state_machine import OrderInfo

from .sync_brackets import BracketSyncAction, BracketSyncError
from .sync_routines import OrderSync, PositionSync

if TYPE_CHECKING:
    from .controller import Controller

log = logging.getLogger(__name__)


class SyncBrokenStateError(Exception):
    """Raised when sync detects unsafe state that must stop trading."""


class PositionsOutOfSync(Exception):
    """Raised when local positions cannot be reconciled to broker positions."""


class SyncCoordinator:
    """Run one broker/local sync pass and report the outcome.

    ``run()`` returns ``True`` only when the current pass completed cleanly.
    It returns ``False`` after any recovery action so the caller can retry from
    fresh broker/local reads.  Retryable broker connection and broker-state
    verification failures also return ``False``.  Terminal safety failures
    raise ``SyncBrokenStateError``; :class:`Controller` owns the decision to
    disable trading.
    """

    def __init__(self, controller: Controller) -> None:
        """Initialize the coordinator for one controller sync run."""
        self.controller = controller
        self._faulty_trades: list[OrderInfo] = []

    async def run(self) -> bool:
        """Run one sync pass against current broker and local state.

        Returns:
            ``True`` when sync completed without recovery actions or terminal
            safety failures.  ``False`` means the controller should retry the
            sync from fresh broker/local reads.

        Raises:
            SyncBrokenStateError: Raised for unsafe state that should stop
                trading immediately.
        """

        if not await verify_broker_position_source(
            self.controller.ib,
            self.controller.broker_request_timeout,
        ):
            return False

        if not await verify_broker_order_source(
            self.controller.ib,
            self.controller.broker_request_timeout,
        ):
            return False

        order_sync = OrderSync(self.controller.ib, self.controller.sm)

        self.controller.release_hold()
        if order_sync.done:
            self.handle_done_trades(order_sync.done)
            await asyncio.sleep(0)
        if order_sync.errors:
            self.handle_error_trades(order_sync.errors)
            await asyncio.sleep(0)
        if order_sync.unknown:
            if self.handle_unknown_trades(order_sync.unknown):
                return False
            return True
        if order_sync.done or order_sync.errors:
            return False

        position_sync = PositionSync(self.controller.ib, self.controller.sm)
        if position_sync.errors:
            try:
                self.handle_error_positions(position_sync.errors)
            except PositionsOutOfSync as exc:
                raise SyncBrokenStateError(
                    "local state does not match broker state"
                ) from exc
            return False

        try:
            BracketSyncAction.from_policy(
                self.controller.missing_brackets,
                self.controller,
            )
        except BracketSyncError as exc:
            raise SyncBrokenStateError("bracket sync failed") from exc

        return True

    def handle_unknown_trades(self, trades: list[ibi.Trade]) -> bool:
        """Cancel unknown broker trades when configured and report if broker changed."""
        log.critical(f"Unknown broker orders during sync: {trades}.")
        if not self.controller.cancel_unknown_trades:
            log.critical(
                "Unknown broker orders left active because "
                "cancel_unknown_trades is False."
            )
            return False

        for trade in trades:
            log.debug(f"Cancelling unknown broker order: {trade.order.orderId}")
            self.controller.cancel(trade)
        return True

    def handle_done_trades(self, trades: list[ibi.Trade]) -> None:
        """
        Events artificially emitted here will trigger registering
        position and saving trade to blotter.
        """
        for trade in trades:
            log.debug(
                f"Back-reporting trade: {trade.contract.symbol} "
                f"{trade.order.action} {misc.trade_fill_price(trade)} "
                f"order id: {trade.order.orderId} {trade.order.permId} "
                f"active?: {trade.isActive()}"
            )
            self.controller.ib.orderStatusEvent.emit(trade)
            for fill in trade.fills:
                self.controller.ib.execDetailsEvent.emit(trade, fill)
            if trade.orderStatus.status == "Filled":
                self.controller.ib.commissionReportEvent.emit(
                    trade, trade.fills[-1], trade.fills[-1].commissionReport
                )

    def handle_error_trades(self, trades: list[ibi.Trade]) -> None:
        """
        Local trades unknown to broker. Local state is corrected, but
        record of trades is kept for further investigation.
        """
        for trade in trades:
            log.error(
                f"Will delete record for trade that IB doesn't know about: "
                f"{trade.order.orderId}"
            )
            self._faulty_trades.append(self.controller.sm._orders[trade.order.orderId])
            self.controller.sm.prune_order(trade.order.orderId)

    def handle_error_positions(self, errors: dict[ibi.Contract, float]) -> None:
        log.error("Will attempt to fix position records")
        for contract, diff in errors.items():
            strategies = self.controller.sm.for_contract.get(contract)
            log.debug(f"Strategies for contract {contract.localSymbol}: {strategies}")
            if strategies and len(strategies) == 1:
                self.controller.sm.strategy[strategies[0]].position -= diff
                log.error(
                    f"Corrected position records for strategy "
                    f"{strategies[0]} by {-diff}"
                )
                self.controller.sm.save_strategies()

            elif (
                strategies
                and self.controller.trader.position_for_contract(contract) == 0
            ):
                for strategy in strategies:
                    self.controller.sm.strategy[strategy].position = 0
                self.controller.sm.save_strategies()
                log.error(
                    f"Position records zeroed for {strategies} "
                    f"to reflect zero position for {contract.symbol}."
                )
            elif strategies:
                strategy_faults = [
                    order_info.strategy for order_info in self._faulty_trades
                ]
                for strategy in strategies:
                    if strategy in strategy_faults:
                        self.controller.sm.strategy[strategy].position = 0
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
                raise PositionsOutOfSync
            self._faulty_trades.clear()


async def verify_broker_position_source(ib: ibi.IB, timeout: float) -> bool:
    """
    Return True when synchronous and requested broker positions agree.

    Practically, if there's an unreported ib_gateway issue,
    ``reqPositionAsync`` typically freezes, which is a sign that we
    cannot rely on information received from broker.
    """
    positions = tuple(ib.positions())
    try:
        requested_positions = tuple(
            await asyncio.wait_for(ib.reqPositionsAsync(), timeout)
        )
    except asyncio.TimeoutError:
        log.warning(f"broker position request timed out after {timeout}s")
        return False
    except Exception as exc:
        log.warning(f"broker position request failed: {exc!r}")
        return False

    positions_dict = {
        position.contract.localSymbol: position.position for position in positions
    }
    requested_positions_dict = {
        position.contract.localSymbol: position.position
        for position in requested_positions
        if position.position
    }
    if positions_dict != requested_positions_dict:
        log.debug(
            f"broker position sources disagree: "
            f"positions={positions_dict} req_positions={requested_positions_dict}"
        )
        return False
    else:
        log.debug(f"broker positions: {positions_dict}")
    return True


async def verify_broker_order_source(ib: ibi.IB, timeout: float) -> bool:
    """Return True when requested broker order data agrees with IB registers."""

    try:
        # reqCompletedOrdersAsync and reqExecutionsAsync should
        # trigger a refresh
        completed_orders, requested_fills = await asyncio.gather(
            asyncio.wait_for(ib.reqCompletedOrdersAsync(False), timeout),
            asyncio.wait_for(ib.reqExecutionsAsync(), timeout),
        )
    except asyncio.TimeoutError:
        log.warning(f"broker order request timed out after {timeout}s")
        return False
    except Exception as exc:
        log.warning(f"broker order request failed: {exc!r}")
        return False

    open_trades = tuple(ib.openTrades())
    known_trades = tuple(ib.trades())
    known_fills = tuple(ib.fills())

    if missing_exec_ids := _missing_requested_fills(requested_fills, known_fills):
        log.warning(
            "broker execution source disagrees with fill register: "
            f"missing execIds={sorted(missing_exec_ids)}"
        )
        return False

    if open_completed := _open_completed_order_keys(open_trades, completed_orders):
        log.warning(
            "broker completed-order source disagrees with open trades: "
            f"completed orders still open={sorted(open_completed)}"
        )
        return False

    if open_filled := _open_filled_trade_keys(open_trades, requested_fills):
        log.warning(
            "broker execution source disagrees with open trades: "
            f"fully executed orders still open={sorted(open_filled)}"
        )
        return False

    if detached_exec_ids := _detached_requested_fills(requested_fills, known_trades):
        log.warning(
            "broker execution source disagrees with trade register: "
            f"detached execIds={sorted(detached_exec_ids)}"
        )
        return False

    return True


def _order_keys(order: ibi.Order) -> set[int]:
    """Return stable non-zero order identifiers from an IB order."""
    return {key for key in (order.orderId, order.permId) if key}


def _trade_keys(trade: ibi.Trade) -> set[int]:
    """Return stable non-zero order identifiers from an IB trade."""
    return _order_keys(trade.order)


def _execution_keys(execution: ibi.Execution) -> set[int]:
    """Return stable non-zero order identifiers from an IB execution."""
    return {key for key in (execution.orderId, execution.permId) if key}


def _missing_requested_fills(
    requested_fills: list[ibi.Fill],
    known_fills: tuple[ibi.Fill, ...],
) -> set[str]:
    known_exec_ids = {fill.execution.execId for fill in known_fills}
    return {
        fill.execution.execId
        for fill in requested_fills
        if fill.execution.execId not in known_exec_ids
    }


def _open_completed_order_keys(
    open_trades: tuple[ibi.Trade, ...],
    completed_orders: list[ibi.Trade],
) -> set[int]:
    open_keys = set().union(*(_trade_keys(trade) for trade in open_trades), set())
    completed_keys = set().union(
        *(_trade_keys(trade) for trade in completed_orders),
        set(),
    )
    return open_keys & completed_keys


def _open_filled_trade_keys(
    open_trades: tuple[ibi.Trade, ...],
    requested_fills: list[ibi.Fill],
) -> set[int]:
    filled_keys: set[int] = set()
    for trade in open_trades:
        total_quantity = trade.order.totalQuantity
        if not total_quantity:
            continue
        filled = sum(
            fill.execution.shares
            for fill in requested_fills
            if _execution_keys(fill.execution) & _trade_keys(trade)
        )
        if filled >= total_quantity:
            filled_keys.update(_trade_keys(trade))
    return filled_keys


def _detached_requested_fills(
    requested_fills: list[ibi.Fill],
    known_trades: tuple[ibi.Trade, ...],
) -> set[str]:
    trades_by_key = {key: trade for trade in known_trades for key in _trade_keys(trade)}
    detached: set[str] = set()
    for fill in requested_fills:
        matching_trades = [
            trades_by_key[key]
            for key in _execution_keys(fill.execution)
            if key in trades_by_key
        ]
        if matching_trades and not any(
            known_fill.execution.execId == fill.execution.execId
            for trade in matching_trades
            for known_fill in trade.fills
        ):
            detached.add(fill.execution.execId)
    return detached
