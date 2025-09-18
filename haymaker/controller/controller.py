from __future__ import annotations

import asyncio
import itertools
import logging
from collections import abc
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from functools import partial
from typing import TYPE_CHECKING, Self

import eventkit as ev  # type: ignore
import ib_insync as ibi

from haymaker import misc
from haymaker.base import Atom
from haymaker.state_machine import OrderInfo, Strategy
from haymaker.trader import FakeTrader, Trader

from .future_roller import FutureRoller
from .sync_routines import (
    ErrorHandlers,
    OrderReconciliationSync,
    OrderSyncStrategy,
    PositionSyncStrategy,
    Terminator,
)

if TYPE_CHECKING:
    from haymaker.blotter import Blotter

log = logging.getLogger(__name__)


class ControllerError(Exception):
    pass


@dataclass
class Controller(Atom):
    """
    Intermediary between execution models (which are off ramps for
    strategies), :class:`Trader` and :class:`StateMachine`.  Use information
    provided by :class:`StateMachine` to make sure that positions held in
    the market reflect what is requested by strategies.

    """

    trader: Trader
    blotter: Blotter | None = None
    cold_start: bool = True
    reset: bool = False
    zero: bool = False
    nuke: bool = False
    cancel_stray_orders: bool = True
    log_order_events: bool = False
    sync_frequency: int = 0
    health_check_frequency: int = 0
    execution_verification_delay: int = 0
    health_check_observables: list[list[Callable[[], bool]]] = field(
        default_factory=list
    )
    _hold: bool = field(default=True, repr=False)
    _sync_timer: ev.Timer | None = None
    _health_check_timer: ev.Timer | None = None
    _order_loggers: OrderLoggers | None = None
    _health_check_functions: list[Callable[[], bool]] = field(
        default_factory=list, repr=False
    )
    _health_check_triggers: list[str] = field(default_factory=list, repr=False)

    @classmethod
    def from_config(
        cls,
        trader: Trader,
        blotter: Blotter | None = None,
        top_config: abc.MutableMapping | None = None,
        health_check_observables: list[list[Callable[[], bool]]] | None = None,
    ) -> Self:
        """
        Extract proper attributes from configuration file and perform
        required initializations based on it.
        """
        log.debug("Initializing Controller with config.")
        if top_config is None:
            top_config = {}

        if health_check_observables is None:
            health_check_observables = []

        config = top_config.get("controller") or {}

        field_names = [field.name for field in fields(cls)]
        for param in config:
            if param not in field_names:
                raise ControllerError(
                    f"Wrong parameter: {param} in 'controller' section of config."
                )

        top_kwargs = {
            i: top_config.get(i, False)
            for i in [
                "cold_start",
                "reset",
                "zero",
                "nuke",
            ]
        }

        return cls(
            trader=trader,
            blotter=blotter,
            health_check_observables=health_check_observables,
            **config,
            **top_kwargs,
        )

    def __post_init__(self) -> None:
        super().__init__()

        # these are essential (non-optional) events
        self.ib.execDetailsEvent.connect(self.onExecDetailsEvent, self._log_event_error)
        self.ib.newOrderEvent.connect(self.onNewOrderEvent, self._log_event_error)
        self.ib.orderStatusEvent.connect(self.onOrderStatusEvent, self._log_event_error)

        # this is for logging
        self.ib.orderStatusEvent.connect(self.log_order_status, self._log_event_error)
        self.ib.errorEvent.connect(self.log_err, self._log_event_error)

        self.set_hold()

        if self.blotter:
            self.ib.commissionReportEvent.connect(
                self.onCommissionReport, self._log_event_error
            )

        if self.sync_frequency:
            self._sync_timer = ev.Timer(self.sync_frequency)
            self._sync_timer.connect(self.sync, error=self._log_event_error)

        if self.health_check_frequency:
            self._health_check_timer = ev.Timer(self.health_check_frequency)
            self._health_check_timer.connect(
                self.run_health_check, error=self._log_event_error
            )

        if self.log_order_events:
            self._order_loggers = OrderLoggers(self.ib)

        if missing_contracts := self.verify_have_contracts_for_positions():
            log.critical(
                f"No qualified contracts for open position: {missing_contracts}"
            )

        self.sync_handlers = ErrorHandlers(self.ib, self.sm, self)
        self.no_future_roll_strategies: list[str] = []
        log.debug(f"Controller initiated: {self}")
        log.debug(f"{self.contracts=}")

    def set_health_check(self, func: Callable[[], bool]) -> None:
        self._health_check_functions.append(func)

    def run_health_check(self, *args) -> None:
        for func in itertools.chain(
            itertools.chain(*self.health_check_observables),
            self._health_check_functions,
        ):
            if not func() and func.__name__ not in self._health_check_triggers:
                log.critical(f"Health check failure for checker: {func.__name__}")
                # prevent repeating same error multiple times
                self._health_check_triggers.append(func.__name__)

    def verify_have_contracts_for_positions(self) -> list[ibi.Contract]:
        return [
            p.contract for p in self.ib.positions() if p.contract not in self.contracts
        ]

    def set_hold(self) -> None:
        self._hold = True
        log.debug("hold set")

    def release_hold(self) -> None:
        if self._hold:
            self._hold = False
            log.debug("hold released")

    def set_no_future_roll_strategies(self, strategies: list[str]) -> None:
        self.no_future_roll_strategies.extend(strategies)

    async def run(self) -> None:
        """
        Main entry point into the programme.  Ensure records up to
        date and any remaining initialization complete.
        """
        log.debug("Running controller...")
        self.set_hold()
        if self.nuke:
            self.run_nuke()

        if self.cold_start:
            log.debug("Starting cold... (state NOT read from db)")
        else:
            try:
                log.debug("Reading from store...")
                await self.sm.read_from_store()
                self.cold_start = True
            except Exception as e:
                log.exception(e)

        log.debug("Will sync...")
        await self.sync()
        log.debug("Sync completed.")

        if self.zero:
            log.debug("Zeroing all records...")
            self.clear_records()
            self.zero = False

        if self.reset:
            await self.execute_stops_and_close_positions()
            self.reset = False
            # zero-out all records
            self.sm.clear_strategies()

        log.debug("Controller run sequence completed successfully.")
        # now Streamers will run

    def roll_futures(self, *args):
        """
        This method is scheduled to run once a day in :class:`.app.App`
        """
        log.info(f"Running roll on controller object: {id(self)}")
        roller = FutureRoller(self)
        roller.roll()

    async def sync(self, *args) -> None:
        if self.ib.isConnected():
            log.debug("--- Sync ---")
            positions = {
                p.contract.localSymbol: p.position for p in self.ib.positions()
            }
            req_positions = {
                p.contract.localSymbol: p.position
                for p in await self.ib.reqPositionsAsync()
                if p.position
            }
            if positions != req_positions:
                log.warning(f"{positions=} {req_positions=}")
            else:
                log.debug(f"{positions=}")

            orders_report = OrderSyncStrategy.run(self.ib, self.sm)
            if not orders_report.is_ok:
                log.debug("Order sync error, will attempt to restart.")
                self.ib.disconnect()
                return
            # IB events will be handled so that matched trades can be sent to blotter
            self.release_hold()
            await self.sync_handlers.handle_orders(orders_report)

            error_position_report = PositionSyncStrategy.run(self.ib, self.sm)
            await self.sync_handlers.handle_positions(error_position_report)
            OrderReconciliationSync.run(self)

            log.debug("--- Sync completed ---")
        else:
            log.debug("No connection. Abandoning sync.")

    def onStart(self, data, *args) -> None:
        # prevent superclass from setting attributes here
        pass

    async def onData(self, data, *args) -> None:
        """
        After obtaining transaction details from execution model,
        verify if the intended effect is the same as achieved effect.
        """
        try:
            strategy = data["strategy"]
            amount = data["amount"]
            target_position = data["target_position"]
            await asyncio.sleep(self.execution_verification_delay)
            await self.verify_transaction_integrity(strategy, amount, target_position)
        except KeyError:
            log.exception(
                "Unable to verify transaction integrity", extra={"data": data}
            )
        contract = data.get("contract")
        self.verify_position_with_broker(contract)

    def trade(
        self,
        strategy_str: str,
        contract: ibi.Contract,
        order: ibi.Order,
        action: str,
        strategy: Strategy,
    ) -> ibi.Trade | None:
        # this will return False if order for the strategy has been repeatedly rejected
        if self.sm.verify_for_rejections(strategy_str):
            trade = self.trader.trade(contract, order)
            self.register_order(strategy_str, action, trade, strategy)
            trade.filledEvent += partial(
                self.log_trade, reason=action, strategy=strategy_str
            )
            return trade
        else:
            return None

    def register_order(
        self, strategy_str: str, action: str, trade: ibi.Trade, strategy: Strategy
    ) -> OrderInfo:
        """
        Register order, register lock, verify that position has been registered.

        Register order that has just been posted to the broker.  If
        it's an order openning a new position register a lock on this
        strategy (the lock may or may not be used by strategy itself,
        it doesn't matter here, locks are registered for all
        positions).  Verify that position has been registered.

        This method is called by :class:`Controller`.
        """
        params = strategy["params"].get(action.lower(), {})
        order_info = OrderInfo(strategy_str, action, trade, params)
        oi = self.sm.save_order(order_info)

        if action.upper() == "OPEN":
            trade.filledEvent += partial(self.register_lock, strategy)
        elif action.upper() == "CLOSE":
            trade.filledEvent += partial(self.remove_lock, strategy)

        log.debug(
            f"{trade.order.orderType} orderId: {trade.order.orderId} "
            f"permId: {trade.order.permId} registered for: "
            f"{trade.contract.localSymbol or trade.contract.symbol} "
        )

        return oi

    def register_lock(self, strategy: Strategy, trade: ibi.Trade) -> None:
        strategy.lock = 1 if trade.order.action == "BUY" else -1

    def remove_lock(self, strategy: Strategy, trade: ibi.Trade) -> None:
        strategy.lock = 0

    def cancel(self, trade: ibi.Trade) -> ibi.Trade | None:
        return self.trader.cancel(trade)

    async def onNewOrderEvent(self, trade: ibi.Trade) -> None:
        # keep this method async; it ensures correct sequence of actions
        """
        Check if the system knows about the order that was just posted
        to the broker.

        This is an event handler (callback).  Connected (subscribed)
        to :meth:`ibi.IB.newOrderEvent` in :meth:`__init__`
        """

        log.debug(f"New order event: {trade.order.orderId, trade.order.permId} ")
        if not (trade.order.orderId < 0 or self.sm.order.get(trade.order.orderId)):
            log.critical(
                f"Unknown trade in the system {trade.order} {trade.contract.symbol}"
            )

    def onOrderStatusEvent(self, trade: ibi.Trade) -> None:

        if self._hold:
            return

        # this will create new order record if it doesn't already exist
        self.sm.save_order_status(trade)

    def register_position(
        self, strategy_str: str, strategy: Strategy, trade: ibi.Trade, fill: ibi.Fill
    ) -> None:
        try:
            if isinstance(trade.contract, ibi.Bag):
                log.debug(
                    f"Combo trade registered for: {trade.contract.symbol}, "
                    f"position kept unchanged at: {strategy.position}"
                )
            elif fill.execution.side == "BOT":
                strategy.position += fill.execution.shares
                log.debug(
                    f"Registered position - orderId: {trade.order.orderId} permId: "
                    f"{trade.order.permId} BUY {trade.order.orderType} "
                    f"for {strategy_str} --> position: {strategy.position} "
                )
            elif fill.execution.side == "SLD":
                strategy.position -= fill.execution.shares
                log.debug(
                    f"Registered position - orderId {trade.order.orderId} permId: "
                    f"{trade.order.permId} SELL {trade.order.orderType} "
                    f"for {strategy_str} --> position: {strategy.position}"
                )
            else:
                log.critical(
                    f"Abiguous fill: {fill} for order: {trade.order} for "
                    f"{trade.contract.localSymbol} strategy: {strategy}"
                )
        except Exception as e:
            log.exception(e)

    async def onExecDetailsEvent(self, trade: ibi.Trade, fill: ibi.Fill) -> None:
        """
        Register position.
        """
        strategy = (
            # check if it's manual trade (id<0)
            self.assign_manual_trade(trade)
            # find trade record
            or self.sm.strategy_for_trade(trade)
            # failing above try to assign
            or self.assign_unknown_trade(trade)
        )
        self.register_position(strategy.strategy, strategy, trade, fill)

    def onCommissionReport(
        self, trade: ibi.Trade, fill: ibi.Fill, report: ibi.CommissionReport
    ) -> None:
        """
        Writing commission on :class:`ibi.Trade` is the final stage of
        order execution.  At this point that trade object is ready for
        storing in blotter.
        """

        # silence emission of all orders from session on startup
        if self._hold:
            return

        order_info = self.sm.save_order_status(trade)

        if order_info:
            try:
                strategy, action, _, params, _ = order_info
                position_id = params.get("position_id") or self.sm.strategy[
                    strategy
                ].get("position_id")

                kwargs = {
                    "strategy": strategy,
                    "action": action,
                    "position_id": position_id,
                    "params": ibi.util.tree(params),
                }
                # optionally set by execution model
                if arrival_price := params.get("arrival_price"):
                    kwargs.update(
                        {
                            "price_time": arrival_price["time"],
                            "bid": arrival_price["bid"],
                            "ask": arrival_price["ask"],
                        }
                    )
            except Exception as e:
                log.error(f"Error while trying to create blotter entry: {e}")

        elif trade.order.totalQuantity == 0:
            log.warning(f"empty CommissionReportEvent emit for trade: {trade}")
            return
        else:
            log.error(
                f"Commission report for unknown trade: {trade.order.orderId} "
                f"{trade.contract.localSymbol}"
            )
            return

        assert self.blotter is not None
        try:
            self.blotter.log_commission(trade, fill, report, **kwargs)
        except Exception as e:
            log.error(f"Error while writing to blotter: {e}")

    async def verify_transaction_integrity(
        self,
        strategy: str,
        amount: float,  # amount in transaction being verified
        target_position: float,  # target direction
    ) -> None:
        """
        Called by :meth:`onData`, which is passing data that was the
        basis for transaction.  The purpose of this method is to
        confirm that the resulting transaction achieved required
        objectives.  Things vefied are: 1.  actual resulting position
        in broker records 2.  records in state_machine

        Any errors are logged but not corrected (may change in
        future).
        """

        data = self.sm.strategy.get(strategy)
        target = target_position * amount

        order_infos = [
            info
            for info in self.sm.orders_for_strategy(strategy)
            if info.action not in ("STOP-LOSS", "TAKE-PROFIT")
        ]
        if order_infos:  # exists an order which is not a sl or tp
            # if order(s) still in execution don't check if position achieved yet
            while any([info.active for info in order_infos]):
                log.debug(
                    f"{strategy} taking long to achive target position of {target}"
                )
                await asyncio.sleep(self.execution_verification_delay)

        log_str = f"target: {target}, position: {data.position}"
        if data:
            records_ok = data.position == (target)
            if records_ok:
                log.debug(f"{strategy} position OK? -> {records_ok} <- " f"{log_str}")
            else:

                log.error(
                    f"Failed to achieve target position for {strategy} - " f"{log_str}"
                )
        else:
            log.critical(f"Attempt to trade for unknown strategy: {strategy}")

    def verify_position_with_broker(self, contract: ibi.Contract) -> None:
        # called by onData after every transaction
        sm_position = self.sm.position.get(contract, 0.0)
        ib_position = self.trader.position_for_contract(contract)
        position_ok = sm_position == ib_position
        log_str_position = f"{sm_position=}, {ib_position=}"
        if position_ok:
            log.debug(
                f"{contract.symbol} records vs broker OK? -> {position_ok} <- "
                f"{log_str_position}"
            )
        else:
            log.error(f"Wrong position for {contract} - {log_str_position}")

    def _assign_trade(self, trade: ibi.Trade) -> Strategy | None:

        # assumed unknown trade is to close a position
        active_strategies_list = [
            s
            for s in self.sm.for_contract[trade.contract]
            if self.sm.strategy[s].active
        ]
        log.debug(
            f"Attemp to assign unknown trade to a strategy: "
            f"{active_strategies_list}"
        )

        if len(active_strategies_list) == 1:
            strategy_str = active_strategies_list[0]
            strategy = self.sm.strategy[strategy_str]

        # if more than 1 active, unknown trade is for the one without
        # resting orders (resting orders are most likely stop-losses or take-profit)
        elif candidate_strategies := [
            s for s in active_strategies_list if not self.sm.orders_for_strategy(s)
        ]:
            log.debug(
                f"Active strategies without resting orders: " f"{candidate_strategies}"
            )
            # if more than one just pick first one
            # there's no way to determine which strategy was meant
            strategy_str = candidate_strategies[0]
        elif active_strategies_list:
            # failing everything else, pick one on which there was most recent operation
            # (probably we're trying to correct a recent error)
            strategy_str = sorted(
                active_strategies_list,
                key=lambda x: self.sm.strategy[x].get("timestamp"),
            )[-1]
        else:
            strategy_str = None

        if strategy_str:
            strategy = self.sm.strategy[strategy_str]
            self.cancel_orders_for_strategy(strategy_str)
        else:
            strategy = None

        return strategy

    def cancel_orders_for_strategy(self, strategy: str) -> None:
        order_infos = self.sm.orders_for_strategy(strategy)
        for oi in order_infos:
            cancelled_trade = self.cancel(oi.trade)
            if cancelled_trade:
                log.debug(
                    f"Cancelled trade: {cancelled_trade.order.orderId} for "
                    f"{cancelled_trade.contract.localSymbol}"
                )

    def _make_strategy(self, trade: ibi.Trade, description: str) -> Strategy:
        """
        Last resort when strategy cannot be matched.  Creating new
        made up strategy to make sure that position change resulting
        from trade will be somehow accounted for.  It should only be
        used if there are no position for the contract, otherwise
        unknown transactions should close existing strategies, rather
        than invent new ones.
        """
        strategy_str = f"{description}_{trade.contract.symbol}"
        strategy = self.sm.strategy[strategy_str]  # this creates new Strategy
        strategy["active_contract"] = trade.contract
        return strategy

    def assign_manual_trade(self, trade: ibi.Trade) -> Strategy | None:

        if trade.order.orderId >= 0:
            return None

        strategy = self._assign_trade(trade) or self._make_strategy(
            trade, "manual_strategy"
        )
        log.debug(f"Manual trade assigned to strategy: {strategy.strategy}.")

        # this will save strategy on OrderInfo
        self.sm.update_strategy_on_order(trade.order.orderId, strategy.strategy)

        return strategy

    def assign_unknown_trade(self, trade: ibi.Trade) -> Strategy:
        strategy = self._assign_trade(trade) or self._make_strategy(trade, "unknown")
        log.critical(f"Unknow trade: {trade}")

        # this will save strategy on OrderInfo
        self.sm.update_strategy_on_order(trade.order.orderId, strategy.strategy)

        return strategy

    def log_order_status(self, trade: ibi.Trade) -> None:
        # Connected to ib.OrderStatusEvent

        if self._hold:
            return

        if trade.order.orderId < 0:
            log.warning(
                f"Manual trade: {trade.order} status update: {trade.orderStatus}"
            )

        elif trade.isDone():
            log.debug(
                f"{trade.contract.symbol}: order {trade.order.orderId} "
                f"{trade.order.orderType} done {trade.orderStatus.status}. "
            )
        else:
            log.info(
                f"{trade.contract.symbol}: OrderStatus ->{trade.orderStatus.status}<-"
                f" for order: {trade.order.orderId} {trade.order.permId}, "
            )

    @staticmethod
    def log_trade(trade: ibi.Trade, reason: str = "", strategy: str = "") -> None:
        log.info(
            f"{reason} trade filled: {trade.contract.localSymbol} "
            f"{trade.order.action} {trade.filled()}"
            f"@{misc.trade_fill_price(trade)} --> {strategy} "
            f"orderId: {trade.order.orderId}, permId: {trade.order.permId} "
        )

    def log_err(
        self, reqId: int, errorCode: int, errorString: str, contract: ibi.Contract
    ) -> None:
        # Connected to ib.errorEvent

        if errorCode < 400:
            # reqId is most likely orderId
            # order rejected is errorCode = 201
            # order cancelled is errorCode = 202
            # 421: Error validating request.-'bN' : cause - Missing order exchange
            order_info = self.sm.order.get(reqId)
            if order_info:
                (strategy, action, trade, *_) = order_info
                order = trade.order
            else:
                strategy, action, trade, order = "", "", "", ""

            if errorCode == 202 and ("YOUR ORDER IS NOT ACCEPTED" not in errorString):
                log.info(
                    f"{errorString} code={errorCode} {contract=} "
                    f"{strategy} | {action} | {order}"
                )
            elif errorCode == 201:
                log.critical(
                    f"ORDER REJECTED: {errorString} {errorCode=} {contract=}, "
                    f"{strategy} | {action} | {order}"
                )
                self.sm.register_rejected_order(strategy.strategy)
            elif errorCode in (321, 322, 323):
                log.info(f"{errorString} {errorCode=}")

            else:
                log.error(
                    f"Error {errorCode}: {errorString} {contract}, "
                    f"{strategy} | {action} | {order}"
                )

    async def execute_stops_and_close_positions(self) -> None:
        await Terminator(self).run()

    def clear_records(self):
        self.sm.clear_strategies()

    def close_positions(self) -> None:
        positions: list[ibi.Position] = self.ib.positions()
        log.debug(f"closing positions: {positions}")
        for position in positions:
            self.ib.qualifyContracts(position.contract)
            self.ib.placeOrder(
                position.contract,
                ibi.MarketOrder(
                    "BUY" if position.position < 0 else "SELL", abs(position.position)
                ),
            )

    def run_nuke(self) -> None:
        """
        Cancel all open orders, close existing positions and prevent
        any further trading.  Response to a critical error or request
        sent by administrator.

        ---> Currently not in use. <---
        """
        # this will ignore any further orders coming from the system
        self.trader = FakeTrader(self.ib)
        self.ib.reqGlobalCancel()
        self.close_positions()

        log.critical("Self nuked!!!!! No more trades will be executed until restart.")


class OrderLoggers:
    """
    These are optional, non-essential loggers that can be switched on
    by :class:`Controller`
    """

    def __init__(self, ib: ibi.IB) -> None:
        self.ib = ib
        ib.cancelOrderEvent += self.log_cancel
        ib.orderModifyEvent += self.log_modification

    @staticmethod
    def log_cancel(trade: ibi.Trade) -> None:
        log.info(
            f"{trade.order.orderType} order {trade.order.action} "
            f"{trade.remaining()} (of "
            f"{trade.order.totalQuantity}) for "
            f"{trade.contract.localSymbol} cancelled"
        )

    @staticmethod
    def log_modification(trade: ibi.Trade) -> None:
        log.debug(f"Order modified: {trade.order}")

    def __repr__(self) -> str:
        return f"OrderLoggers({self.ib})"
