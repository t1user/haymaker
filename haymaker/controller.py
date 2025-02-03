from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import Generator

import eventkit as ev  # type: ignore
import ib_insync as ibi

from . import misc
from .base import Atom
from .blotter import Blotter
from .config import CONFIG
from .misc import sign
from .state_machine import OrderInfo, Strategy
from .sync_routines import (
    ErrorHandlers,
    OrderSyncStrategy,
    PositionSyncStrategy,
    Terminator,
)
from .trader import FakeTrader, Trader

log = logging.getLogger(__name__)


class Controller(Atom):
    """
    Intermediary between execution models (which are off ramps for
    strategies), :class:`Trader` and :class:`StateMachine`.  Use information
    provided by :class:`StateMachine` to make sure that positions held in
    the market reflect what is requested by strategies.

    """

    blotter: Blotter | None
    config = CONFIG.get("controller") or {}

    def __init__(
        self,
        trader: Trader | None = None,
    ):
        super().__init__()
        self.trader = trader or Trader(self.ib)
        # these are essential (non-optional) events
        self.ib.execDetailsEvent.connect(self.onExecDetailsEvent, self._log_event_error)
        self.ib.newOrderEvent.connect(self.onNewOrderEvent, self._log_event_error)
        self.ib.orderStatusEvent.connect(self.onOrderStatusEvent, self._log_event_error)
        # TODO: self.ib.orderModifyEvent
        # TODO: self.ib.orderCancelEvent
        # consider whether these are essential
        self.ib.orderStatusEvent.connect(self.log_order_status, self._log_event_error)
        self.ib.errorEvent.connect(self.log_err, self._log_event_error)

        self.set_hold()

        if self.config.get("use_blotter"):
            self.blotter = Blotter()
            self.ib.commissionReportEvent.connect(
                self.onCommissionReport, self._log_event_error
            )
        else:
            self.blotter = None

        if sync_frequency := self.config.get("sync_frequency"):
            self.sync_timer = ev.Timer(sync_frequency)
            self.sync_timer.connect(self.sync, error=self._log_event_error)

        if self.config.get("log_IB_events"):
            self._attach_logging_events()

        self.cold_start = CONFIG.get("coldstart")
        self.reset = CONFIG.get("reset")
        self.zero = CONFIG.get("zero")
        self.nuke_ = CONFIG.get("nuke")
        self.cancel_stray_orders = self.config.get("cancel_stray_orders")

        self.sync_handlers = ErrorHandlers(self.ib, self.sm, self)
        self.no_future_roll_strategies: list[str] = []
        log.debug(f"Controller initiated: {self}")

    def _attach_logging_events(self):
        # these are non-essential events
        self.ib.newOrderEvent += self.log_new_order
        self.ib.cancelOrderEvent += self.log_cancel
        self.ib.orderModifyEvent += self.log_modification

    def set_hold(self) -> None:
        self.hold = True
        log.debug("hold set")

    def release_hold(self) -> None:
        if self.hold:
            self.hold = False
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
        if self.nuke_:
            self.nuke()

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
            self.sm.clear_models()

        self.roller = FutureRoller(self)

        # Only subsequently Streamers will run (I think...)
        log.debug("Controller run sequence completed successfully.")

    def roll_futures(self, *args):
        """
        This method is scheduled to run once a day in :class:`.app.App`
        """
        log.info(f"Running roll: {args} on controller object: {id(self)}")
        self.roller.roll()

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
        # super().onData(data)
        if not (delay := self.config.get("execution_verification_delay")):
            return

        try:
            strategy = data["strategy"]
            amount = data["amount"]
            target_position = data["target_position"]
            # TODO: Check transaction integrity only after order is done!
            await asyncio.sleep(delay)
            await self.verify_transaction_integrity(strategy, amount, target_position)
        except KeyError:
            log.exception(
                "Unable to verify transaction integrity", extra={"data": data}
            )
        contract = data.get("contract")
        self.verify_position_with_broker(contract)

    def trade(
        self,
        strategy: str,
        contract: ibi.Contract,
        order: ibi.Order,
        action: str,
        data: Strategy,
    ) -> ibi.Trade:
        trade = self.trader.trade(contract, order)
        self.sm.register_order(strategy, action, trade, data)
        trade.filledEvent += partial(self.log_trade, reason=action, strategy=strategy)
        return trade

    def cancel(self, trade: ibi.Trade) -> ibi.Trade | None:
        return self.trader.cancel(trade)

    async def onNewOrderEvent(self, trade: ibi.Trade) -> None:
        """
        Check if the system knows about the order that was just posted
        to the broker.

        This is an event handler (callback).  Connected (subscribed)
        to :meth:`ibi.IB.newOrderEvent` in :meth:`__init__`
        """
        # Consider whether essential or optional
        # Maybe some order verification here?
        log.debug(f"New order event: {trade.order.orderId, trade.order.permId}")
        if not (trade.order.orderId < 0 or self.sm.order.get(trade.order.orderId)):
            log.critical(
                f"Unknown trade in the system {trade.order} {trade.contract.symbol}"
            )

    async def onOrderStatusEvent(self, trade: ibi.Trade) -> None:
        if self.hold:
            return

        # this will create new order record if it doesn't already exist
        await self.sm.save_order_status(trade)

    def register_position(
        self, strategy_str: str, strategy: Strategy, trade: ibi.Trade, fill: ibi.Fill
    ) -> None:
        try:
            if fill.execution.side == "BOT":
                strategy.position += fill.execution.shares
                log.debug(
                    f"Registered orderId: {trade.order.orderId} permId: "
                    f"{trade.order.permId} BUY {trade.order.orderType} "
                    f"for {strategy_str} --> position: {strategy.position}"
                )
            elif fill.execution.side == "SLD":
                strategy.position -= fill.execution.shares
                log.debug(
                    f"Registered orderId {trade.order.orderId} permId: "
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

    def _cleanup_obsolete_orders(self, strategy: Strategy) -> None:
        """Delete stop/take-profit/close orders for a strategy that has no position."""
        order_infos = self.sm.orders_for_strategy(strategy.strategy)
        for oi in order_infos:
            if (oi.action != "OPEN") and oi.active:
                log.debug(f"Resting order cleanup: {oi.action, oi.trade.order.orderId}")
                self.trader.cancel(oi.trade)

    async def onExecDetailsEvent(self, trade: ibi.Trade, fill: ibi.Fill) -> None:
        """
        Register position.
        """
        strategy = (
            self.assign_manual_trade(trade)
            or self.sm.strategy_for_trade(trade)
            or self.assign_unknown_trade(trade)
        )
        self.register_position(strategy.strategy, strategy, trade, fill)

    def onCommissionReport(
        self, trade: ibi.Trade, fill: ibi.Fill, report: ibi.CommissionReport
    ) -> None:
        """
        Writing commission on :class:`ibi.Trade` is the final stage of
        order execution.  After that trade object is ready for storing
        in blotter.
        """

        # silence emission of all orders from session on startup
        if self.hold:
            return

        data = self.sm.order.get(trade.order.orderId)
        if data:
            strategy, action, _, params, _ = data
            position_id = params.get("position_id") or self.sm.strategy[strategy].get(
                "position_id"
            )

            kwargs = {
                "strategy": strategy,
                "action": action,
                "position_id": position_id,
                "params": ibi.util.tree(params),
            }

            if arrival_price := params.get("arrival_price"):
                kwargs.update(
                    {
                        "price_time": arrival_price["time"],
                        "bid": arrival_price["bid"],
                        "ask": arrival_price["ask"],
                    }
                )

        elif trade.order.totalQuantity == 0:
            return

        assert self.blotter is not None
        self.blotter.log_commission(trade, fill, report, **kwargs)

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

        Any errors are logged but not corrected (may be changed in
        future).
        """
        delay = self.config.get("execution_verification_delay")
        assert delay

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
                log.warning(
                    f"{strategy} taking long to achive target position of {target}..."
                )
                await asyncio.sleep(delay)

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
            if candidate_strategies:
                strategy_str = candidate_strategies[0]
                strategy = self.sm.strategy[strategy_str]
        else:
            strategy = None

        return strategy

    def _make_strategy(self, trade: ibi.Trade, description: str) -> Strategy:
        """
        Last resort when strategy cannot be matched.  Creating new
        made up strategy to make sure that position change resulting
        from trade will be somehow accounted for.
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
        # Connected to onOrderStatusEvent

        if self.hold:
            return

        if trade.order.orderId < 0:
            log.warning(
                f"Manual trade: {trade.order} status update: {trade.orderStatus}"
            )

        elif trade.isDone():
            log.debug(
                f"{trade.contract.symbol}: order {trade.order.orderId} "
                f"{trade.order.orderType} done {trade.orderStatus.status}."
            )
        else:
            log.info(
                f"{trade.contract.symbol}: OrderStatus ->{trade.orderStatus.status}<-"
                f" for order: {trade.order},"
            )

    @staticmethod
    def log_new_order(trade: ibi.Trade) -> None:
        log.debug(f"New order {trade.order} for {trade.contract.localSymbol}")

    @staticmethod
    def log_trade(trade: ibi.Trade, reason: str = "", strategy: str = "") -> None:
        log.info(
            f"{reason} trade filled: {trade.contract.localSymbol} "
            f"{trade.order.action} {trade.filled()}"
            f"@{misc.trade_fill_price(trade)} --> {strategy} "
            f"orderId: {trade.order.orderId}, permId: {trade.order.permId}"
        )

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

    def log_err(
        self, reqId: int, errorCode: int, errorString: str, contract: ibi.Contract
    ) -> None:
        # Connected to ib.errorEvent

        if errorCode < 400:
            # reqId is most likely orderId
            # order rejected is errorCode = 201
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

            else:
                log.error(
                    f"Error {errorCode}: {errorString} {contract}, "
                    f"{strategy} | {action} | {order}"
                )

    async def execute_stops_and_close_positions(self) -> None:
        await Terminator(self).run()

    def clear_records(self):
        self.sm.clear_models()

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

    def nuke(self) -> None:
        """
        Cancel all open orders, close existing positions and prevent
        any further trading.  Response to a critical error or request
        sent by administrator.

        ---> Currently not in use. <---
        """
        # this will ignore any further orders coming from the system
        # don't use self.trader any more
        self.trader = FakeTrader(self.ib)
        self.ib.reqGlobalCancel()
        self.close_positions()

        log.critical("Self nuked!!!!! No more trades will be executed until restart.")

    # def __repr__(self) -> str:
    #     return f"{self.__class__.__name__}({self.sm, self.ib, self.blotter})"


class FutureRoller:

    def __init__(self, controller: Controller, excluded_strategies: list[str] = []):
        self.controller = controller
        self.sm = controller.sm
        self.excluded_strategies = excluded_strategies
        self.trade_generator: dict[ibi.Contract, Generator] = {}

    @property
    def futures(self) -> set[ibi.Future]:
        """
        List of active contracts that are futures.
        """
        return set([f for f in self.controller.contracts if isinstance(f, ibi.Future)])

    @property
    def strategies(self) -> dict[ibi.Future, list[str]]:
        """
        Dict of future contracts with corresponding lists of
        strategies for those contracts regardless of whether those
        strategies have open positions.
        """
        strategies = {
            fut: [i for i in strategy_list if i not in self.excluded_strategies]
            for fut, strategy_list in self.sm.strategy.strategies_by_contract().items()
            if isinstance(fut, ibi.Future)
        }
        log.debug(f"{strategies=}")
        return strategies

    @property
    def positions(self) -> dict[ibi.Future, float]:
        """
        Dict of positions for every future contract regardless of
        whether the contract is expiring (and hence should be rolled).

        Where strategies cancel each other, contract will still have
        an entry in the returned dictionary with position value equal to zero.
        Even though those positions don't need to be rolled, their
        orders must be rolled, so it's crucial to have an entry in the
        dict for such contracts.
        """
        return {
            fut: sum(
                [p for name in strategy_names if (p := self.sm.strategy[name].position)]
            )
            for fut, strategy_names in self.strategies.items()
        }

    @property
    def contracts_to_roll(self) -> set:
        """
        Set of futures that need to be rolled, i.e. these are the futures
        we have positions for, but they're not active contracts any
        more and they are not for strategies that we explicitly
        excluded from rolling.

        All previous properties are intermediate steps to get this one
        piece of information.
        """
        # difference between contracts we hold vs active contracts
        contracts_to_roll = set(self.positions.keys()) - self.futures
        return contracts_to_roll

    def positions_by_strategy_for_contract(
        self, contract: ibi.Future
    ) -> dict[str, float]:
        """
        Given an expiring contract, return a dict of strategies with
        their rollable holdings.  It's not necessarily number of
        contracts to be rolled, this is to be determined by other
        methods, it's a number of contracts that should be considered
        for rolling given which contracts are expiring and any
        limitations about strategies excluded from rolling.
        """
        return {
            strategy_str: self.sm.strategy[strategy_str].position
            for strategy_str in self.strategies[contract]
        }

    def match_old_to_new_future(self, old_future: ibi.Future) -> ibi.Future:
        try:
            return next(
                (
                    new_future
                    for new_future in self.futures
                    if (
                        (old_future.symbol == new_future.symbol)
                        and (old_future.exchange == new_future.exchange)
                        and (old_future.multiplier == new_future.multiplier)
                        and (old_future.conId != new_future.conId)
                    )
                )
            )
        except StopIteration:
            log.error(f"No replacement contract for expiring: {old_future}")
            return old_future

    def roll(self) -> None:
        """
        This is the main entry point into the rolling strategy and
        orchestrator of all actions.
        """
        if contracts := self.contracts_to_roll:
            log.warning(
                f"Contracts will be rolled: {[c.localSymbol for c in contracts]}"
            )
        for old_contract in contracts:
            new_contract = self.match_old_to_new_future(old_contract)
            if new_contract.conId != old_contract.conId:
                self.trade_generator[new_contract] = self.execute(
                    old_contract, new_contract
                )
                try:
                    next(self.trade_generator[new_contract])
                except StopIteration:
                    del self.trade_generator[new_contract]
            else:
                log.error(f"Abandoning roll, no replacement found: {old_contract}")

    def execute(
        self, old_contract: ibi.Future, new_contract: ibi.Future
    ) -> Generator[None, None, None]:
        # THIS IS A GENERATOR
        # positions for expiring contract that are not excluded from rolling
        total = self.positions[old_contract]
        strategies_to_trade = self.figure_out_strategies_to_trade(old_contract, total)
        for strategy_str in self.strategies[old_contract]:
            strategy = self.sm.strategy[strategy_str]
            if total and (strategy_str in strategies_to_trade):
                size = strategy.position
                trade = self.trade(
                    strategy_str, strategy, old_contract, new_contract, size
                )
                # if we're changing orders and records connected to a strategy
                # that we trade, make those changes when and if the order is filled
                trade.filledEvent += partial(
                    self.adjust_records_and_orders_for_strategy,
                    strategy_str,
                    strategy,
                    old_contract,
                    new_contract,
                )
                trade.filledEvent += self.trade_callback
                # don't issue next order until this one is done
                yield
            else:
                # no trading here
                self.adjust_records_and_orders_for_strategy(
                    strategy_str, strategy, old_contract, new_contract
                )

    def trade_callback(self, trade: ibi.Trade):
        log.debug(f"Roll trade filled for: {trade.contract.localSymbol}")
        try:
            gen = self.trade_generator.get(trade.contract)
            assert gen
            next(gen)
        except StopIteration:
            del self.trade_generator[trade.contract]

    def adjust_records_and_orders_for_strategy(
        self,
        strategy_str: str,
        strategy: Strategy,
        old_contract: ibi.Future,
        new_contract: ibi.Future,
    ) -> None:
        self.adjust_strategy_records(strategy_str, strategy, old_contract, new_contract)
        self.adjust_strategy_orders(strategy_str, strategy, old_contract, new_contract)

    def figure_out_strategies_to_trade(
        self, contract: ibi.Future, total_position: float
    ) -> list[str]:
        """
        This method determines the actual number of contracts to trade
        for each strategy.  Strategies may not be the same as the
        strategies for which resting orders are rolled (as some
        strategies may have positions cancelling each other).

        The aim is to trade as little as possible and still be able to
        assign each trade to a particular strategy.  In some cases it
        may mean that opposing orders made.  Therefore, orders are
        sent sequentially, each new order is sent only after previous
        one has been filled.

        If we're trading and changing resting orders (i.e. stop-losses
        or take profits) for the same strategy, order adjustments are
        done only after the trade is executed in order to avoid a
        situation where the order rolling position would be rejected,
        but resting stop-losses would be rolled.
        """
        # strategies with positions to roll
        positions: dict[str, float] = self.positions_by_strategy_for_contract(contract)
        # some positions have opposing signs, i.e. cancel each other
        cancelling_strategies: bool = sum(positions.values()) != sum(
            [abs(p) for p in positions.values()]
        )
        # in this case try to figure out the least number of contracts to trade
        if cancelling_strategies:
            ascending_search = self.strategy_search(total_position, positions)
            if len(ascending_search) == 1:
                return ascending_search
            descending_search = self.strategy_search(
                total_position, positions, descending=True
            )
            # return the list with fewer strategies
            if ascending_search and descending_search:
                strategies = sorted(
                    [ascending_search, descending_search], key=lambda x: len(x)
                )[0]
            else:
                # or one that is not empty or if both are empty list of all strategies
                strategies = (
                    ascending_search or descending_search or list(positions.keys())
                )
        else:
            # if strategies are not cancelling each other, just trade all of them
            strategies = list(positions.keys())

        return strategies

    @staticmethod
    def strategy_search(
        total_position: float, positions: dict[str, float], descending: bool = False
    ) -> list[str]:
        position_sign = sign(total_position)
        accumulator = 0.0
        acc_strategies = []
        for strategy_str, position in dict(
            sorted(positions.items(), key=lambda i: abs(i[1]), reverse=descending)
        ).items():
            # rolling this will solve the problem
            if position == total_position:
                return [strategy_str]
            if (sign(position) == position_sign) and (
                abs(accumulator + position) <= abs(total_position)
            ):
                accumulator += position
                acc_strategies.append(strategy_str)
                if accumulator == total_position:
                    return acc_strategies
        return []

    def verify_done(self, old_contract: ibi.Future):
        size = self.positions[old_contract]
        positions = {}
        for strategy_str in self.strategies[old_contract]:
            strategy = self.sm.strategy[strategy_str]
            positions[strategy_str] = strategy.position
        if (size == 0) and (len(positions) == 0):
            log.debug(f"All positions for {old_contract.symbol} rolled.")
        else:
            log.critical(
                f"Future roll error; left-over position for "
                f"{old_contract.symbol}: {size}."
                f"strategy positions: {positions}"
            )

    def trade(
        self,
        strategy_str: str,
        strategy: Strategy,
        old_contract: ibi.Future,
        new_contract: ibi.Future,
        size: float,
    ) -> ibi.Trade:
        combo = self.make_combo(old_contract, new_contract)
        strategy["params"]["rolled"] = True
        order = ibi.MarketOrder("BUY" if size > 0 else "SELL", abs(size))
        return self.controller.trade(
            strategy_str, combo, order, "FUTURE_ROLL", strategy
        )

    @staticmethod
    def make_combo(oc: ibi.Future, nc: ibi.Future) -> ibi.Bag:
        return ibi.Bag(
            symbol=nc.symbol,
            exchange=nc.exchange,
            currency=nc.currency,
            multiplier=nc.multiplier,
            comboLegs=[
                ibi.ComboLeg(
                    conId=oc.conId,
                    ratio=1,
                    action="SELL",
                    exchange=oc.exchange,
                ),
                ibi.ComboLeg(
                    conId=nc.conId,
                    ratio=1,
                    action="BUY",
                    exchange=nc.exchange,
                ),
            ],
        )

    def adjust_strategy_records(
        self,
        strategy_str: str,
        strategy: Strategy,
        old_contract: ibi.Future,
        new_contract: ibi.Future,
    ) -> None:
        strategy.active_contract = new_contract

    def adjust_strategy_orders(
        self,
        strategy_str: str,
        strategy: Strategy,
        old_contract: ibi.Future,
        new_contract: ibi.Future,
    ) -> None:
        for oi in self.sm.orders_for_strategy(strategy_str):
            old_trade = oi.trade
            old_trade.cancelledEvent += partial(
                self.issue_new_order,
                oi=oi,
                new_contract=new_contract,
                strategy_str=strategy_str,
                strategy=strategy,
            )
            cancelled_trade = self.controller.cancel(oi.trade)
            if cancelled_trade:
                log.debug(
                    f"Cancelled order {cancelled_trade.order.orderId}: "
                    f"for: {cancelled_trade.contract.localSymbol} {strategy_str} "
                    f"{oi.action}"
                )

    def issue_new_order(
        self,
        cancelled_trade: ibi.Trade,
        oi: OrderInfo,
        new_contract: ibi.Future,
        strategy_str: str,
        strategy: Strategy,
    ) -> None:
        order_kwarg_dict = ibi.util.dataclassNonDefaults(cancelled_trade.order)

        for key in ("orderId", "permId", "softDollarTier", "clientId"):
            if order_kwarg_dict.get(key):
                del order_kwarg_dict[key]

        if order_kwarg_dict["orderType"] == "FIX PEGGED":
            order_kwarg_dict["orderType"] = "TRAIL"
            order_kwarg_dict["auxPrice"] = misc.round_tick(
                (oi.params.get("trail_multiple") or oi.params.get("adjusted_multiple"))
                * oi.params["sl_points"],
                oi.params["min_tick"],
            )
        new_order = ibi.Order(**order_kwarg_dict)
        new_trade = self.controller.trade(
            strategy_str, new_contract, new_order, oi.action, strategy
        )
        log.debug(
            f"New rolled order {(new_trade.order.orderId, new_trade.order.permId)}"
            f"{new_trade.contract.localSymbol} {strategy_str} {oi.action}"
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(controller={self.controller}, "
            f"excluded_strategies={self.excluded_strategies})"
        )
