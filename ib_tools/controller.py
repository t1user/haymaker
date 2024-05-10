from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import TYPE_CHECKING, Callable, Optional, Union

import ib_insync as ibi

from . import misc
from .base import Atom
from .blotter import Blotter
from .config import CONFIG
from .misc import Signal, sign
from .startup_routines import ErrorHandlers, OrderSyncStrategy, PositionSyncStrategy
from .state_machine import Strategy

# from .manager import IB, STATE_MACHINE
from .trader import FakeTrader, Trader

if TYPE_CHECKING:
    from .execution_models import AbstractExecModel


log = logging.getLogger(__name__)


class Controller(Atom):
    """
    Intermediary between execution models (which are off ramps for
    strategies), :class:`Trader` and :class:`StateMachine`.  Use information
    provided by :class:`StateMachine` to make sure that positions held in
    the market reflect what is requested by strategies.

    """

    blotter: Optional[Blotter]

    def __init__(
        self,
        trader: Optional[Trader] = None,
    ):
        super().__init__()
        self.trader = trader or Trader(self.ib)
        # these are essential (non-optional) events
        self.ib.execDetailsEvent.connect(self.onExecDetailsEvent)
        self.ib.newOrderEvent.connect(self.onNewOrderEvent)
        self.ib.orderStatusEvent.connect(self.onOrderStatusEvent)

        self.set_hold()

        if CONFIG.get("use_blotter"):
            self.blotter = Blotter()
            self.ib.commissionReportEvent += self.onCommissionReport
        else:
            self.blotter = None

        if CONFIG.get("log_IB_events"):
            self._attach_logging_events()

        self.cold_start = CONFIG.get("coldstart")
        self.sync_handlers = ErrorHandlers(self.ib, self.sm, self)
        log.debug(f"Controller initiated: {self}")

    def _attach_logging_events(self):
        # these are non-essential events
        self.ib.newOrderEvent += self.log_new_order
        self.ib.cancelOrderEvent += self.log_cancel
        self.ib.orderModifyEvent += self.log_modification
        self.ib.errorEvent += self.log_error

    def set_hold(self) -> None:
        self.hold = True
        log.debug("hold set")

    def release_hold(self) -> None:
        if self.hold:
            self.hold = False
            log.debug("hold released")

    async def run(self) -> None:

        self.set_hold()

        if self.cold_start:
            log.debug("Starting cold... (state NOT read from db)")
        else:
            try:
                log.debug("Reading from store...")
                await self.sm.read_from_store()
                self.cold_start = True
            except Exception as e:
                log.exception(e)

        await self.sync()

    async def sync(self) -> None:

        orders_report = OrderSyncStrategy.run(self.ib, self.sm)
        # IB events will be handled so that matched trades can be sent to blotter
        self.release_hold()
        await self.sync_handlers.handle_orders(orders_report)

        error_position_report = PositionSyncStrategy.run(self.ib, self.sm)
        await self.sync_handlers.handle_positions(error_position_report)

        log.debug("Sync completed.")

    async def onData(self, data, *args) -> None:
        """
        After obtaining transaction details from execution model,
        verify if the intended effect is the same as achieved effect.
        """
        super().onData(data)
        try:
            strategy = data["strategy"]
            amount = data["amount"]
            target_position = data["target_position"]
            await asyncio.sleep(15)
            self.verify_transaction_integrity(strategy, amount, target_position)
        except KeyError:
            log.exception(
                "Unable to verify transaction integrity", extra={"data": data}
            )

    def verify_transaction_integrity(
        self,
        strategy: str,
        amount: float,
        target_position: Signal,
    ) -> None:
        """
        Is the postion resulting from transaction the same as was
        required?
        """
        data = self.sm.strategy.get(strategy)
        if data:
            # TODO: doesn't work for REVERSE
            log.debug(
                f"Transaction OK? ->{sign(data.position) == target_position}<- "
                f"target_position: {target_position}, "
                f"position: {sign(data.position)}"
                f"->> {strategy}"
            )
            log.debug(
                f"Diff my position vs. IB position: "
                f"{data.active_contract.symbol}: "
                f"{self.verify_position_for_contract(data.active_contract)}"
            )
            try:
                assert sign(data.position) == target_position
                # Investigate why this may be necessary:
                # assert exec_model.position == abs(amount)
            except AssertionError:
                log.critical(f"Wrong position for {strategy}", exc_info=True)
        else:
            log.critical(f"Attempt to trade for unknow strategy: {strategy}")

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

    def cancel(
        self,
        trade: ibi.Trade,
        exec_model: AbstractExecModel,
        callback: Optional[Callable[[ibi.Trade], None]] = None,
    ) -> None:
        trade = self.trader.cancel(trade)
        if callback is not None:
            callback(trade)
        # orders are cancelled by callbacks so this is duplicating
        # self.sm.register_cancel(trade, exec_model)

    async def onNewOrderEvent(self, trade: ibi.Trade) -> None:
        """
        Check if the system knows about the order that was just posted
        to the broker.

        This is an event handler (callback).  Connected (subscribed)
        to :meth:`ibi.IB.newOrderEvent` in :meth:`__init__`
        """

        log.debug(f"New order event: {trade.order.orderId, trade.order.permId}")
        try:
            existing_order_record = self.sm.order.get(trade.order.orderId)
            if not existing_order_record:
                log.critical(f"Unknown trade in the system {trade.order}")
        except Exception as e:
            log.exception(e)
        if trade.order.orderId < 0:
            log.warn(f"manual order: {trade.order}")

    async def onOrderStatusEvent(self, trade: ibi.Trade) -> None:
        if self.hold:
            return
        # log.debug(
        #     f"Reporting order status: {trade.order.orderId} {trade.order.permId} "
        #     f"{trade.orderStatus.status}"
        # )
        await self.sm.save_order_status(trade)

        # Below is logging only
        if trade.order.orderId < 0:
            log.warning(
                f"Manual trade: {trade.order} status update: {trade.orderStatus}"
            )
        elif trade.orderStatus.status == ibi.OrderStatus.Inactive:
            messages = ";".join([m.message for m in trade.log])
            log.error(f"Rejected order: {trade.order}, messages: {messages}")
        elif trade.isDone():
            log.debug(
                f"{trade.contract.symbol}: order {trade.order.orderId} "
                f"{trade.order.orderType} done {trade.orderStatus.status}."
            )
        else:
            log.debug(
                f"{trade.contract.symbol}: OrderStatus ->{trade.orderStatus.status}<-"
                f" for order: {trade.order},"
            )

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

    def onExecDetailsEvent(self, trade: ibi.Trade, fill: ibi.Fill) -> None:
        """
        Register position.
        """
        try:
            order_info = self.sm.order.get(trade.order.orderId)
            if order_info:
                strategy_str: str = order_info.strategy
                strategy: Strategy = self.sm.strategy.get(strategy_str)
        except Exception as e:
            log.exception(e)

        if trade.order.orderId < 0:  # this is MANUAL TRADE
            try:
                log.debug("Will try to assign manual trade to strategy.")
                # assumed manual trade is to close a position
                active_strategies_list = [
                    s
                    for s in self.sm.for_contract[trade.contract]
                    if self.sm.strategy[s].active
                ]
                log.debug(f"{active_strategies_list=}")
                if len(active_strategies_list) == 1:
                    strategy_str = active_strategies_list[0]
                    strategy = self.sm.strategy[strategy_str]

                # if more than 1 active, manual trade is for the one without
                # resting orders (resting orders are most likely stop-losses)
                elif candidate_strategies := [
                    s for s in active_strategies_list if self.sm.orders_for_strategy(s)
                ]:
                    log.debug(
                        f"Active strategies without resting orders: "
                        f"{candidate_strategies}"
                    )
                    # if more than one just pick first one
                    # there's no way to determine which strategy was meant
                    if candidate_strategies:
                        strategy_str = candidate_strategies[0]
                        strategy = self.sm.strategy[strategy_str]

                # if cannot be matched, it's a separate strategy
                # position must always be recorded for a strategy
                else:
                    strategy_str = f"manual_strategy_{trade.contract.symbol}"
                    log.debug(f"{strategy_str=}")
                    strategy = self.sm._data[strategy_str]
                    strategy["active_contract"] = trade.contract
                    log.debug(f"{strategy=}")
                log.debug(f"Manual trade matched to strategy: {strategy_str}")
            except Exception as e:
                log.exception(e)
                raise

        if strategy:
            self.register_position(strategy_str, strategy, trade, fill)
        else:
            log.critical(f"Unknow trade: {trade}")

    def onCommissionReport(
        self, trade: ibi.Trade, fill: ibi.Fill, report: ibi.CommissionReport
    ) -> None:
        """
        Writing commission on :class:`ibi.Trade` is the final stage of order
        execution.  After that trade object is ready for stroring in
        blotter.
        """

        # prevent writing all orders from session on startup
        if self.hold:
            return
        data = self.sm.order.get(trade.order.orderId)
        if data:
            strategy, action, _, params, _ = data
            position_id = params.get("position_id")

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
        elif trade.order.orderId < 0:  # MANUAL TRADE
            strategies_list = self.sm.for_contract[trade.contract]
            if len(strategies_list) == 1:
                strategy = strategies_list[0]
            else:
                strategy = "UNKNOWN"
            kwargs = {"strategy": strategy, "action": "MANUAL TRADE"}
        else:
            kwargs = {"strategy": "unknown", "action": "UNKNOWN"}
            log.debug(
                f"Missing strategy records in `state machine`. "
                f"Incomplete data for blotter."
                f"orderId: {trade.order.orderId} symbol: {trade.contract.symbol} "
                f"orderType: {trade.order.orderType}"
            )

        assert self.blotter is not None
        self.blotter.log_commission(trade, fill, report, **kwargs)

    def log_new_order(self, trade: ibi.Trade) -> None:
        log.debug(f"New order {trade.order} for {trade.contract.localSymbol}")

    def log_trade(self, trade: ibi.Trade, reason: str = "", strategy: str = "") -> None:
        log.info(
            f"{reason} trade filled: {trade.contract.localSymbol} "
            f"{trade.order.action} {trade.filled()}"
            f"@{misc.trade_fill_price(trade)} --> {strategy} "
            f"orderId: {trade.order.orderId}, permId: {trade.order.permId}"
        )

    def log_cancel(self, trade: ibi.Trade) -> None:
        log.info(
            f"{trade.order.orderType} order {trade.order.action} "
            f"{trade.remaining()} (of "
            f"{trade.order.totalQuantity}) for "
            f"{trade.contract.localSymbol} cancelled"
        )

    def log_modification(self, trade: ibi.Trade) -> None:
        log.debug(f"Order modified: {trade.order}")

    def log_error(
        self, reqId: int, errorCode: int, errorString: str, contract: ibi.Contract
    ) -> None:
        pass
        if errorCode < 400:
            # reqId is most likely orderId
            # order rejected is errorCode = 201
            order_info = self.sm.order.get(reqId)
            if order_info:
                strategy, action, trade, _ = order_info
                order = trade.order
            else:
                strategy, action, trade, order = "", "", "", ""  # type: ignore

            log.error(
                f"Error {errorCode}: {errorString} {contract}, "
                f"{strategy} | {action} | {order}"
            )

    def verify_position_for_contract(
        self, contract: ibi.Contract
    ) -> Union[bool, float]:
        my_position = self.sm.position.get(contract, 0.0)
        ib_position = self.ib_position_for_contract(contract)
        return (my_position == ib_position) * 0 or (my_position - ib_position)

    def ib_position_for_contract(self, contract: ibi.Contract) -> float:
        # CONSIDER MOVING TO TRADER
        return next(
            (v.position for v in self.ib.positions() if v.contract == contract), 0
        )

    def nuke(self):
        """
        Cancel all open orders, close existing positions and prevent
        any further trading.  Response to a critical error or request
        sent by administrator.

        ---> Currently not in use. <---
        """
        for order in self.ib.openOrders():
            self.ib.cancelOrder(order)
        for position in self.ib.positions:
            self.trade(
                position.contract,
                ibi.MarketOrder(
                    "BUY" if position.position < 0 else "SELL", abs(position.position)
                ),
            )
        self.trader = FakeTrader()
        log.critical("Self nuked!!!!! No more trades will be executed until restart.")

    def __repr__(self):
        return f"{__class__.__name__}({self.sm, self.ib, self.blotter})"
