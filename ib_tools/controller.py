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
from .startup_routines import OrderSyncStrategy, PositionSyncStrategy
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

    async def sync(self, *args, **kwargs) -> None:
        log.debug("Sync...")

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

        try:
            report = OrderSyncStrategy.run(self.ib, self.sm)
            log.debug(
                f"Trades on re-start -> unknown: {len(report.unknown)}, "
                f"done: {len(report.done)}, error: {len(report.errors)}"
            )
        except Exception as e:
            log.exception(e)
            raise

        for unknown_trade in report.unknown:
            log.critical(f"Unknow trade in the system: {unknown_trade}.")

        # From here IB events will be handled...
        self.hold = False
        log.debug("hold released")
        # ...so that matched trades can be sent to blotter

        try:
            for done_trade in report.done:
                self.report_done_trade(done_trade)
        except Exception as e:
            log.exception(f"Error with done trade: {e}")

        await asyncio.sleep(0)
        try:
            # don't know what to do with that yet:
            self.clear_error_trades(report.errors)

        except Exception as e:
            log.exception(f"Error handling inactive trades: {e}")

        await asyncio.sleep(0)
        try:
            if error_position_report := PositionSyncStrategy.run(
                self.ib, self.sm
            ).errors:
                log.critical(f"Wrong positions on restart: {error_position_report}")
                self.handle_error_positions(error_position_report)
            else:
                log.info("Positions matched to broker?: --> OK <--")
        except Exception as e:
            log.exception(f"Error handling wrong position on restart: {e}")

        log.debug("Sync completed.")

    def handle_error_positions(self, report: dict[ibi.Contract, float]) -> None:
        # too many externalities... refactor
        log.error("Will attempt to fix position records")
        for contract, diff in report.items():
            strategies = self.sm.for_contract.get(contract)
            if strategies and len(strategies) == 1:
                self.sm.strategy[strategies[0]].position -= diff
                log.error(
                    f"Corrected position records for strategy {strategies[0]} by {diff}"
                )
                self.sm.save_model(self.sm._data.encode())
                log.debug("Will attempt to identify missing order. UNTESTED.")
                if (
                    len(
                        missing_trade := [
                            t
                            for t in self.ib.trades()
                            if all(
                                [
                                    t.contract == contract,
                                    t.orderStatus.status == "Filled",
                                    t.order.filledQuantity == abs(diff),
                                ]
                            )
                        ]
                    )
                    == 1
                ):
                    log.debug(
                        f"Missing trade found: {missing_trade[0]}, "
                        f"will try to save and report."
                    )
                    try:
                        self.sm.register_order(
                            strategies[0],
                            "UNKNOWN",
                            missing_trade[0],
                            self.sm.strategy[strategies[0]],
                        )
                        self.report_done_trade(missing_trade[0])
                    except Exception as e:
                        log.exception(e)
                elif missing_trade:
                    # if more than one assume it's the latest one
                    try:
                        self.report_done_trade(
                            sorted(missing_trade, key=lambda x: x.log[-1].time)[-1]
                        )
                    except Exception as e:
                        log.exception(e)

            elif strategies and self.ib_position_for_contract(contract) == 0:
                for strategy in strategies:
                    self.sm.strategy[strategy].position = 0
                self.sm.save_model(self.sm._data.encode())
                log.error(
                    f"Position records zeroed for {strategies} "
                    f"to reflect zero position for {contract.symbol}."
                )
            else:
                log.critical(
                    f"More than 1 stratey for contract {contract.symbol}, "
                    f"cannot fix position records."
                )

    def clear_error_trades(self, trades: list[ibi.Trade]) -> None:
        """
        Should rather beUsed for cold restarts, but currently used at all times.
        Trades that we have as active but IB doesn't know about them.
        """
        for trade in trades:
            log.error(
                f"Will delete record for trade that IB doesn't known about: "
                f"{trade.order.orderId}"
            )
            self.sm.delete_trade_record(trade)

    def report_done_trade(self, trade: ibi.Trade):
        """
        Used during sync.
        """
        log.debug(
            f"Back-reporting trade: {trade.contract.symbol} "
            f"{trade.order.action} {misc.trade_fill_price(trade)} "
            f"order id: {trade.order.orderId} {trade.order.permId} "
            f"active?: {trade.isActive()}"
        )
        self.ib.orderStatusEvent.emit(trade)
        for fill in trade.fills:
            self.ib.execDetailsEvent.emit(trade, fill)
        if trade.orderStatus.status == "Filled":
            self.ib.commissionReportEvent.emit(
                trade, trade.fills[-1], trade.fills[-1].commissionReport
            )

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
                f"Position records correct for contract "
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
        # THIS IS ALL POINTLESS
        # TODO
        # no permId here at all
        # THIS IS A TEST IF IT WORKS
        # await asyncio.sleep(0.1)
        # log.debug(f"New order event: {trade.order.orderId, trade.order.permId}")
        existing_order_record = self.sm.order.get(trade.order.orderId)
        if not existing_order_record:
            log.critical(f"Unknown trade in the system {trade.order}")

        # Give exec_model a chance to save bracket
        # await asyncio.sleep(0.5)
        self.sm.report_new_order(trade)

    async def onOrderStatusEvent(self, trade: ibi.Trade) -> None:
        log.debug("inside onOrderStatusEvent")
        if self.hold:
            return
        log.debug(
            f"Reporting order status: {trade.order.orderId} {trade.order.permId} "
            f"{trade.orderStatus.status}"
        )
        await self.sm.save_order_status(trade)
        if trade.order.orderId < 0:
            log.error(f"Manual trade: {trade.order} status update: {trade.orderStatus}")
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

    # Combine with register_position
    # half of work here is linked to registering position
    def onExecDetailsEvent(self, trade: ibi.Trade, fill: ibi.Fill) -> None:
        """
        Register position.
        """
        order_info = self.sm.order.get(trade.order.orderId)
        if order_info:
            strategy_str = order_info.strategy
        strategy = self.sm.strategy.get(strategy_str)

        if trade.order.orderId < 0:  # this is MANUAL TRADE
            strategies_list = self.sm.for_contract[trade.contract]
            if len(strategies_list) == 1:
                strategy = strategies_list[0]

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
        log.debug("Inside onCommissionReport")
        # prevent writing all orders from session on startup
        if self.hold:
            return
        log.debug(
            f"Will report commission for {trade.order.orderId} {trade.order.permId}"
        )
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
        return (my_position == ib_position) or (my_position - ib_position)

    def ib_position_for_contract(self, contract: ibi.Contract) -> float:
        # CONSIDER MOVING TO TRADER
        return next(
            (v.position for v in self.ib.positions() if v.contract == contract), 0
        )

        # positions = {p.contract: p.position for p in self.ib.positions()}
        # return positions.get(contract, 0.0)

    def nuke(self):
        """
        Cancel all open orders, close existing positions and prevent
        any further trading.  Response to a critical error or request
        sent by administrator.
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
