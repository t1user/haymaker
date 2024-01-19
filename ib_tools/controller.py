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

# from .manager import IB, STATE_MACHINE
from .trader import FakeTrader, Trader

if TYPE_CHECKING:
    from .execution_models import AbstractExecModel
    from .state_machine import StateMachine, Strategy

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
        self.report = {
            "unknown": self.unknown,
            "done": self.done,
            "errors": self.errors,
        }

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
        for orderId, oi in self.sm._orders.copy().items():
            if orderId not in ib_open_trades:
                # if inactive it's already been dealt with before restart
                if oi.active:
                    # this is a trade that we have as active in self.orders
                    # but IB doesn't have it in open orders
                    # we have to figure out what happened to this trade
                    # while we were disconnected and report it as appropriate
                    self.inactive.append(oi.trade)
                else:
                    # this order is no longer of interest
                    # it's inactive in our orders and inactive in IB
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

        done = {
            trade.order.orderId: trade
            for trade in self.inactive
            if trade.order.orderId in ib_known_trades
        }
        self.errors = [
            trade for trade in self.inactive if trade.order.orderId not in done
        ]
        self.done = list(done.values())
        return self


class PositionSyncStrategy:
    """
    Must be called after :class:`.OrderSyncStrategy`
    """

    def __init__(self, ib: ibi.IB, sm: StateMachine):
        self.ib = ib
        self.sm = sm
        self.errors: dict[ibi.Contract, float] = {}
        self.report = {"errors": self.errors}

    @classmethod
    def run(cls, ib: ibi.IB, sm: StateMachine):
        return cls(ib, sm).verify_positions()

    def verify_positions(self) -> PositionSyncStrategy:
        """
        Compare positions actually held with broker with position
        records.  Return differences if any and log an error.
        """

        broker_positions_dict = {i.contract: i.position for i in self.ib.positions()}
        my_positions_dict = self.sm.strategy.total_positions()
        # log.debug(f"Broker positions: {broker_positions_dict}")
        # log.debug(f"My positions: {my_positions_dict}")
        diff = {
            i: (
                (my_positions_dict.get(i) or 0.0)
                - (broker_positions_dict.get(i) or 0.0)
            )
            for i in set([*broker_positions_dict.keys(), *my_positions_dict.keys()])
        }
        self.errors = {k: v for k, v in diff.items() if v != 0}

        return self


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

        log.debug("hold set")
        self.hold = True

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

    async def sync(self, *args, **kwargs) -> None:
        log.debug("Sync...")

        if self.cold_start:
            log.debug("Starting cold...")
        else:
            try:
                log.debug("Reading from store...")
                await self.sm.read_from_store()
                self.cold_start = True
            except Exception as e:
                log.exception(e)

        try:
            report = OrderSyncStrategy.run(self.ib, self.sm).report
            log.debug(
                f"Trades on re-start - unknown: {len(report['unknown'])}, "
                f"done: {len(report['done'])}, error: {len(report['errors'])}"
            )
        except Exception as e:
            log.exception(e)
            raise

        for error_trade in report["unknown"]:
            log.critical(f"Unknow trade in the system: {error_trade}.")

        # From here IB events will be handled...
        self.hold = False
        log.debug("hold released")
        # ...so that matched trades can be sent to blotter

        try:
            for trade in report["done"]:
                self.report_done_trade(trade)
        except Exception as e:
            log.exception(f"Error with done trade: {e}")

        try:
            # don't know what to do with that yet:
            self.handle_inactive_trades(report["errors"])

        except Exception as e:
            log.exception(f"Error handling inactive trades: {e}")

        try:
            if error_position_report := PositionSyncStrategy.run(
                self.ib, self.sm
            ).report["errors"]:
                log.critical(f"Wrong positions on restart: {error_position_report}")
            else:
                log.info("Positions matched to broker?: --> OK <--")
        except Exception as e:
            log.exception(f"Error handling wrong position on restart: {e}")

        log.debug("Sync completed.")

    def handle_inactive_trades(self, trades: list[ibi.Trade]) -> None:
        """
        Trades that we have as active but IB doesn't know about them.
        Used for cold restarts.
        """
        for trade in trades:
            log.error(
                f"Will delete record for trade that IB doesn't known about: "
                f"{trade.order.orderId}"
            )
            self.sm.delete_trade_record(trade)

    def report_done_trade(self, trade: ibi.Trade):
        log.debug(
            f"Back-reporting trade: {trade.contract.symbol} "
            f"{trade.order.action} {trade.order.totalQuantity} "
            f"order id: {trade.order.orderId}"
        )

        self.ib.orderStatusEvent.emit(trade)
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
                f"{self.sm.verify_position_for_contract(data.active_contract)}"
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
        if self.hold:
            return

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
                f"Registered {trade.order.orderId} SELL {trade.order.orderType} "
                f"{trade.order.permId} BUY {trade.order.orderType} "
                f"for {strategy_str} --> position: {strategy.position}"
            )
        else:
            log.critical(
                f"Abiguous fill: {fill} for order: {trade.order} for "
                f"{trade.contract.localSymbol} strategy: {strategy}"
            )

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
            f"{trade.order.action} {trade.orderStatus.filled}"
            f"@{trade.orderStatus.avgFillPrice} --> {strategy}"
            f"orderId: {trade.order.orderId}, permId: {trade.order.permId}"
        )

    def log_cancel(self, trade: ibi.Trade) -> None:
        log.info(
            f"{trade.order.orderType} order {trade.order.action} "
            f"{trade.orderStatus.remaining} (of "
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
        my_position = self.position.get(contract, 0.0)
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
