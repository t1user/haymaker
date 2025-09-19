from __future__ import annotations

import logging
from functools import cached_property, partial
from typing import TYPE_CHECKING, Generator

import ib_insync as ibi

if TYPE_CHECKING:
    from .controller import Controller

from haymaker import misc
from haymaker.base import Atom
from haymaker.state_machine import OrderInfo, Strategy

log = logging.getLogger(__name__)


class FutureRoller:

    def __init__(
        self, controller: Controller, excluded_strategies: list[str] = []
    ) -> None:
        self.controller = controller
        self.sm = controller.sm
        self.excluded_strategies = excluded_strategies
        self._trade_generator: dict[int, Generator] = {}
        log.debug("FutureRoller initiated")

    @cached_property
    def futures(self) -> set[ibi.Future]:
        """
        List of active contracts that are futures.
        """
        # controller is an Atom, so `controller.contracts` is a descriptor
        # on :class:`Atom`
        futures = set(
            [f for f in self.controller.contracts if isinstance(f, ibi.Future)]
        )
        log.debug(f"Current active futures: {[fut.localSymbol for fut in futures]}")
        return futures

    @cached_property
    def strategies(self) -> dict[ibi.Future, list[str]]:
        """
        Dict of future contracts with corresponding lists of
        strategies for those contracts excluding strategies that
        shouldn't be rolled.  Future will appear in the dict only if
        at least one strategy has active position in it.
        """
        strategies = {
            fut: [
                i
                for i in strategy_list
                if (i not in self.excluded_strategies) and strategy_list
            ]
            for fut, strategy_list in self.sm.strategy.strategies_by_contract().items()
            if isinstance(fut, ibi.Future)
        }

        debug_string = {
            fut.localSymbol: strategy_list for fut, strategy_list in strategies.items()
        }
        log.debug(f"strategies: {debug_string}")

        return strategies

    @cached_property
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
        positions = {
            fut: sum([self.sm.strategy[name].position for name in strategy_names])
            for fut, strategy_names in self.strategies.items()
        }
        debug_string = {
            fut.localSymbol: position for fut, position in positions.items()
        }
        log.debug(f"positions: {debug_string}")
        return positions

    @cached_property
    def contracts_to_roll(self) -> set:
        """
        Set of futures that need to be rolled, i.e. these are the futures
        we have positions for, but they're not active contracts any
        more and they are not for strategies that we explicitly
        excluded from rolling.

        All previous properties are intermediate steps to get this one
        piece of information.
        """
        ctr = set(self.positions) - self.futures
        log.debug(f"contracts to roll: {[c.localSymbol for c in ctr]}")
        return ctr

    def positions_by_strategy_for_contract(
        self, contract: ibi.Future
    ) -> dict[str, float]:
        """
        Given an expiring contract, return a dict of strategies with
        their rollable holdings.  It's not necessarily number of
        contracts to be rolled, which is to be determined by other
        methods, it's a number of contracts that should be considered
        for rolling.
        """
        return {
            strategy_str: position
            for strategy_str in self.strategies[contract]
            if (position := self.sm.strategy[strategy_str].position)
        }

    def active_strategies(self, contract: ibi.Future) -> list[str]:
        return [
            strategy_str
            for strategy_str in self.strategies[contract]
            if self.sm.strategy[strategy_str].position
        ]

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
        else:
            log.debug("No contracts to roll.")

        for old_contract in contracts:
            new_contract = self.match_old_to_new_future(old_contract)
            log.debug(
                f"new contract found: {new_contract.localSymbol} for: "
                f"{old_contract.localSymbol}"
            )
            if new_contract.conId == old_contract.conId:
                log.error(f"Abandoning roll, no replacement found: {old_contract}")
            elif not Atom.contract_details[new_contract].is_open():
                log.error(f"Abandoning roll, {new_contract} is not trading now.")
            else:
                self.execute(old_contract, new_contract)

    def execute(self, old_contract: ibi.Future, new_contract: ibi.Future) -> None:
        """
        Execution is done strategy by strategy to allow for updating
        of strategy records.  Some strategies may cancel each other,
        in which case effort will be made to limit trading.

        Strategy records are updated after trade is filled (as
        usual), so is updating of resting stop and take-profit orders.
        """

        # positions for expiring contract that are not excluded from rolling
        total = self.positions[old_contract]
        strategies_to_trade = self.figure_out_strategies_to_trade(old_contract, total)
        strategies_not_to_trade = list(
            set(self.active_strategies(old_contract)) - set(strategies_to_trade)
        )
        log.debug(
            f"Roll {old_contract.localSymbol} -> {new_contract.localSymbol}; total "
            f"position for: {old_contract.localSymbol}: {total}, "
            f"{strategies_to_trade=} {strategies_not_to_trade=}"
        )

        # contract traded will be a `bag` don't use `new_contract` as key
        # because it will not be accessible in callback
        self._trade_generator[new_contract.conId] = self.handle_strategies_to_trade(
            strategies_to_trade, old_contract, new_contract
        )

        try:
            # only initializing the generator, subsequent `next`s in callback
            next(self._trade_generator[new_contract.conId])
        except StopIteration:
            log.error(
                f"Newly created generator for {new_contract} " f"does not exist, WTF?"
            )
            del self._trade_generator[new_contract.conId]

        self.handle_strategies_not_to_trade(
            strategies_not_to_trade, old_contract, new_contract
        )

    def handle_strategies_to_trade(
        self, strategies: list[str], old_contract, new_contract
    ) -> Generator[None, None, None]:
        # THIS IS A GENERATOR!!!
        for strategy_str in strategies:
            strategy = self.sm.strategy[strategy_str]
            size = strategy.position
            trade = self.trade(strategy_str, strategy, old_contract, new_contract, size)
            # if we're changing orders and records connected to a strategy
            # that we trade, make those changes when and if the order is filled
            # `self.adjust_records_and_orders_for_strategy` doesn't accept `Trade`
            # so don't just wrap it in `partial`!
            assert trade
            trade.filledEvent += (
                lambda trade: self.adjust_records_and_orders_for_strategy(
                    strategy_str,
                    strategy,
                    old_contract,
                    new_contract,
                )
            )
            # generate commission report so that blotter record is created
            # trade.filledEvent += lambda trade: self.controller.onCommissionReport(
            #     trade, trade.fills[-1], trade.fills[-1].commissionReport
            # )
            # # this just calls `next` on the generator
            trade.filledEvent += self.trade_callback
            # don't issue next order until this one is done
            log.debug(f"{strategy_str} will yield")
            yield

    def handle_strategies_not_to_trade(
        self, strategies: list[str], old_contract, new_contract
    ) -> None:
        for strategy_str in strategies:
            strategy = self.sm.strategy[strategy_str]
            log.debug(f"No roll trade for {strategy_str}")
            self.adjust_records_and_orders_for_strategy(
                strategy_str, strategy, old_contract, new_contract
            )

    def trade_callback(self, trade: ibi.Trade):
        """
        Callback for `ibi.Bag` contract, which rolled `ibi.Future`.

        The purpose of this function is to call roll execution for next
        strategy.

        .. note:: `trade.contract` is a `ibi.Bag` and not `ibi.Future`
        """
        # contract is a BAG (which has comboLegs with actual contacts)
        log.debug(f"Roll trade filled for: {trade.contract}")
        # `comboLegs` is the attribute to get contracts (list)
        new_contract = trade.contract.comboLegs[-1]
        try:
            gen = self._trade_generator.get(new_contract.conId)
            log.debug(f"Generator obtained, calling next on {trade.contract.symbol}")
            assert gen
            next(gen)
        except StopIteration:
            del self._trade_generator[new_contract.conId]
            log.debug(f"StopIteration on {trade.contract.symbol}. No more trading.")
        except AssertionError:
            log.critical(f"Wrong contract: {new_contract}")
        except Exception as e:
            log.exception(e)

    def adjust_records_and_orders_for_strategy(
        self,
        strategy_str: str,
        strategy: Strategy,
        old_contract: ibi.Future,
        new_contract: ibi.Future,
    ) -> None:
        log.debug(
            f"Post-roll adjustment for {strategy_str} contract update from "
            f"{old_contract.localSymbol} to {new_contract.localSymbol}"
        )
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
        may mean that opposing orders are issued.  Therefore, orders
        are sent sequentially, each new order is sent only after
        previous one has been filled.

        If we're trading and changing resting orders (i.e. stop-losses
        or take profits) for the same strategy, order adjustments are
        done only after the trade is executed in order to avoid a
        situation where the order rolling position would be rejected,
        but resting stop-losses would be rolled.
        """
        log.debug(f"Figuring out strategies to trade for: {contract.localSymbol}")
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
        log.debug(f"Roll strategies to trade on {contract.localSymbol}: {strategies}")
        return strategies

    @staticmethod
    def strategy_search(
        total_position: float, positions: dict[str, float], descending: bool = False
    ) -> list[str]:
        position_sign = misc.sign(total_position)
        accumulator = 0.0
        acc_strategies = []
        for strategy_str, position in dict(
            sorted(positions.items(), key=lambda i: abs(i[1]), reverse=descending)
        ).items():
            # rolling this will solve the problem
            if position == total_position:
                return [strategy_str]
            if (misc.sign(position) == position_sign) and (
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
    ) -> ibi.Trade | None:
        combo = self.make_combo(old_contract, new_contract)
        try:
            log.debug(f"position_id from strategy: {strategy.position_id}")
        except Exception:
            log.error("Didn't get position id from strategy.")
        try:
            position_id = strategy["params"]["open"]["position_id"]
        except KeyError as e:
            log.error(e)
            position_id = None
        params = {
            "from_to_roll": f"{old_contract.localSymbol} -> {new_contract.localSymbol}",
            "old": old_contract,
            "new": new_contract,
            "contract": combo,
            "position_id": position_id,
        }
        strategy["params"]["future-roll"] = params
        order = ibi.MarketOrder("BUY" if size > 0 else "SELL", abs(size))
        return self.controller.trade(
            strategy_str, combo, order, "FUTURE-ROLL", strategy
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
        strategy.active_contract = new_contract  # this never was called
        log.debug(
            f"{strategy_str} adjusted to new contract: {new_contract.localSymbol}"
        )

    def adjust_strategy_orders(
        self,
        strategy_str: str,
        strategy: Strategy,
        old_contract: ibi.Future,
        new_contract: ibi.Future,
    ) -> None:
        log.debug(f"Will adjust resting orders for: {strategy_str}")
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
        assert new_trade
        log.debug(
            f"New roll order: {(new_trade.order.orderId, new_trade.order.permId)} "
            f"{new_trade.contract.localSymbol} {strategy_str} {oi.action}"
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(controller={self.controller}, "
            f"excluded_strategies={self.excluded_strategies})"
        )
