"""Controller-owned futures position and resting-order rolling."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Generator, Mapping
from dataclasses import replace
from datetime import datetime, timezone
from functools import cached_property, partial
from typing import TYPE_CHECKING

import ib_insync as ibi

from haymaker import misc
from haymaker.async_wrappers import create_background_task
from haymaker.book import OrderInfo, PositionState
from haymaker.components.messages import StandardOrderRole

if TYPE_CHECKING:
    from .controller import Controller

log = logging.getLogger(__name__)


class FutureRoller:
    """Roll held futures outside the current ACTIVE/NEXT contract set."""

    def __init__(
        self,
        controller: Controller,
        future_roll_policies: Mapping[str, bool] | None = None,
    ) -> None:
        self.controller = controller
        self.book = controller.book
        self.future_roll_policies = dict(future_roll_policies or {})
        self._trade_generators: dict[int, Generator[None, None, None]] = {}

    @cached_property
    def futures(self) -> set[ibi.Future]:
        """Return every selector's current ACTIVE and NEXT Future."""

        return {
            contract
            for contract in self.controller.contract_registry.current_contracts
            if isinstance(contract, ibi.Future)
        }

    @cached_property
    def sources(self) -> dict[ibi.Future, list[str]]:
        """Group roll-enabled non-flat PositionStates by held Future."""

        grouped: dict[ibi.Future, list[str]] = {}
        for source_key, state in self.book.position_states().items():
            if not state.quantity or not isinstance(state.contract, ibi.Future):
                continue
            if not self.future_roll_policies.get(source_key, True):
                continue
            grouped.setdefault(state.contract, []).append(source_key)
        undeclared = sorted(
            source
            for sources in grouped.values()
            for source in sources
            if source not in self.future_roll_policies
        )
        if undeclared:
            log.warning(
                "Automatic futures-roll policy undeclared for %s; enabled.",
                undeclared,
            )
        return grouped

    @cached_property
    def positions(self) -> dict[ibi.Future, float]:
        """Return aggregate roll-enabled logical quantity by held Future."""

        positions: dict[ibi.Future, float] = {}
        for contract, sources in self.sources.items():
            quantity = 0.0
            for source in sources:
                state = self.book.position_state(source)
                if state is not None:
                    quantity += state.quantity
            positions[contract] = quantity
        return positions

    @cached_property
    def contracts_to_roll(self) -> set[ibi.Future]:
        """Return held Futures outside both current ACTIVE and NEXT."""

        return set(self.positions) - self.futures

    def match_old_to_new_future(self, old: ibi.Future) -> ibi.Future:
        """Return the matching current Future, preferring selector ACTIVE."""

        candidates = [
            future
            for future in self.futures
            if future.symbol == old.symbol
            and future.exchange == old.exchange
            and future.multiplier == old.multiplier
            and future.conId != old.conId
        ]
        if not candidates:
            log.error("No replacement contract for expiring %s", old)
            return old
        active_contracts = {
            selector.active_contract
            for selector in self.controller.contract_registry.selectors
        }
        return next(
            (
                candidate
                for candidate in candidates
                if candidate in active_contracts
            ),
            candidates[0],
        )

    def roll(self) -> None:
        """Start rolling every held Contract that left ACTIVE/NEXT."""

        if not self.contracts_to_roll:
            log.debug("No futures contracts require rolling.")
            return
        log.warning(
            "Contracts will be rolled: %s",
            [contract.localSymbol for contract in self.contracts_to_roll],
        )
        for old in self.contracts_to_roll:
            new = self.match_old_to_new_future(old)
            if new.conId == old.conId:
                continue
            details = self.controller.contract_registry.details.get(new)
            if details is not None and not details.is_open():
                log.error("Abandoning roll while replacement is closed: %s", new)
                continue
            self.execute(old, new)

    def execute(self, old: ibi.Future, new: ibi.Future) -> None:
        """Trade the smallest attributable source set and adjust the rest."""

        total = self.positions[old]
        sources_to_trade = self.figure_out_sources_to_trade(old, total)
        sources_without_trade = list(
            set(self.sources[old]) - set(sources_to_trade)
        )
        generator = self._trade_sources(sources_to_trade, old, new)
        self._trade_generators[new.conId] = generator
        try:
            next(generator)
        except StopIteration:
            self._trade_generators.pop(new.conId, None)
        create_background_task(
            self._adjust_sources_without_trade(sources_without_trade, old, new),
            name="future-roll-record-adjustment",
        )

    def _trade_sources(
        self,
        sources: list[str],
        old: ibi.Future,
        new: ibi.Future,
    ) -> Generator[None, None, None]:
        for source_key in sources:
            state = self.book.position_state(source_key)
            assert state is not None
            trade = self._trade(source_key, state, old, new)
            if trade is None:
                log.error("Roll submission failed for source %s", source_key)
                continue
            trade.filledEvent += (
                lambda completed, key=source_key: self._adjust_source(
                    key,
                    old,
                    new,
                    completed.orderStatus.avgFillPrice,
                )
            )
            trade.filledEvent += self._trade_callback
            yield

    async def _adjust_sources_without_trade(
        self,
        sources: list[str],
        old: ibi.Future,
        new: ibi.Future,
    ) -> None:
        combo = self.make_combo(old, new)
        if not sources:
            return
        price = await self.request_data(combo, self.controller.ib)
        try:
            if price == price:
                for source_key in sources:
                    self._adjust_source(source_key, old, new, price)
            else:
                log.error("Failed to obtain roll adjustment for %s", combo.symbol)
        finally:
            self.controller.ib.cancelMktData(combo)

    @staticmethod
    async def request_data(contract: ibi.Contract, ib: ibi.IB) -> float:
        """Return a short-lived combo market price or NaN on timeout."""

        ticker = ib.reqMktData(contract, "221")
        for _ in range(500):
            price = ticker.marketPrice()
            if price == price:
                return price
            await asyncio.sleep(0.01)
        return float("nan")

    def _trade_callback(self, trade: ibi.Trade) -> None:
        new_leg = trade.contract.comboLegs[-1]
        generator = self._trade_generators.get(new_leg.conId)
        if generator is None:
            return
        try:
            next(generator)
        except StopIteration:
            self._trade_generators.pop(new_leg.conId, None)

    def _adjust_source(
        self,
        source_key: str,
        old: ibi.Future,
        new: ibi.Future,
        fill_price: float,
    ) -> None:
        state = self.book.position_state(source_key)
        if state is None:
            return
        self.book.update_position(
            replace(
                state,
                contract=new,
                updated_at=datetime.now(timezone.utc),
            )
        )
        self._adjust_source_orders(source_key, new, fill_price)
        log.debug(
            "Rolled source %s from %s to %s",
            source_key,
            old.localSymbol,
            new.localSymbol,
        )

    def figure_out_sources_to_trade(
        self, contract: ibi.Future, total_position: float
    ) -> list[str]:
        """Select an attributable source set that nets to broker roll size."""

        positions = {
            source: state.quantity
            for source in self.sources[contract]
            if (state := self.book.position_state(source)) is not None
            and state.quantity
        }
        cancelling = sum(positions.values()) != sum(
            abs(position) for position in positions.values()
        )
        if not cancelling:
            return list(positions)
        ascending = self.source_search(total_position, positions)
        if len(ascending) == 1:
            return ascending
        descending = self.source_search(
            total_position, positions, descending=True
        )
        if ascending and descending:
            return min((ascending, descending), key=len)
        return ascending or descending or list(positions)

    @staticmethod
    def source_search(
        total_position: float,
        positions: dict[str, float],
        descending: bool = False,
    ) -> list[str]:
        """Find a same-direction subset that sums to total position."""

        direction = misc.sign(total_position)
        accumulated = 0.0
        sources: list[str] = []
        for source, position in sorted(
            positions.items(), key=lambda item: abs(item[1]), reverse=descending
        ):
            if position == total_position:
                return [source]
            if misc.sign(position) == direction and abs(
                accumulated + position
            ) <= abs(total_position):
                accumulated += position
                sources.append(source)
                if accumulated == total_position:
                    return sources
        return []

    def _trade(
        self,
        source_key: str,
        state: PositionState,
        old: ibi.Future,
        new: ibi.Future,
    ) -> ibi.Trade | None:
        combo = self.make_combo(old, new)
        params = {
            "from_to_roll": f"{old.localSymbol} -> {new.localSymbol}",
            "old": old,
            "new": new,
        }
        return self.controller.trade(
            combo,
            ibi.MarketOrder(
                "BUY" if state.quantity > 0 else "SELL",
                abs(state.quantity),
            ),
            role=StandardOrderRole.ROLL,
            execution_model_name=state.execution_model_name,
            source_key=source_key,
            position_id=state.position_id,
            params=params,
        )

    @staticmethod
    def make_combo(old: ibi.Future, new: ibi.Future) -> ibi.Bag:
        """Construct the two-leg spread used to roll one Future."""

        return ibi.Bag(
            symbol=new.symbol,
            exchange=new.exchange,
            currency=new.currency,
            multiplier=new.multiplier,
            comboLegs=[
                ibi.ComboLeg(
                    conId=old.conId,
                    ratio=1,
                    action="SELL",
                    exchange=old.exchange,
                ),
                ibi.ComboLeg(
                    conId=new.conId,
                    ratio=1,
                    action="BUY",
                    exchange=new.exchange,
                ),
            ],
        )

    def _adjust_source_orders(
        self, source_key: str, new: ibi.Future, fill_price: float
    ) -> None:
        for info in self.book.active_orders(source_key=source_key):
            if info.role == StandardOrderRole.ROLL:
                continue
            info.trade.cancelledEvent += partial(
                self._issue_replacement_order,
                info=info,
                new_contract=new,
                fill_price=fill_price,
            )
            self.controller.cancel(info.trade)

    def _issue_replacement_order(
        self,
        cancelled_trade: ibi.Trade,
        *,
        info: OrderInfo,
        new_contract: ibi.Future,
        fill_price: float,
    ) -> None:
        options = ibi.util.dataclassNonDefaults(cancelled_trade.order)
        for key in ("orderId", "permId", "softDollarTier", "clientId"):
            options.pop(key, None)
        if options.get("orderType") == "FIX PEGGED":
            options["orderType"] = "TRAIL"
            options["auxPrice"] = misc.round_tick(
                (
                    info.params.get("trail_multiple")
                    or info.params.get("adjusted_multiple")
                )
                * info.params["sl_points"],
                info.params["min_tick"],
            )
        for field_name in ("lmtPrice", "trailStopPrice", "adjustedStopPrice"):
            if options.get(field_name):
                options[field_name] += fill_price
        self.controller.trade(
            new_contract,
            ibi.Order(**options),
            role=info.role,
            execution_model_name=info.execution_model_name,
            source_key=info.source_key,
            position_id=info.position_id,
            params=info.params,
        )

    def __repr__(self) -> str:
        return (
            f"FutureRoller(controller={self.controller!r}, "
            f"future_roll_policies={self.future_roll_policies!r})"
        )
