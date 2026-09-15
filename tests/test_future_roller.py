"""Focused futures-roll discovery, execution, and recovery tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from typing import Any

import ib_insync as ibi
import pytest

from haymaker.book import (
    FutureRollMode,
    FutureRollStage,
    OrderInfo,
    PositionState,
    RollParticipant,
    RollState,
    TargetState,
)
from haymaker.components import (
    BracketFutureRollExecutor,
    DirectFutureRollExecutor,
    FutureRollExecutor,
    FutureRollPolicy,
    PastToActiveRollPolicy,
    RollDecision,
    RollHolding,
    StandardOrderRole,
)
from haymaker.controller.future_roller import FutureRoller
from haymaker.contract_selector import FutureSelector


def future(con_id: int, local_symbol: str) -> ibi.Future:
    """Build one qualified natural-gas Future for roll tests."""

    return ibi.Future(
        conId=con_id,
        symbol="NG",
        exchange="NYMEX",
        currency="USD",
        multiplier="10000",
        localSymbol=local_symbol,
    )


class FakeRegistry:
    """Expose explicit series ownership and ACTIVE/NEXT selection."""

    def __init__(
        self,
        old: ibi.Future,
        active: ibi.Future,
        next_contract: ibi.Future,
    ) -> None:
        self._contracts = {contract.conId for contract in (old, active, next_contract)}
        self.active = active
        self.next_contract = next_contract
        self.chain = list({c.conId: c for c in (old, active, next_contract)}.values())
        for contract, offset in ((old, -30), (active, 30), (next_contract, 90)):
            contract.lastTradeDateOrContractMonth = (
                datetime.now(timezone.utc) + timedelta(days=offset)
            ).strftime("%Y%m%d")

    def selector_for_series(self, series_key: str) -> FutureSelector:
        """Expose a real date-aware chain without replacing qualified objects."""
        return FutureSelector.from_contracts(self.chain)

    def series_key(self, contract: ibi.Future) -> str:
        """Return the registered series for known concrete expiries."""

        if contract.conId not in self._contracts:
            raise KeyError(contract.conId)
        return "ng-series"

    def active_for_series(self, series_key: str) -> ibi.Future:
        """Return the selected ACTIVE expiry."""

        assert series_key == "ng-series"
        return self.active

    def current_for_series(self, series_key: str) -> tuple[ibi.Future, ibi.Future]:
        """Return the accepted ACTIVE/NEXT expiry pair."""

        assert series_key == "ng-series"
        return self.active, self.next_contract

    def get_details(self, contract: ibi.Contract) -> None:
        """Disable market-hours filtering in focused tests."""

        return None


class FakeTrader:
    """Return the test's authoritative broker quantity by conId."""

    def __init__(self, positions: dict[int, float]) -> None:
        self.positions = positions

    def position_for_contract(self, contract: ibi.Contract) -> float:
        """Return one concrete broker position."""

        return self.positions.get(contract.conId, 0.0)


class FakeController:
    """Register roll orders in Book while recording broker side effects."""

    def __init__(self, book, registry: FakeRegistry, positions: dict[int, float]):
        self.book = book
        self.contract_registry = registry
        self.trader = FakeTrader(positions)
        self.future_roll_policies: dict[str, bool] = {}
        self.ib = SimpleNamespace()
        self.trades: list[ibi.Trade] = []
        self.cancelled: list[ibi.Trade] = []

    def trade(
        self,
        contract: ibi.Contract,
        order: ibi.Order,
        *,
        role: str,
        execution_model_name: str,
        source_key: str | None = None,
        position_id: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> ibi.Trade:
        """Create an active Trade and persist its complete attribution."""

        order.orderId = len(self.trades) + 100
        order.permId = order.orderId + 1000
        trade = ibi.Trade(
            contract=contract,
            order=order,
            orderStatus=ibi.OrderStatus(
                orderId=order.orderId,
                status=ibi.OrderStatus.Submitted,
                remaining=order.totalQuantity,
            ),
        )
        self.trades.append(trade)
        self.book.save_order(
            OrderInfo(
                trade=trade,
                role=role,
                submitted_at=datetime.now(timezone.utc),
                execution_model_name=execution_model_name,
                source_key=source_key,
                position_id=position_id,
                params=params or {},
            )
        )
        return trade

    def cancel(self, trade: ibi.Trade) -> ibi.Trade:
        """Make cancellation immediate while preserving normal callbacks."""

        self.cancelled.append(trade)
        trade.orderStatus.status = ibi.OrderStatus.Cancelled
        trade.cancelledEvent.emit(trade)
        return trade


def make_controller(book, old: ibi.Future, active: ibi.Future, next_: ibi.Future):
    """Build the narrow Controller surface used by the new FutureRoller."""

    return FakeController(
        book,
        FakeRegistry(old, active, next_),
        {old.conId: book.positions.quantity(old)},
    )


def execution_fill(
    trade: ibi.Trade,
    quantity: float,
    exec_id: str,
) -> ibi.Fill:
    """Build one normalized execution for a concrete or combo order."""

    return ibi.Fill(
        contract=trade.contract,
        execution=ibi.Execution(
            execId=exec_id,
            orderId=trade.order.orderId,
            permId=trade.order.permId,
            side="BOT" if trade.order.action == "BUY" else "SLD",
            shares=quantity,
            price=1.25,
            time=datetime.now(timezone.utc),
        ),
        commissionReport=ibi.CommissionReport(execId=exec_id),
        time=datetime.now(timezone.utc),
    )


def apply_fill(book, trade: ibi.Trade, quantity: float, exec_id: str) -> None:
    """Persist one Fill and emit completion only after total quantity fills."""

    fill = execution_fill(trade, quantity, exec_id)
    trade.fills.append(fill)
    trade.orderStatus.filled += quantity
    trade.orderStatus.remaining = max(
        trade.order.totalQuantity - trade.orderStatus.filled,
        0,
    )
    if trade.orderStatus.remaining == 0:
        trade.orderStatus.status = ibi.OrderStatus.Filled
        trade.orderStatus.avgFillPrice = fill.execution.price
    book.apply_fill(trade, fill)
    if trade.orderStatus.status == ibi.OrderStatus.Filled:
        trade.filledEvent.emit(trade)


def persist_direct_position(
    book,
    contract: ibi.Future,
    *,
    quantity: float = 2,
) -> None:
    """Create direct TargetState plus authoritative completed Fill evidence."""

    book.update_target(
        TargetState(
            execution_model_name="serial",
            contract=contract,
            target_quantity=quantity,
            target_created_at=datetime.now(timezone.utc),
        )
    )
    order = ibi.MarketOrder(
        "BUY" if quantity > 0 else "SELL",
        abs(quantity),
        orderId=1 + len(book.orders.query()),
    )
    order.permId = order.orderId + 1000
    trade = ibi.Trade(
        contract=contract,
        order=order,
        orderStatus=ibi.OrderStatus(
            orderId=order.orderId,
            status=ibi.OrderStatus.Submitted,
            remaining=abs(quantity),
        ),
    )
    book.save_order(
        OrderInfo(
            trade=trade,
            role=StandardOrderRole.TARGET_ADJUSTMENT,
            submitted_at=datetime.now(timezone.utc),
            execution_model_name="serial",
        )
    )
    apply_fill(book, trade, abs(quantity), f"entry-{contract.conId}")


def persist_bracket_position(
    book,
    contract: ibi.Future,
    *,
    source_key: str = "alpha",
    quantity: float = 1,
    with_stop: bool = True,
) -> ibi.Trade | None:
    """Create a one-to-one episode and optional active critical stop."""

    book.update_position(
        PositionState(
            source_key=source_key,
            execution_model_name=f"{source_key}-brackets",
            contract=contract,
            quantity=quantity,
            target_quantity=quantity,
            target_created_at=datetime.now(timezone.utc),
            position_id=f"{source_key}-episode",
            blocked_direction=1 if quantity > 0 else -1,
            bracket_inputs={"atr": 0.25},
        )
    )
    if not with_stop:
        return None
    order = ibi.StopOrder(
        "SELL" if quantity > 0 else "BUY",
        abs(quantity),
        stopPrice=2.5,
        orderId=20 + len(book.orders.query()),
    )
    order.permId = order.orderId + 1000
    order.ocaGroup = "old-oca"
    trade = ibi.Trade(
        contract=contract,
        order=order,
        orderStatus=ibi.OrderStatus(
            orderId=order.orderId,
            status=ibi.OrderStatus.Submitted,
            remaining=abs(quantity),
        ),
    )
    book.save_order(
        OrderInfo(
            trade=trade,
            role=StandardOrderRole.STOP_LOSS,
            submitted_at=datetime.now(timezone.utc),
            execution_model_name=f"{source_key}-brackets",
            source_key=source_key,
            position_id=f"{source_key}-episode",
            params={"atr": 0.25},
        )
    )
    return trade


def test_direct_roll_moves_fill_evidence_and_target_contract(book):
    old = future(1, "NGQ26")
    active = future(2, "NGU26")
    next_ = future(3, "NGV26")
    persist_direct_position(book, old)
    controller = make_controller(book, old, active, next_)
    roller = FutureRoller(controller)
    roller.register_executor(
        FutureRollMode.DIRECT,
        DirectFutureRollExecutor(name="direct-roll"),
    )

    roller.roll()

    roll_trade = controller.trades[-1]
    info = book.orders.by_id(roll_trade.order.orderId)
    assert isinstance(roll_trade.contract, ibi.Bag)
    assert roll_trade.order.action == "BUY"
    assert roll_trade.order.totalQuantity == 2
    assert info.role == StandardOrderRole.ROLL
    assert info.source_key is None
    assert info.execution_model_name == "direct-roll"
    assert book.rolls.for_series("ng-series").stage is FutureRollStage.ROLL_ORDER_ACTIVE

    apply_fill(book, roll_trade, 2, "direct-roll-fill")

    assert book.positions.quantity(old) == 0
    assert book.positions.quantity(active) == 2
    assert book.targets.for_contract(active).contract is active
    assert book.rolls.for_series("ng-series").stage is FutureRollStage.COMPLETE


def test_pending_bracket_allocation_is_rebuilt_for_all_sources(book):
    """A changed offsetting source changes which sources need physical trades."""
    old, active, next_ = [future(i, f"NG{i}") for i in range(1, 4)]
    persist_bracket_position(book, old, source_key="alpha", quantity=2)
    persist_bracket_position(book, old, source_key="beta", quantity=-1)
    controller = make_controller(book, old, active, next_)
    roller = FutureRoller(controller)
    executor = roller.register_executor(FutureRollMode.BRACKET)
    state = executor.create_state("ng-series", old, active, executor.holdings())
    book.update_roll(state)
    book.update_position(
        replace(book.positions.for_source("beta"), quantity=0, position_id=None)
    )
    refreshed = executor._refresh_pending(state)
    assert [
        (p.source_key, p.quantity, p.requires_trade) for p in refreshed.participants
    ] == [
        ("alpha", 2, True),
        ("beta", 0, False),
    ]
    assert RollState.decode(refreshed.encode()) == refreshed


def test_completed_nonphysical_source_offset_survives_pending_refresh(book):
    """Do not turn the second half of a net-zero logical roll into a real BAG."""
    old, active, next_ = [future(i, f"NG{i}") for i in range(1, 4)]
    persist_bracket_position(book, active, source_key="alpha", quantity=1)
    persist_bracket_position(book, old, source_key="beta", quantity=-1)
    controller = make_controller(book, old, active, next_)
    executor = FutureRoller(controller).register_executor(FutureRollMode.BRACKET)
    state = RollState(
        series_key="ng-series",
        mode=FutureRollMode.BRACKET,
        executor_name=executor.name,
        old_contract=old,
        new_contract=active,
        participant_index=1,
        participants=(
            RollParticipant(
                source_key="alpha",
                position_id="alpha-episode",
                execution_model_name="alpha-brackets",
                quantity=1,
                requires_trade=False,
            ),
            RollParticipant(
                source_key="beta",
                position_id="beta-episode",
                execution_model_name="beta-brackets",
                quantity=-1,
                requires_trade=False,
            ),
        ),
    )
    book.update_roll(state)
    refreshed = executor._refresh_pending(state)
    assert not refreshed.current_participant.requires_trade
    book.update_position(replace(book.positions.for_source("beta"), quantity=-2))
    assert executor._refresh_pending(refreshed) is None
    assert book.rolls.for_series("ng-series").stage is FutureRollStage.BLOCKED
    assert not controller.trades


def test_default_policy_retains_every_eligible_expiry(book):
    """The third expiry is not stale merely because it is neither ACTIVE nor NEXT."""
    old, active, next_, later = [future(i, f"NG{i}") for i in range(1, 5)]
    registry = FakeRegistry(old, active, next_)
    later.lastTradeDateOrContractMonth = (
        datetime.now(timezone.utc) + timedelta(days=180)
    ).strftime("%Y%m%d")
    registry.chain.append(later)
    registry._contracts.add(later.conId)
    for contract in (active, next_, later):
        persist_direct_position(book, contract)
    controller = FakeController(book, registry, {})
    roller = FutureRoller(controller)
    roller.register_executor(FutureRollMode.DIRECT)
    roller.roll()
    assert controller.trades == []
    assert book.rolls.all() == ()


class FixedSuccessorPolicy(FutureRollPolicy):
    """Keep a fixed schedule condition true to exercise occurrence deduplication."""

    def plan(self, holding, selector, *, now):
        chain = [wrapper.contract for wrapper in selector.all_contracts]
        index = next(
            i for i, c in enumerate(chain) if c.conId == holding.contract.conId
        )
        return RollDecision(destination=chain[index + 1], occurrence="fixed-schedule")


def test_fixed_schedule_does_not_cascade_after_completion_or_recovery(book):
    """A fresh coordinator honors Book's completed occurrence, not old callbacks."""
    old, active, next_ = [future(i, f"NG{i}") for i in range(1, 4)]
    persist_direct_position(book, active)
    controller = make_controller(book, old, active, next_)
    controller.trader.positions[active.conId] = 2
    roller = FutureRoller(controller)
    roller.register_executor(FutureRollMode.DIRECT)
    roller.register_policy(FixedSuccessorPolicy(), model_name="serial")
    roller.roll()
    state = book.rolls.for_series("ng-series")
    assert state.old_contract == active
    assert state.new_contract == next_
    # A successor beyond NEXT lets the repeated condition propose another roll.
    later = future(4, "NG4")
    later.lastTradeDateOrContractMonth = (
        datetime.now(timezone.utc) + timedelta(days=180)
    ).strftime("%Y%m%d")
    controller.contract_registry.chain.append(later)
    controller.contract_registry._contracts.add(4)
    apply_fill(book, controller.trades[-1], 2, "scheduled-fill")
    completed = book.rolls.for_series("ng-series")
    assert len(completed.completed_occurrences) == 2
    assert RollState.decode(completed.encode()) == completed
    fresh = FutureRoller(controller)
    fresh.register_executor(FutureRollMode.DIRECT)
    fresh.register_policy(FixedSuccessorPolicy(), model_name="serial")
    fresh.roll()
    assert len(controller.trades) == 1


def test_custom_check_refreshes_selector_date_without_mutating_graph(book):
    """Custom early checks use their time, not the last workload start date."""
    old, active, next_ = [future(i, f"NG{i}") for i in range(1, 4)]
    registry = FakeRegistry(old, active, next_)
    selector = registry.selector_for_series("ng-series")
    holding = RollHolding(contract=active, quantity=1, execution_model_name="serial")
    policy = PastToActiveRollPolicy()
    now = datetime.now(timezone.utc)
    assert policy.plan(holding, selector, now=now) is None
    future_time = now + timedelta(days=45)
    refreshed = replace(selector, today=future_time.replace(tzinfo=None))
    assert policy.plan(holding, refreshed, now=future_time).destination == next_
    assert selector.active_contract == active


@pytest.mark.parametrize("result", ["invalid", "same", "unknown"])
def test_invalid_policy_plan_has_no_broker_or_state_side_effect(book, result):
    """Bad endpoints fail before new roll state or orders are created."""
    old, active, next_ = [future(i, f"NG{i}") for i in range(1, 4)]
    persist_direct_position(book, old)
    controller = make_controller(book, old, active, next_)
    roller = FutureRoller(controller)
    roller.register_executor(FutureRollMode.DIRECT)

    class BadPolicy(FutureRollPolicy):
        """Return one deliberately invalid user decision."""

        def plan(self, holding, selector, *, now):
            if result == "invalid":
                return "not a RollDecision"
            return RollDecision(
                destination=old if result == "same" else future(99, "NG99")
            )

    roller.register_policy(BadPolicy(), model_name="serial")
    with pytest.raises((TypeError, ValueError, KeyError)):
        roller.roll()
    assert not controller.trades
    assert not book.rolls.all()


def test_direct_roll_waits_for_active_target_adjustment(book):
    old = future(1, "NGQ26")
    active = future(2, "NGU26")
    next_ = future(3, "NGV26")
    persist_direct_position(book, old)
    controller = make_controller(book, old, active, next_)
    active_adjustment = controller.trade(
        old,
        ibi.MarketOrder("BUY", 1),
        role=StandardOrderRole.TARGET_ADJUSTMENT,
        execution_model_name="serial",
    )
    roller = FutureRoller(controller)
    roller.register_executor(FutureRollMode.DIRECT)

    roller.roll()

    assert (
        book.rolls.for_series("ng-series").stage
        is FutureRollStage.WAITING_FOR_ACTIVE_WORK
    )
    assert len(controller.trades) == 1

    controller.cancel(active_adjustment)

    assert len(controller.trades) == 2
    assert book.orders.by_id(controller.trades[-1].order.orderId).role == "ROLL"


def test_direct_transfer_recovery_after_only_first_target_was_written(
    book, monkeypatch
):
    """A crash between absolute writes cannot add to the destination twice."""
    old, active, next_ = [future(i, f"NG{i}") for i in range(1, 4)]
    persist_direct_position(book, old)
    persist_direct_position(book, active, quantity=3)
    controller = make_controller(book, old, active, next_)
    roller = FutureRoller(controller)
    executor = roller.register_executor(FutureRollMode.DIRECT)
    roller.roll()
    trade = controller.trades[-1]
    # Suppress the callback, then reproduce persisted ROLL_FILLED recovery.
    trade.filledEvent.clear()
    apply_fill(book, trade, 2, "transfer-crash")
    state = replace(
        book.rolls.for_series("ng-series"), stage=FutureRollStage.ROLL_FILLED
    )
    book.update_roll(state)
    original = book.update_target

    def interrupted(target):
        """Fail the destination write after successfully writing the old zero."""
        if target.contract == active:
            raise RuntimeError("simulated process interruption")
        return original(target)

    monkeypatch.setattr(book, "update_target", interrupted)
    with pytest.raises(RuntimeError, match="simulated"):
        executor.advance(state)
    assert book.targets.for_contract(old).target_quantity == 0
    assert book.targets.for_contract(active).target_quantity == 3
    monkeypatch.setattr(book, "update_target", original)
    # Round-trip the journal to discard any reliance on the old Python object.
    book.update_roll(RollState.decode(book.rolls.for_series("ng-series").encode()))
    roller.recover()
    assert book.targets.for_contract(active).target_quantity == 5
    roller.recover()
    assert book.targets.for_contract(active).target_quantity == 5


def test_partial_bracket_roll_fill_projects_both_concrete_positions(book, monkeypatch):
    old = future(1, "NGQ26")
    active = future(2, "NGU26")
    next_ = future(3, "NGV26")
    persist_bracket_position(book, old, quantity=2)
    controller = make_controller(book, old, active, next_)
    roller = FutureRoller(controller)
    roller.register_executor(FutureRollMode.BRACKET)
    roller.set_policies({"alpha": True})

    roller.roll()
    roll_trade = controller.trades[-1]
    apply_fill(book, roll_trade, 1, "partial-roll")

    def no_query_reconstruction(*args):
        """An in-flight roll must already have updated the shared balances."""
        raise AssertionError("position query reconstructed roll fills")

    monkeypatch.setattr(book.positions, "_roll_contribution", no_query_reconstruction)
    assert book.positions.quantity(old) == 1
    assert book.positions.quantity(active) == 1
    assert book.positions.for_source("alpha").contract is old
    assert book.rolls.for_series("ng-series").stage is FutureRollStage.ROLL_ORDER_ACTIVE


def test_bracket_roll_projection_ignores_intermediate_logical_contract_moves(book):
    old = future(1, "NGQ26")
    active = future(2, "NGU26")
    created_at = datetime.now(timezone.utc)
    for source_key, quantity in (("long-a", 1), ("long-b", 1), ("short", -1)):
        book.update_position(
            PositionState(
                source_key=source_key,
                execution_model_name="brackets",
                contract=active,
                quantity=quantity,
                target_quantity=quantity,
                target_created_at=created_at,
                position_id=f"{source_key}-episode",
            )
        )
    state = book.update_roll(
        RollState(
            series_key="ng-series",
            mode=FutureRollMode.BRACKET,
            executor_name="bracket-roll",
            old_contract=old,
            new_contract=active,
            participants=tuple(
                RollParticipant(
                    execution_model_name="brackets",
                    source_key=source_key,
                    quantity=quantity,
                    position_id=f"{source_key}-episode",
                    requires_trade=source_key == "long-a",
                )
                for source_key, quantity in (
                    ("long-a", 1),
                    ("long-b", 1),
                    ("short", -1),
                )
            ),
            stage=FutureRollStage.ROLL_ORDER_ACTIVE,
            created_at=created_at,
            updated_at=created_at,
        )
    )
    controller = make_controller(book, old, active, future(3, "NGV26"))
    roll_trade = controller.trade(
        FutureRollExecutor.make_combo(old, active),
        ibi.MarketOrder("BUY", 1),
        role=StandardOrderRole.ROLL,
        execution_model_name=state.executor_name,
        source_key="long-a",
        position_id="long-a-episode",
        params={
            "roll_state_key": state.series_key,
            "old_contract": old,
            "new_contract": active,
        },
    )

    apply_fill(book, roll_trade, 1, "offset-roll-fill")

    assert book.positions.quantity(old) == 0
    assert book.positions.quantity(active) == 1
    assert book.positions.by_contract() == {active: 1}


def test_bracket_roll_reinstalls_stop_before_completion(book):
    old = future(1, "NGQ26")
    active = future(2, "NGU26")
    next_ = future(3, "NGV26")
    old_stop = persist_bracket_position(book, old)
    controller = make_controller(book, old, active, next_)
    roller = FutureRoller(controller)
    roller.register_executor(
        FutureRollMode.BRACKET,
        BracketFutureRollExecutor(name="bracket-roll"),
    )
    roller.set_policies({"alpha": True})

    roller.roll()
    roll_trade = controller.trades[-1]
    apply_fill(book, roll_trade, 1, "bracket-roll-fill")

    state = book.positions.for_source("alpha")
    assert state.contract is active
    assert state.quantity == 1
    assert state.position_id == "alpha-episode"
    assert old_stop in controller.cancelled
    replacement_stops = book.orders.active(
        source_key="alpha",
        contract=active,
        role=StandardOrderRole.STOP_LOSS,
    )
    assert len(replacement_stops) == 1
    assert replacement_stops[0].position_id == "alpha-episode"
    assert book.rolls.for_series("ng-series").stage is FutureRollStage.COMPLETE


def test_bracket_roll_recovers_protection_cancellation_stage(book):
    old = future(1, "NGQ26")
    active = future(2, "NGU26")
    next_ = future(3, "NGV26")
    old_stop = persist_bracket_position(book, old)
    position = book.positions.for_source("alpha")
    book.update_position(replace(position, contract=active))
    created_at = datetime.now(timezone.utc)
    book.update_roll(
        RollState(
            series_key="ng-series",
            mode=FutureRollMode.BRACKET,
            executor_name="bracket-roll",
            old_contract=old,
            new_contract=active,
            participants=(
                RollParticipant(
                    execution_model_name="alpha-brackets",
                    source_key="alpha",
                    quantity=1,
                    position_id="alpha-episode",
                    requires_trade=False,
                ),
            ),
            stage=FutureRollStage.CANCELLING_PROTECTION,
            old_protection_order_ids=(old_stop.order.orderId,),
            reference_price=1.25,
            created_at=created_at,
            updated_at=created_at,
        )
    )
    controller = make_controller(book, old, active, next_)
    roller = FutureRoller(controller)
    roller.register_executor(
        FutureRollMode.BRACKET,
        BracketFutureRollExecutor(name="bracket-roll"),
    )

    assert roller.recover()

    assert old_stop in controller.cancelled
    assert book.rolls.for_series("ng-series").stage is FutureRollStage.COMPLETE
    replacement = book.orders.active(
        source_key="alpha",
        contract=active,
        role=StandardOrderRole.STOP_LOSS,
    )
    assert len(replacement) == 1


def test_bracket_roll_blocks_if_normalized_fill_evidence_is_missing(book):
    old = future(1, "NGQ26")
    active = future(2, "NGU26")
    persist_bracket_position(book, old)
    created_at = datetime.now(timezone.utc)
    book.update_roll(
        RollState(
            series_key="ng-series",
            mode=FutureRollMode.BRACKET,
            executor_name="bracket-roll",
            old_contract=old,
            new_contract=active,
            participants=(
                RollParticipant(
                    execution_model_name="alpha-brackets",
                    source_key="alpha",
                    quantity=1,
                    position_id="alpha-episode",
                ),
            ),
            stage=FutureRollStage.ROLL_FILLED,
            roll_order_id=99,
            reference_price=1.25,
            created_at=created_at,
            updated_at=created_at,
        )
    )
    missing_fill_trade = ibi.Trade(
        contract=FutureRollExecutor.make_combo(old, active),
        order=ibi.MarketOrder("BUY", 1, orderId=99),
        orderStatus=ibi.OrderStatus(
            orderId=99,
            status=ibi.OrderStatus.Filled,
            filled=1,
            remaining=0,
        ),
    )
    book.save_order(
        OrderInfo(
            trade=missing_fill_trade,
            role=StandardOrderRole.ROLL,
            submitted_at=created_at,
            execution_model_name="bracket-roll",
            source_key="alpha",
            position_id="alpha-episode",
            params={
                "roll_state_key": "ng-series",
                "old_contract": old,
                "new_contract": active,
            },
        )
    )
    controller = make_controller(book, old, active, future(3, "NGV26"))
    roller = FutureRoller(controller)
    roller.register_executor(
        FutureRollMode.BRACKET,
        BracketFutureRollExecutor(name="bracket-roll"),
    )

    assert roller.recover()

    state = book.rolls.for_series("ng-series")
    assert state.stage is FutureRollStage.BLOCKED
    assert (
        state.failure_reason == "Bracket roll lacks complete normalized Fill evidence"
    )
    assert book.positions.for_source("alpha").contract is old


def test_bracket_roll_blocks_if_critical_stop_is_missing(book):
    old = future(1, "NGQ26")
    active = future(2, "NGU26")
    next_ = future(3, "NGV26")
    persist_bracket_position(book, old, with_stop=False)
    controller = make_controller(book, old, active, next_)
    roller = FutureRoller(controller)
    roller.register_executor(FutureRollMode.BRACKET)
    roller.set_policies({"alpha": True})

    roller.roll()
    apply_fill(book, controller.trades[-1], 1, "unprotected-roll")

    state = book.rolls.for_series("ng-series")
    assert state.stage is FutureRollStage.BLOCKED
    assert state.failure_reason == "Rolled bracket position has no critical stop"


def test_bracket_roll_selects_smallest_subset_equal_to_broker_net():
    old = future(1, "NGQ26")
    holdings = (
        RollHolding(
            contract=old,
            quantity=1,
            execution_model_name="brackets",
            source_key="long-a",
        ),
        RollHolding(
            contract=old,
            quantity=1,
            execution_model_name="brackets",
            source_key="long-b",
        ),
        RollHolding(
            contract=old,
            quantity=-1,
            execution_model_name="brackets",
            source_key="short",
        ),
    )

    assert BracketFutureRollExecutor._physical_sources(holdings) == {"long-a"}


def test_executor_registration_reuses_one_mode_and_rejects_conflicts(book):
    old = future(1, "NGQ26")
    active = future(2, "NGU26")
    controller = make_controller(book, old, active, future(3, "NGV26"))
    roller = FutureRoller(controller)

    direct = roller.register_executor(FutureRollMode.DIRECT)

    assert roller.register_executor(FutureRollMode.DIRECT) is direct
    with pytest.raises(ValueError, match="cannot be mixed"):
        roller.register_executor(FutureRollMode.BRACKET)
    with pytest.raises(ValueError, match="different FutureRollExecutor"):
        roller.register_executor(
            FutureRollMode.DIRECT,
            DirectFutureRollExecutor(name="replacement"),
        )


def test_recovery_rebinds_active_roll_and_completes_from_fill(book):
    old = future(1, "NGQ26")
    active = future(2, "NGU26")
    next_ = future(3, "NGV26")
    persist_direct_position(book, old)
    controller = make_controller(book, old, active, next_)
    initial = FutureRoller(controller)
    initial.register_executor(
        FutureRollMode.DIRECT,
        DirectFutureRollExecutor(name="direct-roll"),
    )
    initial.roll()
    original = controller.trades[-1]
    rebound = ibi.Trade(
        contract=original.contract,
        order=original.order,
        orderStatus=original.orderStatus,
    )
    book.rebind_trade(rebound)

    recovered = FutureRoller(controller)
    recovered.register_executor(
        FutureRollMode.DIRECT,
        DirectFutureRollExecutor(name="direct-roll"),
    )

    assert recovered.recover()
    apply_fill(book, rebound, 2, "recovered-roll-fill")

    assert book.rolls.for_series("ng-series").stage is FutureRollStage.COMPLETE
    assert book.positions.quantity(active) == 2
    assert book.targets.for_contract(active).contract is active


def test_recovery_fails_closed_for_missing_executor_name(book):
    old = future(1, "NGQ26")
    active = future(2, "NGU26")
    next_ = future(3, "NGV26")
    persist_direct_position(book, old)
    controller = make_controller(book, old, active, next_)
    book.update_roll(
        RollState(
            series_key="ng-series",
            mode=FutureRollMode.DIRECT,
            executor_name="removed-executor",
            old_contract=old,
            new_contract=active,
            participants=(
                RollParticipant(
                    execution_model_name="serial",
                    quantity=2,
                ),
            ),
        )
    )
    roller = FutureRoller(controller)
    roller.register_executor(FutureRollMode.DIRECT)

    with pytest.raises(RuntimeError, match="requires removed-executor"):
        roller.recover()


def test_direct_series_can_hold_an_existing_destination_position(book):
    """A direct roll adds to the destination instead of claiming the whole series."""
    old = future(1, "NGQ26")
    active = future(2, "NGU26")
    next_ = future(3, "NGV26")
    persist_direct_position(book, old, quantity=1)
    persist_direct_position(book, active, quantity=2)
    controller = make_controller(book, old, active, next_)
    roller = FutureRoller(controller)
    roller.register_executor(FutureRollMode.DIRECT)
    roller.roll()
    state = book.rolls.for_series("ng-series")
    assert state.stage is FutureRollStage.ROLL_ORDER_ACTIVE
    assert len(controller.trades) == 1
    assert state.target_transfers[1].target_quantity == 3
