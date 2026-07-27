"""Focused futures-roll invariants for typed Book state."""

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast

import ib_insync as ibi

from haymaker.book import OrderInfo, PositionState
from haymaker.components import StandardOrderRole
from haymaker.controller import Controller
from haymaker.controller.future_roller import FutureRoller


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


def position(
    source_key: str,
    contract: ibi.Future,
    *,
    quantity: float = 1,
) -> PositionState:
    """Build one attributed logical futures position."""

    return PositionState(
        source_key=source_key,
        execution_model_name=f"{source_key}_brackets",
        contract=contract,
        quantity=quantity,
        target_quantity=quantity,
        target_created_at=datetime.now(timezone.utc),
        position_id=f"{source_key}-episode",
        blocked_direction=1,
        bracket_inputs={"atr": 0.25},
    )


def roller_controller(book, active: ibi.Future, next_: ibi.Future, **methods):
    """Build the narrow Controller surface used by FutureRoller."""

    registry = SimpleNamespace(
        current_contracts={active, next_},
        selectors=[SimpleNamespace(active_contract=active)],
        details={},
    )
    return cast(
        Controller,
        SimpleNamespace(
            book=book,
            contract_registry=registry,
            **methods,
        ),
    )


def test_rolls_only_held_future_outside_active_and_next(book, caplog):
    old = future(1, "NGQ26")
    active = future(2, "NGU26")
    next_ = future(3, "NGV26")
    book.update_position(position("automatic", old))
    book.update_position(position("manual", old))
    book.update_position(position("active", active))
    book.update_position(position("next", next_))
    controller = roller_controller(book, active, next_)

    roller = FutureRoller(
        controller,
        {"automatic": True, "manual": False, "active": True, "next": True},
    )

    assert roller.sources[old] == ["automatic"]
    assert roller.contracts_to_roll == {old}
    assert roller.match_old_to_new_future(old) is active


def test_roll_order_preserves_source_episode_model_and_role(book):
    old = future(1, "NGQ26")
    active = future(2, "NGU26")
    next_ = future(3, "NGV26")
    state = position("alpha", old, quantity=-2)
    book.update_position(state)
    calls: list[tuple[ibi.Contract, ibi.Order, dict[str, Any]]] = []

    def trade(
        contract: ibi.Contract,
        order: ibi.Order,
        **kwargs: Any,
    ) -> ibi.Trade:
        calls.append((contract, order, kwargs))
        return ibi.Trade(contract=contract, order=order)

    controller = roller_controller(book, active, next_, trade=trade)

    FutureRoller(controller)._trade("alpha", state, old, active)

    contract, order, attribution = calls[0]
    assert isinstance(contract, ibi.Bag)
    assert order.action == "SELL"
    assert order.totalQuantity == 2
    assert attribution["role"] == StandardOrderRole.ROLL
    assert attribution["execution_model_name"] == "alpha_brackets"
    assert attribution["source_key"] == "alpha"
    assert attribution["position_id"] == "alpha-episode"


def test_source_adjustment_preserves_logical_episode_state(book):
    old = future(1, "NGQ26")
    active = future(2, "NGU26")
    next_ = future(3, "NGV26")
    original = position("alpha", old, quantity=3)
    book.update_position(original)
    controller = roller_controller(
        book,
        active,
        next_,
        cancel=lambda trade: None,
    )

    FutureRoller(controller)._adjust_source("alpha", old, active, 1.5)

    adjusted = book.position_state("alpha")
    assert adjusted is not None
    assert adjusted.contract is active
    assert adjusted.quantity == original.quantity
    assert adjusted.target_quantity == original.target_quantity
    assert adjusted.execution_model_name == original.execution_model_name
    assert adjusted.position_id == original.position_id
    assert adjusted.blocked_direction == original.blocked_direction
    assert adjusted.bracket_inputs == original.bracket_inputs


def test_replacement_protective_order_keeps_attribution_and_role(book):
    old = future(1, "NGQ26")
    active = future(2, "NGU26")
    next_ = future(3, "NGV26")
    old_trade = ibi.Trade(
        contract=old,
        order=ibi.StopOrder(
            "SELL",
            1,
            stopPrice=2.5,
            orderId=10,
        ),
    )
    info = OrderInfo(
        trade=old_trade,
        role=StandardOrderRole.STOP_LOSS,
        submitted_at=datetime.now(timezone.utc),
        execution_model_name="alpha_brackets",
        source_key="alpha",
        position_id="alpha-episode",
        params={"atr": 0.25},
    )
    calls: list[tuple[ibi.Contract, ibi.Order, dict[str, Any]]] = []

    def trade(
        contract: ibi.Contract,
        order: ibi.Order,
        **kwargs: Any,
    ) -> ibi.Trade:
        calls.append((contract, order, kwargs))
        return ibi.Trade(contract=contract, order=order)

    controller = roller_controller(book, active, next_, trade=trade)

    FutureRoller(controller)._issue_replacement_order(
        old_trade,
        info=info,
        new_contract=active,
        fill_price=1,
    )

    contract, order, attribution = calls[0]
    assert contract is active
    assert order.orderId == 0
    assert order.auxPrice == 2.5
    assert attribution["role"] == StandardOrderRole.STOP_LOSS
    assert attribution["execution_model_name"] == "alpha_brackets"
    assert attribution["source_key"] == "alpha"
    assert attribution["position_id"] == "alpha-episode"
