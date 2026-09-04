import asyncio
import logging
from datetime import datetime, timezone

import ib_insync as ibi
import pytest

from haymaker.book import PositionState, TargetState
from haymaker.components import (
    BracketExecutionModel,
    ExecutionModel,
    ExecutionRouter,
    ExecutionRule,
    FixedStop,
    PositionIntent,
    PositionTarget,
    SerialTargetExecutionModel,
    StandardOrderRole,
    TakeProfitAsStopMultiple,
    symbol_is,
    target_key_is,
)
from haymaker.controller import Controller


class FakeTrader:
    def __init__(self):
        self.trades = []
        self.cancelled = []

    def trade(self, contract, order):
        order.orderId = len(self.trades) + 1
        order.permId = 100 + order.orderId
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
        return trade

    def cancel(self, trade):
        self.cancelled.append(trade)
        trade.orderStatus.status = ibi.OrderStatus.Cancelled
        trade.cancelledEvent.emit(trade)
        return trade

    def position_for_contract(self, contract):
        return 0

    def positions(self):
        return {}


@pytest.fixture
def execution_runtime(atom_runtime):
    trader = FakeTrader()
    controller = Controller(trader=trader)
    atom_runtime.bind_controller(controller)
    return atom_runtime, controller, trader


def contract(symbol="ES", con_id=1):
    return ibi.Future(
        conId=con_id,
        symbol=symbol,
        exchange="CME",
        localSymbol=f"{symbol}M6",
    )


class SeriesRegistry:
    """Resolve several concrete test Futures to one registered series."""

    def __init__(self, current: tuple[ibi.Future, ibi.Future]) -> None:
        self.current = current
        self.details: dict[ibi.Contract, object] = {}

    def series_key(self, contract: ibi.Future) -> str:
        """Return one explicit series identity for the test contracts."""

        if contract.symbol != "ES":
            raise KeyError(contract.symbol)
        return "es-series"

    def current_for_series(self, series_key: str) -> tuple[ibi.Future, ibi.Future]:
        """Return the configured ACTIVE/NEXT pair."""

        assert series_key == "es-series"
        return self.current

    def get_details(self, contract: ibi.Contract):
        """Disable market-hours filtering in focused tests."""

        return None


def target(
    quantity,
    *,
    source_key=None,
    intent=None,
    symbol="ES",
    con_id=1,
    metadata=None,
    created_at=None,
    target_key=None,
):
    return PositionTarget(
        contract=contract(symbol, con_id),
        target_quantity=quantity,
        target_key=(
            None if source_key is not None else target_key or f"{symbol.lower()}-target"
        ),
        source_key=source_key,
        intent=intent,
        metadata=metadata or {},
        created_at=created_at or datetime.now(timezone.utc),
    )


def execution_fill(trade, quantity, exec_id):
    return ibi.Fill(
        contract=trade.contract,
        execution=ibi.Execution(
            execId=exec_id,
            orderId=trade.order.orderId,
            permId=trade.order.permId,
            side="BOT" if trade.order.action == "BUY" else "SLD",
            shares=quantity,
            price=100,
            time=datetime.now(timezone.utc),
        ),
        commissionReport=ibi.CommissionReport(execId=exec_id),
        time=datetime.now(timezone.utc),
    )


def apply_fill(controller, trade, quantity, exec_id="exec-1", complete=True):
    fill = execution_fill(trade, quantity, exec_id)
    trade.fills.append(fill)
    trade.orderStatus.filled += quantity
    trade.orderStatus.remaining -= quantity
    if complete:
        trade.orderStatus.status = ibi.OrderStatus.Filled
    info = controller.book.order_by_id(trade.order.orderId)
    controller.register_position(info, fill)
    return fill


def working_adjustment(
    runtime,
    controller,
    owner,
    *,
    target_quantity=1,
    symbol="ES",
    con_id=1,
    target_key="es-target",
):
    """Persist a direct target and register its active adjustment order."""

    target_contract = contract(symbol, con_id)
    runtime.book.update_target(
        TargetState(
            target_key=target_key,
            execution_model_name=owner,
            contract=target_contract,
            target_quantity=target_quantity,
            target_created_at=datetime.now(timezone.utc),
        )
    )
    return controller.trade(
        target_contract,
        ibi.MarketOrder("BUY", abs(target_quantity) or 1),
        role=StandardOrderRole.TARGET_ADJUSTMENT,
        execution_model_name=owner,
        target_key=target_key,
    )


def test_target_key_predicate_matches_only_the_requested_identity():
    predicate = target_key_is("es-target")

    assert predicate(target(1, target_key="es-target"))
    assert not predicate(target(1, target_key="other"))


def test_execution_model_string_includes_name_and_class(execution_runtime):
    model = SerialTargetExecutionModel(name="eurex")

    assert str(model) == "eurex[SerialTargetExecutionModel]"


def test_serial_model_submits_absolute_adjustment(execution_runtime):
    _, _, trader = execution_runtime
    model = SerialTargetExecutionModel(name="serial")

    model.onData(target(3))

    assert trader.trades[0].order.action == "BUY"
    assert trader.trades[0].order.totalQuantity == 3
    assert model.book.order_by_id(1).role == StandardOrderRole.TARGET_ADJUSTMENT


def test_serial_model_rejects_another_live_key_for_same_contract(
    execution_runtime,
):
    model = SerialTargetExecutionModel(name="serial")
    model.onData(target(1, target_key="first"))

    with pytest.raises(ValueError, match="already owned by 'first'"):
        model.onData(target(1, target_key="second"))


def test_serial_target_key_can_follow_concrete_expiry_within_one_series(
    execution_runtime,
):
    runtime, _, trader = execution_runtime
    old = contract(con_id=1)
    active = contract(con_id=2)
    runtime.contract_registry = SeriesRegistry((old, active))
    model = SerialTargetExecutionModel(name="serial")
    model.onData(target(0, con_id=1))

    model.onData(target(1, con_id=2))

    assert trader.trades[-1].contract.conId == 2


def test_serial_resizes_on_held_active_or_next_expiry(execution_runtime):
    runtime, controller, trader = execution_runtime
    held = contract(con_id=1)
    next_contract = contract(con_id=2)
    runtime.contract_registry = SeriesRegistry((held, next_contract))
    model = SerialTargetExecutionModel(name="serial")
    model.onData(target(1, con_id=1))
    apply_fill(controller, trader.trades[-1], 1)

    model.onData(target(2, con_id=2))

    assert trader.trades[-1].contract.conId == held.conId
    assert trader.trades[-1].order.totalQuantity == 1


def test_serial_pauses_adjustment_while_held_expiry_needs_roll(
    execution_runtime,
):
    runtime, controller, trader = execution_runtime
    held = contract(con_id=1)
    active = contract(con_id=2)
    next_contract = contract(con_id=3)
    runtime.contract_registry = SeriesRegistry((active, next_contract))
    model = SerialTargetExecutionModel(name="serial")
    model.onData(target(1, con_id=1))
    apply_fill(controller, trader.trades[-1], 1)

    model.onData(target(2, con_id=2))

    assert len(trader.trades) == 1
    assert model.book.target_state("es-target").target_quantity == 2


def test_serial_model_rejects_wrong_message_at_runtime(execution_runtime):
    with pytest.raises(TypeError, match="only PositionTarget"):
        SerialTargetExecutionModel(name="serial").onData({"quantity": 1})


def test_serial_model_supports_same_side_resizing(execution_runtime):
    _, controller, trader = execution_runtime
    model = SerialTargetExecutionModel(name="serial")
    model.onData(target(1))
    first = trader.trades[0]
    apply_fill(controller, first, 1)
    first.filledEvent.emit(first)

    assert len(trader.trades) == 1

    model.onData(target(3))

    assert trader.trades[1].order.action == "BUY"
    assert trader.trades[1].order.totalQuantity == 2


@pytest.mark.asyncio
async def test_serial_target_supersedes_while_order_active(execution_runtime):
    _, controller, trader = execution_runtime
    model = SerialTargetExecutionModel(name="serial")
    model.onData(target(1))
    model.onData(target(3))

    assert len(trader.trades) == 1

    first = trader.trades[0]
    apply_fill(controller, first, 1)
    first.filledEvent.emit(first)
    await asyncio.sleep(0)

    assert len(trader.trades) == 2
    assert trader.trades[1].order.totalQuantity == 2


def test_serial_recovery_resumes_persisted_target(execution_runtime):
    runtime, _, trader = execution_runtime
    model = SerialTargetExecutionModel(name="serial")
    model.book.update_target(
        TargetState(
            target_key="es-target",
            execution_model_name="serial",
            contract=contract(),
            target_quantity=2,
            target_created_at=datetime.now(timezone.utc),
        )
    )
    runtime.workload_generation = 1

    model.onStart({})

    assert trader.trades[0].order.totalQuantity == 2


def test_bracket_model_registers_automatic_future_roll_by_default(
    execution_runtime,
):
    runtime, _, _ = execution_runtime

    BracketExecutionModel("alpha", name="brackets", stop=FixedStop(2))

    assert runtime.future_roll_policies == {"alpha": True}


def test_bracket_model_can_disable_automatic_future_roll(execution_runtime):
    runtime, _, _ = execution_runtime

    BracketExecutionModel(
        "alpha",
        name="brackets",
        stop=FixedStop(2),
        auto_roll_futures=False,
    )

    assert runtime.future_roll_policies == {"alpha": False}


def test_bracket_model_rejects_invalid_or_conflicting_future_roll_policy(
    execution_runtime,
):
    runtime, _, _ = execution_runtime

    with pytest.raises(TypeError, match="auto_roll_futures must be a bool"):
        BracketExecutionModel(
            "invalid",
            name="invalid",
            stop=FixedStop(2),
            auto_roll_futures=1,  # type: ignore[arg-type]
        )
    assert runtime.future_roll_policies == {}

    BracketExecutionModel("alpha", name="first", stop=FixedStop(2))
    with pytest.raises(ValueError, match="Conflicting futures-roll policy"):
        BracketExecutionModel(
            "alpha",
            name="second",
            stop=FixedStop(2),
            auto_roll_futures=False,
        )


def test_bracket_model_rejects_missing_or_stale_intent(execution_runtime):
    model = BracketExecutionModel("alpha", name="brackets", stop=FixedStop(2))

    with pytest.raises(ValueError, match="requires PositionIntent"):
        model.onData(target(1, source_key="alpha", metadata={"atr": 5}))
    with pytest.raises(ValueError, match="inconsistent"):
        model.onData(
            target(
                1,
                source_key="alpha",
                intent=PositionIntent.CLOSE,
                metadata={"atr": 5},
            )
        )

    assert model.book.position_state("alpha") is None


def test_bracket_model_rejects_same_side_resize(execution_runtime):
    runtime, _, _ = execution_runtime
    runtime.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=contract(),
            quantity=1,
            target_quantity=1,
            position_id="episode",
            bracket_inputs={"atr": 5},
        )
    )
    model = BracketExecutionModel("alpha", name="brackets", stop=FixedStop(2))

    with pytest.raises(ValueError, match="same-side"):
        model.onData(
            target(
                2,
                source_key="alpha",
                intent=PositionIntent.OPEN,
                metadata={"atr": 5},
            )
        )


def test_bracket_model_rejects_invalid_oca_type(execution_runtime):
    with pytest.raises(ValueError, match="oca_type"):
        BracketExecutionModel(
            "alpha",
            name="brackets",
            stop=FixedStop(2),
            oca_type=0,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "five"])
def test_bracket_model_validates_recovery_inputs(execution_runtime, value):
    model = BracketExecutionModel("alpha", name="brackets", stop=FixedStop(2))

    with pytest.raises((TypeError, ValueError), match="Bracket input"):
        model.onData(
            target(
                1,
                source_key="alpha",
                intent=PositionIntent.OPEN,
                metadata={"atr": value},
            )
        )

    assert model.book.position_state("alpha") is None


def test_brackets_attach_only_after_complete_entry_fill(execution_runtime):
    _, controller, trader = execution_runtime
    model = BracketExecutionModel(
        "alpha",
        name="brackets",
        stop=FixedStop(2),
    )
    model.onData(
        target(
            2,
            source_key="alpha",
            intent=PositionIntent.OPEN,
            metadata={"atr": 5},
        )
    )
    entry = trader.trades[0]
    apply_fill(controller, entry, 1, complete=False)
    entry.filledEvent.emit(entry)

    assert len(trader.trades) == 1

    apply_fill(controller, entry, 1, exec_id="exec-2")
    entry.orderStatus.avgFillPrice = 100
    entry.filledEvent.emit(entry)

    assert len(trader.trades) == 2
    assert controller.book.order_by_id(2).role == StandardOrderRole.STOP_LOSS
    assert controller.book.order_by_id(2).position_id == (
        controller.book.position_state("alpha").position_id
    )


def test_regular_close_joins_active_bracket_oca_group(execution_runtime):
    """A regular close lets IB cancel stop and take-profit through OCA."""

    _, controller, trader = execution_runtime
    model = BracketExecutionModel(
        "alpha",
        name="brackets",
        stop=FixedStop(2),
        take_profit=TakeProfitAsStopMultiple(2, 3),
    )
    model.onData(
        target(
            1,
            source_key="alpha",
            intent=PositionIntent.OPEN,
            metadata={"atr": 5},
        )
    )
    entry = trader.trades[0]
    apply_fill(controller, entry, 1)
    entry.orderStatus.avgFillPrice = 100
    entry.filledEvent.emit(entry)
    brackets = trader.trades[1:3]
    oca_group = brackets[0].order.ocaGroup

    model.onData(
        target(
            0,
            source_key="alpha",
            intent=PositionIntent.CLOSE,
        )
    )

    close = trader.trades[3]
    assert oca_group
    assert {trade.order.ocaGroup for trade in brackets} == {oca_group}
    assert close.order.ocaGroup == oca_group
    assert close.order.ocaType == model.oca_type
    assert trader.cancelled == []


@pytest.mark.asyncio
async def test_entry_fill_continues_to_newer_close_target(execution_runtime):
    _, controller, trader = execution_runtime
    model = BracketExecutionModel(
        "alpha",
        name="brackets",
        stop=FixedStop(2),
    )
    model.onData(
        target(
            1,
            source_key="alpha",
            intent=PositionIntent.OPEN,
            metadata={"atr": 5},
        )
    )
    entry = trader.trades[0]
    model.onData(
        target(
            0,
            source_key="alpha",
            intent=PositionIntent.CLOSE,
        )
    )

    apply_fill(controller, entry, 1)
    entry.orderStatus.avgFillPrice = 100
    entry.filledEvent.emit(entry)
    await asyncio.sleep(0)

    assert [
        controller.book.order_by_id(trade.order.orderId).role for trade in trader.trades
    ] == [
        StandardOrderRole.OPEN,
        StandardOrderRole.STOP_LOSS,
        StandardOrderRole.CLOSE,
    ]
    assert trader.trades[-1].order.action == "SELL"


@pytest.mark.asyncio
async def test_reversal_closes_then_opens_new_episode(execution_runtime):
    runtime, controller, trader = execution_runtime
    runtime.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=contract(),
            quantity=1,
            target_quantity=1,
            position_id="old-episode",
            bracket_inputs={"atr": 5},
        )
    )
    model = BracketExecutionModel("alpha", name="brackets", stop=FixedStop(2))

    model.onData(
        target(
            -1,
            source_key="alpha",
            intent=PositionIntent.REVERSE,
            metadata={"atr": 5},
        )
    )

    close = trader.trades[0]
    assert close.order.action == "SELL"
    apply_fill(controller, close, 1)
    close.filledEvent.emit(close)
    await asyncio.sleep(0)

    assert trader.trades[1].order.action == "SELL"
    assert runtime.book.position_state("alpha").position_id != "old-episode"
    assert runtime.book.position_state("alpha").bracket_inputs == {"atr": 5}


def test_stop_fill_closes_episode_and_prevents_restart_reentry(
    execution_runtime,
):
    runtime, controller, trader = execution_runtime
    runtime.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=contract(),
            quantity=1,
            target_quantity=1,
            position_id="episode",
            bracket_inputs={"atr": 5},
        )
    )
    model = BracketExecutionModel("alpha", name="brackets", stop=FixedStop(2))
    stop = controller.trade(
        contract(),
        ibi.Order(action="SELL", totalQuantity=1, orderType="STP"),
        role=StandardOrderRole.STOP_LOSS,
        execution_model_name=model.name,
        source_key="alpha",
        position_id="episode",
    )

    assert stop is not None
    apply_fill(controller, stop, 1)

    state = runtime.book.position_state("alpha")
    assert state.quantity == 0
    assert state.target_quantity == 0
    assert state.position_id is None
    assert state.blocked_direction == 1
    model.recover()
    assert trader.trades == [stop]


class RecordingModel(ExecutionModel):
    def __init__(self, **kwargs):
        self.accepted = []
        self.recoveries = 0
        super().__init__(**kwargs)

    def accept(self, incoming):
        self.accepted.append(incoming)
        return True

    def recover(self):
        self.recoveries += 1


def test_router_rejects_wrong_message_at_runtime(execution_runtime):
    router = ExecutionRouter(
        [
            ExecutionRule(
                predicate=lambda target: True,
                model=RecordingModel(name="model"),
            )
        ]
    )

    with pytest.raises(TypeError, match="only PositionTarget"):
        router.onData({"quantity": 1})


def test_router_first_match_and_default(execution_runtime):
    first = RecordingModel(name="first")
    second = RecordingModel(name="second")
    fallback = RecordingModel(name="fallback")
    router = ExecutionRouter(
        [
            ExecutionRule(predicate=symbol_is("ES"), model=first),
            ExecutionRule(predicate=symbol_is("NQ"), model=second),
        ],
        default_model=fallback,
    )

    router.onData(target(1, symbol="ES"))
    router.onData(target(1, symbol="NQ", con_id=2))
    router.onData(target(1, symbol="YM", con_id=3))

    assert len(first.accepted) == 1
    assert len(second.accepted) == 1
    assert len(fallback.accepted) == 1


def test_router_fails_closed_without_default(execution_runtime):
    model = RecordingModel(name="model")
    router = ExecutionRouter([ExecutionRule(predicate=symbol_is("NQ"), model=model)])

    with pytest.raises(LookupError, match="No ExecutionModel"):
        router.onData(target(1))


def test_router_rejects_duplicate_model_names(execution_runtime):
    with pytest.raises(ValueError, match="Duplicate"):
        ExecutionRouter(
            [
                ExecutionRule(
                    predicate=lambda target: True,
                    model=RecordingModel(name="same"),
                ),
                ExecutionRule(
                    predicate=lambda target: False,
                    model=RecordingModel(name="same"),
                ),
            ]
        )


def test_router_starts_every_model_once_per_generation(execution_runtime):
    runtime, _, _ = execution_runtime
    first = RecordingModel(name="first")
    second = RecordingModel(name="second")
    router = ExecutionRouter(
        [
            ExecutionRule(predicate=lambda target: True, model=first),
            ExecutionRule(predicate=lambda target: False, model=second),
        ]
    )
    runtime.workload_generation = 1

    router.onStart({})
    router.onStart({})
    runtime.workload_generation = 2
    router.onStart({})

    assert first.recoveries == 2
    assert second.recoveries == 2


def test_router_ignores_one_to_one_working_orders(execution_runtime):
    runtime, controller, _ = execution_runtime
    current_rule = RecordingModel(name="current")
    router = ExecutionRouter(
        [
            ExecutionRule(
                predicate=lambda target: True,
                model=current_rule,
            )
        ]
    )
    runtime.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="brackets",
            contract=contract(),
            quantity=1,
            target_quantity=1,
        )
    )
    working = controller.trade(
        contract(),
        ibi.MarketOrder("SELL", 1),
        role=StandardOrderRole.CLOSE,
        execution_model_name="brackets",
        source_key="alpha",
    )
    incoming = target(0)
    runtime.workload_generation = 1

    router.onStart({})
    router.onData(incoming)

    assert working.orderStatus.status == ibi.OrderStatus.Submitted
    assert current_rule.recoveries == 1
    assert current_rule.accepted == [incoming]


def test_router_current_rules_own_held_position_without_working_order(
    execution_runtime,
):
    runtime, _, _ = execution_runtime
    old = RecordingModel(name="old")
    current = RecordingModel(name="current")
    router = ExecutionRouter(
        [ExecutionRule(predicate=lambda target: True, model=current)],
        default_model=old,
    )
    runtime.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="old",
            contract=contract(),
            quantity=1,
            target_quantity=1,
        )
    )
    incoming = target(0)

    router.onData(incoming)

    assert current.accepted == [incoming]
    assert old.accepted == []


def test_router_recovers_when_active_order_still_selects_owner(execution_runtime):
    runtime, controller, _ = execution_runtime
    owner = RecordingModel(name="owner")
    working_adjustment(runtime, controller, owner.name)
    router = ExecutionRouter(
        [ExecutionRule(predicate=lambda target: True, model=owner)]
    )
    incoming = target(2)
    runtime.workload_generation = 1

    router.onStart({})
    router.onData(incoming)

    assert owner.recoveries == 1
    assert owner.accepted == [incoming]


def test_router_blocks_missing_active_order_owner(
    execution_runtime,
    caplog,
):
    runtime, controller, _ = execution_runtime
    working_adjustment(runtime, controller, "missing")
    configured = RecordingModel(name="configured")
    router = ExecutionRouter(
        [
            ExecutionRule(
                predicate=lambda target: True,
                model=configured,
            )
        ]
    )
    runtime.workload_generation = 1

    with caplog.at_level(
        logging.CRITICAL, logger="haymaker.components.execution.router"
    ):
        router.onStart({})
        router.onData(target(2))

    assert configured.recoveries == 0
    assert configured.accepted == []
    assert "belongs to 'missing', but current rules select 'configured'" in caplog.text
    assert "PositionTarget suppressed while ExecutionRouter is blocked" in caplog.text


def test_router_blocks_owner_still_configured_for_another_route(
    execution_runtime,
    caplog,
):
    runtime, controller, _ = execution_runtime
    working_adjustment(runtime, controller, "old")
    old = RecordingModel(name="old")
    current = RecordingModel(name="current")
    router = ExecutionRouter(
        [
            ExecutionRule(predicate=symbol_is("ES"), model=current),
            ExecutionRule(predicate=symbol_is("NQ"), model=old),
        ]
    )
    runtime.workload_generation = 1

    with caplog.at_level(
        logging.CRITICAL, logger="haymaker.components.execution.router"
    ):
        router.onStart({})

    assert old.recoveries == 0
    assert current.recoveries == 0
    assert "belongs to 'old', but current rules select 'current'" in caplog.text


def test_router_blocks_active_order_without_target_state(
    execution_runtime,
    caplog,
):
    runtime, controller, _ = execution_runtime
    controller.trade(
        contract(),
        ibi.MarketOrder("BUY", 1),
        role=StandardOrderRole.TARGET_ADJUSTMENT,
        execution_model_name="owner",
        target_key="es-target",
    )
    owner = RecordingModel(name="owner")
    router = ExecutionRouter(
        [ExecutionRule(predicate=lambda target: True, model=owner)]
    )
    runtime.workload_generation = 1

    with caplog.at_level(
        logging.CRITICAL, logger="haymaker.components.execution.router"
    ):
        router.onStart({})

    assert owner.recoveries == 0
    assert "has no recoverable TargetState" in caplog.text


def test_router_blocks_ambiguous_active_adjustment_owners(
    execution_runtime,
    caplog,
):
    runtime, controller, _ = execution_runtime
    working_adjustment(runtime, controller, "first")
    controller.trade(
        contract(),
        ibi.MarketOrder("BUY", 1),
        role=StandardOrderRole.TARGET_ADJUSTMENT,
        execution_model_name="second",
        target_key="es-target",
    )
    first = RecordingModel(name="first")
    second = RecordingModel(name="second")
    router = ExecutionRouter(
        [ExecutionRule(predicate=lambda target: True, model=first)],
        default_model=second,
    )
    runtime.workload_generation = 1

    with caplog.at_level(
        logging.CRITICAL, logger="haymaker.components.execution.router"
    ):
        router.onStart({})

    assert first.recoveries == 0
    assert second.recoveries == 0
    assert "owned by multiple models: ['first', 'second']" in caplog.text


def test_router_blocks_unroutable_active_adjustment(
    execution_runtime,
    caplog,
):
    runtime, controller, _ = execution_runtime
    working_adjustment(runtime, controller, "owner")
    owner = RecordingModel(name="owner")
    router = ExecutionRouter([ExecutionRule(predicate=symbol_is("NQ"), model=owner)])
    runtime.workload_generation = 1

    with caplog.at_level(
        logging.CRITICAL, logger="haymaker.components.execution.router"
    ):
        router.onStart({})

    assert owner.recoveries == 0
    assert "cannot be routed: LookupError" in caplog.text


def test_router_suppresses_live_model_change_until_order_is_terminal(
    execution_runtime,
    caplog,
):
    runtime, controller, _ = execution_runtime
    owner = RecordingModel(name="owner")
    other = RecordingModel(name="other")
    router = ExecutionRouter(
        [
            ExecutionRule(
                predicate=lambda incoming: incoming.target_quantity > 0, model=owner
            )
        ],
        default_model=other,
    )
    working = working_adjustment(runtime, controller, owner.name)
    runtime.workload_generation = 1
    router.onStart({})
    flatten = target(0)

    with caplog.at_level(
        logging.CRITICAL, logger="haymaker.components.execution.router"
    ):
        router.onData(flatten)

    assert owner.accepted == []
    assert other.accepted == []
    assert "target suppressed" in caplog.text

    working.orderStatus.status = ibi.OrderStatus.Filled
    router.onData(flatten)

    assert other.accepted == [flatten]


def test_router_does_not_partially_reassign_unroutable_idle_targets(
    execution_runtime,
    caplog,
):
    runtime, _, _ = execution_runtime
    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for symbol, con_id in (("ES", 1), ("NQ", 2)):
        runtime.book.update_target(
            TargetState(
                target_key=f"{symbol.lower()}-target",
                execution_model_name="old",
                contract=contract(symbol, con_id),
                target_quantity=1,
                target_created_at=created_at,
            )
        )
    current = RecordingModel(name="current")
    router = ExecutionRouter([ExecutionRule(predicate=symbol_is("ES"), model=current)])
    runtime.workload_generation = 1

    with caplog.at_level(
        logging.CRITICAL, logger="haymaker.components.execution.router"
    ):
        router.onStart({})

    assert current.recoveries == 0
    assert runtime.book.target_state("es-target").execution_model_name == "old"
    assert "idle target recovery could not be routed" in caplog.text


def test_router_hands_idle_recovered_target_to_current_model(execution_runtime):
    runtime, _, _ = execution_runtime
    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    runtime.book.update_target(
        TargetState(
            target_key="es-target",
            execution_model_name="old",
            contract=contract(),
            target_quantity=2,
            target_created_at=created_at,
        )
    )
    current = RecordingModel(name="current")
    router = ExecutionRouter(
        [ExecutionRule(predicate=lambda target: True, model=current)]
    )
    runtime.workload_generation = 1

    router.onStart({})

    state = runtime.book.target_state("es-target")
    assert state.execution_model_name == "current"
    assert state.target_quantity == 2
    assert state.target_created_at == created_at


def test_serial_recovery_handoff_converges_from_existing_quantity(
    execution_runtime,
):
    runtime, controller, trader = execution_runtime
    filled = controller.trade(
        contract(),
        ibi.MarketOrder("BUY", 1),
        role=StandardOrderRole.TARGET_ADJUSTMENT,
        execution_model_name="old",
        target_key="es-target",
    )
    apply_fill(controller, filled, 1)
    runtime.book.update_target(
        TargetState(
            target_key="es-target",
            execution_model_name="old",
            contract=contract(),
            target_quantity=2,
            target_created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
    )
    current = SerialTargetExecutionModel(name="current")
    router = ExecutionRouter(
        [ExecutionRule(predicate=lambda target: True, model=current)]
    )
    runtime.workload_generation = 1

    router.onStart({})

    adjustment = trader.trades[-1]
    assert adjustment is not filled
    assert adjustment.order.action == "BUY"
    assert adjustment.order.totalQuantity == 1
    assert runtime.book.target_state("es-target").execution_model_name == "current"


def test_stale_serial_target_is_not_emitted_as_accepted(execution_runtime):
    model = SerialTargetExecutionModel(name="serial")
    newer = target(
        2,
        created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    older = target(
        1,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    accepted = []
    model.dataEvent += lambda incoming, _name: accepted.append(incoming)

    model.onData(newer)
    model.onData(older)

    assert accepted == [newer]


def test_serial_recovery_rebinds_active_order_completion(execution_runtime):
    runtime, _, trader = execution_runtime
    model = SerialTargetExecutionModel(name="serial")
    model.onData(target(1))
    active = trader.trades[0]
    rebound = ibi.Trade(
        contract=active.contract,
        order=active.order,
        orderStatus=active.orderStatus,
    )
    runtime.book.rebind_trade(rebound)
    runtime.workload_generation = 1

    model.onStart({})

    assert len(rebound.filledEvent) == 1


def test_bracket_recovery_rebinds_active_entry_fill(execution_runtime):
    runtime, _, trader = execution_runtime
    model = BracketExecutionModel("alpha", name="brackets", stop=FixedStop(2))
    model.onData(
        target(
            1,
            source_key="alpha",
            intent=PositionIntent.OPEN,
            metadata={"atr": 5},
        )
    )
    entry = trader.trades[0]
    rebound = ibi.Trade(
        contract=entry.contract,
        order=entry.order,
        orderStatus=entry.orderStatus,
    )
    runtime.book.rebind_trade(rebound)
    runtime.workload_generation = 1

    model.onStart({})

    assert len(rebound.filledEvent) == 1
