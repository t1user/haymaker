import asyncio
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
    symbol_is,
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


def target(
    quantity,
    *,
    source_key=None,
    intent=None,
    symbol="ES",
    con_id=1,
    metadata=None,
    created_at=None,
):
    return PositionTarget(
        contract=contract(symbol, con_id),
        target_quantity=quantity,
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
            execution_model_name="serial",
            contract=contract(),
            target_quantity=2,
            target_created_at=datetime.now(timezone.utc),
        )
    )
    runtime.workload_generation = 1

    model.onStart({})

    assert trader.trades[0].order.totalQuantity == 2


def test_bracket_model_rejects_missing_or_stale_intent(execution_runtime):
    model = BracketExecutionModel(
        "alpha", name="brackets", stop=FixedStop(2)
    )

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
    model = BracketExecutionModel(
        "alpha", name="brackets", stop=FixedStop(2)
    )

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
    model = BracketExecutionModel(
        "alpha", name="brackets", stop=FixedStop(2)
    )

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
        controller.book.order_by_id(trade.order.orderId).role
        for trade in trader.trades
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
    model = BracketExecutionModel(
        "alpha", name="brackets", stop=FixedStop(2)
    )

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
    model = BracketExecutionModel(
        "alpha", name="brackets", stop=FixedStop(2)
    )
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
    router = ExecutionRouter(
        [ExecutionRule(predicate=symbol_is("NQ"), model=model)]
    )

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


def test_router_preserves_source_affinity_until_flat(execution_runtime):
    runtime, _, _ = execution_runtime
    owner = RecordingModel(name="owner")
    current_rule = RecordingModel(name="current")
    router = ExecutionRouter(
        [
            ExecutionRule(
                predicate=lambda target: True,
                model=current_rule,
            )
        ],
        default_model=owner,
    )
    runtime.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="owner",
            contract=contract(),
            quantity=1,
            target_quantity=1,
        )
    )
    incoming = target(0, source_key="alpha")

    router.onData(incoming)

    assert owner.accepted == [incoming]
    assert current_rule.accepted == []

    runtime.book.update_position(
        PositionState(
            source_key="alpha",
            execution_model_name="owner",
            contract=contract(),
        )
    )
    router.onData(incoming)

    assert current_rule.accepted == [incoming]


def test_router_fails_closed_for_missing_recovery_model(execution_runtime):
    runtime, _, _ = execution_runtime
    runtime.book.update_target(
        TargetState(
            execution_model_name="missing",
            contract=contract(),
            target_quantity=1,
            target_created_at=datetime.now(timezone.utc),
        )
    )
    router = ExecutionRouter(
        [
            ExecutionRule(
                predicate=lambda target: True,
                model=RecordingModel(name="configured"),
            )
        ]
    )
    runtime.workload_generation = 1

    with pytest.raises(RuntimeError, match="affinity"):
        router.onStart({})


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
    model = BracketExecutionModel(
        "alpha", name="brackets", stop=FixedStop(2)
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
    rebound = ibi.Trade(
        contract=entry.contract,
        order=entry.order,
        orderStatus=entry.orderStatus,
    )
    runtime.book.rebind_trade(rebound)
    runtime.workload_generation = 1

    model.onStart({})

    assert len(rebound.filledEvent) == 1
