"""Independent contract_execution Portfolio-to-broker concrete-target scenarios."""

from collections.abc import Iterable
from datetime import datetime, timezone

import ib_insync as ibi
import pytest
from signal_execution_harness import PositionBroker, settle_events

from haymaker.components import (
    ExecutionRouter,
    ExecutionRule,
    Portfolio,
    PositionTarget,
    SerialTargetExecutionModel,
    Signal,
    SignalType,
    symbol_is,
)
from haymaker.controller import Controller


class AllocatingPortfolio(Portfolio):
    """Example signal allocations that can change several concrete targets."""

    def __init__(self):
        super().__init__()
        self.allocations: dict[str, tuple[ibi.Contract, float]] = {}

    def process(self, signal: Signal) -> Iterable[PositionTarget]:
        """Replace a signal allocation and emit totals for old and new Contracts."""
        if not isinstance(signal.value, float):
            raise TypeError("This Portfolio requires scalar Signals")
        previous = self.allocations.get(signal.signal_key)
        self.allocations[signal.signal_key] = (signal.contract, float(signal.value))
        affected = {signal.contract}
        if previous is not None:
            affected.add(previous[0])
        for contract in sorted(affected, key=lambda c: c.conId):
            yield PositionTarget(
                contract=contract,
                target_quantity=sum(
                    quantity
                    for member, quantity in self.allocations.values()
                    if member.conId == contract.conId
                ),
            )


def signal(source: str, con_id: int, value: float) -> Signal:
    """Build one source observation for a concrete expiry."""
    return Signal(
        signal_key=source,
        contract=ibi.Future(
            "ES", conId=con_id, exchange="CME", localSymbol=f"ES-{con_id}"
        ),
        value=value,
        signal_type=SignalType.STATE,
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def contract_execution(atom_runtime_factory):
    """Use real contract_execution components and the independent IB event harness."""
    broker = PositionBroker()
    runtime = atom_runtime_factory(ib=broker)
    controller = Controller(trader=runtime.trader)
    runtime.bind_controller(controller)
    portfolio = AllocatingPortfolio()
    model = SerialTargetExecutionModel(name="serial")
    router = ExecutionRouter([ExecutionRule(predicate=symbol_is("ES"), model=model)])
    portfolio.connect(router)
    return runtime, broker, portfolio, model, router


async def test_contract_execution_expiries_are_independent_during_target_supersession(
    contract_execution,
):
    """A newer A target never adjusts an already held or working B expiry."""
    runtime, broker, portfolio, model, router = contract_execution
    portfolio.onData(signal("alpha", 101, 3))
    portfolio.onData(signal("beta", 102, 2))
    first, second = broker.submitted
    await broker.fill(first, 1)
    portfolio.onData(signal("alpha", 101, 1))
    assert len(broker.submitted) == 2
    await broker.fill(first, 2)
    reduction = broker.submitted[-1]
    assert reduction.contract == first.contract
    assert reduction.order.action == "SELL"
    assert reduction.order.totalQuantity == 2
    await broker.fill(reduction)
    await broker.fill(second)
    assert runtime.book.positions.filled_quantity(first.contract) == 1
    assert runtime.book.positions.filled_quantity(second.contract) == 2
    assert broker.quantities == dict(runtime.book.positions.by_contract())
    assert runtime.book.positions._for_signal("alpha") is None
    # A replayed execution is not another position change.
    broker.execDetailsEvent.emit(first, first.fills[0])
    await settle_events()
    assert runtime.book.positions.filled_quantity(first.contract) == 1


async def test_one_signal_can_reallocate_between_concrete_contracts(contract_execution):
    """Portfolio explicitly requests both the old zero and new absolute target."""
    runtime, broker, portfolio, model, router = contract_execution
    portfolio.onData(signal("alpha", 101, 2))
    old = broker.submitted[-1]
    await broker.fill(old)
    portfolio.onData(signal("alpha", 102, 3))
    close, opening = broker.submitted[-2:]
    assert close.contract == old.contract
    assert close.order.action == "SELL"
    assert opening.contract.conId == 102
    await broker.fill(close)
    await broker.fill(opening)
    assert runtime.book.targets.for_contract(old.contract).target_quantity == 0
    assert runtime.book.positions.filled_quantity(old.contract) == 0
    assert runtime.book.positions.filled_quantity(opening.contract) == 3


async def test_completion_feedback_observes_accounted_fills_and_repeats_on_restart(
    contract_execution,
):
    """A Portfolio can wait for convergence without depending on Trade callbacks."""
    runtime, broker, portfolio, model, router = contract_execution
    completed = []

    def onFeedback(target):
        """Observe Book quantities at the moment completion is delivered."""
        completed.append(
            (target, runtime.book.positions.filled_quantity(target.contract))
        )

    portfolio.feedbackEvent += onFeedback
    portfolio.onData(signal("alpha", 101, 2))
    await broker.fill(broker.submitted[0], 1)
    assert completed == []
    await broker.fill(broker.submitted[0], 1)
    assert len(completed) == 1
    assert completed[0][1] == 2
    assert completed[0][0].target_quantity == 2
    model.recover()
    assert len(completed) == 1
    runtime.workload_generation += 1
    router.onStart({})
    assert len(completed) == 2
    assert len(broker.submitted) == 1


async def test_portfolio_can_wait_for_flat_before_opening_another_contract(
    contract_execution,
):
    """The user's feedback policy, not the execution model, sequences expiries."""
    runtime, broker, portfolio, model, router = contract_execution
    portfolio.onData(signal("alpha", 101, 2))
    await broker.fill(broker.submitted[0])

    def onFeedback(target):
        """Open B only once the requested zero in A has actually converged."""
        if target.contract.conId == 101 and target.target_quantity == 0:
            portfolio.onData(signal("alpha", 102, 3))

    portfolio.feedbackEvent += onFeedback
    portfolio.onData(signal("alpha", 101, 0))
    close = broker.submitted[-1]
    await broker.fill(close, 1)
    assert len(broker.submitted) == 2
    await broker.fill(close, 1)
    assert broker.submitted[-1].contract.conId == 102
    assert runtime.book.positions.filled_quantity(close.contract) == 0
