"""Independent direct Portfolio-to-broker concrete-target scenarios."""

from collections.abc import Iterable
from datetime import datetime, timezone

import ib_insync as ibi
import pytest
from episode_harness import EpisodeBroker, settle_events

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
    """Example source allocations that can change several concrete targets."""

    def __init__(self):
        super().__init__()
        self.allocations: dict[str, tuple[ibi.Contract, float]] = {}

    def process(self, signal: Signal) -> Iterable[PositionTarget]:
        """Replace a source allocation and emit totals for old and new Contracts."""
        if not isinstance(signal.value, float):
            raise TypeError("This Portfolio requires scalar Signals")
        previous = self.allocations.get(signal.source_key)
        self.allocations[signal.source_key] = (signal.contract, float(signal.value))
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
        source_key=source,
        contract=ibi.Future(
            "ES", conId=con_id, exchange="CME", localSymbol=f"ES-{con_id}"
        ),
        value=value,
        signal_type=SignalType.STATE,
        created_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def direct(atom_runtime_factory):
    """Use real direct components and the independent IB event harness."""
    broker = EpisodeBroker()
    runtime = atom_runtime_factory(ib=broker)
    controller = Controller(trader=runtime.trader)
    runtime.bind_controller(controller)
    controller.release_hold()
    portfolio = AllocatingPortfolio()
    model = SerialTargetExecutionModel(name="serial")
    router = ExecutionRouter([ExecutionRule(predicate=symbol_is("ES"), model=model)])
    portfolio.connect(router)
    return runtime, broker, portfolio, model, router


async def test_direct_expiries_are_independent_during_target_supersession(direct):
    """A newer A target never adjusts an already held or working B expiry."""
    runtime, broker, portfolio, model, router = direct
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
    assert runtime.book.direct_quantity(first.contract) == 1
    assert runtime.book.direct_quantity(second.contract) == 2
    assert broker.quantities == dict(runtime.book.direct_positions())
    assert runtime.book.position_state("alpha") is None
    # A replayed execution is not another position change.
    broker.execDetailsEvent.emit(first, first.fills[0])
    await settle_events()
    assert runtime.book.direct_quantity(first.contract) == 1


async def test_one_source_can_reallocate_between_concrete_contracts(direct):
    """Portfolio explicitly requests both the old zero and new absolute target."""
    runtime, broker, portfolio, model, router = direct
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
    assert runtime.book.target_state(old.contract).target_quantity == 0
    assert runtime.book.direct_quantity(old.contract) == 0
    assert runtime.book.direct_quantity(opening.contract) == 3
