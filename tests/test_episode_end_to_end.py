"""Independent signal-to-broker tests; keep the older integration suite too."""

import ib_insync as ibi
import pytest
from episode_harness import (
    EpisodeBroker,
    EpisodeSignalModel,
    observation,
    settle_events,
)

from haymaker.base import Pipe
from haymaker.components import (
    BinarySignalProcessor,
    BracketExecutionModel,
    FixedSizeAllocator,
    FixedStop,
    OpposingSignalPolicy,
    PortfolioWrapper,
    SignalType,
    StandardOrderRole,
    TakeProfitAsStopMultiple,
)
from haymaker.controller import Controller


@pytest.fixture
def episode(atom_runtime_factory):
    """Compose the production one-to-one path around an independent broker."""
    broker = EpisodeBroker()
    runtime = atom_runtime_factory(ib=broker)
    controller = Controller(trader=runtime.trader)
    runtime.bind_controller(controller)
    controller.release_hold()
    contract = ibi.Future("ES", conId=101, exchange="CME", localSymbol="ESU6")
    signal = EpisodeSignalModel("alpha", contract, SignalType.EVENT)
    processor = BinarySignalProcessor(
        opposing=OpposingSignalPolicy.REVERSE, respect_blocked_direction=True
    )
    model = BracketExecutionModel(
        "alpha",
        name="brackets",
        stop=FixedStop(2),
        take_profit=TakeProfitAsStopMultiple(2, 2),
    )
    pipe = Pipe(signal, processor, PortfolioWrapper(FixedSizeAllocator(2)), model)
    return runtime, broker, signal, model, pipe


async def test_entry_partial_fill_protection_and_commission_without_blotter(episode):
    """Observe independently accounted fills and full-fill-only protection."""
    runtime, broker, signal, model, pipe = episode
    signal.onData(observation(1))
    await settle_events()
    entry = broker.submitted[0]
    first = await broker.fill(entry, 1)
    assert runtime.book.position_state("alpha").quantity == 1
    assert broker.quantities[entry.contract] == 1
    assert len(broker.submitted) == 1
    await broker.fill(entry, 1)
    assert len(broker.submitted) == 3
    assert runtime.book.position_state("alpha").quantity == 2
    assert {
        runtime.book.order_by_id(t.order.orderId).role for t in broker.submitted[1:]
    } == {StandardOrderRole.STOP_LOSS, StandardOrderRole.TAKE_PROFIT}
    await broker.commission(entry, first)
    info = runtime.book.order_by_id(entry.order.orderId)
    assert info.fills[0].commission_report.commission == 1.25
    assert runtime.book.blotter is None


@pytest.mark.parametrize("exit_index", [1, 2])
async def test_protective_exit_oca_and_direction_block(episode, exit_index):
    """Both protective exits cancel their sibling and block same-side reentry."""
    runtime, broker, signal, model, pipe = episode
    signal.onData(observation(1))
    await broker.fill(broker.submitted[0])
    await broker.fill(broker.submitted[exit_index])
    assert not broker.openTrades()
    assert not broker.positions()
    assert runtime.book.position_state("alpha").blocked_direction == 1
    signal.onData(observation(1))
    await settle_events()
    assert len(broker.submitted) == 3
    signal.onData(observation(-1))
    await broker.fill(broker.submitted[-1], 1)
    assert runtime.book.position_state("alpha").blocked_direction is None
