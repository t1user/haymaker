"""Independent signal-to-broker tests; keep the older integration suite too."""

import ib_insync as ibi
import pytest
from copy import deepcopy
from episode_harness import (
    EpisodeBroker,
    EpisodeSignalModel,
    observation,
    settle_events,
)

from haymaker.base import Pipe
from haymaker.book import Book
from haymaker.components import (
    BinarySignalProcessor,
    BracketExecutionModel,
    FixedSizeAllocator,
    FixedStop,
    OpposingSignalPolicy,
    PositionIntent,
    PositionTarget,
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
    assert runtime.book.position_state("alpha").blocked_direction == 1
    await broker.fill(broker.submitted[-1], 1)
    assert runtime.book.position_state("alpha").blocked_direction is None


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("pending_entry", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
async def test_differing_contract_exit_and_reversal(
    episode, direction, pending_entry, reverse
):
    """The wrapper forwards B; execution alone closes A before opening B."""
    runtime, broker, signal, model, pipe = episode
    signal.onData(observation(direction))
    entry = broker.submitted[0]
    if not pending_entry:
        await broker.fill(entry)
    original = runtime.book.position_state("alpha")
    incoming = ibi.Future("ES", conId=102, exchange="CME", localSymbol="ESZ6")
    signal.contract = incoming
    # EVENT zero is intentionally ignored; STATE zero explicitly requests flat.
    signal.signal_type = SignalType.STATE
    signal.onData(observation(-direction if reverse else 0, atr=9))
    accepted = runtime.book.position_state("alpha")
    assert accepted.contract == entry.contract
    assert accepted.target_contract == incoming
    assert accepted.bracket_inputs == {"atr": 5}
    if pending_entry:
        assert len(broker.submitted) == 1
        await broker.fill(entry)
    close = next(
        t
        for t in broker.submitted
        if runtime.book.order_by_id(t.order.orderId).role == StandardOrderRole.CLOSE
    )
    assert close.contract == entry.contract
    assert close.order.ocaGroup == broker.submitted[1].order.ocaGroup
    await broker.fill(close, 1)
    assert not any(t.contract == incoming for t in broker.submitted)
    await broker.fill(close, 1)
    assert broker.quantities[entry.contract] == 0
    if reverse:
        new_entry = broker.submitted[-1]
        assert new_entry.contract == incoming
        state = runtime.book.position_state("alpha")
        assert state.position_id != original.position_id
        assert state.bracket_inputs == {"atr": 9}
        await broker.fill(new_entry)
        assert broker.quantities[incoming] == -direction * 2
        assert runtime.book.position_state("alpha").quantity == -direction * 2
    else:
        assert not broker.positions()
        assert not broker.openTrades()


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("close_filled", [0, 1, 2])
async def test_reversal_recovers_new_contract_without_close_filled_callback(
    episode, atom_runtime_factory, order_saver, state_saver, direction, close_filled
):
    """A detached persisted snapshot resumes the opening without old events."""
    runtime, broker, signal, model, pipe = episode
    signal.onData(observation(direction))
    await broker.fill(broker.submitted[0])
    incoming = ibi.Future("ES", conId=102, exchange="CME", localSymbol="ESZ6")
    model.onData(
        PositionTarget(
            contract=incoming,
            target_quantity=-direction * 2,
            source_key="alpha",
            intent=PositionIntent.REVERSE,
            metadata={"atr": 9},
        )
    )
    if close_filled:
        await broker.fill(broker.submitted[-1], close_filled, notify_filled=False)
    assert len(broker.submitted) == 4
    # Detached documents and a new IB/Controller/model graph: no old closures.
    order_saver.store[order_saver.collection] = deepcopy(
        order_saver.store[order_saver.collection]
    )
    state_saver.store[state_saver.collection] = deepcopy(
        state_saver.store[state_saver.collection]
    )
    recovered = Book(
        order_saver=order_saver, state_saver=state_saver, save_async=False, restore=True
    )
    replacement = broker.restarted()
    fresh = atom_runtime_factory(ib=replacement, book_=recovered)
    controller = Controller(trader=fresh.trader)
    fresh.bind_controller(controller)
    controller.release_hold()
    for trade in replacement.submitted:
        recovered.rebind_trade(trade)
    resumed = BracketExecutionModel("alpha", name="brackets", stop=FixedStop(2))
    resumed.recover()
    if close_filled < 2:
        assert len(replacement.submitted) == 4
        await replacement.fill(replacement.submitted[-1])
    assert len(replacement.submitted) == 5
    new_entry = replacement.submitted[-1]
    assert new_entry.contract == incoming
    await replacement.fill(new_entry)
    assert recovered.position_state("alpha").quantity == -direction * 2
    assert replacement.quantities[incoming] == -direction * 2
