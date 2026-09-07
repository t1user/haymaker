from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Literal

import ib_insync as ibi
import pytest

from haymaker.components import (
    FixedSizeAllocator,
    Portfolio,
    PortfolioStateMixin,
    PortfolioWrapper,
    PositionIntent,
    PositionProposal,
    PositionTarget,
    Signal,
    SignalType,
)


class SavedPortfolio(PortfolioStateMixin, Portfolio):
    """Opt-in state storage without changing the Portfolio processing contract."""

    portfolio_key = "test_allocations"

    def process(self, signal: Signal) -> Iterable[PositionTarget]:
        """No trades are needed to test the persistence boundary."""
        return ()


def test_portfolio_state_mixin_loads_only_when_explicitly_requested(atom_runtime):
    first = SavedPortfolio()
    assert first.load_state() is None
    first.save_state({"allocations": {"alpha": 2}})
    second = SavedPortfolio()
    assert second.load_state() == {"allocations": {"alpha": 2}}
    assert (
        atom_runtime.book.load_portfolio_state("test_allocations")
        == second.load_state()
    )


def test_portfolio_state_mixin_allows_independent_backend(atom_runtime):
    """A user can replace storage without inventing another store protocol."""

    class Independent(SavedPortfolio):
        def load_state(self):
            """Read the user's independently owned state."""
            return self.saved

        def save_state(self, state):
            """Write only the user's independently owned state."""
            self.saved = dict(state)

    portfolio = Independent()
    portfolio.save_state({"alpha": 1})
    assert portfolio.load_state() == {"alpha": 1}
    assert atom_runtime.book.load_portfolio_state(portfolio.portfolio_key) is None


def test_portfolio_blueprint_positions_query_filled_not_desired_quantities(
    atom_runtime,
):
    """The helper combines registry membership with accounting, not allocations."""
    from haymaker.book import PositionState

    blueprint = ibi.Stock("AAPL", "SMART", "USD")
    held = ibi.Stock("AAPL", "SMART", "USD", conId=11)
    atom_runtime.contract_registry.register_blueprint(blueprint)
    atom_runtime.contract_registry.reset_data([[ibi.ContractDetails(contract=held)]])
    atom_runtime.book.update_position(
        PositionState(
            source_key="source",
            execution_model_name="brackets",
            contract=held,
            quantity=2,
            target_quantity=5,
        )
    )
    portfolio = SavedPortfolio()
    assert portfolio.positions_for_blueprint(blueprint) == {held: 2}
    assert portfolio.positions_for_blueprint(held) == {held: 2}
    with pytest.raises(TypeError):
        portfolio.positions_for_blueprint(held)[held] = 7


def signal(source_key: str = "alpha", value: float = 1) -> Signal:
    return Signal(
        source_key=source_key,
        contract=ibi.Future(conId=1, symbol="ES", exchange="CME"),
        value=value,
        signal_type=SignalType.STATE,
        metadata={"atr": 10},
    )


def proposal(
    direction: Literal[-1, 0, 1] = 1,
    intent: PositionIntent = PositionIntent.OPEN,
) -> PositionProposal:
    return PositionProposal(
        signal=signal(),
        target_direction=direction,
        intent=intent,
    )


def test_fixed_size_allocator_preserves_proposal_inputs_for_wrapper():
    target = FixedSizeAllocator(3).target_for(proposal(-1))

    assert target.target_quantity == -3
    assert target.source_key == "alpha"
    assert target.contract == signal().contract
    assert target.intent is None
    assert target.metadata == {"atr": 10}


def test_fixed_size_allocator_uses_source_mapping():
    allocator = FixedSizeAllocator({"alpha": 2, "beta": 4})

    assert allocator.target_for(proposal()).target_quantity == 2


def test_fixed_size_allocator_missing_source_mapping_raises():
    with pytest.raises(KeyError):
        FixedSizeAllocator({"beta": 2}).target_for(proposal())


def test_fixed_size_allocator_callable_receives_proposal():
    allocator = FixedSizeAllocator(lambda incoming: incoming.signal.metadata["atr"] / 2)

    assert allocator.target_for(proposal()).target_quantity == 5


@pytest.mark.parametrize("size", [float("nan"), float("inf")])
def test_fixed_size_allocator_rejects_non_finite_size(size):
    with pytest.raises(ValueError, match="finite"):
        FixedSizeAllocator(size).target_for(proposal())


def test_portfolio_wrapper_emits_zero_or_one_target(atom_runtime):
    wrapper = PortfolioWrapper(FixedSizeAllocator(2))
    output = []
    wrapper.dataEvent += output.append

    wrapper.onData(proposal(direction=-1, intent=PositionIntent.REVERSE))

    assert len(output) == 1
    assert output[0].target_quantity == -2
    assert output[0].intent is PositionIntent.REVERSE


def test_portfolio_wrapper_rejects_wrong_message_at_runtime(atom_runtime):
    with pytest.raises(TypeError, match="only PositionProposal"):
        PortfolioWrapper(FixedSizeAllocator()).onData({"direction": 1})


def test_portfolio_wrapper_rejects_allocator_returning_multiple_targets(
    atom_runtime,
):
    class InvalidAllocator:
        def target_for(self, proposal):
            return [
                PositionTarget(contract=proposal.signal.contract, target_quantity=1)
            ]

    with pytest.raises(TypeError, match="multiple"):
        PortfolioWrapper(InvalidAllocator()).onData(proposal())


def test_portfolio_wrapper_rejects_changed_identity(atom_runtime):
    class InvalidAllocator:
        def target_for(self, proposal):
            return PositionTarget(
                contract=proposal.signal.contract,
                target_quantity=1,
                source_key="other",
            )

    with pytest.raises(ValueError, match="source_key"):
        PortfolioWrapper(InvalidAllocator()).onData(proposal())


class EchoPortfolio(Portfolio):
    def process(self, incoming: Signal) -> Iterable[PositionTarget]:
        value = incoming.value
        assert isinstance(value, float)
        return (
            PositionTarget(
                contract=incoming.contract,
                target_quantity=value,
            ),
        )


def test_direct_portfolio_emits_targets_and_omits_intent(atom_runtime):
    portfolio = EchoPortfolio()
    output = []
    portfolio.dataEvent += output.append

    portfolio.onData(signal(value=2))

    assert output[0].target_quantity == 2
    assert output[0].intent is None


def test_direct_portfolio_rejects_wrong_message_at_runtime(atom_runtime):
    with pytest.raises(TypeError, match="only Signal"):
        EchoPortfolio().onData({"value": 1})


def test_registered_portfolio_rejects_unknown_source(atom_runtime):
    portfolio = EchoPortfolio(sources={"alpha", "beta"})

    with pytest.raises(KeyError, match="Unknown"):
        portfolio.onData(signal(source_key="gamma"))

    assert portfolio.expected_sources == frozenset({"alpha", "beta"})


def test_dynamic_portfolio_has_no_completeness_policy(atom_runtime):
    portfolio = EchoPortfolio(sources=None)

    portfolio.onData(signal(source_key="new"))

    assert portfolio.expected_sources is None


def test_portfolio_rejects_unsupported_signal_type(atom_runtime):
    portfolio = EchoPortfolio(supported_signal_types={SignalType.STATE})
    event = Signal(
        source_key="alpha",
        contract=signal().contract,
        value=1,
        signal_type=SignalType.EVENT,
    )

    with pytest.raises(ValueError, match="does not support"):
        portfolio.onData(event)


class AnalogAggregatePortfolio(Portfolio):
    def __init__(self):
        super().__init__(sources={"alpha", "beta"})
        self.values = {}
        self.observation_times = []

    def process(self, incoming: Signal) -> Iterable[PositionTarget]:
        self.observation_times.append(incoming.as_of)
        if incoming.signal_type is SignalType.STATE:
            self.values[incoming.source_key] = incoming.value
        else:
            self.values[incoming.source_key] = (
                self.values.get(incoming.source_key, 0) + incoming.value
            )
        total = sum(self.values.values())
        return (
            PositionTarget(
                contract=incoming.contract,
                target_quantity=total,
            ),
            PositionTarget(
                contract=ibi.Future(
                    conId=2,
                    symbol="NQ",
                    exchange="CME",
                ),
                target_quantity=-total,
            ),
        )


def test_direct_analog_portfolio_replaces_state_and_accumulates_events(
    atom_runtime,
):
    portfolio = AnalogAggregatePortfolio()
    output = []
    portfolio.dataEvent += output.append
    observed_at = datetime(2026, 1, 1, tzinfo=timezone.utc)

    portfolio.onData(
        Signal(
            source_key="alpha",
            contract=signal().contract,
            value=2,
            signal_type=SignalType.STATE,
            as_of=observed_at,
        )
    )
    portfolio.onData(
        Signal(
            source_key="beta",
            contract=signal().contract,
            value=-0.5,
            signal_type=SignalType.EVENT,
        )
    )
    portfolio.onData(
        Signal(
            source_key="beta",
            contract=signal().contract,
            value=-0.5,
            signal_type=SignalType.EVENT,
        )
    )

    assert [target.target_quantity for target in output[-2:]] == [1, -1]
    assert portfolio.observation_times[0] is observed_at


class BinaryAggregatePortfolio(Portfolio):
    def __init__(self):
        super().__init__(supported_signal_types={SignalType.STATE})
        self.directions = {}

    def process(self, incoming: Signal) -> Iterable[PositionTarget]:
        value = incoming.value
        assert isinstance(value, float)
        self.directions[incoming.source_key] = (
            1 if value > 0 else -1 if value < 0 else 0
        )
        return (
            PositionTarget(
                contract=incoming.contract,
                target_quantity=sum(self.directions.values()),
            ),
        )


def test_direct_binary_portfolio_aggregates_source_directions(atom_runtime):
    portfolio = BinaryAggregatePortfolio()
    output = []
    portfolio.dataEvent += output.append

    portfolio.onData(signal(source_key="alpha", value=10))
    portfolio.onData(signal(source_key="beta", value=-0.1))

    assert output[-1].target_quantity == 0
