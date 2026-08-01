from collections.abc import Iterable
from datetime import datetime, timezone

import ib_insync as ibi

from haymaker.base import Atom, Pipe
from haymaker.components import (
    BracketExecutionModel,
    ExecutionRouter,
    ExecutionRule,
    FixedSizeAllocator,
    FixedStop,
    BinarySignalProcessor,
    Portfolio,
    PortfolioWrapper,
    PositionTarget,
    SerialTargetExecutionModel,
    Signal,
    SignalCalculation,
    SignalModel,
    SignalType,
)
from haymaker.controller import Controller


class FakeTrader:
    def __init__(self):
        self.trades = []

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
        return trade

    def position_for_contract(self, contract):
        return 0

    def positions(self):
        return {}


class Source(Atom):
    output_type = object

    def run(self, data):
        self.startEvent.emit({})
        self.dataEvent.emit(data)


class IntegrationSignalModel(SignalModel):
    def calculate_signal(self, data):
        return SignalCalculation(
            value=data,
            metadata={"atr": 5},
        )


def contract(symbol="ES", con_id=1):
    return ibi.Future(
        conId=con_id,
        symbol=symbol,
        exchange="CME",
        localSymbol=f"{symbol}M6",
    )


def test_one_to_one_pipeline_submits_attributed_open(atom_runtime):
    trader = FakeTrader()
    controller = Controller(trader=trader)
    atom_runtime.bind_controller(controller)
    source = Source()
    signal_model = IntegrationSignalModel("alpha", contract(), SignalType.STATE)
    processor = BinarySignalProcessor(respect_blocked_direction=True)
    wrapper = PortfolioWrapper(FixedSizeAllocator(2))
    execution = BracketExecutionModel(
        "alpha",
        name="alpha_brackets",
        stop=FixedStop(2),
    )
    Pipe(source, signal_model, processor, wrapper, execution)

    source.run(1)

    assert len(trader.trades) == 1
    trade = trader.trades[0]
    assert trade.order.action == "BUY"
    assert trade.order.totalQuantity == 2
    info = atom_runtime.book.order_by_id(trade.order.orderId)
    assert info.source_key == "alpha"
    assert info.execution_model_name == "alpha_brackets"
    assert info.position_id == atom_runtime.book.position_state("alpha").position_id


class AggregatePortfolio(Portfolio):
    def __init__(self):
        super().__init__(sources={"alpha", "beta"})
        self.values: dict[str, float] = {}

    def process(self, signal: Signal) -> Iterable[PositionTarget]:
        if not isinstance(signal.value, float):
            raise TypeError("AggregatePortfolio requires scalar Signals")
        self.values[signal.source_key] = signal.value
        total = sum(self.values.values())
        return (
            PositionTarget(
                contract=contract(),
                target_quantity=total,
                metadata={"sources": dict(self.values)},
            ),
        )


def test_direct_pipeline_routes_aggregate_target(atom_runtime):
    trader = FakeTrader()
    controller = Controller(trader=trader)
    atom_runtime.bind_controller(controller)
    portfolio = AggregatePortfolio()
    model = SerialTargetExecutionModel(name="serial")
    router = ExecutionRouter(
        [ExecutionRule(predicate=lambda target: True, model=model)]
    )
    portfolio.connect(router)

    portfolio.onData(
        Signal(
            source_key="alpha",
            contract=contract(),
            value=2,
            signal_type=SignalType.STATE,
        )
    )
    portfolio.onData(
        Signal(
            source_key="beta",
            contract=contract(),
            value=-1,
            signal_type=SignalType.STATE,
            created_at=datetime.now(timezone.utc),
        )
    )

    assert len(trader.trades) == 1
    assert atom_runtime.book.target_state("serial", contract()).target_quantity == 1
