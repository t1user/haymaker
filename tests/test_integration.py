from datetime import datetime, timezone

import ib_insync as ibi
import pytest
from test_brick import data_for_df, df_brick  # noqa

from ib_tools.base import Pipe
from ib_tools.bracket_legs import FixedStop
from ib_tools.controller import Controller
from ib_tools.execution_models import BaseExecModel, EventDrivenExecModel
from ib_tools.portfolio import AbstractBasePortfolio, FixedPortfolio, PortfolioWrapper
from ib_tools.signals import BinarySignalProcessor
from ib_tools.state_machine import Strategy, StrategyContainer


@pytest.fixture
def portfolio():
    portfolio = FixedPortfolio()

    yield portfolio
    AbstractBasePortfolio.instance = None


@pytest.fixture
def pipe(df_brick, data_for_df, portfolio, atom):  # noqa
    class FakeStateMachine:
        strategy = StrategyContainer()

        def locked(self, key):
            return 0

    sm = FakeStateMachine()

    class FakeController(atom):
        out = None

        def trade(self, strategy, contract, order, action, data):
            self.out = strategy, contract, order, action, data

    controller = FakeController()
    # signal is 1, contract is NQ
    brick = df_brick
    # so this should result in action "OPEN"
    signal = BinarySignalProcessor(state_machine=sm)

    # on which exec_model should act by issuing Buy order
    exec_model = EventDrivenExecModel(stop=FixedStop(5), controller=controller)

    class SourceAtom(atom):
        def run(self):
            # this should ensure setting "strategy" attr on exec_model
            # brick has this attr so it will emit it on start
            # and every subsequent Atom down the chain should set it on start
            self.startEvent.emit({})
            self.dataEvent.emit(data_for_df)

    source = SourceAtom()

    Pipe(source, brick, signal, PortfolioWrapper(), exec_model)
    source.run()

    return controller.out


def test_strategy_is_strategy(pipe):
    strategy, contract, order, action, data = pipe
    # this is the strategy that was set on Brick object
    assert strategy == "eska_NQ"


def test_data_is_dict(pipe):
    strategy, contract, order, action, data = pipe

    assert isinstance(data, Strategy)


def test_contract_is_contract(pipe):
    strategy, contract, order, action, data = pipe
    assert isinstance(contract, ibi.Contract)


def test_required_fields_in_data_present(pipe):
    _, _, _, _, data = pipe
    data_ = data.params["open"]
    assert set(["strategy", "contract", "amount", "signal", "action"]).issubset(
        set(data_.keys())
    )


def test_order_is_order(pipe):
    _, _, order, _, _ = pipe
    assert isinstance(order, ibi.Order)


def test_order_is_a_buy_order(pipe):
    _, _, order, _, _ = pipe
    assert order.action == "BUY"


def test_order_is_for_one_contract(pipe):
    _, _, order, _, __ = pipe
    assert order.totalQuantity == 1


# #####################################
# Test position recorded
# #####################################


@pytest.fixture
def new_setup(atom):
    class FakeTrader:
        def trade(self, contract: ibi.Contract, order: ibi.Order):
            return ibi.Trade(contract, order)

    class FakeController(Controller):
        trade_object = None

        def trade(self, *args, **kwargs):
            self.trade_object = super().trade(*args, **kwargs)

    controller = FakeController(trader=FakeTrader())

    class Source(atom):
        pass

    source = Source()
    em = BaseExecModel(controller=controller)

    source += em

    source.startEvent.emit({"strategy": "xxx"})
    return controller, source, em


def test_buy_position_registered(new_setup):
    controller, source, em = new_setup

    data = {
        "signal": 1,
        "action": "OPEN",
        "amount": 1,
        "target_position": 1,
        "contract": ibi.Future("NQ", "CME"),
    }
    source.dataEvent.emit(data)
    trade_object = controller.trade_object
    trade_object.fills.append(
        ibi.Fill(
            contract=trade_object.contract,
            execution=ibi.Execution(
                execId="0000e1a7.656447c6.01.01",
                shares=trade_object.order.totalQuantity,
                side="SLD" if trade_object.order.action == "SELL" else "BOT",
            ),
            commissionReport=ibi.CommissionReport(commission=1.0, realizedPNL=0.0),
            time=datetime.now(timezone.utc),
        )
    )
    controller.ib.execDetailsEvent.emit(trade_object, trade_object.fills[-1])
    assert em.data.position == 1


def test_sell_position_registered(new_setup):
    controller, source, em = new_setup

    data = {
        "signal": -1,
        "action": "OPEN",
        "amount": 1,
        "target_position": -1,
        "contract": ibi.Future("NQ", "CME"),
    }
    source.dataEvent.emit(data)
    trade_object = controller.trade_object
    trade_object.fills.append(
        ibi.Fill(
            contract=trade_object.contract,
            execution=ibi.Execution(
                execId="0000e1a7.656447c6.01.01",
                shares=trade_object.order.totalQuantity,
                side="SLD" if trade_object.order.action == "SELL" else "BOT",
            ),
            commissionReport=ibi.CommissionReport(commission=1.0, realizedPNL=0.0),
            time=datetime.now(timezone.utc),
        )
    )
    controller.ib.execDetailsEvent.emit(trade_object, trade_object.fills[-1])
    assert em.data.position == -1
