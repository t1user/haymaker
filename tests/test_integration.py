from datetime import datetime, timezone

import ib_insync as ibi
import pytest
from test_brick import data_for_df, df_brick  # noqa

from ib_tools.base import Atom, Pipe
from ib_tools.bracket_legs import FixedStop
from ib_tools.controller import Controller
from ib_tools.execution_models import BaseExecModel, EventDrivenExecModel
from ib_tools.manager import STATE_MACHINE
from ib_tools.portfolio import FixedPortfolio
from ib_tools.signals import BinarySignalProcessor


@pytest.fixture
def pipe(df_brick, data_for_df):  # noqa
    class FakeStateMachine:
        def position(self, key):
            return 0

        def locked(self, key):
            return 0

    sm = FakeStateMachine()

    class FakeController:
        out = None

        def trade(self, contract, order, action, exec_model):
            self.out = contract, order, action, exec_model

    controller = FakeController()

    # signal is 1, contract is NQ
    brick = df_brick
    # so this should result in action "OPEN"
    signal = BinarySignalProcessor(sm)
    # this porfolio does fixed amount = 1
    portfolio = FixedPortfolio()
    # on which exec_model should act by issuing Buy order
    exec_model = EventDrivenExecModel(stop=FixedStop(5), controller=controller)

    class SourceAtom(Atom):
        def run(self):
            self.dataEvent.emit(data_for_df)

    source = SourceAtom()

    Pipe(source, brick, signal, portfolio, exec_model)
    source.run()

    return controller.out


def test_exec_model_is_exec_model(pipe):
    contract, order, action, exec_model = pipe
    assert isinstance(exec_model, EventDrivenExecModel)


def test_contract_is_contract(pipe):
    contract, order, action, exec_model = pipe
    assert isinstance(contract, ibi.Contract)


def test_required_fields_in_data_present(pipe):
    _, _, _, exec_model = pipe
    data = exec_model.params["open"]
    assert set(["strategy", "contract", "amount", "signal", "action"]).issubset(
        set(data.keys())
    )


def test_order_is_order(pipe):
    _, order, _, _ = pipe
    assert isinstance(order, ibi.Order)


def test_order_is_a_buy_order(pipe):
    _, order, _, __ = pipe
    assert order.action == "BUY"


def test_order_is_for_one_contract(pipe):
    _, order, _, __ = pipe
    assert order.totalQuantity == 1


# #####################################
# Test sending orders
# #####################################


# THIS IS A SHITTY TEST BECAUSE IT WORKS WITH GLOBAL STATE
@pytest.fixture
def new_setup():
    class FakeTrader:
        def trade(self, contract: ibi.Contract, order: ibi.Order):
            return ibi.Trade(contract, order)

    class FakeController(Controller):
        trade_object = None

        def trade(self, *args, **kwargs):
            print(f"trade received: {args} {kwargs}")
            self.trade_object = super().trade(*args, **kwargs)

    ib = ibi.IB()
    controller = FakeController(STATE_MACHINE, ib, trader=FakeTrader())

    class Source(Atom):
        pass

    source = Source()
    em = BaseExecModel(controller=controller)
    source += em

    source.startEvent.emit({"strategy": "xxx", "execution_model": em})

    return ib, controller, source, em


def test_buy_position_registered(new_setup):
    ib, controller, source, em = new_setup

    data = {
        "signal": 1,
        "action": "OPEN",
        "amount": 1,
        "target_position": 1,
        "contract": ibi.ContFuture("NQ", "CME"),
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
    ib.execDetailsEvent.emit(trade_object, trade_object.fills[-1])

    assert em.position == 1


def test_sell_position_registered(new_setup):
    ib, controller, source, em = new_setup

    data = {
        "signal": -1,
        "action": "OPEN",
        "amount": 1,
        "target_position": -1,
        "contract": ibi.ContFuture("NQ", "CME"),
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
    ib.execDetailsEvent.emit(trade_object, trade_object.fills[-1])

    assert em.position == -1
