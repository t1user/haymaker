import ib_insync as ibi
import pytest
from test_brick import data_for_df, df_brick  # noqa

from ib_tools.base import Atom, Pipe
from ib_tools.bracket_legs import FixedStop
from ib_tools.execution_models import EventDrivenExecModel
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
        def trade(self, contract, order, action, exec_model, callback):
            self.out = contract, order, action, exec_model, callback

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
    contract, order, action, exec_model, callback = pipe
    assert isinstance(exec_model, EventDrivenExecModel)


def test_contract_is_contract(pipe):
    contract, order, action, exec_model, callback = pipe
    assert isinstance(contract, ibi.Contract)


def test_required_fields_in_data_present(pipe):
    _, _, _, exec_model, _ = pipe
    data = exec_model.params["open"]
    assert set(
        ["strategy", "contract", "exec_model", "amount", "signal", "action"]
    ).issubset(set(data.keys()))


def test_order_is_order(pipe):
    _, order, _, _, _ = pipe
    assert isinstance(order, ibi.Order)


def test_order_is_a_buy_order(pipe):
    _, order, _, _, _ = pipe
    assert order.action == "BUY"


def test_order_is_for_one_contract(pipe):
    _, order, _, _, _ = pipe
    assert order.totalQuantity == 1


def test_callback_is_callable(pipe):
    _, _, _, _, callback = pipe
    assert callable(callback)
