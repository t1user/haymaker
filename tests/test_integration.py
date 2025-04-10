import asyncio
from datetime import datetime, timezone

import ib_insync as ibi
import pytest
from test_brick import data_for_df, df_brick  # noqa

from haymaker.base import Pipe
from haymaker.bracket_legs import FixedStop
from haymaker.controller import Controller
from haymaker.execution_models import BaseExecModel, EventDrivenExecModel
from haymaker.portfolio import AbstractBasePortfolio, FixedPortfolio, PortfolioWrapper
from haymaker.signals import BinarySignalProcessor
from haymaker.state_machine import Strategy, StrategyContainer
from haymaker.trader import Trader


@pytest.fixture
def portfolio():
    portfolio = FixedPortfolio()

    yield portfolio
    AbstractBasePortfolio.instance = None


@pytest.fixture
def pipe(df_brick, data_for_df, portfolio, Atom, strategy_saver):  # noqa
    class FakeStateMachine:
        strategy = StrategyContainer(strategy_saver)

        def locked(self, key):
            return 0

        def position_and_order_for_strategy(self, strategy_str: str):
            return 0

    sm = FakeStateMachine()

    class FakeController(Atom):
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

    class SourceAtom(Atom):
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
def new_setup(Atom):
    class FakeTrader:
        def trade(self, contract: ibi.Contract, order: ibi.Order):
            return ibi.Trade(contract, order)

    class FakeController(Controller):
        trade_object = None

        def trade(self, *args, **kwargs):
            self.trade_object = super().trade(*args, **kwargs)
            return self.trade_object

    controller = FakeController(trader=FakeTrader())

    class Source(Atom):
        pass

    source = Source()
    em = BaseExecModel(controller=controller)

    source += em

    source.startEvent.emit({"strategy": "xxx"})
    return controller, source, em


@pytest.mark.asyncio
async def test_buy_position_registered(new_setup):
    controller, source, em = new_setup

    data = {
        "signal": 1,
        "action": "OPEN",
        "amount": 1,
        "target_position": 1,
        "contract": ibi.Future(
            conId=551601561,
            symbol="ES",
            lastTradeDateOrContractMonth="20240621",
            multiplier="50",
            exchange="CME",
            currency="USD",
            localSymbol="ESM4",
            tradingClass="ES",
        ),
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
    await asyncio.sleep(0)
    assert em.data.position == 1


@pytest.mark.asyncio
async def test_sell_position_registered(new_setup):
    controller, source, em = new_setup

    data = {
        "signal": -1,
        "action": "OPEN",
        "amount": 1,
        "target_position": -1,
        "contract": ibi.Future(
            conId=551601561,
            symbol="ES",
            lastTradeDateOrContractMonth="20240621",
            multiplier="50",
            exchange="CME",
            currency="USD",
            localSymbol="ESM4",
            tradingClass="ES",
        ),
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
    trade_object.order.permId = 12345
    controller.ib.execDetailsEvent.emit(trade_object, trade_object.fills[-1])
    await asyncio.sleep(0)
    assert em.data.position == -1


@pytest.mark.asyncio
async def test_manual_order_created(Atom):

    class A(Atom):
        pass

    a = A()
    contract = ibi.Future(
        conId=551601561,
        symbol="ES",
        lastTradeDateOrContractMonth="20240621",
        multiplier="50",
        exchange="CME",
        currency="USD",
        localSymbol="ESM4",
        tradingClass="ES",
    )
    trade_object = ibi.Trade(
        contract=contract,
        order=ibi.Order(action="BUY", totalQuantity=1, orderId=-1, permId=12345),
    )
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
    controller = Controller(trader=Trader(Atom.ib))
    controller.release_hold()
    controller.ib.orderStatusEvent.emit(trade_object)
    controller.ib.execDetailsEvent.emit(trade_object, trade_object.fills[-1])
    await asyncio.sleep(0)
    assert a.sm._strategies.total_positions()[trade_object.contract] == 1
    assert "manual_strategy_ES" in a.sm._strategies
