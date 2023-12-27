import logging
import random
from datetime import datetime, timezone

import ib_insync as ibi
import pytest

from ib_tools.base import Atom
from ib_tools.bracket_legs import FixedStop, TakeProfitAsStopMultiple, TrailingStop
from ib_tools.execution_models import (
    AbstractExecModel,
    BaseExecModel,
    EventDrivenExecModel,
)
from ib_tools.state_machine import Model


def test_AbstraExecModel_is_abstract():
    with pytest.raises(TypeError):
        AbstractExecModel()


def test_BaseExecModel_instantiates():
    bem = BaseExecModel()
    assert isinstance(bem, BaseExecModel)


def test_EventDrivenExecModel_instantiates():
    edem = EventDrivenExecModel(stop=FixedStop(1))
    assert isinstance(edem, EventDrivenExecModel)


def test_EventDrivenExecModel_requires_stop():
    with pytest.raises(TypeError):
        EventDrivenExecModel()


def test_BaseExecModel_order_validator_works_with_correct_keys():
    open_order = {"orderType": "LMT", "lmtPrice": 5}
    bem = BaseExecModel({"open_order": open_order})
    assert bem.open_order == open_order


def test_BaseExecModel_order_validator_raises_with_incorrect_keys():
    open_order = {"orderType": "LMT", "price123": 5}
    with pytest.raises(ValueError) as excinfo:
        BaseExecModel({"open_order": open_order})
    assert "price123" in str(excinfo.value)


def test_position_id():
    em = EventDrivenExecModel(stop=FixedStop(10))
    em.onStart({"strategy": "xxx"})
    id1 = em.get_position_id()
    id2 = em.get_position_id()
    assert id1 == id2


def test_position_id_reset():
    em = EventDrivenExecModel(stop=FixedStop(10))
    em.onStart({"strategy": "xxx"})
    id1 = em.get_position_id()
    id2 = em.get_position_id(True)
    assert id1 != id2


def test_oca_group_EventDrivenExecModel():
    e = EventDrivenExecModel(
        stop=FixedStop(1), take_profit=TakeProfitAsStopMultiple(1, 2)
    )
    e.onStart({"strategy": "xxx"})
    oca_group = e.oca_group_generator()
    assert isinstance(oca_group, str)
    assert len(oca_group) > 10


def test_oca_group_unique_EventDrivenExecModel():
    e = EventDrivenExecModel(
        stop=FixedStop(1), take_profit=TakeProfitAsStopMultiple(1, 2)
    )
    e.onStart({"strategy": "xxx"})
    oca_group1 = e.oca_group_generator()
    oca_group2 = e.oca_group_generator()
    assert oca_group1 != oca_group2


def test_oca_group_is_not_position_id():
    e = EventDrivenExecModel(
        stop=FixedStop(1), take_profit=TakeProfitAsStopMultiple(1, 2)
    )
    e.onStart({"strategy": "xxx"})
    oca_group = e.oca_group_generator()
    position_id = e.get_position_id()
    assert oca_group != position_id


@pytest.fixture
def objects():
    class FakeController:
        contract = None
        order = None
        action = None
        trade_object = None

        def trade(
            self,
            strategy: str,
            contract: ibi.Contract,
            order: ibi.Order,
            action: str,
            data: Model,
        ):
            order.orderId = random.randint(1, 100)
            trade_object = ibi.Trade(order=order, contract=contract)
            self.contract = contract
            self.order = order
            self.action = action
            fill = ibi.Fill(
                contract,
                ibi.Execution(
                    execId="0000e1a7.656447c6.01.01",
                    time=datetime.now(timezone.utc),
                    side="SLD" if order.action == "SELL" else "BOT",
                    shares=order.totalQuantity,
                ),
                ibi.CommissionReport(),
                datetime.now(timezone.utc),
            )
            trade_object.fills.append(fill)
            self.trade_object = trade_object
            self.fill = fill
            return trade_object

        def trade_done(self):
            self.trade_object.fillEvent.emit(self.trade_object, self.fill)
            self.trade_object.filledEvent.emit(self.trade_object)

    class Source(Atom):
        pass

    controller = FakeController()
    source = Source()

    return controller, source


def test_EventDrivenExecModel_brackets_have_same_oca(objects):
    controller, source = objects
    em = EventDrivenExecModel(
        stop=TrailingStop(3),
        take_profit=TakeProfitAsStopMultiple(3, 3),
        controller=controller,
    )
    em.onStart({"strategy": "xxx"})
    source += em
    source.dataEvent.emit(
        {
            "signal": 1,
            "action": "OPEN",
            "amount": 1,
            "target_position": 1,
            "atr": 5,
            "contract": ibi.Future("NQ", "CME", conId=1),
        }
    )
    controller.trade_done()
    brackets = list(em.data.brackets.values())
    assert brackets[0].trade.order.ocaGroup == brackets[1].trade.order.ocaGroup


def test_EventDrivenExecModel_close_has_same_oca_as_brackets(objects):
    controller, source = objects
    em = EventDrivenExecModel(
        stop=TrailingStop(3),
        take_profit=TakeProfitAsStopMultiple(3, 3),
        controller=controller,
    )
    em.onStart({"strategy": "xxx"})
    source += em
    source.dataEvent.emit(
        {
            "signal": 1,
            "action": "OPEN",
            "amount": 1,
            "target_position": 1,
            "atr": 5,
            "contract": ibi.ContFuture("NQ", "CME", conId=1),
        }
    )
    controller.trade_done()
    brackets = list(em.data.brackets.values())
    em.position = 1
    source.dataEvent.emit(
        {
            "signal": -1,
            "action": "CLOSE",
            "amount": 1,
            "target_position": 0,
        }
    )
    controller.trade_done()
    assert controller.order.ocaGroup == brackets[0].trade.order.ocaGroup


def test_BaseExecModel_open_signal_generates_order(objects):
    controller, source = objects
    em = BaseExecModel(controller=controller)
    source += em
    data = {
        "signal": 1,
        "action": "OPEN",
        "amount": 1,
        "target_position": 1,
        "contract": ibi.ContFuture("NQ", "CME"),
    }
    source.startEvent.emit({"strategy": "xxx"})
    source.dataEvent.emit(data)
    assert controller.order.action == "BUY"


def test_BaseExecModel_no_close_order_without_position(objects):
    controller, source = objects
    em = BaseExecModel(controller=controller)
    em.onStart({"strategy": "xxx"})
    source += em

    data = {
        "signal": -1,
        "action": "CLOSE",
        "amount": 1,
        "target_position": 0,
        "contract": ibi.ContFuture("NQ", "CME"),
    }
    source.dataEvent.emit(data)
    assert controller.order is None


def test_BaseExecModel_faulty_close_order_logs(objects, caplog):
    controller, source = objects
    em = BaseExecModel(controller=controller)
    source += em
    em.onStart({"strategy": "xxx"})

    data = {
        "signal": -1,
        "action": "CLOSE",
        "amount": 1,
        "target_position": 0,
        "contract": ibi.ContFuture("NQ", "CME"),
    }
    source.dataEvent.emit(data)
    assert caplog.record_tuples[0][1] == logging.ERROR


def test_BaseExecModel_close_signal_generates_order(objects):
    controller, source = objects
    em = BaseExecModel(controller=controller)
    em.onStart({"strategy": "xxx"})
    source += em

    data_open = {
        "signal": 1,
        "action": "OPEN",
        "amount": 1,
        "target_position": 1,
        "contract": ibi.Future("NQ", "CME"),
    }
    source.dataEvent.emit(data_open)
    em.data.position = 1
    data_close = {
        "signal": -1,
        "action": "CLOSE",
        "amount": 1,
        "target_position": 0,
        "contract": ibi.Future("NQ", "CME"),
    }
    source.dataEvent.emit(data_close)
    assert controller.order.action == "SELL"


# def test_BaseExecModel_position_updated_after_BUY_trade(objects):
#     controller, source, em = objects
#     data = {
#         "signal": 1,
#         "action": "OPEN",
#         "amount": 1,
#         "target_position": 1,
#         "contract": ibi.ContFuture("NQ", "CME"),
#     }
#     source.dataEvent.emit(data)
#     assert em.position == 1


# def test_BaseExecModel_position_updated_after_SELL_trade(objects):
#     controller, source, em = objects
#     data = {
#         "signal": -1,
#         "action": "OPEN",
#         "amount": 1,
#         "target_position": -1,
#         "contract": ibi.ContFuture("NQ", "CME"),
#     }
#     source.dataEvent.emit(data)
#     assert em.position == -1


def test_passed_order_kwargs_update_defaults():
    class FakeController:
        contract = None
        order = None
        action = None

        def trade(
            self,
            strategy: str,
            contract: ibi.Contract,
            order: ibi.Order,
            action: str,
            model: Model,
        ):
            self.contract = contract
            self.order = order
            self.action = action

    controller = FakeController()
    em = BaseExecModel(orders={"open_order": {"algoParams": ""}}, controller=controller)

    class Source(Atom):
        pass

    source = Source()
    source += em
    source.startEvent.emit({"strategy": "xxx"})
    source.dataEvent.emit(
        {
            "signal": 1,
            "action": "OPEN",
            "amount": 1,
            "target_position": 1,
            "contract": ibi.ContFuture("NQ", "CME"),
        }
    )
    assert controller.order.algoParams == ""
