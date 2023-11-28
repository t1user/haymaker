import logging
import random
from datetime import datetime, timezone
from typing import Optional

import ib_insync as ibi
import pytest

from ib_tools import bracket_legs, misc
from ib_tools.base import Atom
from ib_tools.bracket_legs import FixedStop, TakeProfitAsStopMultiple
from ib_tools.execution_models import (
    AbstractExecModel,
    BaseExecModel,
    EventDrivenExecModel,
    OcaExecModel,
)


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
    id1 = em.position_id()
    id2 = em.position_id()
    assert id1 == id2


def test_position_id_reset():
    em = EventDrivenExecModel(stop=FixedStop(10))
    id1 = em.position_id()
    id2 = em.position_id(True)
    assert id1 != id2


def test_oca_group_OcaExecModel():
    e = OcaExecModel(stop=FixedStop(1), take_profit=TakeProfitAsStopMultiple(1, 2))
    oca_group = e.oca_group()
    assert isinstance(oca_group, str)
    assert len(oca_group) > 10


def test_oca_group_unique_OcaExecModel():
    e = OcaExecModel(stop=FixedStop(1), take_profit=TakeProfitAsStopMultiple(1, 2))
    oca_group1 = e.oca_group()
    oca_group2 = e.oca_group()
    assert oca_group1 != oca_group2


def test_oca_group_is_not_position_id():
    e = OcaExecModel(stop=FixedStop(1), take_profit=TakeProfitAsStopMultiple(1, 2))
    oca_group = e.oca_group
    position_id = e.position_id()
    assert oca_group != position_id


def test_OcaExecModel_requires_two_brackets():
    with pytest.raises(TypeError):
        OcaExecModel(stop=FixedStop(1))


@pytest.fixture
def objects():
    class FakeController:
        contract = None
        order = None
        action = None

        def trade(
            self,
            contract: ibi.Contract,
            order: ibi.Order,
            action: str,
            exec_model: AbstractExecModel,
            callback: Optional[misc.Callback],
        ):
            order.orderId = random.randint(1, 100)
            trade = ibi.Trade(order=order, contract=contract)

            if callback is not None:
                callback(trade)

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
            trade.fills.append(fill)
            trade.fillEvent.emit(trade, fill)
            trade.filledEvent.emit(trade)

    class Source(Atom):
        pass

    controller = FakeController()
    source = Source()
    em = BaseExecModel(controller=controller)
    source += em

    return controller, source, em


def test_OcaExecModel_brackets_have_same_oca(objects):
    controller, source, _ = objects
    em = OcaExecModel(
        stop=bracket_legs.TrailingStop(3),
        take_profit=bracket_legs.TakeProfitAsStopMultiple(3, 3),
        controller=controller,
    )
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
    brackets = list(em.brackets.values())
    assert brackets[0].trade.order.ocaGroup == brackets[1].trade.order.ocaGroup


def test_BaseExecModel_open_signal_generates_order(objects):
    controller, source, em = objects
    data = {
        "signal": 1,
        "action": "OPEN",
        "amount": 1,
        "target_position": 1,
        "contract": ibi.ContFuture("NQ", "CME"),
    }
    source.dataEvent.emit(data)
    assert controller.order.action == "BUY"


def test_BaseExecModel_no_close_order_without_position(objects):
    controller, source, em = objects
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
    controller, source, em = objects
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
    controller, source, em = objects
    data_open = {
        "signal": 1,
        "action": "OPEN",
        "amount": 1,
        "target_position": 1,
        "contract": ibi.ContFuture("NQ", "CME"),
    }
    source.dataEvent.emit(data_open)
    data_close = {
        "signal": -1,
        "action": "CLOSE",
        "amount": 1,
        "target_position": 0,
        "contract": ibi.ContFuture("NQ", "CME"),
    }
    source.dataEvent.emit(data_close)
    assert controller.order.action == "SELL"


def test_BaseExecModel_position_updated_after_BUY_trade(objects):
    controller, source, em = objects
    data = {
        "signal": 1,
        "action": "OPEN",
        "amount": 1,
        "target_position": 1,
        "contract": ibi.ContFuture("NQ", "CME"),
    }
    source.dataEvent.emit(data)
    assert em.position == 1


def test_BaseExecModel_position_updated_after_SELL_trade(objects):
    controller, source, em = objects
    data = {
        "signal": -1,
        "action": "OPEN",
        "amount": 1,
        "target_position": -1,
        "contract": ibi.ContFuture("NQ", "CME"),
    }
    source.dataEvent.emit(data)
    assert em.position == -1


def test_passed_order_kwargs_update_defaults():
    class FakeController:
        contract = None
        order = None
        action = None

        def trade(
            self,
            contract: ibi.Contract,
            order: ibi.Order,
            action: str,
            exec_model: AbstractExecModel,
            callback: Optional[misc.Callback],
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
