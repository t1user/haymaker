import logging
import random
from datetime import datetime, timezone

import ib_insync as ibi
import pytest

from haymaker.bracket_legs import FixedStop, TakeProfitAsStopMultiple, TrailingStop
from haymaker.controller import Controller
from haymaker.execution_models import (
    AbstractExecModel,
    BaseExecModel,
    EventDrivenExecModel,
)
from haymaker.state_machine import Strategy


def test_AbstraExecModel_is_abstract(controller):
    with pytest.raises(TypeError):
        AbstractExecModel(controller=controller)


def test_BaseExecModel_instantiates(controller):
    bem = BaseExecModel(controller=controller)
    assert isinstance(bem, BaseExecModel)


def test_EventDrivenExecModel_instantiates(controller):
    edem = EventDrivenExecModel(stop=FixedStop(1), controller=controller)
    assert isinstance(edem, EventDrivenExecModel)


def test_EventDrivenExecModel_requires_stop(controller):
    with pytest.raises(TypeError):
        EventDrivenExecModel(controller=controller)


def test_BaseExecModel_order_validator_works_with_correct_keys(controller):
    open_order = {"orderType": "LMT", "lmtPrice": 5}
    bem = BaseExecModel({"open_order": open_order}, controller=controller)
    assert bem.open_order == open_order


def test_BaseExecModel_order_validator_raises_with_incorrect_keys(controller):
    open_order = {"orderType": "LMT", "price123": 5}
    with pytest.raises(ValueError) as excinfo:
        BaseExecModel({"open_order": open_order}, controller=controller)
    assert "price123" in str(excinfo.value)


def test_position_id(controller):
    em = EventDrivenExecModel(stop=FixedStop(10), controller=controller)
    em.onStart({"strategy": "xxx"})
    id1 = em.get_position_id()
    id2 = em.get_position_id()
    assert id1 == id2


def test_position_id_reset(controller):
    em = EventDrivenExecModel(stop=FixedStop(10), controller=controller)
    em.onStart({"strategy": "xxx"})
    id1 = em.get_position_id()
    id2 = em.get_position_id(True)
    assert id1 != id2


def test_oca_group_EventDrivenExecModel(controller):
    e = EventDrivenExecModel(
        stop=FixedStop(1),
        take_profit=TakeProfitAsStopMultiple(1, 2),
        controller=controller,
    )
    e.onStart({"strategy": "xxx"})
    oca_group = e.oca_group_generator()
    assert isinstance(oca_group, str)
    assert len(oca_group) > 10


def test_oca_group_unique_EventDrivenExecModel(controller):
    e = EventDrivenExecModel(
        stop=FixedStop(1),
        take_profit=TakeProfitAsStopMultiple(1, 2),
        controller=controller,
    )
    e.onStart({"strategy": "xxx"})
    oca_group1 = e.oca_group_generator()
    oca_group2 = e.oca_group_generator()
    assert oca_group1 != oca_group2


def test_oca_group_is_not_position_id(controller):
    e = EventDrivenExecModel(
        stop=FixedStop(1),
        take_profit=TakeProfitAsStopMultiple(1, 2),
        controller=controller,
    )
    e.onStart({"strategy": "xxx"})
    oca_group = e.oca_group_generator()
    position_id = e.get_position_id()
    assert oca_group != position_id


@pytest.fixture
def objects(Atom):

    class Data:
        contract = None
        order = None
        action = None
        trade_object = None
        fill = None
        position = None

        def trade_done(self):
            self.trade_object.fillEvent.emit(self.trade_object, self.fill)
            self.trade_object.filledEvent.emit(self.trade_object)

    output_data = Data()

    class FakeTrader:
        """
        All it cares about is accepting correct data and returning
        Trade object with desired properties.
        """

        def trade(self, contract, order) -> ibi.Trade:
            order.orderId = random.randint(1, 100)
            trade_object = ibi.Trade(order=order, contract=contract)
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
            output_data.trade_object = trade_object
            output_data.fill = fill
            return trade_object

    class FakeController(Controller):
        """
        It's a regular `Controller object, the only modification being
        that it records data passed to :meth:`Trade`
        """

        def trade(
            self,
            strategy: str,
            contract: ibi.Contract,
            order: ibi.Order,
            action: str,
            data: Strategy,
        ):
            output_data.contract = contract
            output_data.order = order
            output_data.action = action
            return super().trade(strategy, contract, order, action, data)

    class Source(Atom):
        pass

    controller = FakeController(FakeTrader())
    source = Source()

    return controller, source, output_data


def test_EventDrivenExecModel_brackets_have_same_oca(objects):
    controller, source, data = objects
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
    data.trade_done()
    brackets = list(em.data.brackets.values())
    assert brackets[0].trade.order.ocaGroup == brackets[1].trade.order.ocaGroup


def test_EventDrivenExecModel_close_has_same_oca_as_brackets(objects):
    controller, source, data = objects
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
    data.trade_done()
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
    data.trade_done()
    assert data.order.ocaGroup == brackets[0].trade.order.ocaGroup


def test_BaseExecModel_open_signal_generates_order(objects):
    controller, source, output_data = objects
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
    assert output_data.order.action == "BUY"


def test_BaseExecModel_no_close_order_without_position(objects):
    controller, source, output_data = objects
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
    assert output_data.order is None


def test_BaseExecModel_faulty_close_order_logs(objects, caplog):
    """
    Execution model logs an attempt to close a non-existing position.
    """
    controller, source, data = objects
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
    assert caplog.record_tuples[-1][1] == logging.ERROR


def test_BaseExecModel_close_signal_generates_order(objects):
    controller, source, data = objects
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
    assert data.order.action == "SELL"


def test_passed_order_kwargs_update_defaults(Atom, objects):
    controller, source, data = objects
    # these are non defaults, so assert will check whether defaults
    # have been successfully overridden
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
    assert data.order.algoParams == ""
