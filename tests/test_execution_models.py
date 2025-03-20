import logging
import random
from dataclasses import dataclass
from datetime import datetime, timezone

import ib_insync as ibi
import pytest

from haymaker.bracket_legs import FixedStop, TakeProfitAsStopMultiple, TrailingStop
from haymaker.controller import Controller
from haymaker.execution_models import (
    AbstractExecModel,
    BaseExecModel,
    EventDrivenExecModel,
    OrderKey,
)
from haymaker.state_machine import Strategy


def test_AbstraExecModel_is_abstract(controller: Controller):
    with pytest.raises(TypeError):
        AbstractExecModel(controller=controller)  # type: ignore


def test_BaseExecModel_instantiates(controller: Controller):
    bem = BaseExecModel(controller=controller)
    assert isinstance(bem, BaseExecModel)


def test_EventDrivenExecModel_instantiates(controller: Controller):
    edem = EventDrivenExecModel(stop=FixedStop(1), controller=controller)
    assert isinstance(edem, EventDrivenExecModel)


def test_EventDrivenExecModel_requires_stop(controller: Controller):
    with pytest.raises(TypeError):
        EventDrivenExecModel(controller=controller)


def test_BaseExecModel_order_validator_works_with_correct_keys(controller: Controller):
    open_order = {"orderType": "LMT", "lmtPrice": 5}
    bem = BaseExecModel(open_order=open_order, controller=controller)
    assert isinstance(bem.open_order, dict)
    assert bem.open_order["lmtPrice"] == 5


def test_BaseExecModel_order_validator_raises_with_incorrect_keys(
    controller: Controller,
):
    open_order = {"orderType": "LMT", "price123": 5}
    with pytest.raises(ValueError) as excinfo:
        BaseExecModel(open_order=open_order, controller=controller)
    assert "price123" in str(excinfo.value)


def test_position_id(controller: Controller):
    em = EventDrivenExecModel(stop=FixedStop(10), controller=controller)
    em.onStart({"strategy": "xxx"})
    id1 = em.get_position_id()
    id2 = em.get_position_id()
    assert id1 == id2


def test_position_id_reset(controller: Controller):
    em = EventDrivenExecModel(stop=FixedStop(10), controller=controller)
    em.onStart({"strategy": "xxx"})
    id1 = em.get_position_id()
    id2 = em.get_position_id(True)
    assert id1 != id2


def test_oca_group_EventDrivenExecModel(controller: Controller):
    e = EventDrivenExecModel(
        stop=FixedStop(1),
        take_profit=TakeProfitAsStopMultiple(1, 2),
        controller=controller,
    )
    e.onStart({"strategy": "xxx"})
    oca_group = e.oca_group_generator()
    assert isinstance(oca_group, str)
    assert len(oca_group) > 10


def test_oca_group_unique_EventDrivenExecModel(controller: Controller):
    e = EventDrivenExecModel(
        stop=FixedStop(1),
        take_profit=TakeProfitAsStopMultiple(1, 2),
        controller=controller,
    )
    e.onStart({"strategy": "xxx"})
    oca_group1 = e.oca_group_generator()
    oca_group2 = e.oca_group_generator()
    assert oca_group1 != oca_group2


def test_oca_group_is_not_position_id(controller: Controller):
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

    @dataclass
    class Data:
        contract: ibi.Contract | None = None
        order: ibi.Order | None = None
        action: str | None = None
        trade_object: ibi.Trade | None = None
        fill: ibi.Fill | None = None
        position: float | None = None

        def trade_done(self):
            assert self.trade_object is not None
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

    controller = FakeController(FakeTrader())  # type: ignore
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
    # em.position = 1
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
    em = BaseExecModel(open_order={"algoParams": ""}, controller=controller)

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


def test_EventDrivenExecModel_bracket_params_override_detaults(Atom, trade):
    """
    Create a setup where `FakeTrader` will record received order that
    we can compare with expectations.

    `stop` parameter passed to `EventDrivenExecModel` is `TrailingStop` so
    resulting bracket order must be `TRAIL` even though default is `STP`.
    """

    class FakeTrader:
        order = None

        def trade(self, contract, order):
            self.order = order
            return ibi.Trade(order=ibi.Order(orderId=1))

    fake_trader = FakeTrader()

    em = EventDrivenExecModel(
        stop=TrailingStop(2, vol_field="my_vol_field"),
        controller=Controller(fake_trader),
    )
    em._attach_bracket(trade, {"my_vol_field": 10})

    assert fake_trader.order.orderType == "TRAIL"
    # basis for stop distance calculation is defined as `my_vol_field`
    # its value is given as 10 and stop multiple is 2
    assert fake_trader.order.auxPrice == 20


def test_OrderKey_picks_correct_order_low_level(controller):
    em = BaseExecModel(open_order={"orderType": "STP"}, controller=controller)
    my_order = em._order(OrderKey.open_order, {})
    assert my_order.orderType == "STP"


def test_OrderKey_picks_correct_order_higher_level(Atom):

    class FakeTrader:
        order = None

        def trade(self, contract, order):
            self.order = order
            return ibi.Trade(order=ibi.Order(orderId=1))

    fake_trader = FakeTrader()

    em = BaseExecModel(
        open_order={"orderType": "STPLMT"},
        close_order={"orderType": "TRAIL"},
        controller=Controller(fake_trader),
    )
    em.open({"contract": ibi.Future("NQ", "CME"), "signal": 1, "amount": 1})
    assert fake_trader.order.orderType == "STPLMT"


def test_OrderKey_picks_correct_order_higher_level_2(Atom):

    class FakeTrader:
        order = None

        def trade(self, contract, order):
            self.order = order
            return ibi.Trade(order=ibi.Order(orderId=1))

    fake_trader = FakeTrader()

    em = BaseExecModel(
        open_order={"orderType": "STPLMT"},
        close_order={"orderType": "TRAIL"},
        controller=Controller(fake_trader),
    )
    em.strategy = "xxx"
    em.open({"contract": ibi.Future("NQ", "CME"), "signal": 1, "amount": 1})
    # brute forcing position here because we're not properly updating StateMachine
    # and execution models abandon close positions for non-existing positions
    em.data.position = 1
    em.close({"contract": ibi.Future("NQ", "CME"), "signal": -1, "amount": 1})
    assert fake_trader.order.orderType == "TRAIL"
