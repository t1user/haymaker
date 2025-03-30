import asyncio
from typing import Any

import eventkit as ev  # type: ignore
import ib_insync as ibi
import pytest
from helpers import wait_for_condition

from haymaker.state_machine import (
    OrderContainer,
    OrderInfo,
    StateMachine,
    Strategy,
    StrategyContainer,
)

strategy_defaults: dict[str, Any] = {
    "active_contract": None,
    "position": 0.0,
    "params": {},
    "lock": 0,
    "position_id": "",
}

ce = ev.Event("strategyChangeEvent")


def test_StateMachine_is_singleton(state_machine):
    """
    Atempt to instantiate :class:`StateMachine` more than once
    should raise `TypeError`
    """
    with pytest.raises(TypeError):
        StateMachine()


def test_OrderInfo_unpackable():
    o1 = OrderInfo("coolstrategy", "OPEN", ibi.Trade(), None)
    strategy, action, trade, exec_model, active = o1
    assert strategy == "coolstrategy"
    assert action == "OPEN"
    assert not active


def test_OrderInfo_iterator():
    trade = ibi.Trade()
    o1 = OrderInfo("coolstrategy", "OPEN", trade, None)
    for i, j in zip(o1, ["coolstrategy", "OPEN", trade, None]):
        assert i == j


def test_OrderInfo_not_active():
    trade1 = ibi.Trade(orderStatus=ibi.OrderStatus(orderId=2, status="Filled"))
    o1 = OrderInfo("coolstrategy", "STOP", trade1, None)
    assert o1.active is False


def test_OrderInfo_active():
    trade1 = ibi.Trade(orderStatus=ibi.OrderStatus(orderId=2, status="Submitted"))
    o1 = OrderInfo("coolstrategy", "STOP", trade1, None)
    assert o1.active is True


def test_OrderInfo_amount_long():
    trade1 = ibi.Trade(
        order=ibi.Order(action="BUY", totalQuantity=1, permId=1234),
        orderStatus=ibi.OrderStatus(orderId=2, status="Submitted"),
    )
    o1 = OrderInfo("coolstrategy", "OPEN", trade1, None)
    assert o1.amount == 1


def test_OrderInfo_amount_shoft():
    trade1 = ibi.Trade(
        order=ibi.Order(action="SELL", totalQuantity=1, permId=1234),
        orderStatus=ibi.OrderStatus(orderId=2, status="Submitted"),
    )
    o1 = OrderInfo("coolstrategy", "OPEN", trade1, None)
    assert o1.amount == -1


def test_OrderContainer_strategy_gets_correct_orders():
    o1 = OrderInfo(
        "coolstrategy",
        "OPEN",
        ibi.Trade(orderStatus=ibi.OrderStatus(orderId=2, status="Submitted")),
        None,
    )
    o2 = OrderInfo(
        "coolstrategy",
        "STOP",
        ibi.Trade(orderStatus=ibi.OrderStatus(orderId=2, status="Submitted")),
        None,
    )
    o3 = OrderInfo(
        "suckystrategy",
        "STOP",
        ibi.Trade(orderStatus=ibi.OrderStatus(orderId=2, status="Submitted")),
        None,
    )

    orders = OrderContainer()
    orders[1] = o1
    orders[2] = o2
    orders[3] = o3

    assert list(orders.strategy("coolstrategy")) == [o1, o2]


def test_OrderContainer_del_works_correctly():
    o1 = OrderInfo(
        "coolstrategy",
        "OPEN",
        ibi.Trade(orderStatus=ibi.OrderStatus(orderId=2, status="Submitted")),
        None,
    )
    o2 = OrderInfo(
        "coolstrategy",
        "STOP",
        ibi.Trade(orderStatus=ibi.OrderStatus(orderId=2, status="Submitted")),
        None,
    )
    o3 = OrderInfo(
        "suckystrategy",
        "STOP",
        ibi.Trade(orderStatus=ibi.OrderStatus(orderId=2, status="Submitted")),
        None,
    )

    orders = OrderContainer()
    orders[1] = o1
    orders[2] = o2
    orders[3] = o3
    del orders[1]

    assert list(orders.strategy("coolstrategy")) == [o2]


def test_OrderContainer_get_order():
    o1 = OrderInfo("coolstrategy", "OPEN", ibi.Trade(), None)
    o2 = OrderInfo("coolstrategy", "STOP", ibi.Trade(), None)
    o3 = OrderInfo("suckystrategy", "STOP", ibi.Trade(), None)

    orders = OrderContainer()
    orders[1] = o1
    orders[2] = o2
    orders[3] = o3
    assert orders.get(1) == o1


def test_OrderContainer_get_order_default():
    o1 = OrderInfo("coolstrategy", "OPEN", ibi.Trade(), None)
    o2 = OrderInfo("coolstrategy", "STOP", ibi.Trade(), None)
    o3 = OrderInfo("suckystrategy", "STOP", ibi.Trade(), None)

    orders = OrderContainer()
    orders[1] = o1
    orders[2] = o2
    orders[3] = o3
    assert orders.get(10, "xyz") == "xyz"


def test_OrderContainer_get_item():
    o1 = OrderInfo("coolstrategy", "OPEN", ibi.Trade(), None)
    o2 = OrderInfo("coolstrategy", "STOP", ibi.Trade(), None)
    o3 = OrderInfo("suckystrategy", "STOP", ibi.Trade(), None)

    orders = OrderContainer()
    orders[1] = o1
    orders[2] = o2
    orders[3] = o3
    assert orders[1] == o1


def test_state_machine_get_order(state_machine):
    o1 = OrderInfo("coolstrategy", "OPEN", ibi.Trade(), None)
    o2 = OrderInfo("coolstrategy", "STOP", ibi.Trade(), None)
    o3 = OrderInfo("suckystrategy", "STOP", ibi.Trade(), None)

    orders = OrderContainer()
    orders[1] = o1
    orders[2] = o2
    orders[3] = o3
    state_machine._orders = orders
    assert state_machine.order.get(1) == o1


def test_state_machine_delete_order(state_machine):
    o1 = OrderInfo("coolstrategy", "OPEN", ibi.Trade(), None)
    o2 = OrderInfo("coolstrategy", "STOP", ibi.Trade(), None)
    o3 = OrderInfo("suckystrategy", "STOP", ibi.Trade(), None)

    orders = OrderContainer()
    orders[1] = o1
    orders[2] = o2
    orders[3] = o3
    state_machine._orders = orders
    state_machine.delete_order(1)
    assert list(state_machine._orders.values()) == [o2, o3]


def test_OrderContainer_active_only_flag_in_get():
    o1 = OrderInfo("coolstrategy", "OPEN", ibi.Trade(), None)
    o2 = OrderInfo(
        "coolstrategy",
        "STOP",
        ibi.Trade(orderStatus=ibi.OrderStatus(orderId=2, status="Filled")),
        None,
    )

    orders = OrderContainer()
    orders[1] = o1
    orders[2] = o2
    assert orders.get(2, active_only=True) is None


def test_OrderContainer_default_in_get_works():
    o1 = OrderInfo("coolstrategy", "OPEN", ibi.Trade(), {})
    o2 = OrderInfo("coolstrategy", "STOP", ibi.Trade, {})

    orders = OrderContainer()
    orders[1] = o1
    orders[2] = o2

    assert orders.get(3, "Not Found") == "Not Found"
    assert orders.get(3, "Not Found", active_only=True) == "Not Found"


def test_OrderContainer_default_in_get_works_if_active_only_not_found():
    o1 = OrderInfo("coolstrategy", "OPEN", ibi.Trade(), {})
    o2 = OrderInfo(
        "coolstrategy",
        "STOP",
        ibi.Trade(orderStatus=ibi.OrderStatus(orderId=2, status="Filled")),
        {},
    )

    orders = OrderContainer()
    orders[1] = o1
    orders[2] = o2
    assert orders.get(2, "Not Found", active_only=True) == "Not Found"


def test_OrderContainer_limited_in_size_and_max_size_parameter_works():
    orders = OrderContainer(
        {
            i: OrderInfo(
                "coolstrategy", "OPEN", ibi.Trade(order=ibi.Order(orderId=i)), None
            )
            for i in range(20)
        },
        max_size=5,
    )

    # done should keep only 10 last items
    assert len(orders) == 5


def test_OrderContainer_drops_oldest():
    # create dict with keys: 0...19, but keep only last 10 elements
    orders = OrderContainer(
        {
            i: OrderInfo(
                "coolstrategy", "OPEN", ibi.Trade(order=ibi.Order(orderId=i)), None
            )
            for i in range(20)
        },
        max_size=10,
    )

    # only last 10 keys kept: 10...19, so first item in dict should be 10
    first_key = list(orders.keys())[0]
    assert first_key == 10

    # this is the last key, which should not have been affected
    last_key = list(orders.keys())[-1]
    assert last_key == 19


def test_OrderContainer_max_size_ignored_if_zero():
    orders = OrderContainer(
        {
            i: OrderInfo(
                "coolstrategy", "OPEN", ibi.Trade(order=ibi.Order(orderId=i)), None
            )
            for i in range(20)
        },
        max_size=0,
    )

    assert len(orders) == 20


@pytest.mark.asyncio
async def test_OrderContainer_recalls_permId():
    orders = OrderContainer(
        {
            i: OrderInfo(
                "coolstrategy",
                "OPEN",
                ibi.Trade(order=ibi.Order(orderId=i, permId=i * 100)),
                None,
            )
            for i in range(1, 3)
        },
        max_size=0,
    )
    await asyncio.sleep(0)
    assert orders[100] == orders[1]


@pytest.mark.asyncio
async def test_OrderContainer_get_recalls_permId():
    orders = OrderContainer(
        {
            i: OrderInfo(
                "coolstrategy",
                "OPEN",
                ibi.Trade(order=ibi.Order(orderId=i, permId=i * 100)),
                None,
            )
            for i in range(1, 3)
        },
        max_size=0,
    )
    await asyncio.sleep(0)
    assert orders.get(100) == orders[1]


def test_Strategy_default_dict_not_shared_among_instances():
    cont = StrategyContainer()
    x, y = cont["x"], cont["y"]
    x.params.update({"a": 1, "b": 2})
    assert "a" not in y.params


def test_Strategy_dot_getitem():
    m = Strategy({"x": 1, "y": 2, "z": 3}, ce)
    assert m.x == 1


def test_Strategy_dot_setitem():
    m = Strategy({"x": 1, "y": 2, "z": 3}, ce)
    m.a = 5
    assert m["a"] == 5


def test_Strategy_dot_get():
    m = Strategy({"x": 1, "y": 2, "z": 3}, ce)
    m.a = 5
    assert m.get("a") == 5


def test_Strategy_dot_delete():
    m = Strategy({"x": 1, "y": 2, "z": 3}, ce)
    del m.y
    # timestamp gets automatically created for every entry
    # it cannot be compared with strategy_defaults
    if m.get("timestamp"):
        del m["timestamp"]
    assert m.data == {"x": 1, "z": 3, **strategy_defaults}


def test_Strategy_update():
    m = Strategy({"x": 1, "y": 2, "z": 3}, ce)
    m.update({"a": 5, "b": 9, "c": 0})
    if m.get("timestamp"):
        del m["timestamp"]
    assert m == {"x": 1, "y": 2, "z": 3, "a": 5, "b": 9, "c": 0, **strategy_defaults}


def test_Strategy_contains():
    m = Strategy({"x": 1, "y": 2, "z": 3}, ce)
    m.a = 8
    assert "a" in m


def test_StrategyContainer_contains_only_Strategies():
    """Saving regular dict must create a strategy"""
    cont = StrategyContainer({"a": {"x": 1, "y": 2}, "b": {"x": 4, "y": 9}})
    assert isinstance(cont["b"], Strategy)
    assert cont["a"].x == 1


def test_StrategyContainer_add_key():
    m = StrategyContainer(
        {
            "a": Strategy({"x": 1, "y": 2}, ce),
            "b": Strategy({"x": 4, "y": 9}, ce),
        }
    )
    m["c"] = Strategy({"x": 2, "c": 3}, ce)
    assert isinstance(m["c"], Strategy)


def test_StrategyContainer_add_key_with_dict():
    m = StrategyContainer(
        {
            "a": Strategy({"x": 1, "y": 2}, ce),
            "b": Strategy({"x": 4, "y": 9}, ce),
        }
    )
    m["c"] = {"x": 2, "c": 3}
    assert isinstance(m["c"], Strategy)


def test_StrategyContainer_inserts_strategy_name():
    cont = StrategyContainer()
    cont["x"] = Strategy(strategyChangeEvent=ce)
    assert cont["x"].strategy == "x"


def test_StrategyContainer_missing():
    m = StrategyContainer(
        {
            "a": Strategy({"x": 1, "y": 2}, ce),
            "b": Strategy({"x": 4, "y": 9}, ce),
        }
    )
    assert isinstance(m["not_set"], Strategy)


def test_StrategyContainer_new_strategy():
    m = StrategyContainer(
        {
            "a": Strategy({"x": 1, "y": 2}, ce),
            "b": Strategy({"x": 4, "y": 9}, ce),
        }
    )
    data = m["c"]
    data["x"] = 5

    assert "c" in m
    assert "x" in m["c"]


def test_access_order_with_square_brackets(state_machine):
    oi = OrderInfo("coolstrategy", "OPEN", ibi.Trade(), None)
    state_machine.order[1234] = oi
    assert state_machine.order[1234] == oi


def test_access_strategy_with_square_brackets(state_machine):
    st = Strategy({"a": {"x": 1, "y": 2}}, ce)
    state_machine.strategy["xyz"] = st
    assert state_machine.strategy["xyz"] == st


def test_access_strategy_with_get_existing_key(state_machine):
    st = Strategy({"a": {"x": 1, "y": 2}}, ce)
    state_machine.strategy["xyz"] = st
    assert state_machine.strategy.get("xyz") == st


def test_access_strategy_with_get_non_existing_key_with_default(state_machine):
    st = Strategy({"a": {"x": 1, "y": 2}}, ce)
    state_machine.strategy["xyz"] = st
    assert state_machine.strategy.get("abc", "default") == "default"


def test_access_strategy_with_get_non_existing_key_with_falsey_default(state_machine):
    st = Strategy({"a": {"x": 1, "y": 2}}, ce)
    state_machine.strategy["xyz"] = st
    assert state_machine.strategy.get("abc", "") == ""


def test_access_strategy_with_get_non_existing_key_no_default(state_machine):

    assert state_machine.strategy.get("abc").strategy == "abc"


def test_access_order_with_get(state_machine):
    oi = OrderInfo("coolstrategy", "OPEN", ibi.Trade(), None)
    state_machine.order[1234] = oi
    assert state_machine.order.get(1234) == oi
    assert state_machine.order.get(5678) is None


def test_access_strategy_with_get(state_machine):
    st = Strategy({"a": {"x": 1, "y": 2}}, ce)
    state_machine.strategy["xyz"] = st
    assert state_machine.strategy.get("xyz") == st
    # get with unknown key creates new entry,
    # so this is not symetrical to previous test


def test_position_and_order_for_strategy_position_no_pending_order(state_machine):
    state_machine.strategy["coolstrategy"].position = 1
    assert state_machine.position_and_order_for_strategy("coolstrategy") == 1


def test_position_and_order_for_strategy_no_position_no_pending_order(state_machine):
    state_machine.strategy["coolstrategy"].position = 0
    assert state_machine.position_and_order_for_strategy("coolstrategy") == 0


def test_position_and_order_for_strategy_no_position_pending_order(state_machine):
    state_machine.strategy["coolstrategy"].position = 0
    oi = OrderInfo(
        "coolstrategy",
        "OPEN",
        ibi.Trade(
            order=ibi.Order(action="BUY", totalQuantity=1),
            orderStatus=ibi.OrderStatus(orderId=2, status="Submitted"),
        ),
        None,
    )
    state_machine.order[1234] = oi
    assert state_machine.position_and_order_for_strategy("coolstrategy") == 1


def test_position_and_order_for_strategy_position_pending_order(state_machine):
    state_machine.strategy["coolstrategy"].position = 1
    oi = OrderInfo(
        "coolstrategy",
        "CLOSE",
        ibi.Trade(
            order=ibi.Order(action="SELL", totalQuantity=1, permId=1234),
            orderStatus=ibi.OrderStatus(orderId=2, status="Submitted"),
        ),
        None,
    )
    state_machine.order[1234] = oi
    assert state_machine.position_and_order_for_strategy("coolstrategy") == 0


def test_position_and_order_for_strategy_position_irrelevant_pending_order(
    state_machine,
):
    state_machine.strategy["coolstrategy"].position = 1
    oi = OrderInfo(
        "coolstrategy",
        "STOP",
        ibi.Trade(
            order=ibi.Order(action="SELL", totalQuantity=1, permId=1234),
            orderStatus=ibi.OrderStatus(orderId=2, status="Submitted"),
        ),
        None,
    )
    state_machine.order[1234] = oi
    assert state_machine.position_and_order_for_strategy("coolstrategy") == 1


def test_active_property_on_Strategy_false_works():
    strat_cont = StrategyContainer()
    st = strat_cont["new_strategy"]
    assert not st.active


def test_active_property_on_Strategy_true_works():
    strat_cont = StrategyContainer()
    st = strat_cont["new_strategy"]
    st.position += 1
    assert st.active


def test_Strategy_change_emits_event():
    class Counter:
        def __init__(self):
            self.count = 0

        def __call__(self):
            self.count += 1

    c = Counter()

    changeEvent = ev.Event("changeEvent")
    changeEvent += c

    st = Strategy({"position": 0}, changeEvent)

    st.position += 1
    # one event emitted when `position` created
    # one when increased
    assert c.count == 2


@pytest.mark.asyncio
async def test_StrategyContainer_gets_events_on_Strategy_change():
    class Counter:
        def __init__(self):
            self.count = 0

        def __call__(self):
            self.count += 1

    c = Counter()
    cont = StrategyContainer(None, save_delay=0.001)

    cont.strategyChangeEvent += c
    strat = cont["new_strategy"]
    strat.position += 1
    # there were 8 events emitted (7 items added to strategy dict
    # because of defaults and 1 change) they were all debounced into
    # one event
    # make sure there is one emit
    assert await wait_for_condition(lambda: c.count == 1)
    # and then however long we wait no more emits
    assert not await wait_for_condition(lambda: c.count == 2)


def test_strategy_has_position_attribute():
    # Tripple checking because mypy doesn't seem to notice
    cont = StrategyContainer(None, save_delay=0.000001)
    strat = cont["new_strategy"]  # noqa
    s = cont["new_strategy"]
    assert isinstance(s.position, float)


def test_empty_strategy_contains_defaults():
    strat = Strategy({"a": 1, "b": 2}, ce)
    assert strat.position == 0


def test_strategy_cannot_be_created_without_change_event():
    with pytest.raises(ValueError):
        Strategy({"a": 1, "b": 2})
