import ib_insync as ibi
import pytest

from ib_tools.state_machine import (
    OrderContainer,
    OrderInfo,
    StateMachine,
    Strategy,
    StrategyContainer,
)


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
    assert active


def test_OrderInfo_iterator():
    trade = ibi.Trade()
    o1 = OrderInfo("coolstrategy", "OPEN", trade, None)
    for i, j in zip(o1, ["coolstrategy", "OPEN", trade, None]):
        assert i == j


def test_OrderInfo_active():
    trade1 = ibi.Trade(orderStatus=ibi.OrderStatus(orderId=2, status="Filled"))
    trade2 = ibi.Trade()
    o1 = OrderInfo("coolstrategy", "STOP", trade1, None)
    o2 = OrderInfo("coolstrategy", "STOP", trade2, None)
    assert o1.active is False
    assert o2.active is True


def test_OrderContainer_strategy_gets_correct_orders():
    o1 = OrderInfo("coolstrategy", "OPEN", ibi.Trade(), None)
    o2 = OrderInfo("coolstrategy", "STOP", ibi.Trade(), None)
    o3 = OrderInfo("suckystrategy", "STOP", ibi.Trade(), None)

    orders = OrderContainer()
    orders[1] = o1
    orders[2] = o2
    orders[3] = o3

    assert list(orders.strategy("coolstrategy")) == [o1, o2]


def test_OrderContainer_del_works_correctly():
    o1 = OrderInfo("coolstrategy", "OPEN", ibi.Trade(), None)
    o2 = OrderInfo("coolstrategy", "STOP", ibi.Trade(), None)
    o3 = OrderInfo("suckystrategy", "STOP", ibi.Trade(), None)

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


def test_OrderContainer_limited_in_size():
    orders = OrderContainer(
        {
            i: OrderInfo(
                "coolstrategy", "OPEN", ibi.Trade(order=ibi.Order(orderId=i)), {}
            )
            for i in range(20)
        }
    )

    # done should keep only 10 last items
    assert len(orders) == 10


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


def test_Strategy_dot_getitem():
    m = Strategy({"x": 1, "y": 2, "z": 3})
    assert m.x == 1


def test_Strategy_dot_setitem():
    m = Strategy({"x": 1, "y": 2, "z": 3})
    m.a = 5
    assert m == {"x": 1, "y": 2, "z": 3, "a": 5}


def test_Strategy_dot_get():
    m = Strategy({"x": 1, "y": 2, "z": 3})
    m.a = 5
    assert m.get("a") == 5


def test_Strategy_dot_delete():
    m = Strategy({"x": 1, "y": 2, "z": 3})
    del m.y
    assert m == {"x": 1, "z": 3}


def test_Strategy_update():
    m = Strategy({"x": 1, "y": 2, "z": 3})
    m.update({"a": 5, "b": 9, "c": 0})
    assert m == {"x": 1, "y": 2, "z": 3, "a": 5, "b": 9, "c": 0}


def test_Strategy_contains():
    m = Strategy({"x": 1, "y": 2, "z": 3})
    m.a = 8
    assert "a" in m


def test_StrategyContainer_contains_only_Strategies():
    m = StrategyContainer({"a": {"x": 1, "y": 2}, "b": {"x": 4, "y": 9}})
    assert isinstance(m["b"], Strategy)
    assert m["b"].x == 4


def test_StrategyContainer_add_key():
    m = StrategyContainer({"a": {"x": 1, "y": 2}, "b": {"x": 4, "y": 9}})
    m["c"] = {"x": 2, "c": 3}
    assert isinstance(m["c"], Strategy)


def test_StrategyContainer_missing():
    m = StrategyContainer({"a": {"x": 1, "y": 2}, "b": {"x": 4, "y": 9}})
    assert isinstance(m["not_set"], Strategy)


def test_StrategyContaner_new_strategy():
    m = StrategyContainer({"a": {"x": 1, "y": 2}, "b": {"x": 4, "y": 9}})
    data = m["c"]
    data["x"] = 5
    assert m.data == {
        "a": {"x": 1, "y": 2},
        "b": {"x": 4, "y": 9},
        "c": {
            "x": 5,
            "active_contract": None,
            "position": 0.0,
            "params": {},
            "lock": 0,
            "position_id": "",
        },
    }


def test_access_order_with_square_brackets(state_machine):
    oi = OrderInfo("coolstrategy", "OPEN", ibi.Trade(), None)
    state_machine.order[1234] = oi
    assert state_machine.order[1234] == oi


def test_access_strategy_with_square_brackets(state_machine):
    st = {"a": {"x": 1, "y": 2}}
    state_machine.strategy["xyz"] = st
    assert state_machine.strategy["xyz"] == st


def test_access_order_with_get(state_machine):
    oi = OrderInfo("coolstrategy", "OPEN", ibi.Trade(), None)
    state_machine.order[1234] = oi
    assert state_machine.order.get(1234) == oi
    assert state_machine.order.get(5678) is None


def test_access_strategy_with_get(state_machine):
    st = {"a": {"x": 1, "y": 2}}
    state_machine.strategy["xyz"] = st
    assert state_machine.strategy.get("xyz") == st
    # get with unknown key creates new entry,
    # so this is not symetrical to previous test
