import ib_insync as ibi
import pytest

from ib_tools.state_machine import OrderContainer, OrderInfo, StateMachine


@pytest.fixture
def state_machine():
    # ensure any existing singleton is destroyed
    if StateMachine._instance:
        StateMachine._instance = None
    return StateMachine()


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


def test_OrderContainer_delete_order():
    o1 = OrderInfo("coolstrategy", "OPEN", ibi.Trade(), None)
    o2 = OrderInfo("coolstrategy", "STOP", ibi.Trade(), None)
    o3 = OrderInfo("suckystrategy", "STOP", ibi.Trade(), None)

    orders = OrderContainer()
    orders[1] = o1
    orders[2] = o2
    orders[3] = o3
    del orders[1]
    assert list(orders.values()) == [o1, o2, o3]


def test_OrderContainer_old_accessible():
    o1 = OrderInfo("coolstrategy", "OPEN", ibi.Trade(), None)
    o2 = OrderInfo("coolstrategy", "STOP", ibi.Trade(), None)
    o3 = OrderInfo("suckystrategy", "STOP", ibi.Trade(), None)

    orders = OrderContainer()
    orders[1] = o1
    orders[2] = o2
    orders[3] = o3
    del orders[1]
    assert orders.done == {1: o1}


def test_OrderContainer_active_accessible():
    o1 = OrderInfo("coolstrategy", "OPEN", ibi.Trade(), None)
    o2 = OrderInfo("coolstrategy", "STOP", ibi.Trade(), None)
    o3 = OrderInfo("suckystrategy", "STOP", ibi.Trade(), None)

    orders = OrderContainer()
    orders[1] = o1
    orders[2] = o2
    orders[3] = o3
    del orders[1]

    assert orders.active == {2: o2, 3: o3}


def test_OrderContainer_old_accessible_from_parent():
    o1 = OrderInfo("coolstrategy", "OPEN", ibi.Trade(), None)
    o2 = OrderInfo("coolstrategy", "STOP", ibi.Trade(), None)
    o3 = OrderInfo("suckystrategy", "STOP", ibi.Trade(), None)

    orders = OrderContainer()
    orders[1] = o1
    orders[2] = o2
    orders[3] = o3
    del orders[1]
    assert list(orders.keys()) == [1, 2, 3]


def test_state_machine_get_order(state_machine):
    o1 = OrderInfo("coolstrategy", "OPEN", ibi.Trade(), None)
    o2 = OrderInfo("coolstrategy", "STOP", ibi.Trade(), None)
    o3 = OrderInfo("suckystrategy", "STOP", ibi.Trade(), None)

    orders = OrderContainer()
    orders[1] = o1
    orders[2] = o2
    orders[3] = o3
    state_machine.orders = orders
    assert state_machine.get_order(1) == o1


def test_state_machine_delete_order(state_machine):
    o1 = OrderInfo("coolstrategy", "OPEN", ibi.Trade(), None)
    o2 = OrderInfo("coolstrategy", "STOP", ibi.Trade(), None)
    o3 = OrderInfo("suckystrategy", "STOP", ibi.Trade(), None)

    orders = OrderContainer()
    orders[1] = o1
    orders[2] = o2
    orders[3] = o3
    state_machine.orders = orders
    state_machine.delete_order(1)
    assert list(state_machine.orders.values()) == [o1, o2, o3]


#### record of done orders included


def test_OrderContainer_del_moves_to_done():
    o1 = OrderInfo("coolstrategy", "OPEN", ibi.Trade(), None)
    o2 = OrderInfo("coolstrategy", "STOP", ibi.Trade(), None)
    o3 = OrderInfo("suckystrategy", "STOP", ibi.Trade(), None)

    orders = OrderContainer()
    orders[1] = o1
    orders[2] = o2
    orders[3] = o3
    del orders[1]

    assert orders.done[1] == o1


def test_OrderContainer_done_included_in_get():
    o1 = OrderInfo("coolstrategy", "OPEN", ibi.Trade(), None)
    o2 = OrderInfo("coolstrategy", "STOP", ibi.Trade(), None)

    orders = OrderContainer()
    orders[1] = o1
    orders[2] = o2

    del orders[1]

    assert orders.get(1) == o1


def test_OrderContainer_active_included_in_get():
    o1 = OrderInfo("coolstrategy", "OPEN", ibi.Trade(), None)
    o2 = OrderInfo("coolstrategy", "STOP", ibi.Trade(), None)

    orders = OrderContainer()
    orders[1] = o1
    orders[2] = o2

    del orders[1]

    assert orders.get(2) == o2


def test_OrderContainer_getting_active_and_done_items():
    o1 = OrderInfo("coolstrategy", "OPEN", ibi.Trade(), None)
    o2 = OrderInfo("coolstrategy", "STOP", ibi.Trade(), None)
    o3 = OrderInfo("suckystrategy", "STOP", ibi.Trade(), None)

    orders = OrderContainer()
    orders[1] = o1
    orders[2] = o2
    orders[3] = o3

    del orders[1]

    assert orders[1] == o1
    assert orders[2] == o2


def test_OrderContainer_sets_correct_active_attributes():
    o1 = OrderInfo("coolstrategy", "OPEN", ibi.Trade(), None)
    o2 = OrderInfo("coolstrategy", "STOP", ibi.Trade(), None)

    orders = OrderContainer()
    orders[1] = o1
    orders[2] = o2

    del orders[1]

    assert not orders[1].active
    assert orders[2].active


def test_OrderContainer_active_only_flag_in_get():
    o1 = OrderInfo("coolstrategy", "OPEN", ibi.Trade(), None)
    o2 = OrderInfo("coolstrategy", "STOP", ibi.Trade(), None)

    orders = OrderContainer()
    orders[1] = o1
    orders[2] = o2

    del orders[1]

    assert orders.get(1, active_only=True) is None


def test_OrderContainer_default_in_get_works():
    o1 = OrderInfo("coolstrategy", "OPEN", ibi.Trade(), None)
    o2 = OrderInfo("coolstrategy", "STOP", ibi.Trade(), None)

    orders = OrderContainer()
    orders[1] = o1
    orders[2] = o2

    del orders[1]

    assert orders.get(3, "Not Found") == "Not Found"
    assert orders.get(3, "Not Found", active_only=True) == "Not Found"


def test_OrderContainer_done_limited_in_size():
    orders = OrderContainer.from_items(
        {
            i: OrderInfo(
                "coolstrategy", "OPEN", ibi.Trade(order=ibi.Order(orderId=i)), None
            )
            for i in range(20)
        },
    )
    # orders with keys 0..14 moved to done
    for i in range(15):
        del orders[i]

    # done should keep only 10 last items
    assert len(orders.done) == 10


def test_OrderContainer_done_limited_in_size_and_keep_parameter_works():
    orders = OrderContainer.from_items(
        {
            i: OrderInfo(
                "coolstrategy", "OPEN", ibi.Trade(order=ibi.Order(orderId=i)), None
            )
            for i in range(20)
        },
        keep=5,
    )
    # orders with keys 0..14 moved to done
    for i in range(15):
        del orders[i]

    # done should keep only 10 last items
    assert len(orders.done) == 5


def test_OrderContainer_done_drops_oldest():
    orders = OrderContainer.from_items(
        {
            i: OrderInfo(
                "coolstrategy", "OPEN", ibi.Trade(order=ibi.Order(orderId=i)), None
            )
            for i in range(20)
        }
    )
    # orders with keys 0..14 moved to done
    for i in range(15):
        del orders[i]

    # orders with keys 0...4 discarded so first kept it 5
    first_key = list(orders.done.keys())[0]
    assert first_key == 5

    # this is the last of keys removed from the main dict
    last_key = list(orders.done.keys())[-1]
    assert last_key == 14
