import random
from datetime import datetime, timezone

import ib_insync as ibi
import pytest

from haymaker.bracket_legs import (
    AdjustableFixedTrailingStop,
    AdjustableTrailingFixedStop,
    FixedStop,
    TrailingStop,
)


@pytest.fixture
def params():
    return {"atr": 10}


@pytest.fixture
def trade():
    contract = ibi.Contract(conId=1, symbol="NQ", exchange="CME")
    order = ibi.Order(totalQuantity=2, orderId=random.randint(1, 100), action="BUY")
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
    order_status = ibi.OrderStatus(
        order.orderId, filled=fill.execution.shares, avgFillPrice=200
    )
    trade_object.orderStatus = order_status
    return trade_object


@pytest.mark.parametrize(
    "key,value",
    [
        ("orderType", "STP"),
        ("auxPrice", 180),  # 200 - 20 (price - 2*atr)
        ("totalQuantity", 2),
        ("action", "SELL"),
    ],
)
def test_FixedStop(params, trade, key, value):
    stop = FixedStop(2)
    memo = {}
    order_kwargs = stop(
        params,
        trade,
        memo,
    )
    assert isinstance(order_kwargs, dict)
    assert memo
    assert order_kwargs[key] == value


@pytest.mark.parametrize(
    "key,value",
    [
        ("orderType", "TRAIL"),
        ("auxPrice", 20),  # 2*atr
        ("totalQuantity", 2),
        ("action", "SELL"),
    ],
)
def test_TrailingStop(params, trade, key, value):
    stop = TrailingStop(2)
    memo = {}
    order_kwargs = stop(
        params,
        trade,
        memo,
    )
    assert isinstance(order_kwargs, dict)
    assert memo
    assert order_kwargs[key] == value


@pytest.mark.parametrize(
    "key,value",
    [
        ("orderType", "TRAIL"),
        ("auxPrice", 10),
        ("totalQuantity", 2),
        ("action", "SELL"),
        ("triggerPrice", 220),
        ("adjustedStopPrice", 190),
        ("adjustedOrderType", "STP"),
    ],
)
def test_AdjustableTrailingFixedStop(params, trade, key, value):
    stop = AdjustableTrailingFixedStop(1, 2, 3)
    memo = {}
    order_kwargs = stop(
        params,
        trade,
        memo,
    )
    assert isinstance(order_kwargs, dict)
    assert order_kwargs[key] == value
    assert memo


@pytest.mark.parametrize(
    "key,value",
    [
        ("orderType", "STP"),
        ("auxPrice", 190),  # 200 - 10, ie. price - 1*atr
        ("totalQuantity", 2),
        ("action", "SELL"),
        ("triggerPrice", 220),
        ("adjustedStopPrice", 190),
        ("adjustedOrderType", "TRAIL"),
    ],
)
def test_AdjustableFixedTrailingStop(params, trade, key, value):
    stop = AdjustableFixedTrailingStop(1, 2, 3)
    memo = {}
    order_kwargs = stop(
        params,
        trade,
        memo,
    )
    assert isinstance(order_kwargs, dict)
    assert order_kwargs[key] == value
    assert memo
