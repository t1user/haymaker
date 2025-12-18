from datetime import timedelta
from typing import Literal

import ib_insync as ibi
import pytest

from haymaker.misc import (
    Counter,
    barSizeSetting_to_timedelta,
    contractAsTuple,
    general_to_specific_contract_class,
    sign,
)


def test_Counter():
    c = Counter()
    num = c()
    assert isinstance(num, str)
    assert num.endswith("00000")


def test_Counter_increments_by_one():
    c = Counter()
    num = c()
    num1 = c()
    assert int(num1[-1]) - int(num[-1]) == 1


def test_Counter_doesnt_duplicate_on_reinstantiation():
    c = Counter()
    num = c()
    d = Counter()
    num1 = d()
    assert num != num1


@pytest.mark.parametrize(
    "input,expected",
    [(0, 0), (5, 1), (-10, -1), (-3.4, -1), (2.234, 1), (-0, 0), (+0, 0), (-0.0000, 0)],
)
def test_sign_function(input: float | Literal[0] | Literal[5] | Literal[-10], expected):
    assert sign(input) == expected


@pytest.mark.parametrize(
    "contract",
    [
        ibi.Stock,
        ibi.Option,
        ibi.Future,
        ibi.ContFuture,
        ibi.Index,
        ibi.CFD,
        ibi.Bond,
        ibi.Commodity,
        ibi.FuturesOption,
        ibi.MutualFund,
        ibi.Warrant,
        ibi.Crypto,
    ],
)
def test_contractAsTuple_works_for_every_contract_type_except_for_bag(contract):
    tuples = contractAsTuple(contract())
    assert isinstance(tuples, tuple)
    assert isinstance(tuples[0], tuple)


def test_general_to_specific_contract_class():
    contract = ibi.Contract(
        secType="FUT",
        conId=657106382,
        symbol="HSI",
        lastTradeDateOrContractMonth="20240130",
        multiplier="50",
        exchange="HKFE",
        currency="HKD",
        localSymbol="HSIF4",
        tradingClass="HSI",
    )
    future = general_to_specific_contract_class(contract)

    assert future == contract
    assert isinstance(future, ibi.Future)


def test_general_to_specific_contract_class_with_contfuture():
    contract = ibi.ContFuture(
        conId=656780482,
        symbol="MGC",
        lastTradeDateOrContractMonth="20250827",
        multiplier="10",
        exchange="COMEX",
        currency="USD",
        localSymbol="MGCQ5",
        tradingClass="MGC",
    )
    future = general_to_specific_contract_class(contract)

    assert future == contract
    assert isinstance(future, ibi.Future)


def test_general_to_specific_contract_class_with_contfuture_Contract():
    contract = ibi.Contract(
        secType="CONTFUT",
        conId=674701641,
        symbol="MGC",
        lastTradeDateOrContractMonth="20251229",
        multiplier="10",
        exchange="COMEX",
        currency="USD",
        localSymbol="MGCZ5",
        tradingClass="MGC",
    )

    future = general_to_specific_contract_class(contract)

    assert future == contract
    assert isinstance(future, ibi.Future)


def test_general_to_specific_contract_class_works_with_non_futures():
    contract = ibi.Contract(
        secType="STK",
        conId=4391,
        symbol="AMD",
        exchange="SMART",
        primaryExchange="NASDAQ",
        currency="USD",
        localSymbol="AMD",
        tradingClass="NMS",
        comboLegs=[],
    )

    modified = general_to_specific_contract_class(contract)

    assert modified == contract
    assert isinstance(modified, ibi.Stock)


def test_general_to_specific_contract_class_doesnt_touch_contract_subclasses():
    contract = ibi.Future(
        conId=637533641,
        symbol="ES",
        lastTradeDateOrContractMonth="20250919",
        multiplier="50",
        exchange="CME",
        currency="USD",
        localSymbol="ESU5",
        tradingClass="ES",
    )
    future = general_to_specific_contract_class(contract)
    assert future is contract


def test_general_to_specific_contract_class_raises_with_non_contracts():
    some_faulty_object = object()
    with pytest.raises(AssertionError):
        general_to_specific_contract_class(some_faulty_object)


def test_barSizeSetting_to_timedelta():
    assert barSizeSetting_to_timedelta("30 secs") == timedelta(seconds=30)
