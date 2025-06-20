from dataclasses import dataclass
from datetime import datetime

import pytest
from contract_details_data import es_details_chain, gold_details_chain  # type: ignore
from ib_insync import Contract, ContractDetails, Future  # noqa

from haymaker import misc
from haymaker.contract_selector import (
    ContFutureSelector,
    FutureSelector,
    GoldComex,
    NoOffset,
    selector_factory,
)

# this comes from CME website:
# https://www.cmegroup.com/markets/metals/precious/gold.calendar.html
gold_parameters = [
    # symbol, last trading day
    ("GCH5", "20250227"),
    ("GCJ5", "20250328"),
    ("GCK5", "20250429"),
    ("GCM5", "20250529"),
    ("GCN5", "20250627"),
    ("GCQ5", "20250730"),
    ("GCU5", "20250828"),
    ("GCV5", "20250929"),
    ("GCX5", "20251030"),
    ("GCZ5", "20251126"),
    ("GCF6", "20251230"),
    ("GCG6", "20260129"),
    ("GCH6", "20260226"),
]

# this is pulled from IB
gold_chain: list[ContractDetails] = misc.decode_tree(gold_details_chain)
gold_details_dict = {
    details.contract.localSymbol: details for details in gold_chain  # type: ignore
}


@pytest.mark.parametrize("symbol,expected_last_trading_day", gold_parameters)
def test_last_trading_day_for_GoldComex(symbol, expected_last_trading_day):
    fut = GoldComex.from_details(gold_details_dict[symbol])
    assert fut.last_trading_day == datetime.strptime(
        expected_last_trading_day, "%Y%m%d"
    )


def test_FutureSelector_active_contract():
    selector = FutureSelector(gold_chain, GoldComex, today=datetime(2025, 3, 20))
    assert selector.active_contract.localSymbol == "GCJ5"


def test_FutureSelector_next_contract():
    selector = FutureSelector(
        gold_chain, GoldComex, roll_margin_bdays=5, today=datetime(2025, 3, 20)
    )
    # k contract is skipped because it's not liquid
    # days till roll is 3, which is smaller than roll_margin_bdays
    assert selector.next_contract.localSymbol == "GCM5"


def test_appropriate_contracts_selected():

    @dataclass
    class NoOffsetSelectorWithSchedule(NoOffset):
        active_months = [6]

    selector = FutureSelector(
        gold_chain,
        NoOffsetSelectorWithSchedule,
        roll_margin_bdays=5,
        today=datetime(2025, 3, 20),
    )
    # selecting only June (M) contracts
    assert selector.active_contract.localSymbol == "GCM5"


es_parameters = [
    # symbol, settlement date
    ("ESM5", "20250620"),
    ("ESU5", "20250919"),
    ("ESZ5", "20251219"),
    ("ESH6", "20260320"),
    ("ESM6", "20260618"),
    ("ESU6", "20260918"),
    ("ESZ6", "20261218"),
    ("ESH7", "20270319"),
    ("ESM7", "20270617"),
    ("ESU7", "20270917"),
    ("ESZ7", "20271217"),
    ("ESH8", "20280317"),
    ("ESM8", "20280616"),
    ("ESU8", "20280915"),
    ("ESZ8", "20281215"),
    ("ESH9", "20290316"),
    ("ESM9", "20290615"),
    ("ESU9", "20290921"),
    ("ESZ9", "20291221"),
    ("ESH0", "20300315"),
    ("ESM0", "20300621"),
]
es_chain: list[ContractDetails] = misc.decode_tree(es_details_chain)
es_details_dict = {
    details.contract.localSymbol: details for details in es_chain  # type: ignore
}


@pytest.mark.parametrize("symbol,expected_last_trading_day", es_parameters)
def test_last_trading_day_for_NoOffset(symbol, expected_last_trading_day):
    fut = NoOffset.from_details(es_details_dict[symbol])
    assert fut.last_trading_day == datetime.strptime(
        expected_last_trading_day, "%Y%m%d"
    )


def test_FutureSelector_active_contract_ES():
    selector = FutureSelector(es_chain, NoOffset, today=datetime(2025, 9, 10))
    assert selector.active_contract.localSymbol == "ESU5"


def test_FutureSelector_next_contract_ES():
    selector = FutureSelector(
        es_chain, NoOffset, roll_margin_bdays=5, today=datetime(2025, 9, 10)
    )
    assert selector.next_contract.localSymbol == "ESZ5"


#####################
# New Tests From Here
#####################


def test_correct_selector_chosen(details_chain):
    selector = selector_factory(details_chain)
    assert isinstance(selector, FutureSelector)


def test_correct_selector_chosen_contfuture(contfuture_details_chain):
    # if details contain ContFuture rather than Future ContFutureSelector
    # should be picked
    selector = selector_factory(contfuture_details_chain)
    assert isinstance(selector, ContFutureSelector)


def test_correct_active_contract(details_chain):
    selector = selector_factory(details_chain)
    selector.today = datetime(2025, 6, 20)
    assert selector.active_contract == Future(
        conId=637533641,
        symbol="ES",
        lastTradeDateOrContractMonth="20250919",
        multiplier="50",
        exchange="CME",
        currency="USD",
        localSymbol="ESU5",
        tradingClass="ES",
    )


def test_correct_active_contract_future_date(details_chain):
    selector = selector_factory(details_chain)
    # first contract in the chain is not active any more on this day
    selector.today = datetime(2025, 9, 17)
    assert selector.active_contract == Future(
        conId=495512563,
        symbol="ES",
        lastTradeDateOrContractMonth="20251219",
        multiplier="50",
        exchange="CME",
        currency="USD",
        localSymbol="ESZ5",
        tradingClass="ES",
    )


def test_set_selector_params(details_chain):
    # roll only one day before last trading day
    selector = selector_factory(details_chain, futures_roll_bdays=1)
    print(selector.roll_bdays)
    # same date as in previous test
    selector.today = datetime(2025, 9, 17)
    # exptect earlier contract to be still active
    assert selector.active_contract == Future(
        conId=637533641,
        symbol="ES",
        lastTradeDateOrContractMonth="20250919",
        multiplier="50",
        exchange="CME",
        currency="USD",
        localSymbol="ESU5",
        tradingClass="ES",
    )


def test_correct_last_trading_day(details_chain):
    selector = selector_factory(details_chain)
    assert selector._active_contract().last_trading_day == datetime(2025, 9, 19, 0, 0)


def test_correct_next_contract(details_chain):
    selector = selector_factory(details_chain)
    selector.today = datetime(2025, 6, 20)
    assert selector.active_contract == Future(
        conId=637533641,
        symbol="ES",
        lastTradeDateOrContractMonth="20250919",
        multiplier="50",
        exchange="CME",
        currency="USD",
        localSymbol="ESU5",
        tradingClass="ES",
    )


def test_correct_next_contract_future_date(details_chain):
    selector = selector_factory(details_chain)
    print(selector.roll_bdays, selector.roll_margin_bdays)
    # 3 days before roll date
    selector.today = datetime(2025, 9, 11)
    # next contract in the chain is `next`
    assert selector.next_contract == Future(
        conId=495512563,
        symbol="ES",
        lastTradeDateOrContractMonth="20251219",
        multiplier="50",
        exchange="CME",
        currency="USD",
        localSymbol="ESZ5",
        tradingClass="ES",
    )


def test_selector_params_for_next_contract(details_chain):
    selector = selector_factory(details_chain, futures_roll_margin_bdays=2)
    # 3 days before roll date (same as previous test)
    selector.today = datetime(2025, 9, 11)
    # next contract is still previous contract, because margin changed to 2 days
    assert selector.next_contract == Future(
        conId=637533641,
        symbol="ES",
        lastTradeDateOrContractMonth="20250919",
        multiplier="50",
        exchange="CME",
        currency="USD",
        localSymbol="ESU5",
        tradingClass="ES",
    )
