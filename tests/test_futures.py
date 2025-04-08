from dataclasses import dataclass
from datetime import datetime

import pytest
from contract_details_data import es_details_chain, gold_details_chain  # type: ignore
from ib_insync import Contract, ContractDetails  # noqa

from haymaker import misc
from haymaker.futures import FutureSelector, GoldComex, NoOffset

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
