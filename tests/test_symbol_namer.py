from dataclasses import FrozenInstanceError

import ib_insync as ibi
import pytest

from haymaker.datastore import (
    BarSizeSymbolNamer,
    StrategySymbolNamer,
    simple_symbol_namer,
)

FUTURE_CONTRACT = ibi.Future(
    conId=730283097,
    symbol="NQ",
    lastTradeDateOrContractMonth="20260320",
    multiplier="20",
    exchange="CME",
    currency="USD",
    localSymbol="NQH6",
    tradingClass="NQ",
)
STOCK_CONTRACT = ibi.Stock(
    conId=265598,
    symbol="AAPL",
    exchange="NASDAQ",
    primaryExchange="NASDAQ",
    currency="USD",
    localSymbol="AAPL",
    tradingClass="NMS",
)


@pytest.mark.parametrize(
    "contract,output",
    [
        (FUTURE_CONTRACT, "NQH6_FUT"),
        (STOCK_CONTRACT, "AAPL_STK"),
    ],
)
def test_simple_symbol_namer(contract, output):
    symbol_name = simple_symbol_namer(contract)
    assert symbol_name == output


@pytest.mark.parametrize(
    "contract,barSizeSetting,output",
    [
        (FUTURE_CONTRACT, "30 secs", "NQH6_FUT_30_sec"),
        (FUTURE_CONTRACT, "30 sec", "NQH6_FUT_30_sec"),
        (FUTURE_CONTRACT, "1 hour", "NQH6_FUT_1_hour"),
        (STOCK_CONTRACT, "30 secs", "AAPL_STK_30_sec"),
        (STOCK_CONTRACT, "30 sec", "AAPL_STK_30_sec"),
        (STOCK_CONTRACT, "1 hour", "AAPL_STK_1_hour"),
    ],
)
def test_symbol_name_from_contract_and_barSizeSetting(contract, barSizeSetting, output):
    namer = BarSizeSymbolNamer(barSizeSetting)
    symbol_name = namer(contract)
    assert symbol_name == output


def test_namer_raises_if_used_with_non_contract():
    namer = BarSizeSymbolNamer("30 secs")
    with pytest.raises(AssertionError):
        namer("NQ")


@pytest.mark.parametrize(
    "namer,field_name,new_value",
    [
        (BarSizeSymbolNamer("30 secs"), "barSizeSetting", "1 hour"),
        (StrategySymbolNamer("alpha"), "strategy", "beta"),
    ],
)
def test_symbol_namers_are_immutable(namer, field_name, new_value):
    with pytest.raises(FrozenInstanceError):
        setattr(namer, field_name, new_value)
