import ib_insync as ibi
import pytest

from haymaker.datastore import (
    CollectionNamerBarsizeSetting,
    simple_collection_namer,
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
def test_simple_collection_namer(contract, output):
    collection_name = simple_collection_namer(contract)
    assert collection_name == output


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
def test_collection_name_from_contract_and_barSizeSetting(
    contract, barSizeSetting, output
):
    namer = CollectionNamerBarsizeSetting(barSizeSetting)
    collection_name = namer(contract)
    assert collection_name == output


def test_namer_raises_if_used_with_non_contract():
    namer = CollectionNamerBarsizeSetting("30 secs")
    with pytest.raises(AssertionError):
        namer("NQ")
