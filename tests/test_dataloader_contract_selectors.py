import ib_insync as ibi
import pytest

from haymaker.dataloader.contract_selectors import (
    ContractSelector,
    CurrentFutureContractSelector,
)
from haymaker.dataloader.pacer import RequestPacing


async def collect_contracts(selector: ContractSelector) -> list[ibi.Contract]:
    """Collect contracts yielded by a selector."""

    return [contract async for contract in selector.objects()]


@pytest.mark.asyncio
async def test_selector_qualifies_with_paced_contract_details():
    """Selectors should route qualification through paced reqContractDetails."""

    class FakeIB:
        def __init__(self) -> None:
            self.details_calls = 0

        async def reqContractDetailsAsync(
            self, contract: ibi.Contract
        ) -> list[ibi.ContractDetails]:
            self.details_calls += 1
            qualified = ibi.Stock("AAPL", "NYSE", "USD", conId=123)
            return [ibi.ContractDetails(contract=qualified)]

        async def qualifyContractsAsync(self, *contracts: ibi.Contract) -> None:
            raise AssertionError("ContractSelector should use reqContractDetailsAsync")

    ib = FakeIB()
    pacing = RequestPacing(ib, "30 secs", "TRADES", no_restriction=True)
    kwargs = {
        "secType": "STK",
        "symbol": "AAPL",
        "exchange": "SMART",
        "currency": "USD",
    }

    first = ContractSelector.from_kwargs(pacing=pacing, **kwargs)
    second = ContractSelector.from_kwargs(pacing=pacing, **kwargs)

    first_contract = (await collect_contracts(first))[0]
    second_contract = (await collect_contracts(second))[0]

    assert first_contract.conId == 123
    assert second_contract.conId == 123
    assert first_contract.exchange == "SMART"
    assert second_contract.exchange == "SMART"
    assert ib.details_calls == 2


@pytest.mark.asyncio
async def test_current_future_selector_requalifies_after_changing_contfuture():
    """Changing CONTFUT into FUT should trigger another contract-details lookup."""

    class FakeIB:
        def __init__(self) -> None:
            self.sec_types: list[str] = []

        async def reqContractDetailsAsync(
            self, contract: ibi.Contract
        ) -> list[ibi.ContractDetails]:
            self.sec_types.append(contract.secType)
            qualified: ibi.Contract
            if contract.secType == "CONTFUT":
                qualified = ibi.ContFuture(
                    symbol="ES",
                    exchange="CME",
                    currency="USD",
                    conId=100,
                    localSymbol="ES",
                )
            else:
                qualified = ibi.Future(
                    symbol="ES",
                    exchange="CME",
                    currency="USD",
                    conId=200,
                    lastTradeDateOrContractMonth="202512",
                    localSymbol="ESZ5",
                )
            return [ibi.ContractDetails(contract=qualified)]

    ib = FakeIB()
    pacing = RequestPacing(ib, "30 secs", "TRADES", no_restriction=True)
    selector = CurrentFutureContractSelector(
        pacing=pacing,
        secType="FUT",
        symbol="ES",
        exchange="CME",
        currency="USD",
    )

    contracts = await collect_contracts(selector)

    assert ib.sec_types == ["CONTFUT", "FUT"]
    assert contracts[0].secType == "FUT"
    assert contracts[0].conId == 200
