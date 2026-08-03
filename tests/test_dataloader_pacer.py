import asyncio
import logging
from datetime import datetime, timedelta, timezone

import ib_insync as ibi
import pytest

from haymaker.dataloader.pacer import (
    BID_ASK_WEIGHT,
    HISTORICAL_GLOBAL_LIMIT,
    HISTORICAL_PROBE_RESERVE,
    HISTORICAL_SAME_KEY_LIMIT,
    RequestKind,
    RequestProfile,
    RequestPacing,
    RollingWindowRule,
)


class FakeIB:
    """Minimal async IB stub for pacer tests."""

    def __init__(self) -> None:
        self.historical_requests = 0
        self.head_timestamp_requests = 0
        self.contract_details_requests = 0
        self.last_historical_kwargs = None

    async def reqHistoricalDataAsync(
        self,
        contract,
        endDateTime,
        durationStr,
        barSizeSetting,
        whatToShow,
        useRTH,
        formatDate=1,
        keepUpToDate=False,
        chartOptions=None,
        timeout=60,
    ):
        """Record one historical request and return a placeholder result."""

        self.historical_requests += 1
        self.last_historical_kwargs = {
            "contract": contract,
            "endDateTime": endDateTime,
            "durationStr": durationStr,
            "barSizeSetting": barSizeSetting,
            "whatToShow": whatToShow,
            "useRTH": useRTH,
            "formatDate": formatDate,
            "keepUpToDate": keepUpToDate,
            "chartOptions": chartOptions,
            "timeout": timeout,
        }
        return ["ok"]

    async def reqHeadTimeStampAsync(self, contract, **kwargs):
        """Record one head-timestamp request."""

        self.head_timestamp_requests += 1
        return datetime(2025, 1, 1, tzinfo=timezone.utc)

    async def reqContractDetailsAsync(self, contract):
        """Record one contract-details request."""

        self.contract_details_requests += 1
        return [ibi.ContractDetails(contract=contract)]


def test_historical_allowance_fraction_scales_capacity():
    """Allowance fraction should scale capacity while reserving probe room."""

    pacing = RequestPacing(FakeIB(), allowance_fraction=2)

    assert (
        pacing.historical.rules[0].capacity
        == HISTORICAL_GLOBAL_LIMIT * 2 - HISTORICAL_PROBE_RESERVE
    )


def test_historical_probe_reserve_does_not_reduce_same_key_limit():
    """Supervisor probe reserve should only affect the global historical bucket."""

    pacing = RequestPacing(FakeIB())

    assert pacing.historical.rules[0].capacity == (
        HISTORICAL_GLOBAL_LIMIT - HISTORICAL_PROBE_RESERVE
    )
    assert pacing.historical.rules[1].capacity == HISTORICAL_SAME_KEY_LIMIT


@pytest.mark.asyncio
async def test_large_bars_do_not_consume_small_bar_rolling_limit():
    """Bars above 30 seconds should bypass IB's lifted hard pacing rules."""

    contract = ibi.Future(symbol="ES", exchange="CME", currency="USD", conId=123)
    pacing = RequestPacing(FakeIB())

    await pacing.historical_data(
        contract,
        endDateTime="",
        durationStr="1 D",
        barSizeSetting="1 min",
        whatToShow="TRADES",
        useRTH=False,
        formatDate=2,
    )

    assert not pacing.historical.rules[0].history
    assert pacing.large_bar_historical.rules == []


@pytest.mark.asyncio
async def test_head_timestamps_use_discovery_pacing():
    """Head timestamps should not consume small-bar historical capacity."""

    contract = ibi.Future(symbol="ES", exchange="CME", currency="USD", conId=123)
    ib = FakeIB()
    pacing = RequestPacing(ib)

    await pacing.head_timestamp(
        contract,
        whatToShow="TRADES",
        useRTH=False,
        formatDate=2,
    )

    assert ib.head_timestamp_requests == 1
    assert not pacing.historical.rules[0].history
    assert len(pacing.metadata.rules[0].history) == 1


@pytest.mark.asyncio
async def test_request_status_distinguishes_waiting_for_ib():
    """Bucket status should expose admitted requests awaiting broker responses."""

    started = asyncio.Event()
    release = asyncio.Event()

    class SlowIB(FakeIB):
        async def reqHistoricalDataAsync(self, *args, **kwargs):
            started.set()
            await release.wait()
            return ["ok"]

    contract = ibi.Future(symbol="ES", exchange="CME", currency="USD", conId=123)
    pacing = RequestPacing(SlowIB(), no_restriction=True)
    request = asyncio.create_task(
        pacing.historical_data(
            contract,
            endDateTime="",
            durationStr="1 D",
            barSizeSetting="30 secs",
            whatToShow="TRADES",
            useRTH=False,
            formatDate=2,
        ),
        name="worker 3",
    )
    await started.wait()

    status = pacing.historical.status()
    assert status.in_flight == 1
    assert status.oldest_in_flight_seconds is not None
    assert status.requests[0].label == "ES"
    assert status.requests[0].owner == "worker 3"

    release.set()
    await request
    assert pacing.historical.status().in_flight == 0


@pytest.mark.asyncio
async def test_long_pacer_wait_names_local_limiting_rule(caplog):
    """A suspended request should identify the local rule responsible."""

    contract = ibi.Future(symbol="ES", exchange="CME", currency="USD", conId=123)
    pacing = RequestPacing(FakeIB())
    rule = pacing.historical.rules[0]
    assert isinstance(rule, RollingWindowRule)
    now = datetime.now(timezone.utc)
    profile = RequestProfile(RequestKind.HISTORICAL_DATA)
    for _ in range(rule.capacity):
        rule.register(profile, now)

    caplog.set_level(logging.DEBUG, logger="haymaker.dataloader.pacer")
    request = asyncio.create_task(
        pacing.historical_data(
            contract,
            endDateTime="",
            durationStr="1 D",
            barSizeSetting="30 secs",
            whatToShow="TRADES",
            useRTH=False,
            formatDate=2,
        )
    )
    await asyncio.sleep(0)

    assert pacing.historical.status().pacing_waiters == 1
    assert "Local pacer delaying" in caplog.text
    assert "global 59 requests/600s" in caplog.text

    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request


def test_bid_ask_historical_profile_uses_weight_two():
    """BID_ASK requests should consume double historical pacing weight."""

    contract = ibi.Future(symbol="ES", exchange="CME", currency="USD", conId=123)
    pacing = RequestPacing(FakeIB())

    profile = pacing._historical_profile(
        RequestKind.HISTORICAL_DATA,
        contract,
        endDateTime="",
        durationStr="1 D",
        barSizeSetting="30 secs",
        whatToShow="BID_ASK",
        useRTH=False,
        formatDate=2,
        keepUpToDate=False,
        chartOptions=[],
    )

    assert profile.weight == BID_ASK_WEIGHT


def test_identical_historical_request_cooldown():
    """Identical historical requests should be delayed by the cooldown rule."""

    contract = ibi.Future(symbol="ES", exchange="CME", currency="USD", conId=123)
    pacing = RequestPacing(FakeIB())
    profile = pacing._historical_profile(
        RequestKind.HISTORICAL_DATA,
        contract,
        endDateTime="20250101 00:00:00",
        durationStr="1 D",
        barSizeSetting="30 secs",
        whatToShow="TRADES",
        useRTH=False,
        formatDate=2,
        keepUpToDate=False,
        chartOptions=[],
    )
    cooldown = pacing.historical.rules[2]
    now = datetime.now(timezone.utc)

    cooldown.register(profile, now)

    assert cooldown.wait_time(profile, now + timedelta(seconds=1)) == pytest.approx(14)


def test_pacing_violation_registry_uses_hashable_contracts():
    """Pacing violation tracking should use ib_insync contract hash semantics."""

    contract = ibi.Future(symbol="ES", exchange="CME", currency="USD", conId=123)
    pacing = RequestPacing(FakeIB())

    pacing.registry.register(1, 162, "pacing violation", contract)

    assert pacing.verify(contract)
    assert not pacing.verify(contract)


def test_historical_profile_rejects_unqualified_ib_contracts():
    """Historical pacing should not synthesize keys for unqualified contracts."""

    contract = ibi.Future(symbol="ES", exchange="CME", currency="USD")
    pacing = RequestPacing(FakeIB())

    with pytest.raises(ValueError, match="can't be hashed"):
        pacing._historical_profile(
            RequestKind.HISTORICAL_DATA,
            contract,
            endDateTime="",
            durationStr="1 D",
            barSizeSetting="30 secs",
            whatToShow="TRADES",
            useRTH=False,
            formatDate=2,
            keepUpToDate=False,
            chartOptions=[],
        )


@pytest.mark.asyncio
async def test_no_restriction_bypasses_identical_request_cooldown():
    """Disabling pacing should bypass all client-side waits."""

    contract = ibi.Future(symbol="ES", exchange="CME", currency="USD", conId=123)
    ib = FakeIB()
    pacing = RequestPacing(ib, no_restriction=True)

    await pacing.historical_data(
        contract,
        endDateTime="",
        durationStr="1 D",
        barSizeSetting="30 secs",
        whatToShow="TRADES",
        useRTH=False,
        formatDate=2,
    )
    await pacing.historical_data(
        contract,
        endDateTime="",
        durationStr="1 D",
        barSizeSetting="30 secs",
        whatToShow="TRADES",
        useRTH=False,
        formatDate=2,
    )

    assert ib.historical_requests == 2


@pytest.mark.asyncio
async def test_contract_details_requests_are_paced_without_caching():
    """Contract-details requests should be paced but not cached by the pacer."""

    contract = ibi.Future(symbol="ES", exchange="CME", currency="USD")
    ib = FakeIB()
    pacing = RequestPacing(ib, no_restriction=True)

    await pacing.contract_details(contract)
    await pacing.contract_details(contract)

    assert ib.contract_details_requests == 2


@pytest.mark.asyncio
async def test_contract_details_records_symbol_timezone_metadata():
    """Contract-details responses should seed session timezone metadata."""

    class MetadataIB(FakeIB):
        async def reqContractDetailsAsync(self, contract):
            self.contract_details_requests += 1
            future = ibi.Future(
                symbol="NQ",
                exchange="CME",
                currency="USD",
                localSymbol="NQU6",
                tradingClass="NQ",
            )
            return [
                ibi.ContractDetails(
                    contract=future,
                    marketName="NQ",
                    timeZoneId="US/Central",
                    tradingHours="20260622:1700-20260623:1600",
                    liquidHours="20260623:0830-20260623:1600",
                    realExpirationDate="20260918",
                )
            ]

    pacing = RequestPacing(MetadataIB(), no_restriction=True)

    await pacing.contract_details(ibi.Future(symbol="NQ", exchange="CME"))

    expired_contract = ibi.Future(
        symbol="NQ",
        exchange="CME",
        currency="USD",
        localSymbol="NQH6",
        tradingClass="NQ",
    )
    assert pacing.contract_timezone(expired_contract) == "US/Central"
    assert pacing.contract_metadata["NQ"].liquid_hours == (
        "20260623:0830-20260623:1600"
    )


@pytest.mark.asyncio
async def test_historical_data_rejects_updating_requests():
    """Dataloader historical requests must be finite requests."""

    contract = ibi.Future(symbol="ES", exchange="CME", currency="USD", conId=123)
    pacing = RequestPacing(FakeIB(), no_restriction=True)

    with pytest.raises(ValueError, match="finite"):
        await pacing.historical_data(
            contract,
            endDateTime="",
            durationStr="1 D",
            barSizeSetting="30 secs",
            whatToShow="TRADES",
            useRTH=False,
            formatDate=2,
            keepUpToDate=True,
        )


@pytest.mark.asyncio
async def test_contfuture_historical_data_requires_empty_end_datetime():
    """Continuous futures should follow IB's empty-endDateTime requirement."""

    contract = ibi.ContFuture(symbol="ES", exchange="CME", currency="USD", conId=123)
    ib = FakeIB()
    pacing = RequestPacing(ib, no_restriction=True)

    with pytest.raises(ValueError, match="empty endDateTime"):
        await pacing.historical_data(
            contract,
            endDateTime="20250101 00:00:00",
            durationStr="1 D",
            barSizeSetting="30 secs",
            whatToShow="TRADES",
            useRTH=False,
            formatDate=2,
        )

    await pacing.historical_data(
        contract,
        endDateTime="",
        durationStr="1 D",
        barSizeSetting="30 secs",
        whatToShow="TRADES",
        useRTH=False,
        formatDate=2,
    )

    assert ib.historical_requests == 1
    assert ib.last_historical_kwargs["endDateTime"] == ""
