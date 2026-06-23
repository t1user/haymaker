"""Client-side pacing for Interactive Brokers dataloader requests.

The dataloader keeps its own pacing scheduler even though recent TWS/Gateway
versions can pace API messages server-side.  The local scheduler gives the
process deterministic control over request ordering, reserves capacity for
other clients through an allowance fraction, and keeps pacing retries out of
the downloader workflow.

The concrete limits are intentionally module-level constants rather than YAML
configuration.  They model current IBKR guidance closely enough for the
dataloader and can be updated in code when IBKR policy changes.  User-facing
configuration is limited to disabling the pacer and scaling the allowance.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable, Hashable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import StrEnum
from typing import Protocol, TypeVar

import ib_insync as ibi

log = logging.getLogger(__name__)
T = TypeVar("T")

# Historical-family requests cover reqHistoricalData, reqHeadTimeStamp, and
# reqHistoricalSchedule. IBKR documents these as sharing historical data pacing.
HISTORICAL_GLOBAL_LIMIT = 60  # 60 weighted historical requests.
HISTORICAL_GLOBAL_WINDOW = 600.0  # Rolling 10-minute window, in seconds.
HISTORICAL_PROBE_RESERVE = 1  # Leave room for supervised connection probes.
HISTORICAL_SAME_KEY_LIMIT = 5  # Same contract/exchange/data type requests.
HISTORICAL_SAME_KEY_WINDOW = 2.0  # Rolling 2-second same-key window.
HISTORICAL_IDENTICAL_COOLDOWN = 15.0  # Exact duplicate request cooldown.
HISTORICAL_MAX_CONCURRENT = 20  # Open historical requests allowed at once.

# Contract metadata requests are not documented as historical-data requests, but
# they still go through TWS/Gateway message handling, so keep a conservative
# lightweight bucket for reqContractDetails.
METADATA_GLOBAL_LIMIT = 10  # Contract-details requests per metadata window.
METADATA_GLOBAL_WINDOW = 1.0  # Metadata rolling window, in seconds.
METADATA_MAX_CONCURRENT = 5  # Open contract-details requests allowed at once.

BID_ASK_WEIGHT = 2  # IBKR counts BID_ASK historical requests twice.
DEFAULT_PACING_RETRY_DELAY = 60.0  # Delay after an IB pacing violation, seconds.


class RequestKind(StrEnum):
    """IB request categories handled by dataloader pacing."""

    HISTORICAL_DATA = "reqHistoricalData"
    HEAD_TIMESTAMP = "reqHeadTimeStamp"
    HISTORICAL_SCHEDULE = "reqHistoricalSchedule"
    CONTRACT_DETAILS = "reqContractDetails"


def scaled_capacity(limit: int, allowance_fraction: float, minimum: int = 1) -> int:
    """Return an integer capacity scaled by the configured allowance fraction.

    Args:
        limit: Base request allowance from the module-level IB policy.
        allowance_fraction: Dataloader-specific multiplier. Values below ``1``
            reserve capacity for other IB clients; values above ``1`` allow a
            deliberately more aggressive local schedule.
        minimum: Smallest capacity the pacer may produce.

    Returns:
        The scaled integer capacity, never below ``minimum``.
    """

    return max(minimum, int(limit * allowance_fraction))


def historical_global_capacity(allowance_fraction: float) -> int:
    """Return historical global capacity after reserving probe allowance.

    The connection supervisor uses a historical-data request as its readiness
    probe. That request is outside this pacer, so the dataloader keeps one
    historical global slot unused rather than trying to account for individual
    probe timings.
    """

    return max(1, scaled_capacity(HISTORICAL_GLOBAL_LIMIT, allowance_fraction) - 1)


@dataclass(frozen=True)
class RequestProfile:
    """Pacing metadata for one IB request.

    Attributes:
        kind: IB request family.
        weight: Capacity units consumed by the request. ``BID_ASK`` historical
            requests consume two units under IBKR policy.
        same_key: Optional key for per-contract/per-exchange/per-data-type
            historical request windows.
        identical_key: Optional key for exact duplicate request cooldowns.
    """

    kind: RequestKind
    weight: int = 1
    same_key: Hashable | None = None
    identical_key: Hashable | None = None


class PacingRule(Protocol):
    """Protocol implemented by request pacing rules."""

    def wait_time(self, profile: RequestProfile, now: datetime) -> float:
        """Return seconds to wait before registering a request."""
        ...

    def register(self, profile: RequestProfile, now: datetime) -> None:
        """Record that a request has been made."""
        ...


class RollingWindowRule:
    """Restrict weighted request count within one rolling time window."""

    def __init__(self, capacity: int, window_seconds: float) -> None:
        """Initialize the rolling window rule."""

        self.capacity = capacity
        self.window = timedelta(seconds=window_seconds)
        self.history: deque[tuple[datetime, int]] = deque()

    def wait_time(self, profile: RequestProfile, now: datetime) -> float:
        """Return seconds to wait before the request can be registered."""

        self._prune(now)
        capacity = max(self.capacity, profile.weight)
        total = sum(weight for _, weight in self.history)
        if total + profile.weight <= capacity:
            return 0.0

        excess = total + profile.weight - capacity
        removed = 0
        for timestamp, weight in self.history:
            removed += weight
            if removed >= excess:
                return max(
                    0.0,
                    (self.window - (now - timestamp)) / timedelta(seconds=1),
                )
        return 0.0

    def register(self, profile: RequestProfile, now: datetime) -> None:
        """Record one request in the rolling window."""

        self._prune(now)
        self.history.append((now, profile.weight))

    def _prune(self, now: datetime) -> None:
        """Remove expired request timestamps."""

        while self.history and now - self.history[0][0] >= self.window:
            self.history.popleft()


class KeyedRollingWindowRule:
    """Restrict weighted request count per request key."""

    def __init__(self, capacity: int, window_seconds: float) -> None:
        """Initialize the keyed rolling window rule."""

        self.capacity = capacity
        self.window = timedelta(seconds=window_seconds)
        self.history: defaultdict[Hashable, deque[tuple[datetime, int]]] = defaultdict(
            deque
        )

    def wait_time(self, profile: RequestProfile, now: datetime) -> float:
        """Return seconds to wait before the keyed request can be registered."""

        if profile.same_key is None:
            return 0.0
        history = self.history[profile.same_key]
        self._prune(history, now)
        capacity = max(self.capacity, profile.weight)
        total = sum(weight for _, weight in history)
        if total + profile.weight <= capacity:
            return 0.0

        excess = total + profile.weight - capacity
        removed = 0
        for timestamp, weight in history:
            removed += weight
            if removed >= excess:
                return max(
                    0.0,
                    (self.window - (now - timestamp)) / timedelta(seconds=1),
                )
        return 0.0

    def register(self, profile: RequestProfile, now: datetime) -> None:
        """Record one keyed request in the rolling window."""

        if profile.same_key is None:
            return
        history = self.history[profile.same_key]
        self._prune(history, now)
        history.append((now, profile.weight))

    def _prune(self, history: deque[tuple[datetime, int]], now: datetime) -> None:
        """Remove expired request timestamps."""

        while history and now - history[0][0] >= self.window:
            history.popleft()


class IdenticalCooldownRule:
    """Prevent identical requests from being repeated too quickly."""

    def __init__(self, cooldown_seconds: float) -> None:
        """Initialize the identical-request cooldown rule."""

        self.cooldown = timedelta(seconds=cooldown_seconds)
        self.last_request: dict[Hashable, datetime] = {}

    def wait_time(self, profile: RequestProfile, now: datetime) -> float:
        """Return seconds to wait for the identical request cooldown."""

        if profile.identical_key is None:
            return 0.0
        last = self.last_request.get(profile.identical_key)
        if last is None:
            return 0.0
        return max(0.0, (self.cooldown - (now - last)) / timedelta(seconds=1))

    def register(self, profile: RequestProfile, now: datetime) -> None:
        """Record the latest timestamp for an identical request key."""

        if profile.identical_key is not None:
            self.last_request[profile.identical_key] = now


class RequestBucket:
    """Apply pacing rules and a concurrency limit to one request family.

    ``RequestBucket`` is intentionally unaware of IB request semantics. Request
    methods build a :class:`RequestProfile`, then the bucket serializes pacing
    registration under a lock while allowing the actual IB awaits to run under a
    bounded semaphore.
    """

    def __init__(self, rules: list[PacingRule], max_concurrent: int) -> None:
        """Initialize the request bucket."""

        self.rules = rules
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.lock = asyncio.Lock()

    async def run(
        self,
        profile: RequestProfile,
        request: Callable[[], Awaitable[T]],
        *,
        disabled: bool,
    ) -> T:
        """Run one request after waiting for applicable pacing rules.

        Args:
            profile: Request metadata used by the bucket rules.
            request: Zero-argument coroutine factory for the actual IB call.
            disabled: If true, bypass all client-side pacing and run the
                request immediately.

        Returns:
            Whatever the wrapped IB coroutine returns.
        """

        if disabled:
            return await request()

        async with self.semaphore:
            await self._wait_and_register(profile)
            return await request()

    async def _wait_and_register(self, profile: RequestProfile) -> None:
        """Wait for all pacing rules, then register the request."""

        async with self.lock:
            while True:
                now = datetime.now(timezone.utc)
                wait_time = max(rule.wait_time(profile, now) for rule in self.rules)
                if wait_time <= 0:
                    break
                log.debug(f"Will throttle {profile.kind}: {round(wait_time, 1)}s")
                await asyncio.sleep(wait_time)

            now = datetime.now(timezone.utc)
            for rule in self.rules:
                rule.register(profile, now)


class PacingViolationRegistry:
    """Record contracts that IB reported as pacing violations.

    IB reports pacing failures through ``errorEvent`` rather than by raising
    from the awaited request. The dataloader therefore records affected
    contracts here and lets the paced request wrapper decide whether an empty
    response should be retried.
    """

    def __init__(self) -> None:
        """Initialize an empty pacing-violation registry."""

        self.data: set[Hashable] = set()

    def register(
        self, reqId: int, errorCode: int, errorString: str, contract: ibi.Contract
    ) -> None:
        """Record a pacing violation error event."""

        log.debug(
            f"Pacing violation registered for {getattr(contract, 'symbol', contract)}"
        )
        hash(contract)
        self.data.add(contract)

    def verify(self, contract: ibi.Contract) -> bool:
        """Return whether a contract has a pending pacing violation marker."""

        try:
            hash(contract)
            self.data.remove(contract)
            return True
        except KeyError:
            return False

    def onError(
        self, reqId: int, errorCode: int, errorString: str, contract: ibi.Contract
    ) -> None:
        """Handle an IB error event."""

        if "pacing violation" in errorString:
            self.register(reqId, errorCode, errorString, contract)


@dataclass
class RequestPacing:
    """Route and pace all IB requests made by one dataloader session.

    Args:
        ib: IB client used for paced requests.
        no_restriction: Disable all local pacing while keeping retry handling
            and call routing in place.
        allowance_fraction: Multiplier applied to module-level request limits.
        pacing_retry_delay: Delay before retrying when IB reports a pacing
            violation via ``errorEvent``.
        registry: Pacing violation registry bound to the session.
    """

    ib: ibi.IB
    no_restriction: bool = False
    allowance_fraction: float = 1.0
    pacing_retry_delay: float = DEFAULT_PACING_RETRY_DELAY
    registry: PacingViolationRegistry = field(
        default_factory=PacingViolationRegistry, repr=False
    )
    historical: RequestBucket = field(init=False, repr=False)
    metadata: RequestBucket = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize request buckets from module-level IB pacing policy."""

        if self.allowance_fraction <= 0:
            raise ValueError("allowance_fraction must be greater than 0")
        self.historical = self._historical_bucket()
        self.metadata = self._metadata_bucket()

    def _historical_bucket(self) -> RequestBucket:
        """Create the historical-data request bucket."""

        return RequestBucket(
            [
                RollingWindowRule(
                    historical_global_capacity(self.allowance_fraction),
                    HISTORICAL_GLOBAL_WINDOW,
                ),
                KeyedRollingWindowRule(
                    scaled_capacity(HISTORICAL_SAME_KEY_LIMIT, self.allowance_fraction),
                    HISTORICAL_SAME_KEY_WINDOW,
                ),
                IdenticalCooldownRule(HISTORICAL_IDENTICAL_COOLDOWN),
            ],
            scaled_capacity(HISTORICAL_MAX_CONCURRENT, self.allowance_fraction),
        )

    def _metadata_bucket(self) -> RequestBucket:
        """Create the contract-metadata request bucket."""

        return RequestBucket(
            [
                RollingWindowRule(
                    scaled_capacity(METADATA_GLOBAL_LIMIT, self.allowance_fraction),
                    METADATA_GLOBAL_WINDOW,
                ),
            ],
            scaled_capacity(METADATA_MAX_CONCURRENT, self.allowance_fraction),
        )

    async def historical_data(
        self,
        contract: ibi.Contract,
        *,
        endDateTime: datetime | date | str,
        durationStr: str,
        barSizeSetting: str,
        whatToShow: str,
        useRTH: bool,
        formatDate: int,
        keepUpToDate: bool = False,
        chartOptions: list[ibi.TagValue] | None = None,
        timeout: float = 0,
    ) -> ibi.BarDataList:
        """Run a historical bar request under IB historical pacing rules.

        Args:
            contract: Contract used by the historical data request.
            endDateTime: Request end timestamp passed to IB.
            durationStr: Request duration passed to IB.
            barSizeSetting: Historical bar size passed to IB.
            whatToShow: Historical data type passed to IB.
            useRTH: Whether IB should restrict bars to regular trading hours.
            formatDate: IB date-format flag.
            keepUpToDate: Must remain false for finite dataloader requests.
            chartOptions: IB chart options. This should normally be empty.
            timeout: IB request timeout in seconds.

        Returns:
            Historical bars returned by IB.
        """

        if keepUpToDate:
            raise ValueError("Dataloader historical requests must be finite.")
        if getattr(contract, "secType", "") == "CONTFUT" and endDateTime not in (
            "",
            None,
        ):
            raise ValueError(
                "Continuous futures historical requests require empty endDateTime."
            )

        options = chartOptions or []
        profile = self._historical_profile(
            RequestKind.HISTORICAL_DATA,
            contract,
            endDateTime=endDateTime,
            durationStr=durationStr,
            barSizeSetting=barSizeSetting,
            whatToShow=whatToShow,
            useRTH=useRTH,
            formatDate=formatDate,
            keepUpToDate=keepUpToDate,
            chartOptions=options,
        )
        return await self._run_with_pacing_retry(
            self.historical,
            profile,
            contract,
            lambda: self.ib.reqHistoricalDataAsync(
                contract,
                endDateTime,
                durationStr,
                barSizeSetting,
                whatToShow,
                useRTH,
                formatDate,
                keepUpToDate,
                options,
                timeout,
            ),
        )

    async def head_timestamp(
        self,
        contract: ibi.Contract,
        *,
        whatToShow: str,
        useRTH: bool,
        formatDate: int,
    ) -> datetime | None:
        """Run a head-timestamp request under IB historical pacing rules.

        Args:
            contract: Contract used by the head-timestamp request.
            whatToShow: Historical data type passed to IB.
            useRTH: Whether IB should restrict the query to regular trading
                hours.
            formatDate: IB date-format flag.

        Returns:
            Head timestamp returned by IB, or ``None`` when IB returns no value.
        """

        profile = self._historical_profile(
            RequestKind.HEAD_TIMESTAMP,
            contract,
            endDateTime="",
            durationStr="",
            barSizeSetting="",
            whatToShow=whatToShow,
            useRTH=useRTH,
            formatDate=formatDate,
            keepUpToDate=False,
            chartOptions=[],
        )
        return await self._run_with_pacing_retry(
            self.historical,
            profile,
            contract,
            lambda: self.ib.reqHeadTimeStampAsync(
                contract,
                whatToShow=whatToShow,
                useRTH=useRTH,
                formatDate=formatDate,
            ),
        )

    async def historical_schedule(
        self,
        contract: ibi.Contract,
        *,
        numDays: int,
        endDateTime: datetime | date | str,
        useRTH: bool,
    ) -> ibi.HistoricalSchedule:
        """Run a historical schedule request under IB historical pacing rules.

        Args:
            contract: Contract used by the schedule request.
            numDays: Number of schedule days requested.
            endDateTime: Schedule request end timestamp.
            useRTH: Whether IB should restrict the schedule to regular trading
                hours.

        Returns:
            Historical schedule returned by IB.
        """

        profile = self._historical_profile(
            RequestKind.HISTORICAL_SCHEDULE,
            contract,
            endDateTime=endDateTime,
            durationStr=f"{numDays} D",
            barSizeSetting="1 day",
            whatToShow="SCHEDULE",
            useRTH=useRTH,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[],
        )
        return await self._run_with_pacing_retry(
            self.historical,
            profile,
            contract,
            lambda: self.ib.reqHistoricalScheduleAsync(
                contract,
                numDays=numDays,
                endDateTime=endDateTime,
                useRTH=useRTH,
            ),
        )

    async def contract_details(
        self,
        contract: ibi.Contract,
    ) -> list[ibi.ContractDetails]:
        """Run a contract-details request under metadata pacing rules.

        Args:
            contract: Possibly partial contract specification to resolve.

        Returns:
            Contract details returned by IB.
        """

        profile = RequestProfile(
            RequestKind.CONTRACT_DETAILS,
        )
        return await self.metadata.run(
            profile,
            lambda: self.ib.reqContractDetailsAsync(contract),
            disabled=self.no_restriction,
        )

    async def _run_with_pacing_retry(
        self,
        bucket: RequestBucket,
        profile: RequestProfile,
        contract: ibi.Contract,
        request: Callable[[], Awaitable[T]],
    ) -> T:
        """Run a request and retry when IB reports it as a pacing violation."""

        while True:
            result = await bucket.run(profile, request, disabled=self.no_restriction)
            if not result and self.verify(contract):
                log.warning(f"Pacing violation for {contract}; retrying.")
                await asyncio.sleep(self.pacing_retry_delay)
                continue
            return result

    def _historical_profile(
        self,
        kind: RequestKind,
        contract: ibi.Contract,
        *,
        endDateTime: datetime | date | str,
        durationStr: str,
        barSizeSetting: str,
        whatToShow: str,
        useRTH: bool,
        formatDate: int,
        keepUpToDate: bool,
        chartOptions: list[ibi.TagValue],
    ) -> RequestProfile:
        """Build pacing metadata for a historical-family request.

        Args:
            kind: Historical-family request method.
            contract: Contract supplied to the IB request.
            endDateTime: Request end timestamp.
            durationStr: Historical request duration.
            barSizeSetting: Historical bar size.
            whatToShow: Historical data type.
            useRTH: Regular-trading-hours flag.
            formatDate: IB date-format flag.
            keepUpToDate: Whether the request subscribes to updating bars.
            chartOptions: IB chart options.

        Returns:
            A request profile that can be evaluated by the historical bucket.
        """

        hash(contract)
        weight = BID_ASK_WEIGHT if whatToShow == "BID_ASK" else 1
        return RequestProfile(
            kind,
            weight=weight,
            same_key=(contract, getattr(contract, "exchange", ""), whatToShow),
            identical_key=(
                kind,
                contract,
                endDateTime,
                durationStr,
                barSizeSetting,
                whatToShow,
                useRTH,
                formatDate,
                keepUpToDate,
                tuple((option.tag, option.value) for option in chartOptions),
            ),
        )

    def verify(self, contract: ibi.Contract) -> bool:
        """Return whether the contract recently hit a pacing violation."""

        return self.registry.verify(contract)

    def onErrEvent(
        self, reqId: int, errorCode: int, errorString: str, contract: ibi.Contract
    ) -> None:
        """Handle ``ib.errorEvent`` updates relevant to pacing violations."""

        self.registry.onError(reqId, errorCode, errorString, contract)
