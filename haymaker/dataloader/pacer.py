"""Client-side pacing for Interactive Brokers dataloader requests.

The dataloader keeps its own pacing scheduler even though recent TWS/Gateway
versions can pace API messages server-side.  The local scheduler gives the
process deterministic control over request ordering, reserves capacity for
other clients through an allowance fraction, and keeps pacing retries out of
the downloader workflow.

IBKR historical-data limits relevant to this package, as documented in July
2026, are summarized below:

* TWS API messages are limited by the account's market-data lines; the default
  allowance is 50 messages per second.
* At most 50 historical-data requests may be open concurrently, and IBKR notes
  that a substantially smaller number is usually more efficient.
* For bars of 30 seconds or less: do not repeat an identical request within 15
  seconds; do not make six or more requests for the same contract, exchange,
  and tick type within two seconds; and do not make more than 60 requests in ten
  minutes. ``BID_ASK`` requests count twice.
* The hard small-bar pacing rules are lifted for bars of one minute or longer,
  but IBKR still applies soft load balancing.
* Request duration and bar size must form a valid step size. IBKR recommends
  returning only a few thousand bars per request; its current maximum-duration
  table separately caps one-second bars at ``2000 S``.
* Bars of 30 seconds or less are unavailable beyond six months. Expired futures
  are unavailable beyond two years after expiry. Expired options, futures
  options, warrants, structured products, and future spreads are unavailable;
  end-of-day data is unavailable for those derivative types.
* Native combo history is unavailable, delisted-security history is unavailable,
  and history before a security changed exchange is often unavailable.
* Continuous futures require an empty ``endDateTime``.

``reqHeadTimeStamp`` is documented separately and IBKR does not state that it
shares the small-bar historical quota. It therefore uses the conservative
discovery bucket together with contract-details requests.

Sources:
https://ibkrcampus.com/campus/ibkr-api-page/twsapi-doc/
https://interactivebrokers.github.io/tws-api/historical_limitations.html
https://ibkrcampus.com/campus/ibkr-api-page/contracts/
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

from .helpers import duration_in_secs

log = logging.getLogger(__name__)
T = TypeVar("T")

# IBKR small-bar rule: at most 60 requests in a rolling ten-minute window.
# One slot remains unused because the supervisor's historical probe is external.
HISTORICAL_GLOBAL_LIMIT = 60  # 60 weighted historical requests.
HISTORICAL_GLOBAL_WINDOW = 600.0  # Rolling 10-minute window, in seconds.
HISTORICAL_PROBE_RESERVE = 1  # Leave room for supervised connection probes.
# IBKR small-bar rule: fewer than six matching requests in two seconds.
HISTORICAL_SAME_KEY_LIMIT = 5  # Same contract/exchange/data type requests.
HISTORICAL_SAME_KEY_WINDOW = 2.0  # Rolling 2-second same-key window.
# IBKR small-bar rule: no identical request within 15 seconds.
HISTORICAL_IDENTICAL_COOLDOWN = 15.0  # Exact duplicate request cooldown.
# IBKR permits 50 open historical requests and recommends using fewer.
HISTORICAL_MAX_CONCURRENT = 20  # Open historical requests allowed at once.
PACING_STATUS_INTERVAL = 60.0  # Repeat long local-wait diagnostics once per minute.

# Head timestamps and contract details are not documented as small-bar
# historical requests. Keep them below the general TWS API message-rate limit.
METADATA_GLOBAL_LIMIT = 10  # Discovery requests per metadata window.
METADATA_GLOBAL_WINDOW = 1.0  # Metadata rolling window, in seconds.
METADATA_MAX_CONCURRENT = 5  # Open contract-details requests allowed at once.

# IBKR small-bar pacing rule: BID_ASK consumes two request units.
BID_ASK_WEIGHT = 2
DEFAULT_PACING_RETRY_DELAY = 60.0  # Delay after an IB pacing violation, seconds.


class RequestKind(StrEnum):
    """IB request categories handled by dataloader pacing."""

    HISTORICAL_DATA = "reqHistoricalData"
    HEAD_TIMESTAMP = "reqHeadTimeStamp"
    HISTORICAL_SCHEDULE = "reqHistoricalSchedule"
    CONTRACT_DETAILS = "reqContractDetails"


@dataclass(frozen=True)
class ContractMetadata:
    """Session metadata extracted from IB contract-details responses.

    Args:
        symbol: Symbol key used by the session cache.
        exchange: Contract exchange from the detail response.
        trading_class: IB trading class when available.
        time_zone_id: Exchange timezone advertised by IB.
        trading_hours: Raw trading-hours string from IB.
        liquid_hours: Raw liquid-hours string from IB.
        real_expiration_date: Real expiration date advertised by IB.
    """

    symbol: str
    exchange: str = ""
    trading_class: str = ""
    time_zone_id: str | None = None
    trading_hours: str = ""
    liquid_hours: str = ""
    real_expiration_date: str = ""


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
        label: Human-readable contract or request label used in diagnostics.
    """

    kind: RequestKind
    weight: int = 1
    same_key: Hashable | None = None
    identical_key: Hashable | None = None
    label: str = ""


class PacingRule(Protocol):
    """Protocol implemented by request pacing rules."""

    def wait_time(self, profile: RequestProfile, now: datetime) -> float:
        """Return seconds to wait before registering a request."""
        ...

    def register(self, profile: RequestProfile, now: datetime) -> None:
        """Record that a request has been made."""
        ...

    @property
    def description(self) -> str:
        """Return a concise diagnostic description of the rule."""
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

    @property
    def description(self) -> str:
        """Return the rolling-window rule in log-friendly form."""

        return f"global {self.capacity} requests/{self.window.total_seconds():g}s"

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

    @property
    def description(self) -> str:
        """Return the keyed rolling-window rule in log-friendly form."""

        return (
            f"same contract/exchange/type {self.capacity} requests/"
            f"{self.window.total_seconds():g}s"
        )

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

    @property
    def description(self) -> str:
        """Return the duplicate cooldown in log-friendly form."""

        return f"identical-request cooldown {self.cooldown.total_seconds():g}s"


@dataclass(frozen=True)
class InFlightRequest:
    """One request submitted to IB and still awaiting its response."""

    label: str
    kind: RequestKind
    owner: str
    elapsed_seconds: float


@dataclass(frozen=True)
class RequestBucketStatus:
    """Current local queue and broker-wait counts for one request bucket."""

    concurrency_waiters: int
    pacing_waiters: int
    in_flight: int
    oldest_in_flight_seconds: float | None
    requests: tuple[InFlightRequest, ...]


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
        self.concurrency_waiters = 0
        self.pacing_waiters = 0
        self.in_flight: dict[asyncio.Task, tuple[float, RequestProfile]] = {}

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
            return await self._run_request(profile, request)

        self.concurrency_waiters += 1
        waiting_for_slot = True
        try:
            async with self.semaphore:
                self.concurrency_waiters -= 1
                waiting_for_slot = False
                self.pacing_waiters += 1
                try:
                    await self._wait_and_register(profile)
                finally:
                    self.pacing_waiters -= 1
                return await self._run_request(profile, request)
        except BaseException:
            if waiting_for_slot:
                self.concurrency_waiters -= 1
            raise

    async def _wait_and_register(self, profile: RequestProfile) -> None:
        """Wait for all pacing rules, then register the request."""

        async with self.lock:
            while True:
                now = datetime.now(timezone.utc)
                waits = [(rule.wait_time(profile, now), rule) for rule in self.rules]
                wait_time = max((wait for wait, _ in waits), default=0.0)
                if wait_time <= 0:
                    break
                limiting_rules = ", ".join(
                    rule.description
                    for wait, rule in waits
                    if abs(wait - wait_time) < 0.001
                )
                sleep_for = min(wait_time, PACING_STATUS_INTERVAL)
                log.debug(
                    "Local pacer delaying %s for %s: %.1fs remaining; limiting "
                    "rule: %s. Rechecking in %.1fs.",
                    profile.kind,
                    profile.label or "unlabelled request",
                    wait_time,
                    limiting_rules,
                    sleep_for,
                )
                await asyncio.sleep(sleep_for)

            now = datetime.now(timezone.utc)
            for rule in self.rules:
                rule.register(profile, now)

    async def _run_request(
        self, profile: RequestProfile, request: Callable[[], Awaitable[T]]
    ) -> T:
        """Submit one admitted request and track time spent waiting for IB."""

        task = asyncio.current_task()
        assert task is not None
        started = asyncio.get_running_loop().time()
        self.in_flight[task] = (started, profile)
        label = profile.label or "unlabelled request"
        log.debug("Submitted %s for %s to IB; awaiting response.", profile.kind, label)
        try:
            result = await request()
        except BaseException:
            elapsed = asyncio.get_running_loop().time() - started
            log.debug(
                "IB request %s for %s ended with an exception after %.1fs.",
                profile.kind,
                label,
                elapsed,
            )
            raise
        else:
            elapsed = asyncio.get_running_loop().time() - started
            log.debug(
                "IB request %s for %s completed after %.1fs.",
                profile.kind,
                label,
                elapsed,
            )
            return result
        finally:
            self.in_flight.pop(task, None)

    def status(self) -> RequestBucketStatus:
        """Return a snapshot suitable for aggregate dataloader diagnostics."""

        now = asyncio.get_running_loop().time()
        requests = tuple(
            sorted(
                (
                    InFlightRequest(
                        profile.label or "unlabelled request",
                        profile.kind,
                        task.get_name(),
                        now - started,
                    )
                    for task, (started, profile) in self.in_flight.items()
                ),
                key=lambda request: request.elapsed_seconds,
                reverse=True,
            )
        )
        return RequestBucketStatus(
            concurrency_waiters=self.concurrency_waiters,
            pacing_waiters=self.pacing_waiters,
            in_flight=len(requests),
            oldest_in_flight_seconds=(
                requests[0].elapsed_seconds if requests else None
            ),
            requests=requests,
        )


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
    contract_metadata: dict[str, ContractMetadata] = field(
        default_factory=dict, repr=False
    )
    historical: RequestBucket = field(init=False, repr=False)
    large_bar_historical: RequestBucket = field(init=False, repr=False)
    metadata: RequestBucket = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize request buckets from module-level IB pacing policy."""

        if self.allowance_fraction <= 0:
            raise ValueError("allowance_fraction must be greater than 0")
        self.historical = self._historical_bucket(small_bar_rules=True)
        self.large_bar_historical = self._historical_bucket(small_bar_rules=False)
        self.metadata = self._metadata_bucket()

    def _historical_bucket(self, *, small_bar_rules: bool) -> RequestBucket:
        """Create the historical-data request bucket."""

        rules: list[PacingRule] = []
        if small_bar_rules:
            rules = [
                RollingWindowRule(
                    historical_global_capacity(self.allowance_fraction),
                    HISTORICAL_GLOBAL_WINDOW,
                ),
                KeyedRollingWindowRule(
                    scaled_capacity(HISTORICAL_SAME_KEY_LIMIT, self.allowance_fraction),
                    HISTORICAL_SAME_KEY_WINDOW,
                ),
                IdenticalCooldownRule(HISTORICAL_IDENTICAL_COOLDOWN),
            ]
        return RequestBucket(
            rules,
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
            # IBKR continuous-future rule: endDateTime must be empty.
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
        # IBKR applies the hard historical pacing rules only through 30 seconds.
        bucket = (
            self.historical
            if duration_in_secs(barSizeSetting) <= 30
            else self.large_bar_historical
        )
        return await self._run_with_pacing_retry(
            bucket,
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
        """Run a head-timestamp request under conservative discovery pacing.

        Args:
            contract: Contract used by the head-timestamp request.
            whatToShow: Historical data type passed to IB.
            useRTH: Whether IB should restrict the query to regular trading
                hours.
            formatDate: IB date-format flag.

        Returns:
            Head timestamp returned by IB, or ``None`` when IB returns no value.
        """

        profile = RequestProfile(
            RequestKind.HEAD_TIMESTAMP, label=_request_label(contract)
        )
        return await self._run_with_pacing_retry(
            self.metadata,
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

        # IB implements SCHEDULE through historical data and publishes no
        # separate pacing quota, so keep schedule calls in the small-bar bucket.
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
            label=_request_label(contract),
        )
        details = await self.metadata.run(
            profile,
            lambda: self.ib.reqContractDetailsAsync(contract),
            disabled=self.no_restriction,
        )
        self.record_contract_metadata(details)
        return details

    def record_contract_metadata(self, details: list[ibi.ContractDetails]) -> None:
        """Record useful contract metadata from an IB details response."""

        for detail in details:
            metadata = _contract_metadata(detail)
            if metadata is None:
                continue
            for key in _metadata_keys(detail, metadata):
                self._store_contract_metadata(key, metadata)

    def contract_timezone(self, contract: ibi.Contract) -> str | None:
        """Return cached timezone for a contract symbol without broker calls."""

        for key in _contract_keys(contract):
            if metadata := self.contract_metadata.get(key):
                if metadata.time_zone_id:
                    return metadata.time_zone_id
        return None

    def _store_contract_metadata(self, key: str, metadata: ContractMetadata) -> None:
        """Store metadata, preferring entries that include timezone data."""

        existing = self.contract_metadata.get(key)
        if existing is None or (metadata.time_zone_id and not existing.time_zone_id):
            self.contract_metadata[key] = metadata

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
            label=_request_label(contract),
        )

    def verify(self, contract: ibi.Contract) -> bool:
        """Return whether the contract recently hit a pacing violation."""

        return self.registry.verify(contract)

    def onErrEvent(
        self, reqId: int, errorCode: int, errorString: str, contract: ibi.Contract
    ) -> None:
        """Handle ``ib.errorEvent`` updates relevant to pacing violations."""

        self.registry.onError(reqId, errorCode, errorString, contract)


def _contract_metadata(detail: ibi.ContractDetails) -> ContractMetadata | None:
    """Return cacheable metadata from one contract-details entry."""

    contract = detail.contract
    if contract is None:
        return None
    symbol = _clean_key(getattr(contract, "symbol", ""))
    if not symbol:
        return None
    return ContractMetadata(
        symbol=symbol,
        exchange=str(getattr(contract, "exchange", "")),
        trading_class=str(getattr(contract, "tradingClass", "")),
        time_zone_id=str(detail.timeZoneId) if detail.timeZoneId else None,
        trading_hours=str(detail.tradingHours or ""),
        liquid_hours=str(detail.liquidHours or ""),
        real_expiration_date=str(detail.realExpirationDate or ""),
    )


def _request_label(contract: ibi.Contract) -> str:
    """Return the shortest useful contract label for request diagnostics."""

    return str(
        getattr(contract, "localSymbol", "")
        or getattr(contract, "symbol", "")
        or contract
    )


def _metadata_keys(detail: ibi.ContractDetails, metadata: ContractMetadata) -> set[str]:
    """Return symbol-family keys for a contract-details entry."""

    contract = detail.contract
    keys = {metadata.symbol}
    if contract is not None:
        keys.update(
            _clean_key(value)
            for value in (
                getattr(contract, "symbol", ""),
                getattr(contract, "tradingClass", ""),
                getattr(contract, "localSymbol", ""),
            )
        )
    keys.update(
        _clean_key(value)
        for value in (
            getattr(detail, "marketName", ""),
            getattr(detail, "underSymbol", ""),
        )
    )
    return {key for key in keys if key}


def _contract_keys(contract: ibi.Contract) -> tuple[str, ...]:
    """Return cache lookup keys for a contract."""

    keys = [
        _clean_key(getattr(contract, "symbol", "")),
        _clean_key(getattr(contract, "tradingClass", "")),
        _clean_key(getattr(contract, "localSymbol", "")),
    ]
    return tuple(key for key in keys if key)


def _clean_key(value: object) -> str:
    """Return a normalized metadata key."""

    return str(value or "").strip().upper()
