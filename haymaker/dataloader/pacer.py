import asyncio
import logging
from collections.abc import Awaitable, Callable
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from types import TracebackType
from typing import Self, TypeVar

import ib_insync as ibi

from .helpers import duration_in_secs

log = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass
class Restriction:
    """Restrict requests using a request timestamp holder owned by one session."""

    requests: int
    seconds: float
    holder: deque[datetime] = field(
        repr=False, default_factory=lambda: deque(maxlen=100)
    )

    def check(self) -> float:
        """Return required sleep time in seconds."""
        # get last six requests that have been let through
        holder_ = deque(self.holder, maxlen=self.requests)
        if len(holder_) < self.requests:
            return 0.0
        else:
            return max(
                (
                    timedelta(seconds=self.seconds)
                    - (datetime.now(timezone.utc) - holder_[0])
                ),
                timedelta(seconds=0),
            ) / timedelta(seconds=1)


@dataclass
class NoRestriction(Restriction):
    requests: int = 0
    seconds: float = 0

    def check(self) -> float:
        return 0.0


class PacingViolationRegistry:
    """
    Keep record of contracts that registered pacing violation so that
    their jobs can be rescheduled.

    * onError: should be connected to ib.onError event, will filter
    pacing violation errors and process them

    * verify: return True if given contract has been affected by
    pacing violation within last second
    """

    def __init__(self) -> None:
        """Initialize an empty pacing-violation registry."""

        self.data: set[ibi.Contract] = set()

    def register(
        self, reqId: int, errorCode: int, errorString: str, contract: ibi.Contract
    ) -> None:
        log.debug(f"Pacing violation registered for {contract.symbol}")
        self.data.add(contract)

    def verify(self, contract: ibi.Contract) -> bool:
        try:
            self.data.remove(contract)
            return True
        except KeyError:
            return False

    def onError(
        self, reqId: int, errorCode: int, errorString: str, contract: ibi.Contract
    ) -> None:
        if "pacing violation" in errorString:
            self.register(reqId, errorCode, errorString, contract)


@dataclass
class RequestPacing:
    """Own request pacing and pacing-violation state for one dataloader session."""

    bar_size: str
    wts: str
    restriction_specs: list[tuple[int, float]] = field(default_factory=list)
    no_restriction: bool = False
    pacing_retry_delay: float = 60
    restriction_threshold: int = 30
    restrictions: list[Restriction] = field(init=False, repr=False)
    registry: PacingViolationRegistry = field(
        default_factory=PacingViolationRegistry, repr=False
    )

    def __post_init__(self) -> None:
        self.restrictions = self._build_restrictions()
        log.debug(f"Request pacing initialized: {self.restrictions}")

    def _build_restrictions(self) -> list[Restriction]:
        """Build rate-limit restrictions for this pacing session."""

        specs = [] if self.no_restriction else self.restriction_specs
        if (not specs) or (
            duration_in_secs(self.bar_size) > self.restriction_threshold
        ):
            return []
        if self.wts == "BID_ASK":
            specs = [(requests, int(seconds / 2)) for requests, seconds in specs]
        holder: deque[datetime] = deque(maxlen=100)
        return [Restriction(*spec, holder) for spec in specs]

    async def __aenter__(self) -> Self:
        await self.wait_for_slot()
        self.register_request_time()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        pass

    async def wait_for_slot(self) -> None:
        """Wait until all configured request restrictions permit another request."""

        while self.restrictions and (
            wait_time := max([restriction.check() for restriction in self.restrictions])
        ):
            log.debug(
                f"Will throtle: {round(wait_time, 1)}s till "
                f"{datetime.now() + timedelta(seconds=wait_time)}"
            )
            await asyncio.sleep(wait_time)

    def register_request_time(self) -> None:
        """Record that a request is about to be sent to IB."""

        for restriction in self.restrictions:
            restriction.holder.append(datetime.now(timezone.utc))

    async def run(
        self, contract: ibi.Contract, request: Callable[[], Awaitable[T]]
    ) -> T:
        """Run one broker request, retrying pacing violations internally."""

        while True:
            async with self:
                result = await request()
            if not result and self.verify(contract):
                log.warning(f"Pacing violation for {contract}; retrying.")
                await asyncio.sleep(self.pacing_retry_delay)
                continue
            return result

    def verify(self, contract: ibi.Contract) -> bool:
        """Return whether the contract recently hit a pacing violation."""

        return self.registry.verify(contract)

    def onErrEvent(
        self, reqId: int, errorCode: int, errorString: str, contract: ibi.Contract
    ) -> None:
        """Handle ``ib.errorEvent`` updates relevant to pacing violations."""

        self.registry.onError(reqId, errorCode, errorString, contract)
