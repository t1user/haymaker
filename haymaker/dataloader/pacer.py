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
    """Restrict requests using a request timestamp holder owned by one pacer."""

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


@dataclass
class Pacer:
    """One object for all workers."""

    restrictions: list[Restriction] = field(default_factory=list)

    async def __aenter__(self):
        # inner context ensures separate wait time for each worker
        while self.restrictions and (
            time := max([restriction.check() for restriction in self.restrictions])
        ):
            log.debug(
                f"Will throtle: {round(time, 1)}s till "
                f"{datetime.now() + timedelta(seconds=time)}"
            )
            await asyncio.sleep(time)

        # register request time right before making the request
        # request should be the next instruction out of this context
        for restriction in self.restrictions:
            restriction.holder.append(datetime.now(timezone.utc))

    async def __aexit__(self, *args):
        pass


def pacer(
    barSize,
    wts,
    *,
    restrictions: list[tuple[int, float]] = [],
    restriction_threshold: int = 30,  # barSize in secs above which restrictions apply
) -> Pacer:
    """
    Factory function returning correct pacer preventing (or rather
    limiting -:)) data pacing restrictions by Interactive Brokers.
    """

    if (not restrictions) or (duration_in_secs(barSize) > restriction_threshold):
        return Pacer()

    else:
        # 'BID_ASK' requests counted as double by ib
        if wts == "BID_ASK":
            restrictions = [
                (restriction[0], int(restriction[1] / 2))
                for restriction in restrictions
            ]
    holder: deque[datetime] = deque(maxlen=100)
    return Pacer([Restriction(*res, holder) for res in restrictions])


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


class PacingViolationError(Exception):
    """Raised when IB reports a pacing violation for a just-finished request."""


@dataclass
class RequestPacing:
    """Own pacing limiter and pacing-violation state for one dataloader session."""

    bar_size: str
    wts: str
    restrictions: list[tuple[int, float]] = field(default_factory=list)
    no_restriction: bool = False
    limiter: Pacer = field(init=False, repr=False)
    registry: PacingViolationRegistry = field(
        default_factory=PacingViolationRegistry, repr=False
    )

    def __post_init__(self) -> None:
        self.limiter = pacer(
            self.bar_size,
            self.wts,
            restrictions=[] if self.no_restriction else self.restrictions,
        )
        log.debug(f"Pacer initialized: {self.limiter}")

    async def __aenter__(self) -> Self:
        await self.limiter.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.limiter.__aexit__(exc_type, exc, traceback)

    async def request(
        self, contract: ibi.Contract, request: Callable[[], Awaitable[T]]
    ) -> T:
        """Run one broker request and raise when an empty result was pacing."""

        async with self:
            result = await request()
        if not result and self.verify(contract):
            raise PacingViolationError(
                "Empty IB response matched a recent pacing violation."
            )
        return result

    def verify(self, contract: ibi.Contract) -> bool:
        """Return whether the contract recently hit a pacing violation."""

        return self.registry.verify(contract)

    def onErrEvent(
        self, reqId: int, errorCode: int, errorString: str, contract: ibi.Contract
    ) -> None:
        """Handle ``ib.errorEvent`` updates relevant to pacing violations."""

        self.registry.onError(reqId, errorCode, errorString, contract)
