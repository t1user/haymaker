import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import ClassVar

import ib_insync as ibi

from .helpers import duration_in_secs

log = logging.getLogger(__name__)


@dataclass
class Restriction:
    holder: ClassVar[deque[datetime]] = deque(maxlen=100)
    requests: int
    seconds: float

    def check(self) -> float:
        """Return required sleep time in seconds."""
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
class InnerContext:
    """Created separately for every wait instance."""

    wait_time: float

    async def __aenter__(self):
        # add up to 1 sec random time to prevent all pacers exiting
        # from wait simultanously
        wait_time = self.wait_time  # + random.randint(0, 100) / 100
        log.debug(
            f"Will throtle: {round(wait_time, 1)}s till "
            f"{datetime.now() + timedelta(seconds=wait_time)}"
        )
        await asyncio.sleep(wait_time)

    async def __aexit__(self, *args):
        pass


@dataclass
class Pacer:
    """One object for all workers."""

    restrictions: list[Restriction] = field(
        default_factory=partial(list, NoRestriction())
    )

    async def __aenter__(self):
        # inner context ensures separate wait time for each worker
        while time := max([restriction.check() for restriction in self.restrictions]):
            async with InnerContext(time):
                pass
        # register request time right before making the request
        # request should be the next instruction out of this context
        Restriction.holder.append(datetime.now(timezone.utc))

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
    return Pacer([Restriction(*res) for res in restrictions])


class PacingViolationRegistry:
    """
    Keep record of contracts that registered pacing violation so that
    their jobs can be rescheduled.

    * onError: should be connected to ib.onError event, will filter
    pacing violation errors and process them

    * verify: return True if given contract has been affected by
    pacing violation within last second
    """

    data: set[ibi.Contract] = set()

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
    pass
