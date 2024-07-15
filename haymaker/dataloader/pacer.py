import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import ClassVar

from .helpers import duration_in_secs

log = logging.getLogger(__name__)


@dataclass
class Restriction:
    holder: ClassVar[deque[datetime]] = deque(maxlen=100)
    requests: int
    seconds: float

    def check(self) -> bool:
        """Return True if pacing restriction neccessary"""
        holder_ = deque(self.holder, maxlen=self.requests)
        if len(holder_) < self.requests:
            return False
        elif (datetime.now(timezone.utc) - holder_[0]) <= timedelta(
            seconds=self.seconds + 0.1
        ):
            return True
        else:
            return False


@dataclass
class NoRestriction(Restriction):
    requests: int = 0
    seconds: float = 0

    def check(self) -> bool:
        return False


@dataclass
class Pacer:
    restrictions: list[Restriction] = field(
        default_factory=partial(list, NoRestriction())
    )

    async def __aenter__(self):
        while any([restriction.check() for restriction in self.restrictions]):
            await asyncio.sleep(0.1)
        # register request time right before exiting the context
        Restriction.holder.append(datetime.now(timezone.utc))
        log.debug("pacer holding...")

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
