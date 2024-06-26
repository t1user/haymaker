import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import ClassVar

from .helpers import duration_in_secs


@dataclass
class Restriction:
    holder: ClassVar[deque[datetime]] = deque(maxlen=100)
    seconds: float
    requests: int

    def check(self) -> bool:
        """Return True if pacing restriction neccessary"""
        holder_ = deque(self.holder, maxlen=self.requests)
        if len(holder_) < self.requests:
            return False
        elif (datetime.now(timezone.utc) - holder_[0]) <= timedelta(
            seconds=self.seconds
        ):
            return True
        else:
            return False


@dataclass
class NoRestriction(Restriction):
    seconds: float = 0
    requests: int = 0

    def check(self) -> bool:
        return False


@dataclass
class Pacer:
    restrictions: list[Restriction] = field(
        default_factory=partial(list, NoRestriction())
    )

    async def __aenter__(self):
        while any([timer.check() for timer in self.timers]):
            await asyncio.sleep(0.1)
        # register request time right before exiting the context
        Restriction.holder.append(datetime.now(timezone.utc))

    async def __aexit__(self, *args):
        pass


def pacer(
    barSize,
    wts,
    *,
    restrictions: list[tuple[float, int]] = [],
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
