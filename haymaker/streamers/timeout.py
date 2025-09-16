from __future__ import annotations

import asyncio
import logging
from collections import UserDict, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

import eventkit as ev  # type: ignore
import ib_insync as ibi

from haymaker.base import Details
from haymaker.config import CONFIG

if TYPE_CHECKING:
    from .streamers import Streamer

log = logging.getLogger(__name__)

# debug means log, otherwise restart
TIMEOUT_DEBUG = CONFIG.get("streamer", {}).get("timeout", {}).get("debug", False)


@dataclass
class Timeout:
    time: float
    event: ev.Event
    ib: ibi.IB
    details: Details | None = None
    name: str = ""
    debug: bool = TIMEOUT_DEBUG
    _timeout: ev.Event | None = field(repr=False, default=None)
    _now: datetime | None = None  # for testing only

    def __post_init__(self):
        if self.time:
            self._set_timeout(self.event)

    def _on_timeout_error(self, *args):
        log.error(f"{self!s} Timeout broken... {args}")

    async def _timeout_callback(self, *args) -> None:
        if not self.details:
            log.error("Cannot set timout, no trading hours details.")
        else:
            log.log(
                5,
                f"{self!s} - No data for {self.time} secs. Market open?: "
                f"{self.details.is_open(self._now)}",
            )
            if self.details.is_open(self._now):
                self.triggered_action()
            else:
                reactivate_time = self.details.next_open(self._now)

                sleep_time = (reactivate_time - datetime.now(tz=timezone.utc)).seconds
                log.log(
                    5,
                    f"{self} will sleep till market reopen at: {reactivate_time} i.e. "
                    f"{sleep_time} seconds",
                )
                await asyncio.sleep(sleep_time)
                self._set_timeout(self.event)

    def triggered_action(self):
        if self.debug:
            log.error(f"Stale streamer {self!s} Reset?")
            return True
        else:
            log.debug(f"Stale streamer {self!s} will disconnect ib...")
            self.ib.disconnect()

    def _set_timeout(self, event: ev.Event) -> None:
        self._timeout = event.timeout(self.time)
        self._timeout.connect(self._timeout_callback)
        log.debug(f"Timeout set: {self!s}")

    def __str__(self) -> str:
        return f"Timeout <{self.time}s> for {self.name}  event id: {id(self._timeout)}"


class TimeoutContainer(UserDict):
    """
    Class that stores and creates timeouts based on parameters extracted from
    passed :class:`Streamer` instance.

    Args:
        streamer: :class:`Streamer` instance, whose streams are being kept alive


    Usage:
    .. code-block:: python

        class MyStreamer(Streamer):

            def some_meth(self):
                ticker = self.streaming_func()
                # this ensures streaming event is kept alive
                # and timer is added to TimeoutContainerDefaultdict
                self.timers["my_timer"] = ticker.updateEvent
                ticker.updateEvent += self.dataEvent

            def other_meth(self):
                # timer is accessible via:
                timer = self.timers["my_timer"]

        s = MyStreamer()
    """

    def __init__(self, streamer: Streamer) -> None:
        self.streamer = streamer
        super().__init__()

    def __setitem__(self, key: str, obj) -> None:
        if isinstance(obj, ev.Event):
            assert isinstance(self.streamer.details, Details)
            obj = Timeout(
                self.streamer.timeout,
                obj,
                self.streamer.ib,
                self.streamer.details,
                f"{str(self.streamer)}-<<{key}>>",
            )
        super().__setitem__(key, obj)


class TimeoutContainerDefaultdict(defaultdict):
    """
    Container for all timeouts.

    This is one container shared among all instances of :class:`Streamer`.
    * keys are :class:`Streamer` instances - each instance keeps its own dict of
      timeouts with unique names; it's essential that each :class:`Streamer` is hashable
    * values are :class:`TimeoutContainer`, which are modified dictionaries, where each
        :class:`Streamer` can keep its timeouts with keys unique for that class.

    Args:
        default_factory: essentially it has to be :type:`TimeoutContainer`
    """

    def __init__(self, default_factory: Callable = TimeoutContainer):
        super().__init__(default_factory)

    def __missing__(self, key: Any) -> Any:
        if self.default_factory:
            self[key] = self.default_factory(key)  # type: ignore
            return self[key]
        raise KeyError(key)
