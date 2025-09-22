from __future__ import annotations

import asyncio
import itertools
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import ClassVar, Self

import eventkit as ev  # type: ignore
import ib_insync as ibi

from haymaker.base import Atom, Details
from haymaker.config import CONFIG as config

log = logging.getLogger(__name__)

CONFIG = config.get("timeout", {})
# debug means log, otherwise restart
TIMEOUT_DEBUG = CONFIG.get("debug", False)
# zero means no timeout
TIMEOUT_TIME = CONFIG.get("time", 0)

_counter = itertools.count().__next__


@dataclass
class Timeout:
    """
    Can be attached to any :eventkit.event:class:`Event` to monitor if
    this event keeps emitting data.  Any breat in emits longer than
    `time` will cause a timeout, which depanding on `debug` parameter
    will either log an error or restart the system.

    Args:
    =====

    event: event to be monitored

    time: time after which timout will be triggered, if not given
    default value will be used from config

    name: string that this object will be refered to in logs

    details: if given, will adjust timeouts to trading calendar
    embedded in details (timouts only during market hours)

    debug: if True, timeout will log an error, otherwise restart the
    system
    """

    instances: ClassVar[list["Timeout"]] = []
    ib: ClassVar[ibi.IB]

    event: ev.Event
    time: float = TIMEOUT_TIME
    name: str = ""
    details: Details | None = None
    debug: bool = TIMEOUT_DEBUG
    _timeout: ev.Event | None = field(repr=False, default=None)
    _now: datetime | None = None  # for testing only

    @classmethod
    def set_ib(cls, ib: ibi.IB):
        cls.ib = ib

    @classmethod
    def from_atom(
        cls, atom: Atom, event: ev.Event, key: str = "", time: float = TIMEOUT_TIME
    ) -> Self:
        """
        Extract relevant information from passed `atom`.

        Args:
        =====

        atom: atom object from which information is to be extracted

        even: event to be monitored

        key: if given, it will be concatinated with atom name to
        create timout name

        time: time after which timeout will be triggered, if not given
        default value will be used from config
        """

        assert isinstance(atom.details, Details), (
            f"{atom} is missing contract details."
            f"`Timeout.from_atom` can be used only with atoms that have details."
        )
        return cls(
            event,
            time,
            f"{str(atom)}-<<{key}>>",
            atom.details,
        )

    @classmethod
    def reset(cls) -> None:
        # Must be run on system restart so that timouts are not duplicated
        n = len(cls.instances)
        cls.instances.clear()
        log.debug(f"{n} timeouts cleared.")

    def __new__(cls, *args, **kwargs) -> Self:
        # Keep track of all :class:`.Timeout` instances created so that they
        # can be cleared and reset after reboot.
        obj = super().__new__(cls)
        cls.instances.append(obj)
        return obj

    def __post_init__(self) -> None:
        assert isinstance(
            self.event, ev.Event
        ), f"{self.event} must be an instance of `eventkit.Event`"

        self.name = self.name or f"<{_counter()}>"

        if self.time:
            self._set_timeout(self.event)

    def _on_timeout_error(self, *args) -> None:
        log.error(f"{self!s} Timeout broken... {args}")

    async def _timeout_callback(self, *args) -> None:
        log.log(5, f"Timeout: {self!s} - No data for {self.time} secs.")

        if not self.details:
            self.triggered_action()
        else:
            log.debug(f"{self!s} Market open?: {self.details.is_open(self._now)}")

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

    def triggered_action(self) -> None:
        if self.debug:
            log.error(f"{self!s} triggered. Possibly system reset needed.")
        else:
            log.debug(f"Stale streamer {self!s} will disconnect ib...")
            self.ib.disconnect()

    def _set_timeout(self, event: ev.Event) -> None:
        self._timeout = event.timeout(self.time)
        self._timeout.connect(self._timeout_callback)
        log.debug(f"Timeout set: {self!s}")

    def __str__(self) -> str:
        return f"Timeout <{self.time}s> for {self.name}  event id: {id(self._timeout)}"
