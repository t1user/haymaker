"""Event inactivity monitors for user-composed Haymaker pipelines."""

from __future__ import annotations

import asyncio
import inspect
import itertools
import logging
import math
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import ClassVar, Self

import eventkit as ev  # type: ignore

from ..base import Atom
from ..details_processor import Details

log = logging.getLogger(__name__)

_counter = itertools.count().__next__
_TimeoutCallback = Callable[[], Awaitable[None] | None]


class EventTimeout:
    """Call a user function after an event stops emitting.

    Use this component to monitor any ``eventkit.Event``, including custom
    events that have no market-data or supervisor semantics. Construction arms
    one inactivity interval. Every source emission restarts that interval. A
    deadline calls ``callback`` once and then remains quiet until another
    source emission begins a new inactivity episode.

    Positive intervals must be constructed while an asyncio event loop is
    running. The owner controls lifetime explicitly with :meth:`cancel`;
    supervised workload restarts do not cancel or recreate a general timeout.
    Ending the source event cancels it automatically.

    Args:
        event: Event whose emission gaps are monitored.
        seconds: Non-negative inactivity interval in seconds. Zero disables
            deadline scheduling while retaining the event connection.
        callback: Synchronous or asynchronous no-argument function called once
            for each inactivity episode.
        name: Optional diagnostic label. An automatic numeric label is used
            when omitted.

    Raises:
        TypeError: If ``event`` is not an eventkit Event, ``seconds`` is not a
            real number, or ``callback`` is not callable.
        ValueError: If ``seconds`` is negative or non-finite.
        RuntimeError: If a positive interval is constructed without a running
            asyncio event loop.

    Example:
        Monitor an application event and release the monitor with its owner::

            stale_events = []
            timeout = EventTimeout(
                updates,
                30,
                callback=lambda: stale_events.append("quotes"),
                name="quote updates",
            )
            try:
                await consume_updates()
            finally:
                timeout.cancel()
    """

    def __init__(
        self,
        event: ev.Event,
        seconds: float,
        *,
        callback: _TimeoutCallback,
        name: str = "",
    ) -> None:
        if not isinstance(event, ev.Event):
            raise TypeError("event must be an eventkit.Event")
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            raise TypeError("seconds must be a real number")
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("seconds must be finite and non-negative")
        if not callable(callback):
            raise TypeError("callback must be callable")

        self.event = event
        self.seconds = float(seconds)
        self.callback = callback
        self.name = name or f"<{_counter()}>"
        self._timer: asyncio.TimerHandle | None = None
        self._callback_tasks: set[asyncio.Task[None]] = set()
        self._cancelled = False
        self._paused = False
        self._triggered = False

        done_callback = self.onEventDone if event.done_event is not None else None
        event.connect(self.onEvent, done=done_callback)
        if event.done():
            self.cancel()
            return
        try:
            self.arm()
        except Exception:
            event.disconnect_obj(self)
            raise

    @property
    def armed(self) -> bool:
        """Return whether an inactivity deadline is currently scheduled."""

        return self._timer is not None and not self._timer.cancelled()

    @property
    def cancelled(self) -> bool:
        """Return whether this timeout has been permanently cancelled."""

        return self._cancelled

    @property
    def triggered(self) -> bool:
        """Return whether the current inactivity episode reached its deadline."""

        return self._triggered

    def arm(self) -> None:
        """Start a complete inactivity interval from the current time.

        Calling this method while armed replaces the existing deadline.
        A cancelled timeout cannot be reused.

        Raises:
            RuntimeError: If the timeout is cancelled or a positive interval
                is armed without a running asyncio event loop.
        """

        if self._cancelled:
            raise RuntimeError("a cancelled EventTimeout cannot be armed")

        self._cancel_timer()
        self._paused = False
        self._triggered = False
        if not self.seconds:
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            raise RuntimeError(
                "positive EventTimeout intervals must be armed while an "
                "asyncio event loop is running"
            ) from exc
        self._timer = loop.call_later(self.seconds, self._deadline_expired)
        log.debug("Timeout armed: %s", self)

    def cancel(self) -> None:
        """Permanently cancel the deadline, callback tasks, and event wiring."""

        if self._cancelled:
            return

        self._cancelled = True
        self._cancel_timer()
        self.event.disconnect_obj(self)
        try:
            current = asyncio.current_task()
        except RuntimeError:
            current = None
        for task in tuple(self._callback_tasks):
            if task is not current and not task.done():
                task.cancel()
        self._callback_tasks.clear()
        log.debug("Timeout cancelled: %s", self)

    def onEvent(self, *args: object) -> None:
        """Restart the interval when the monitored event emits.

        Args:
            *args: Values emitted by the monitored event. They are ignored.
        """

        if self._cancelled or self._paused:
            return
        self.arm()

    def onEventDone(self, event: ev.Event) -> None:
        """Cancel when the monitored event declares itself complete.

        Args:
            event: Completed monitored event.
        """

        self.cancel()

    def _pause(self) -> None:
        """Disarm without making cancellation terminal."""

        self._cancel_timer()
        self._paused = True

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _deadline_expired(self) -> None:
        self._timer = None
        if self._cancelled or self._paused:
            return

        self._triggered = True
        task = asyncio.create_task(
            self._invoke_callback(),
            name=f"event-timeout-callback:{self.name}",
        )
        self._callback_tasks.add(task)
        task.add_done_callback(self._callback_finished)

    async def _invoke_callback(self) -> None:
        try:
            result = self.callback()
            if inspect.isawaitable(result):
                await result
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Timeout callback failed: %s", self)

    def _callback_finished(self, task: asyncio.Task[None]) -> None:
        self._callback_tasks.discard(task)

    def __str__(self) -> str:
        """Return a compact diagnostic description."""

        return f"{type(self).__name__}<{self.seconds:g}s:{self.name}>"


class MarketDataTimeout(EventTimeout):
    """Monitor one market-data event with session and restart awareness.

    This is the stale-subscription monitor used by Haymaker streamers. It adds
    the Atom's Contract details, runtime timeout policy, and supervisor restart
    callback to :class:`EventTimeout`.

    If a deadline occurs while the market is closed, monitoring pauses until
    the next session opens and then starts a full new interval. During an open
    session, ``log_only=True`` logs once and waits for fresh data before
    rearming. Restart mode requests one supervised workload rebuild and remains
    disarmed even if that request is rejected because another lifecycle
    transition is already underway.

    Instances are workload-scoped. Haymaker cancels all active market-data
    timeouts when the supervised live workload stops; the next workload
    generation creates fresh instances from streamer ``onStart`` processing.
    Use :meth:`EventTimeout.cancel` for explicit early cancellation.

    Args:
        event: Market-data event whose emission gaps are monitored.
        seconds: Non-negative inactivity interval in seconds.
        details: Contract details supplying the trading-session calendar.
        request_restart: Supervisor callback used in restart mode.
        log_only: If true, log stale data rather than request a restart.
        name: Optional diagnostic label.

    Raises:
        TypeError: If ``details`` is not
            :class:`~haymaker.details_processor.Details` or a supplied restart
            callback is not callable.
        RuntimeError: If restart mode has a positive interval but no supervisor
            callback.

    Note:
        Prefer :meth:`from_atom` inside ``onStart`` for user-defined
        market-data Atoms. Construct :class:`EventTimeout` directly for events
        that do not need market-session or supervisor behavior.
    """

    _instances: ClassVar[list[MarketDataTimeout]] = []

    def __init__(
        self,
        event: ev.Event,
        seconds: float,
        *,
        details: Details,
        request_restart: Callable[[str], bool | None] | None = None,
        log_only: bool = False,
        name: str = "",
    ) -> None:
        if not isinstance(details, Details):
            raise TypeError("details must be haymaker.details_processor.Details")
        if request_restart is not None and not callable(request_restart):
            raise TypeError("request_restart must be callable or None")
        restart_enabled = (
            isinstance(seconds, (int, float))
            and not isinstance(seconds, bool)
            and math.isfinite(seconds)
            and seconds > 0
        )
        if restart_enabled and not log_only and request_restart is None:
            raise RuntimeError(
                "restart-enabled MarketDataTimeout requires a bound supervisor "
                "restart callback"
            )

        self.details = details
        self.request_restart = request_restart
        self.log_only = log_only
        self._now: datetime | None = None
        super().__init__(
            event,
            seconds,
            callback=self._handle_timeout,
            name=name,
        )
        if not self.cancelled:
            self._instances.append(self)

    @classmethod
    def from_atom(
        cls,
        atom: Atom,
        event: ev.Event,
        key: str = "",
        seconds: float | None = None,
    ) -> Self:
        """Create a workload-scoped market-data timeout from Atom services.

        Call this after contract qualification and supervisor binding, normally
        from ``onStart``. The Atom supplies its resolved Contract details, the
        configured timeout policy, and the current restart callback.

        Args:
            atom: Atom owning the monitored market-data subscription.
            event: Subscription update event to monitor.
            key: Optional label appended to the Atom's diagnostic name.
            seconds: Explicit interval in seconds. ``None`` uses the runtime
                timeout policy.

        Returns:
            Configured market-data timeout.

        Raises:
            ValueError: If the Atom has no resolved Contract details.
            RuntimeError: If restart mode is enabled before the supervisor
                restart callback is bound.
        """

        policy = atom.runtime.timeout_policy
        interval = policy.seconds if seconds is None else seconds
        details = atom.contract_registry.get_details(atom.contract)
        if details is None:
            raise ValueError(
                f"{atom} has no resolved Contract details; create "
                "MarketDataTimeout.from_atom() from onStart or later"
            )

        return cls(
            event,
            interval,
            details=details,
            request_restart=atom.request_restart,
            log_only=policy.log_only,
            name=f"{atom}-<<{key}>>",
        )

    @classmethod
    def _cancel_all(cls) -> None:
        """Cancel every market-data timeout owned by the current workload."""

        count = len(cls._instances)
        for instance in tuple(cls._instances):
            instance.cancel()
        log.debug("%s market-data timeouts cancelled.", count)

    def cancel(self) -> None:
        """Cancel this monitor and remove it from workload ownership."""

        super().cancel()
        try:
            self._instances.remove(self)
        except ValueError:
            pass

    async def _handle_timeout(self) -> None:
        now = self._now or datetime.now(timezone.utc)
        log.log(5, "%s detected no data for %s seconds.", self, self.seconds)

        if self.details.is_open(now):
            if self.log_only:
                log.error("%s triggered; market data may be stale.", self)
                return

            self._pause()
            assert self.request_restart is not None
            log.debug("Stale market data will request restart: %s", self)
            self.request_restart(f"stale market data: {self}")
            return

        next_open = self.details.next_open(now)
        if next_open is None:
            self._pause()
            log.debug("%s has no future market session to monitor.", self)
            return

        self._pause()
        delay = max((next_open - now).total_seconds(), 0)
        log.log(
            5,
            "%s paused until market reopens at %s (%s seconds).",
            self,
            next_open,
            delay,
        )
        await asyncio.sleep(delay)
        if not self.cancelled:
            self.arm()


__all__ = ["EventTimeout", "MarketDataTimeout"]
