from __future__ import annotations

import itertools
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from functools import cached_property
from typing import Awaitable, ClassVar

import eventkit as ev  # type: ignore
import ib_insync as ibi

from haymaker.misc import format_timestamp

from ..base import Atom
from ..datastore import AsyncDataStore
from ..details_processor import typical_session_length
from ..durationStr_converters import (
    datapoints_to_durationStr,
    date_to_delta_wrapper,
)
from ..validators import wts_validator
from .timeouts import MarketDataTimeout

log = logging.getLogger(__name__)


_counter = itertools.count().__next__


_REALTIME_BAR_DATA_TYPES = frozenset({"TRADES", "MIDPOINT", "BID", "ASK"})


class Streamer(Atom, ABC):
    """Base class for broker subscriptions that emit live market data.

    Concrete streamers wrap one Interactive Brokers subscription, attach its
    update event to ``Atom.dataEvent``, and remain active while the runtime
    IB client is connected. Instantiate streamers during strategy composition;
    the live runtime starts every registered instance once per workload.

    Attributes:
        timeout: ``True`` uses the runtime timeout policy, a positive number
            sets an explicit interval in seconds, and ``False`` disables stale
            data monitoring.

    Note:
        Streamer instances are process-global and are not reset for same-process
        application reuse.
    """

    instances: ClassVar[list["Streamer"]] = []
    timeout: bool | float = True

    def __new__(cls, *args, **kwargs):
        # Keep track of all :class:`.Streamer` instances created so that they
        # can be re-started on reboot.
        obj = super().__new__(cls)
        cls.instances.append(obj)
        return obj

    @classmethod
    def awaitables(cls) -> list[Awaitable]:
        """
        Coroutines from all instantiated streamers.  Can be passed to
        :py:func:`asyncio.gather`
        """
        return [s.run() for s in cls.instances]

    @abstractmethod
    def streaming_func(self):
        raise NotImplementedError

    async def run(self) -> None:
        """
        Start subscription and start emitting data.  This is the main
        entry point into the streamer.
        """
        # this will check if contract changed and trigger
        # `onContractChanged` if necessary
        self.onStart({})
        ticker = self.streaming_func()
        ticker.updateEvent.disconnect(self.dataEvent)
        ticker.updateEvent += self.dataEvent
        try:
            self._set_timeout(ticker.updateEvent, "ticks")
            await self._wait_until_disconnected()
        finally:
            ticker.updateEvent.disconnect(self.dataEvent)

    async def _wait_until_disconnected(self) -> None:
        """Wait without polling until IB emits its next disconnection."""
        if self.ib.isConnected():
            await self.ib.disconnectedEvent

    def _set_timeout(self, event: ev.Event, name: str) -> None:
        """Install stale-market-data monitoring for one subscription event.

        Args:
            event: Broker update event to monitor.
            name: Diagnostic label appended to the streamer name.
        """
        if self.timeout and isinstance(self.timeout, bool):
            MarketDataTimeout.from_atom(self, event, name)
        elif self.timeout:
            MarketDataTimeout.from_atom(self, event, name, self.timeout)

    @cached_property
    def _id(self) -> int:
        return _counter()

    def __str__(self) -> str:
        identifier = [str(self._id)]
        if contract := getattr(self, "contract", None):
            identifier.append(contract.symbol)
        if name := getattr(self, "name", None):
            identifier.append(name)
        return f"{self.__class__.__name__}<{'><'.join(identifier)}>"


@dataclass(eq=False)
class HistoricalDataStreamer(Streamer):
    """Stream IB historical bars and continue with completed live bars.

    Use this source when a strategy needs an initial bar history followed by
    updates from ``reqHistoricalDataAsync(keepUpToDate=True)``. It emits a
    list of completed ``BarData`` objects only when a new valid completed bar
    is available; the final work-in-progress bar is excluded.

    Args:
        contract: Contract blueprint registered with the runtime.
        durationStr: IB duration string or desired historical bar count.
        barSizeSetting: IB bar-size string such as ``"1 hour"``.
        whatToShow: IB data type such as ``"TRADES"``.
        useRTH: Whether to request regular-trading-hours data only.
        formatDate: IB timestamp format. ``2`` is the supported aware-UTC
            setting.
        datastore: ``False`` disables persisted-endpoint lookup. ``True`` uses
            the runtime-default market-data store. An
            :class:`~haymaker.datastore.AsyncDataStore` uses that custom store
            instead. Defaults to ``False``.
        timeout: Runtime timeout policy override in seconds, ``True`` for the
            configured default, or ``False`` to disable monitoring.

    Note:
        IB updates keep-up-to-date historical subscriptions approximately
        every five seconds. Use a tick streamer when that latency is unsuitable.
    """

    # field() supplies no default: it keeps this dataclass argument required
    # while allowing Atom.contract's inherited descriptor to remain active.
    contract: ibi.Contract = field()
    durationStr: str | int  # can be given as number of required datapoints
    barSizeSetting: str
    whatToShow: str
    useRTH: bool = False
    formatDate: int = 2  # should be 2 for utc timestamp
    datastore: bool | AsyncDataStore = False
    timeout: bool | float = True
    _last_bar_date: date | datetime | None = None

    def __post_init__(self) -> None:
        self.whatToShow = wts_validator(self.whatToShow)
        if self.datastore is None:
            raise TypeError(
                "HistoricalDataStreamer datastore must be False, True, "
                "or an AsyncDataStore"
            )
        Atom.__init__(self)
        if self.datastore is True:
            self.datastore = self.runtime.market_data_store_factory(
                bar_size_setting=self.barSizeSetting,
                what_to_show=self.whatToShow,
                use_rth=self.useRTH,
            )

    def streaming_func(self) -> Awaitable:
        return self.ib.reqHistoricalDataAsync(
            self.contract,
            endDateTime="",
            durationStr=self._durationStr,
            barSizeSetting=self.barSizeSetting,
            whatToShow=self.whatToShow,
            useRTH=self.useRTH,
            formatDate=self.formatDate,
            keepUpToDate=True,
            timeout=0,
        )

    async def last_db_point(self) -> date | datetime | None:
        """Return the date of the last persisted bar.

        Returns:
            Last persisted bar date in its original date or datetime category,
            or ``None`` when no persisted data is available.
        """
        store = self.datastore
        if store is False:
            return None
        if store is True:
            raise RuntimeError("HistoricalDataStreamer datastore was not initialized")

        if up_to := (await store.read_metadata(self.contract)).get("up_to"):
            log.debug(f"{self!s} retrieved last date from datastore: {up_to}")
            return self._restore_bar_date(up_to)

        df = await store.read(self.contract)
        try:
            return df.index[-1]  # type: ignore
        except (AttributeError, IndexError):
            return None

    @staticmethod
    def _restore_bar_date(value: date | datetime | str) -> date | datetime:
        """Restore a persisted bar timestamp to its temporal category.

        Args:
            value: Intraday datetime, calendar date, or persisted date string.

        Returns:
            Calendar date for date-only values, otherwise a datetime.
        """
        if not isinstance(value, str):
            return value
        if len(value) == 10:
            try:
                return date.fromisoformat(value)
            except ValueError:
                pass
        if len(value) == 8 and value.isdigit():
            return datetime.strptime(value, "%Y%m%d").date()
        return format_timestamp(value)

    def _ensure_durationStr(self) -> str:
        """
        Accept durationStr as either ready str to be passed to
        :meth:`ib_insync.IB.reqHistoricalData` or if it's passed as a
        number of required datapoints int, convert it to the correct str.
        """
        return (
            datapoints_to_durationStr(
                self.durationStr,
                self.barSizeSetting,
                typical_session_length(self.contract_details.trading_hours),
            )
            if isinstance(self.durationStr, int)
            else self.durationStr
        )

    @property
    def _durationStr(self) -> str:
        if self._last_bar_date is None:
            return self._ensure_durationStr()
        start_date = (
            self._last_bar_date
            if isinstance(self._last_bar_date, datetime)
            else datetime.combine(self._last_bar_date, datetime.min.time())
        )
        return date_to_delta_wrapper(start_date, self.barSizeSetting, margin=2)

    async def sync_last_bar_date(self) -> None:
        if self._last_bar_date is None:
            self._last_bar_date = await self.last_db_point()

    async def run(self) -> None:
        self.onStart({})

        # this starts subscription so that current price is readily available from ib
        stream = self.ib.reqMktData(self.contract, "221")
        self._set_timeout(stream.updateEvent, "ticks")

        await self.sync_last_bar_date()
        log.debug(f"{self!s} requesting bars {self._durationStr=}")
        bars = await self.streaming_func()
        log.debug(
            f"{self!s} received historical bars, last bar date: "
            f"{bars[-1].date if bars else bars}"
        )
        self._set_timeout(bars.updateEvent, "bars")

        try:
            async for bars_, hasNewBar in bars.updateEvent:
                if not hasNewBar:
                    continue
                completed_bar_date = bars_[-2].date
                if (not self._last_bar_date) or (
                    completed_bar_date > self._last_bar_date
                ):
                    self._last_bar_date = completed_bar_date
                    self.on_new_bar(bars_[:-1])
        except ValueError as e:
            log.debug(f"Empty emit for {self!s}: {e}")

    def on_new_bar(self, bars: list[ibi.BarData]) -> None:
        """Emit a completed historical snapshot without invalid bars."""
        if not bars or not self._is_valid_bar(bars[-1]):
            return
        self.dataEvent.emit([bar for bar in bars if self._is_valid_bar(bar)])

    def _is_valid_bar(self, bar: ibi.BarData) -> bool:
        """Return whether a historical bar has finite applicable prices."""
        prices = [bar.open, bar.high, bar.low, bar.close]
        if self.whatToShow == "TRADES":
            prices.append(bar.average)
        return all(math.isfinite(price) for price in prices)

    def onContractChanged(
        self, old_contract: ibi.Contract, new_contract: ibi.Contract
    ) -> None:
        self._last_bar_date = None
        super().onContractChanged(old_contract, new_contract)


@dataclass(eq=False)
class MktDataStreamer(Streamer):
    """Stream IB level-one market-data ticker updates.

    Args:
        contract: Contract blueprint to subscribe to.
        tickList: IB generic tick-list string.
        timeout: Runtime timeout policy override.

    Emits:
        Updated IB ``Ticker`` objects from ``reqMktData``.
    """

    # field() supplies no default: it keeps this dataclass argument required
    # while allowing Atom.contract's inherited descriptor to remain active.
    contract: ibi.Contract = field()
    tickList: str
    timeout: bool | float = True

    def __post_init__(self) -> None:
        Atom.__init__(self)

    def streaming_func(self) -> ibi.Ticker:
        return self.ib.reqMktData(self.contract, self.tickList)


@dataclass(eq=False)
class RealTimeBarsStreamer(Streamer):
    """Stream completed IB real-time five-second bars.

    Args:
        contract: Contract blueprint to subscribe to.
        whatToShow: IB real-time bar data type.
        useRTH: Whether to emit regular-trading-hours bars only.
        realTimeBarsOptions: Optional IB request TagValues.
        timeout: Runtime timeout policy override.

    Emits:
        ``RealTimeBarList`` updates whose latest bar has passed basic price
        validation. IB supports only a five-second interval for this request.
    """

    # field() supplies no default: it keeps this dataclass argument required
    # while allowing Atom.contract's inherited descriptor to remain active.
    contract: ibi.Contract = field()
    whatToShow: str
    useRTH: bool
    realTimeBarsOptions: list[ibi.TagValue] = field(default_factory=list)
    timeout: bool | float = True

    def __post_init__(self) -> None:
        if self.whatToShow not in _REALTIME_BAR_DATA_TYPES:
            raise ValueError(
                "Real-time bar whatToShow must be one of "
                f"{sorted(_REALTIME_BAR_DATA_TYPES)}, not {self.whatToShow!r}"
            )
        Atom.__init__(self)

    def streaming_func(self) -> ibi.RealTimeBarList:
        return self.ib.reqRealTimeBars(
            self.contract,
            5,
            self.whatToShow,
            self.useRTH,
            realTimeBarsOptions=self.realTimeBarsOptions,
        )

    async def run(self) -> None:
        """
        The difference to superclass is that here we connect the
        `updateEvent` to intermediary function `onUpdateEvent`, which needs
        to perform some additional checks before emitting `dataEvent`.
        """
        self.onStart({})
        bars = self.streaming_func()
        bars.updateEvent.disconnect(self.onUpdateEvent)
        bars.updateEvent += self.onUpdateEvent
        try:
            self._set_timeout(bars.updateEvent, "bars")
            await self._wait_until_disconnected()
        finally:
            bars.updateEvent.disconnect(self.onUpdateEvent)

    def onUpdateEvent(self, bars: ibi.RealTimeBarList, hasNewBar: bool) -> None:
        """Emit a completed real-time bar list after price validation."""
        if hasNewBar and self._is_valid_bar(bars[-1]):
            self.dataEvent.emit(bars)

    def _is_valid_bar(self, bar: ibi.RealTimeBar) -> bool:
        """Return whether a real-time bar has finite applicable prices."""
        prices = [bar.open_, bar.high, bar.low, bar.close]
        if self.whatToShow == "TRADES":
            prices.append(bar.wap)
        return all(math.isfinite(price) for price in prices)


@dataclass(eq=False)
class TickByTickStreamer(Streamer):
    """Stream IB tick-by-tick ticker updates.

    Args:
        contract: Contract blueprint to subscribe to.
        tickType: IB tick type, for example ``"Last"`` or ``"BidAsk"``.
        numberOfTicks: Historical ticks requested before live continuation.
        ignoreSize: Whether IB should omit same-price size-only changes.
        timeout: Runtime timeout policy override.

    Emits:
        Updated IB ``Ticker`` objects from
        ``reqTickByTickData``.
    """

    # field() supplies no default: it keeps this dataclass argument required
    # while allowing Atom.contract's inherited descriptor to remain active.
    contract: ibi.Contract = field()
    tickType: str
    numberOfTicks: int = 0
    ignoreSize: bool = False
    timeout: bool | float = True

    def __post_init__(self) -> None:
        Atom.__init__(self)

    def streaming_func(self) -> ibi.Ticker:
        return self.ib.reqTickByTickData(
            contract=self.contract,
            tickType=self.tickType,
            numberOfTicks=self.numberOfTicks,
            ignoreSize=self.ignoreSize,
        )


__all__ = [
    "HistoricalDataStreamer",
    "MktDataStreamer",
    "RealTimeBarsStreamer",
    "Streamer",
    "TickByTickStreamer",
]
