from __future__ import annotations

import asyncio
import itertools
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
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
from ..timeout import Timeout

log = logging.getLogger(__name__)


_counter = itertools.count().__next__


def bar_filter(bar: ibi.BarData) -> bool:
    """Return whether an IB bar contains a non-positive price.

    Use this predicate with bar aggregators when malformed market-data bars
    should be excluded before downstream calculations.

    Args:
        bar: Broker bar to validate.

    Returns:
        ``True`` when any OHLC or average price is non-positive.
    """
    return any(
        (
            bar.close <= 0,
            bar.open <= 0,
            bar.high <= 0,
            bar.low <= 0,
            bar.average <= 0,
        )
    )


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
        ticker.updateEvent.clear()
        ticker.updateEvent += self.dataEvent
        self._set_timeout(ticker.updateEvent, "ticks")
        while self.ib.isConnected():
            await asyncio.sleep(0)

    def _set_timeout(self, event: ev.Event, name: str) -> None:
        """
        Automatically monitor event for stale data.  Can be switched
        off by overriding class variable `set_timeout`
        """
        if self.timeout and isinstance(self.timeout, bool):
            Timeout.from_atom(self, event, name)
        elif self.timeout:
            Timeout.from_atom(self, event, name, self.timeout)

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


@dataclass
class HistoricalDataStreamer(Streamer):
    """Stream IB historical bars and continue with completed live bars.

    Use this source when a strategy needs an initial bar history followed by
    updates from ``reqHistoricalDataAsync(keepUpToDate=True)``. It emits a
    ``BarDataList`` only when a new completed bar is available; the final
    work-in-progress bar is excluded.

    Args:
        contract: Contract blueprint registered with the runtime.
        durationStr: IB duration string or desired historical bar count.
        barSizeSetting: IB bar-size string such as ``"1 hour"``.
        whatToShow: IB data type such as ``"TRADES"``.
        useRTH: Whether to request regular-trading-hours data only.
        formatDate: IB timestamp format. ``2`` is the supported aware-UTC
            setting.
        datastore: Optional awaited store used to find the last persisted bar.
            Its symbol naming must match ``barSizeSetting``.
        timeout: Runtime timeout policy override in seconds, ``True`` for the
            configured default, or ``False`` to disable monitoring.

    Note:
        IB updates keep-up-to-date historical subscriptions approximately
        every five seconds. Use a tick streamer when that latency is unsuitable.
    """

    contract: ibi.Contract
    durationStr: str | int  # can be given as number of required datapoints
    barSizeSetting: str
    whatToShow: str
    useRTH: bool = False
    formatDate: int = 2  # should be 2 for utc timestamp
    datastore: AsyncDataStore | None = None
    timeout: bool | float = True
    _last_bar_date: datetime | None = None

    def __post_init__(self) -> None:
        if isinstance(self.datastore, bool):
            raise TypeError(
                "datastore must be an AsyncDataStore or None; "
                "boolean shortcuts are not supported"
            )
        Atom.__init__(self)

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

    async def last_db_point(self) -> datetime | None:
        """
        Return datetime for the last bar availble in the datastore for
        given contract.

        start_date: how far back should available data be searched
        """
        if (store := self.datastore) is None:
            return None

        if up_to := (await store.read_metadata(self.contract)).get("up_to"):
            log.debug(f"{self!s} retrieved last date from datastore: {up_to}")
            return format_timestamp(up_to)

        df = await store.read(self.contract)
        try:
            return df.index[-1]  # type: ignore
        except (AttributeError, IndexError):
            return None

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
        return (
            date_to_delta_wrapper(self._last_bar_date, self.barSizeSetting, margin=2)
            if self._last_bar_date
            else self._ensure_durationStr()
        )

    async def sync_last_bar_date(self) -> None:
        if self._last_bar_date is None:
            self._last_bar_date = await self.last_db_point()

    async def run(self) -> None:
        self.onStart({})

        # this starts subscription so that current price is readily available from ib
        stream = self.ib.reqMktData(self.contract, "221")
        self._set_timeout(stream.updateEvent, "ticks")

        log.debug(f"{self!s} requesting bars {self._durationStr=}")
        await self.sync_last_bar_date()
        bars = await self.streaming_func()
        log.debug(
            f"{self!s} received historical bars, last bar date: "
            f"{bars[-1].date if bars else bars}"
        )
        self._set_timeout(bars.updateEvent, "bars")

        try:
            async for bars_, hasNewBar in bars.updateEvent:
                if hasNewBar and (
                    (not self._last_bar_date) or (bars_[-2].date > self._last_bar_date)
                ):
                    self._last_bar_date = bars_[-2].date
                    self.on_new_bar(bars_[:-1])
        except ValueError as e:
            log.debug(f"Empty emit for {self!s}: {e}")

    def on_new_bar(self, bars: ibi.BarDataList) -> None:
        if bar_filter(bars[-1]):
            return
        else:
            self.dataEvent.emit(bars)

    def onContractChanged(
        self, old_contract: ibi.Contract, new_contract: ibi.Contract
    ) -> None:
        self._last_bar_date = None
        super().onContractChanged(old_contract, new_contract)


@dataclass
class MktDataStreamer(Streamer):
    """Stream IB level-one market-data ticker updates.

    Args:
        contract: Contract blueprint to subscribe to.
        tickList: IB generic tick-list string.
        timeout: Runtime timeout policy override.

    Emits:
        Updated IB ``Ticker`` objects from ``reqMktData``.
    """

    contract: ibi.Contract
    tickList: str
    timeout: bool | float = True

    def __post_init__(self):
        Atom.__init__(self)

    def streaming_func(self) -> ibi.Ticker:
        return self.ib.reqMktData(self.contract, self.tickList)


@dataclass
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

    contract: ibi.Contract
    whatToShow: str
    useRTH: bool
    realTimeBarsOptions: list[ibi.TagValue] = field(default_factory=list)
    timeout: bool | float = True

    def __post_init__(self):
        Atom.__init__(self)

    def streaming_func(self):
        return self.ib.reqRealTimeBars(
            self.contract,
            5,
            self.whatToShow,
            self.useRTH,
        )

    async def run(self):
        """
        The difference to superclass is that here we connect the
        `updateEvent` to intermediary function `onUpdate`, which needs
        to perform some additional checks before emitting `dataEvent`.
        """
        self.onStart({})
        bars = self.streaming_func()
        bars.updateEvent.clear()
        bars.updateEvent += self.onUpdate
        self._set_timeout(bars.updateEvent, "bars")
        while self.ib.isConnected():
            await asyncio.sleep(0)

    def onUpdate(self, bars, hasNewBar):
        # No need to filter out the last bar
        # emits are every 5 secs, hasNewBar always True
        # last bar is ready and not modified after emit
        if hasNewBar and not bar_filter(bars[-1]):
            self.dataEvent.emit(bars)


@dataclass
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

    contract: ibi.Contract
    tickType: str
    numberOfTicks: int = 0
    ignoreSize: bool = False
    timeout: bool | float = True

    def __post_init__(self):
        Atom.__init__(self)

    def streaming_func(self) -> ibi.Ticker:
        return self.ib.reqTickByTickData(
            contract=self.contract,
            tickType=self.tickType,
            numberOfTicks=self.numberOfTicks,
            ignoreSize=self.ignoreSize,
        )
