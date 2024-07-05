import logging
from typing import List, Optional, Union

import eventkit as ev  # type: ignore
import ib_insync as ibi

from .base import Atom

log = logging.getLogger(__name__)


class BarAggregator(Atom):
    def __init__(
        self,
        filter: Union["CountBars", "VolumeBars", "TimeBars", "NoFilter"],
        incremental_only: bool = False,
        debug: bool = False,
    ):
        Atom.__init__(self)
        self._incremental_only = incremental_only
        self._filter = filter
        self._filter += self.onDataBar
        self._log_level = log.level
        self._debug = debug

    def onStart(self, data, *args):
        if isinstance(data, dict):
            startup = data.get("startup")
            # prevent logging messages during startup phase
            if startup and not self._debug:
                log.setLevel(logging.ERROR)
            else:
                log.setLevel(self._log_level)
        super().onStart(data, *args)

    def onDataBar(self, bars, *args):
        if self._incremental_only:
            try:
                self.dataEvent.emit(bars[-1])
            except KeyError:
                log.debug(f"Empty input from filter {self._filter}")
        else:
            self.dataEvent.emit(bars)

    def onData(self, data, *args) -> None:
        if isinstance(data, ibi.BarDataList):
            # Streamers with incremental_only=False have not been properly tested!
            log.critical("WE SHOULD NOT BE HERE")
            try:
                data = data[-1]
            except KeyError:
                log.debug("Empty input from streamer.")
        self._filter.on_source(data)


class BarList(List[ibi.BarData]):
    def __init__(self, *args):
        super().__init__(*args)
        self.updateEvent = ibi.Event("updateEvent")

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)


class CountBars(ev.Op):
    __slots__ = ("_count", "bars", "label")

    bars: BarList

    def __init__(
        self, count: int, source: Optional[ev.Event] = None, *, label: str = ""
    ):
        ev.Op.__init__(self, source)
        self._count = count
        self.bars = BarList()

    def on_source(self, new_bar: ibi.BarData, *args) -> None:
        if not self.bars or self.bars[-1].barCount == self._count:
            bar = new_bar
            bar.average = new_bar.average * new_bar.volume
            bar.barCount = 1
            self.bars.append(bar)
        else:
            bar = self.bars[-1]
            bar.high = max(bar.high, new_bar.high)
            bar.low = min(bar.low, new_bar.low)
            bar.close = new_bar.close
            bar.volume += new_bar.volume
            bar.average = new_bar.average * new_bar.volume
            bar.barCount += 1
        if bar.barCount == self._count:
            bar.average = bar.average / bar.volume
            self.bars.updateEvent.emit(self.bars, True)
            self.emit(self.bars)


class VolumeBars(ev.Op):
    __slots__ = ("_volume", "bars", "label")

    bars: BarList

    def __init__(
        self, volume: int, source: Optional[ev.Event] = None, *, label: str = ""
    ) -> None:
        ev.Op.__init__(self, source)
        self._volume = volume
        self.bars = BarList()
        self.label = label

    def on_source(self, new_bar: ibi.BarData, *args) -> None:
        # filter out faulty bars emitted as first daily bars
        if new_bar.volume < 0 or new_bar.barCount < 0:
            return
        if not self.bars or self.bars[-1].volume >= self._volume:
            bar = new_bar
            bar.average = bar.average * bar.volume
            self.bars.append(bar)
        else:
            bar = self.bars[-1]
            bar.high = max(bar.high, new_bar.high)
            bar.low = min(bar.low, new_bar.low)
            bar.close = new_bar.close
            bar.volume += new_bar.volume
            bar.average = new_bar.average * new_bar.volume
            bar.barCount += new_bar.barCount
        log.log(
            5,
            f"{new_bar.date} --> accumulated volume: "
            f"{bar.volume:6.0f} /{self._volume:6.0f} += {new_bar.volume:5.0f},"
            f" --> {self.label}",
        )
        if bar.volume >= self._volume:
            bar.average = bar.average / bar.volume
            log.log(5, f"New candle for {self.label} -> {self.bars[-1]}")
            self.bars.updateEvent.emit(self.bars, True)
            self.emit(self.bars)


class TickBars(ev.Op):
    __slots__ = ("_count", "bars", "label")

    bars: BarList

    def __init__(
        self, count: int, source: Optional[ev.Event] = None, *, label: str = ""
    ) -> None:
        ev.Op.__init__(self, source)
        self._count = count
        self.bars = BarList()
        self.label = label

    def on_source(self, new_bar: ibi.BarData, *args) -> None:
        if not self.bars or self.bars[-1].barCount == self._count:
            bar = new_bar
            new_bar.average = new_bar.average * new_bar.volume
            self.bars.append(bar)
        else:
            bar = self.bars[-1]
            bar.high = max(bar.high, new_bar.high)
            bar.low = min(bar.low, new_bar.low)
            bar.close = new_bar.close
            bar.volume += new_bar.volume
            bar.average = new_bar.average * new_bar.volume
            bar.barCount += new_bar.barCount
        if bar.barCount == self._count:
            bar.average = bar.average / bar.volume
            self.bars.updateEvent.emit(self.bars, True)
            self.emit(self.bars)


class TimeBars(ev.Op):
    __slots__ = ("_timer", "bars", "_running_price_volume", "label")

    bars: BarList

    def __init__(
        self, timer, source: Optional[ev.Event] = None, *, label: str = ""
    ) -> None:
        ev.Op.__init__(self, source)
        self._timer = timer
        self._timer.connect(self._on_timer, None, self._on_timer_done)
        self.bars = BarList()
        self._running_price_volume = 0.0
        self.label = label

    def on_source(self, new_bar: ibi.BarData, *args) -> None:
        if not self.bars:
            return
        bar = self.bars[-1]
        if bar.open == 0:
            bar = new_bar
        self._running_price_volume += new_bar.average * new_bar.volume
        bar.high = max(bar.high, new_bar.high)
        bar.low = min(bar.low, new_bar.low)
        bar.close = new_bar.close
        bar.volume += new_bar.volume
        bar.average = self._running_price_volume / bar.volume
        bar.barCount += new_bar.barCount

        self.bars.updateEvent.emit(self.bars, False)

    def _on_timer(self, time) -> None:
        if self.bars:
            bar = self.bars[-1]
            if bar.close == 0 and len(self.bars) > 1:
                bar.open = bar.high = bar.low = bar.close = self.bars[-2].close
            self.bars.updateEvent.emit(self.bars, True)
            self.emit(bar)
        self.bars.append(ibi.BarData(time))

    def _on_timer_done(self, timer) -> None:
        self._timer = None
        self.set_done()


class NoFilter(ev.Op):
    """
    This works as an accumulator making sure that no bars are lost
    during restarts.
    """

    __slots__ = ("bars", "label")

    bars: BarList

    def __init__(self, source: Optional[ev.Event] = None, *, label: str = "") -> None:
        ev.Op.__init__(self, source)
        self.bars = BarList()
        self.label = label

    def on_source(self, new_bar: ibi.BarData, *args) -> None:
        self.bars.append(new_bar)
        log.log(5, f"New candle for {self.label} -> {self.bars[-1]}")
        self.bars.updateEvent.emit(self.bars, True)
        self.emit(self.bars)
