from typing import List, Union

import eventkit as ev  # type: ignore
import ib_insync as ibi
from logbook import Logger  # type: ignore

from ib_tools.base import Atom

log = Logger(__name__)


class BarAggregator(Atom):
    def __init__(self, filter: Union["CountBars", "VolumeBars", "TimeBars"]):
        Atom.__init__(self)
        self._filter = filter
        self._filter += self.onDataBar

    def onDataBar(self, bars, *args):
        self.dataEvent.emit(bars[-1])

    def onData(self, data, *args) -> None:
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
    __slots__ = ("_count", "bars")

    bars: BarList

    def __init__(self, count, source=None):
        ev.Op.__init__(self, source)
        self._count = count
        self.bars = BarList()

    def on_source(self, new_bar: ibi.BarData, *args):
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
    __slots__ = ("_volume", "bars")

    bars: BarList

    def __init__(self, volume, source=None):
        ev.Op.__init__(self, source)
        self._volume = volume
        self.bars = BarList()

    def on_source(self, new_bar: ibi.BarData, *args):
        if not self.bars or abs(self.bars[-1].volume) >= abs(
            self._volume
        ):  # remove abs!!!
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
        if bar.volume == self._volume:
            bar.average = bar.average / bar.volume
            self.bars.updateEvent.emit(self.bars, True)
            self.emit(self.bars)


class TickBars(ev.Op):
    __slots__ = ("_count", "bars")

    bars: BarList

    def __init__(self, count, source=None):
        ev.Op.__init__(self, source)
        self._count = count
        self.bars = BarList()

    def on_source(self, new_bar: ibi.BarData, *args):
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
    __slots__ = ("_timer", "bars", "_running_price_volume")

    bars: BarList

    def __init__(self, timer, source=None):
        ev.Op.__init__(self, source)
        self._timer = timer
        self._timer.connect(self._on_timer, None, self._on_timer_done)
        self.bars = BarList()
        self._running_price_volume = 0

    def on_source(self, new_bar: ibi.BarData, *args):
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

    def _on_timer(self, time):
        if self.bars:
            bar = self.bars[-1]
            if bar.close == 0 and len(self.bars) > 1:
                bar.open = bar.high = bar.low = bar.close = self.bars[-2].close
            self.bars.updateEvent.emit(self.bars, True)
            self.emit(bar)
        self.bars.append(ibi.BarData(time))

    def _on_timer_done(self, timer):
        self._timer = None
        self.set_done()
