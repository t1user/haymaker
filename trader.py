import pandas as pd
import eventkit as ev


class Consolidator():

    events = ('newCandle')
    newCandle = ev.Event('newCandle')

    __iadd__ = newCandle.connect
    __isub__ = newCandle.disconnect
    __or__ = newCandle.pipe
    connect = newCandle.connect
    disconnect = newCandle.disconnect
    clear = newCandle.clear

    def __init__(self, volume, bars):
        self.volume = volume
        self.new_bars = []
        self.subscribe(bars)

    def subscribe(self, bars):
        def onNewBar(bars, hasNewBar):
            if hasNewBar:
                self.new_bars.append(bars[-1])
                self.aggregate(bars[-1])
            else:
                print('old bar ', bars[-1])

        bars.updateEvent += onNewBar

    def aggregate(self, bar):
        raise NotImplementedError
