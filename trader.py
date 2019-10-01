import pandas as pd
import eventkit as ev
from ib_insync import util


class BaseStrategy:

    def __init__(self, ib):
        #ib.openOrderEvent += self.onOpenOrderEvent
        #ib.orderStatusEvent += self.onOrderStatusEvent
        #ib.execDetailsEvent += self.onExecDetailsEvent
        #ib.updatePortfolioEvent += self.onUpdatePortfolioEvent
        #ib.positionEvent += self.onPositionEvent
        #ib.pnlSingleEvent += self.onPnlSigleEvent
        ib.errorEvent += self.onErrorEvent
        #ib.updateEvent += cls.onUpdateEvent
        #ib.barUpdateEvent += cls.onBarUpdateEvent
        self.ib = ib

    def onOpenOrderEvent(self, *args):
        print('OpenOrderEvent')
        print(args)
        print('---------')

    def onOrderStatusEvent(self, *args):
        print('OrderStatusEvent')
        print(args)
        print('---------')

    def onExecDetailsEvent(self, *args):
        print('ExecDetailsEvent')
        print(args)
        print('---------')

    def onUpdatePortfolioEvent(self, *args):
        print('UpdatePortfolioEvent')
        print(args)
        print('---------')

    def onPositionEvent(self, position):
        print('PositionEvent')
        print(position)
        #trades = self.ib.openOrders()
        #print('trades: ', trades)
        #print(position.position, position.contract)
        print('---------')

    def onPnlSigleEvent(self, *args):
        print('PnlSingleEvent')
        print(args)
        print('---------')

    def onErrorEvent(self, *args):
        print('ErrorEvent')
        print(args)
        print('---------')

    def onUpdateEvent(self, *args):
        print('UpdateEvent')
        print(args)
        print('---------')

    def onBarUpdateEvent(self, *args):
        print('BarUpdateEvent')
        print(args)
        print('---------')


class Consolidator:

    events = ('newCandle')
    newCandle = ev.Event('newCandle')

    #__iadd__ = newCandle.connect
    __isub__ = newCandle.disconnect
    __or__ = newCandle.pipe
    connect = newCandle.connect
    disconnect = newCandle.disconnect
    clear = newCandle.clear
    pipe = newCandle.pipe

    def __init__(self, bars):
        self.bars = bars
        self.new_bars = []
        self.backfill = True

    def __iadd__(self,  *args):
        self.newCandle.connect(*args)
        self.init()

    def init(self):
        for bar in self.bars:
            self.new_bars.append(bar)
            self.aggregate(bar)
        self.backfill = False
        self.subscribe(self.bars)

    def subscribe(self, bars):
        def onNewBar(bars, hasNewBar):
            if hasNewBar:
                self.new_bars.append(bars[-2])
                self.aggregate(bars[-2])

        bars.updateEvent += onNewBar

    def aggregate(self, bar):
        raise NotImplementedError


"""
            if hasNewBar:
                if bars[-1].volume < 0:
                    print('---FAULTY BAR---')
                    print(bars[-1])
                    print('---END FAULTY BAR---')
                    return
            # print(bars[-1])
            self.new_bars.append(bars[-1])
            self.aggregate(bars[-1])
"""
