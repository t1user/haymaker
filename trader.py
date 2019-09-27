import pandas as pd
import eventkit as ev


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

    def onOpenOrderEvent(self, *args, **kwargs):
        print('OpenOrderEvent')
        print(args, kwargs)
        print('---------')

    def onOrderStatusEvent(self, *args, **kwargs):
        print('OrderStatusEvent')
        print(args, kwargs)
        print('---------')

    def onExecDetailsEvent(self, *args, **kwargs):
        print('ExecDetailsEvent')
        print(args, kwargs)
        print('---------')

    def onUpdatePortfolioEvent(self, *args, **kwargs):
        print('UpdatePortfolioEvent')
        print(args, kwargs)
        print('---------')

    def onPositionEvent(self, position):
        print('PositionEvent')
        print(position)
        #trades = self.ib.openOrders()
        #print('trades: ', trades)
        #print(position.position, position.contract)
        print('---------')

    def onPnlSigleEvent(self, *args, **kwargs):
        print('PnlSingleEvent')
        print(args, kwargs)
        print('---------')

    def onErrorEvent(self, *args, **kwargs):
        print('ErrorEvent')
        print(args, kwargs)
        print('---------')

    def onUpdateEvent(self, *args, **kwargs):
        print('UpdateEvent')
        print(args, kwargs)
        print('---------')

    def onBarUpdateEvent(self, *args, **kwargs):
        print('BarUpdateEvent')
        print(args, kwargs)
        print('---------')


class Consolidator:

    events = ('newCandle')
    newCandle = ev.Event('newCandle')

    __iadd__ = newCandle.connect
    __isub__ = newCandle.disconnect
    __or__ = newCandle.pipe
    connect = newCandle.connect
    disconnect = newCandle.disconnect
    clear = newCandle.clear
    pipe = newCandle.pipe

    def __init__(self, bars):
        self.new_bars = []
        self.subscribe(bars)

    def subscribe(self, bars):
        def onNewBar(bars, hasNewBar):
            if hasNewBar:
                if bars[-1].volume < 0:
                    print('---FAULTY BAR---')
                    print(bars[-1])
                    print('---END FAULTY BAR---')
                    return
            # print(bars[-1])
            self.new_bars.append(bars[-1])
            self.aggregate(bars[-1])

        bars.updateEvent += onNewBar

    def aggregate(self, bar):
        raise NotImplementedError
