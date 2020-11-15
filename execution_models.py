from collection import defaultdict


from trader import Trader
from ib_insync import Contract, Trade, Order, MarketOrder, StopOrder, TagValue
from logger import Logger


log = Logger(__name__)


class BaseExecModel:

    contracts = {}

    # TODO
    # EACH ORDER TYPE PASSED TO INIT
    # ??? RECONCILE ON RE-CONNECT ????
    def __init__(self, trader: Trader):
        self.trader = trader

    def entry_order(self, signal, amount):
        assert signal in [-1, 1], 'Invalid trade signal'
        return MarketOrder('BUY' if signal == 1 else 'SELL',
                           amount, algoStrategy='Adaptive',
                           algoParams=[
                               TagValue('adaptivePriority', 'Normal')],
                           tif='Day')

    close_order = entry_order

    def onEntry(self, contract: Contract, signal: int, atr: float,
                amount: int) -> Trade:
        self.contracts[contract.symbol].atr = atr
        trade = self.trader.trade(contract, self.entry_order(signal, amount),
                                  'ENTRY')
        trade.filledEvent += self.attach_sl
        return trade

    def onClose(self, contract: Contract, signal: int, atr: float,
                amount: int) -> Trade:
        # TODO can sl close position before being removed?
        # make sure sl didn't close the position before being removed
        self.remove_sl(contract)
        trade = self.trader.trade(contract, self.close_order(signal, amount),
                                  'CLOSE')
        return trade

    def attach_sl(self, trade: Trade) -> None:
        sl_points = self.contracts[self.contract.symbol].atr
        order = xxx
        self.trader.trade(contract, order, 'STOP-LOSS')

    def remove_sl(self, contract: Contract) -> None:
        open_trades = self.ib.openTrades()
        orders = defaultdict(list)
        for t in open_trades:
            orders[t.contract.localSymbol].append(t.order)
        for order in orders[contract.localSymbol]:
            if order.orderType in ('STP', 'TRAIL'):
                self.ib.cancelOrder(order)
                log.debug(f'stop loss removed for {contract.localSymbol}')


class Stop:
    def __init__(self, trade: Trade, sl_points: int):
        self.contract = trade.contract
        self.action = trade.order.action
        assert self.action in ('BUY', 'SELL')
        self.reverseAction = 'BUY' if self.action == 'SELL' else 'SELL'
        self.direction = 1 if self.reverseAction == 'BUY' else -1
        self.amount = trade.orderStatus.filled
        self.price = trade.orderStatus.avgFillPrice
        self.sl_points = sl_points

    def order(self):
        raise NotImplementedError


class FixedStop(Stop):

    def order(self):
        sl_price = round_tick(
            self.price + self.sl_points * self.direction *
            self.contracts[contract.symbol].sl_atr,
            self.contracts[contract.symbol].details.minTick)
        log.info(f'STOP LOSS PRICE: {sl_price}')
        return StopOrder(reverseAction, amount, sl_price,
                         outsideRth=True, tif='GTC')


class TrailingStop(Stop):

    def order(self):
        distance = self.round_tick(
            sl_points * self.contracts[contract.symbol].sl_atr,
            self.contracts[contract.symbol].details.minTick)
        log.info(f'TRAILING STOP LOSS DISTANCE: {distance}')
        order = Order(orderType='TRAIL', action=reverseAction,
                      totalQuantity=amount, auxPrice=distance,
                      outsideRth=True, tif='GTC')


class TrailingFixedStop(TrailingStop):

    def order(self):
        sl = super().order()
        log.debug(sl)
        sl.adjustedOrderType = 'STP'
        sl.adjustedStopPrice = (price - direction * 2 * distance)
        # self.contracts[contract.symbol].details.minTick)
        log.debug(f'adjusted stop price: {sl.adjustedStopPrice}')
        log.debug(f'DISTANCE: {distance}')
        sl.triggerPrice = sl.adjustedStopPrice - direction * distance
        log.debug(f'stop loss for {contract.localSymbol} '
                  f'fixed at {sl.triggerPrice}')


def round_tick(price: float, tick_size: float) -> float:
    floor = price // tick_size
    remainder = price % tick_size
    if remainder > (tick_size / 2):
        floor += 1
    return round(floor * tick_size, 4)
