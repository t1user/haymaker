from typing import Set

from ib_insync import (IB, Trade, Fill, Contract, CommissionReport,
                       BarDataList, PortfolioItem, AccountValue, PnL,
                       PnLSingle, Ticker, NewsTick, NewsBulletin,
                       ScanData, Position, Event)
from ib_insync.ibcontroller import Watchdog
from logbook import Logger


log = Logger(__name__)


class WatchdogHandlers:

    def __init__(self, dog: Watchdog):
        dog.startingEvent += self.onStarting
        dog.startedEvent += self.onStarted
        dog.stoppingEvent += self.onStopping
        dog.stoppedEvent += self.onStopped
        dog.softTimeoutEvent += self.onSoftTimeout
        dog.hardTimeoutEvent += self.onHardTimeout
        self.dog = dog

    def onStarting(self, *args):
        log.debug(f'StartingEvent {args}')

    def onStarted(self, *args):
        log.debug(f'StartedEvent {args}')

    def onStopping(self, *args):
        log.debug(f'StoppingEvent {args}')

    def onStopped(self, *args):
        log.debug(f'StoppedEvent {args}')

    def onSoftTimeout(self, *args):
        log.debug(f'SoftTimeoutEvent {args}')

    def onHardTimeout(self, *args):
        log.debug(f'HardTimeoutEvent {args}')


class IBHandlers:

    def __init__(self, ib: IB):
        ib.connectedEvent += self.onConnected
        ib.disconnectedEvent += self.onDisconnected
        ib.updateEvent += self.onUpdate
        ib.pendingTickersEvent += self.onPendingTickers
        ib.barUpdateEvent += self.onBarUpdate
        ib.newOrderEvent += self.onNewOrder
        ib.orderModifyEvent += self.onOrderModify
        ib.cancelOrderEvent += self.onCancelOrder
        ib.openOrderEvent += self.onOpenOrder
        ib.orderStatusEvent += self.onOrderStatus
        ib.execDetailsEvent += self.onExecDetails
        ib.commissionReportEvent += self.onCommissionReport
        ib.updatePortfolioEvent += self.onUpdatePortfolio
        ib.positionEvent += self.onPosition
        ib.accountValueEvent += self.onAccountValue
        ib.accountSummaryEvent += self.onAccountSummary
        ib.pnlEvent += self.onPnl
        ib.pnlSingleEvent += self.onPnlSingle
        ib.tickNewsEvent += self.onTickNews
        ib.newsBulletinEvent += self.onNewsBulletin
        ib.scannerDataEvent += self.onScannerData
        ib.errorEvent += self.onError
        ib.timeoutEvent += self.onTimeout
        scheduledUpdate = Event().timerange(300, None, 600)
        scheduledUpdate += self.onScheduledUpdate
        self.ib = ib
        self.portfolio_items = {}

    def onConnected(self):
        log.debug('Connection established')
        self.account = self.ib.client.getAccounts()[0]
        self.ib.accountSummary()
        self.ib.reqPnL(self.account)

    def onDisconnected(self):
        log.debug('Connection lost')

    def onUpdate(self):
        pass

    def onPendingTickers(self, tickers: Set[Ticker]):
        pass

    def onBarUpdate(self, bars: BarDataList, hasNewBar: bool):
        pass

    def onNewOrder(self, trade: Trade):
        log.info(f'New order: {trade.contract.localSymbol} {trade.order}')

    def onOrderModify(self, trade: Trade):
        log.info(
            f'Order modified: {trade.contract.localSymbol} {trade.order}')

    def onCancelOrder(self, trade: Trade):
        log.info(
            f'Order canceled: {trade.contract.localSymbol} {trade.order}')

    def onOpenOrder(self, trade: Trade):
        pass

    def onOrderStatus(self, trade: Trade):
        log.info(f'Order status {trade.contract.localSymbol} '
                 f'{trade.order.action} {trade.order.totalQuantity} '
                 f'{trade.order.orderType} - '
                 f'{trade.orderStatus.status} - '
                 f'(t: {trade.order.totalQuantity} '
                 f'f: {trade.orderStatus.filled} '
                 f'r: {trade.orderStatus.remaining})')

    def onExecDetails(self, trade: Trade, fill: Fill):
        pass

    def onCommissionReport(self, trade: Trade, fill: Fill,
                           report: CommissionReport):
        # log.debug(f'Commission report: {report}')
        pass

    def onUpdatePortfolio(self, item: PortfolioItem):
        realized = round(item.realizedPNL, 2)
        unrealized = round(item.unrealizedPNL, 2)
        total = round(realized + unrealized)
        report = (item.contract.localSymbol, realized, unrealized, total)
        log.debug(f'Portfolio item: {report}')
        self.portfolio_items[item.contract.localSymbol] = (
            realized, unrealized, total)

    def onPosition(self, position: Position):
        log.debug(f'Position update: {position.contract.localSymbol}: '
                  f'{position.position}, avg cost: {position.avgCost}')

    def onAccountValue(self, value: AccountValue):
        if value.tag == 'NetLiquidation':
            log.debug(value)

    def onAccountSummary(self, value: AccountValue):
        """
        tags = ['UnrealizedPnL', 'RealizedPnL', 'FuturesPNL',
                'NetLiquidationByCurrency']
        """
        # tags = ['NetLiquidationByCurrency']
        # if value.tag in tags:
        #    log.info(f'{value.tag}: {value.value}')
        pass

    def onPnl(self, entry: PnL):
        pass

    def onPnlSingle(self, entry: PnLSingle):
        pass

    def onTickNews(self, news: NewsTick):
        pass

    def onNewsBulletin(self, bulletin: NewsBulletin):
        pass

    def onScannerData(self, data: ScanData):
        pass

    def onError(self, reqId: int, errorCode: int, errorString: str,
                contract: Contract):
        if errorCode not in (2157, 2158, 2119, 2104, 2106, 165, 2108,
                             2103, 2105, 10182, 1100):
            log.error(f'ERROR: {errorCode} {errorString} {contract}')

    def onTimeout(self, idlePeriod: float):
        pass
        #log.error(f'Timeout! Idle period: {idlePeriod}')

    def onScheduledUpdate(self, time):
        log.info(f'pnl: {self.ib.pnl()}')
        summary = [0, 0, 0]
        for contract, value in self.portfolio_items.items():
            summary[0] += value[0]
            summary[1] += value[1]
            summary[2] += value[2]
        message = (f'realized: {summary[0]}, '
                   f'unrealized: {summary[1]}, total: {summary[2]}')
        log.info(message)
        positions = [(p.contract.localSymbol, p.position)
                     for p in self.ib.positions()]
        log.info(f'POSITIONS: {positions}')
        self.manager.onScheduledUpdate()


class Handlers(WatchdogHandlers, IBHandlers):
    def __init__(self, ib, dog):
        IBHandlers.__init__(self, ib)
        WatchdogHandlers.__init__(self, dog)
