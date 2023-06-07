from typing import Dict, Set, Tuple

import ib_insync as ibi
from ib_insync.ibcontroller import Watchdog
from logbook import Logger  # type: ignore

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
        log.debug(f"StartingEvent {args}")

    def onStarted(self, *args):
        log.debug(f"StartedEvent {args}")

    def onStopping(self, *args):
        log.debug(f"StoppingEvent {args}")

    def onStopped(self, *args):
        log.debug(f"StoppedEvent {args}")

    def onSoftTimeout(self, *args):
        log.debug(f"SoftTimeoutEvent {args}")

    def onHardTimeout(self, *args):
        log.debug(f"HardTimeoutEvent {args}")


class IBHandlers:
    def __init__(self, ib: ibi.IB):
        ib.connectedEvent += self.onConnected
        ib.disconnectedEvent += self.onDisconnected
        ib.updateEvent += self.onUpdate
        ib.pendingTickersEvent += self.onPendingTickers
        ib.barUpdateEvent += self.onBarUpdate
        ib.newOrderEvent += self.onNewOrder
        ib.orderModifyEvent += self.onModifyOrder
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
        scheduledUpdate = ibi.Event().timerange(300, None, 600)
        scheduledUpdate += self.onScheduledUpdate
        self.ib = ib
        self.portfolio_items: Dict[str, Tuple[float, float, float]] = {}

    def onConnected(self):
        log.debug("Connection established")
        self.account = self.ib.client.getAccounts()[0]
        self.ib.accountSummary()
        self.ib.reqPnL(self.account)

    def onDisconnected(self):
        log.warning("Connection lost")

    def onUpdate(self):
        pass

    def onPendingTickers(self, tickers: Set[ibi.Ticker]):
        pass

    def onBarUpdate(self, bars: ibi.BarDataList, hasNewBar: bool):
        pass

    def onNewOrder(self, trade: ibi.Trade):
        log.info(f"New order: {trade.contract.localSymbol} {trade.order}")

    def onModifyOrder(self, trade: ibi.Trade):
        log.info(f"Order modified: {trade.contract.localSymbol} {trade.order}")

    def onCancelOrder(self, trade: ibi.Trade):
        log.info(f"Order canceled: {trade.contract.localSymbol} {trade.order}")

    def onOpenOrder(self, trade: ibi.Trade):
        pass

    def onOrderStatus(self, trade: ibi.Trade):
        log.info(
            f"Order status {trade.contract.localSymbol} "
            f"{trade.order.action} {trade.order.totalQuantity} "
            f"{trade.order.orderType} - "
            f"{trade.orderStatus.status} - "
            f"(t: {trade.order.totalQuantity} "
            f"f: {trade.orderStatus.filled} "
            f"r: {trade.orderStatus.remaining})"
        )

    def onExecDetails(self, trade: ibi.Trade, fill: ibi.Fill):
        pass

    def onCommissionReport(
        self, trade: ibi.Trade, fill: ibi.Fill, report: ibi.CommissionReport
    ):
        # log.debug(f'Commission report: {report}')
        pass

    def onUpdatePortfolio(self, item: ibi.PortfolioItem):
        realized = round(item.realizedPNL, 2)
        unrealized = round(item.unrealizedPNL, 2)
        total = round(realized + unrealized)
        report = (item.contract.localSymbol, realized, unrealized, total)
        log.debug(f"Portfolio item: {report}")
        self.portfolio_items[item.contract.localSymbol] = (realized, unrealized, total)

    def onPosition(self, position: ibi.Position):
        log.debug(
            f"Position update: {position.contract.localSymbol}: "
            f"{position.position}, avg cost: {position.avgCost}"
        )

    def onAccountValue(self, value: ibi.AccountValue):
        if value.tag == "NetLiquidation":
            log.debug(value)

    def onAccountSummary(self, value: ibi.AccountValue):
        """
        tags = ['UnrealizedPnL', 'RealizedPnL', 'FuturesPNL',
                'NetLiquidationByCurrency']
        """
        # tags = ['NetLiquidationByCurrency']
        # if value.tag in tags:
        #    log.info(f'{value.tag}: {value.value}')
        pass

    def onPnl(self, entry: ibi.PnL):
        pass

    def onPnlSingle(self, entry: ibi.PnLSingle):
        pass

    def onTickNews(self, news: ibi.NewsTick):
        pass

    def onNewsBulletin(self, bulletin: ibi.NewsBulletin):
        pass

    def onScannerData(self, data: ibi.ScanData):
        pass

    def onError(
        self, reqId: int, errorCode: int, errorString: str, contract: ibi.Contract
    ):
        if errorCode not in (
            2157,
            2158,
            2119,
            2104,
            2106,
            165,
            2108,
            2103,
            2105,
            10182,
            1100,
        ):
            log.error(f"ERROR: {errorCode} {errorString} {contract}")

    def onTimeout(self, idlePeriod: float):
        pass
        # log.error(f'Timeout! Idle period: {idlePeriod}')

    def onScheduledUpdate(self, time):
        log.info(f"pnl: {self.ib.pnl()}")
        summary = [0, 0, 0]
        for _contract, value in self.portfolio_items.items():
            summary[0] += value[0]
            summary[1] += value[1]
            summary[2] += value[2]
        message = (
            f"realized: {summary[0]}, " f"unrealized: {summary[1]}, total: {summary[2]}"
        )
        log.info(message)
        positions = [(p.contract.localSymbol, p.position) for p in self.ib.positions()]
        log.info(f"POSITIONS: {positions}")
        self.manager.onScheduledUpdate()


class Handlers(WatchdogHandlers, IBHandlers):
    def __init__(self, ib, dog):
        IBHandlers.__init__(self, ib)
        WatchdogHandlers.__init__(self, dog)
