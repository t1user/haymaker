"""
This module connects all `ib_insync` and `watchdog` events to logger:
`broker`.  Basically, it logs all broker messages.
"""

import logging
from typing import Dict, Set, Tuple

import ib_insync as ibi


class IBHandlers:
    def __init__(self, ib: ibi.IB):

        self.log = logging.getLogger("broker")

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
        self.log.info("Connection established")
        self.account = self.ib.client.getAccounts()[0]
        self.ib.accountSummary()
        self.ib.reqPnL(self.account)

    def onDisconnected(self):
        self.log.warning("Connection lost")

    def onUpdate(self):
        pass

    def onPendingTickers(self, tickers: Set[ibi.Ticker]):
        pass

    def onBarUpdate(self, bars: ibi.BarDataList, hasNewBar: bool):
        pass

    def onNewOrder(self, trade: ibi.Trade):
        self.log.info(f"New order: {trade.contract.localSymbol} {trade.order}")

    def onModifyOrder(self, trade: ibi.Trade):
        self.log.info(f"Order modified: {trade.contract.localSymbol} {trade.order}")

    def onCancelOrder(self, trade: ibi.Trade):
        self.log.info(f"Order canceled: {trade.contract.localSymbol} {trade.order}")

    def onOpenOrder(self, trade: ibi.Trade):
        self.log.debug(f"Open order: {trade}")

    def onOrderStatus(self, trade: ibi.Trade):
        self.log.info(
            f"Order status {trade.contract.localSymbol} "
            f"{trade.order.action} {trade.order.totalQuantity} "
            f"{trade.order.orderType} - "
            f"{trade.orderStatus.status} - "
            f"(t: {trade.order.totalQuantity} "
            f"f: {trade.orderStatus.filled} "
            f"r: {trade.orderStatus.remaining})"
        )

    def onExecDetails(self, trade: ibi.Trade, fill: ibi.Fill):
        self.log.info(f"execution details: {fill}")

    def onCommissionReport(
        self, trade: ibi.Trade, fill: ibi.Fill, report: ibi.CommissionReport
    ):
        self.log.info(f"Commission report: {report}")

    def onUpdatePortfolio(self, item: ibi.PortfolioItem):
        realized = round(item.realizedPNL, 2)
        unrealized = round(item.unrealizedPNL, 2)
        total = round(realized + unrealized)
        report = (item.contract.localSymbol, realized, unrealized, total)
        self.log.info(f"Portfolio item: {report}")
        self.portfolio_items[item.contract.localSymbol] = (realized, unrealized, total)

    def onPosition(self, position: ibi.Position):
        self.log.info(
            f"Position update: {position.contract.localSymbol}: "
            f"{position.position}, avg cost: {position.avgCost}"
        )

    def onAccountValue(self, value: ibi.AccountValue):
        if value.tag == "NetLiquidation":
            self.log.info(value)

    def onAccountSummary(self, value: ibi.AccountValue):
        """
        tags = ['UnrealizedPnL', 'RealizedPnL', 'FuturesPNL',
                'NetLiquidationByCurrency']
        """
        tags = ["NetLiquidationByCurrency"]
        if value.tag in tags:
            self.log.info(f"{value.tag}: {value.value}")

    def onPnl(self, entry: ibi.PnL):
        self.log.debug(f"pnl: {entry}")

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
        # if errorCode not in (
        #     2157,
        #     2158,
        #     2119,
        #     2104,
        #     2106,
        #     165,
        #     2108,
        #     2103,
        #     2105,
        #     10182,
        #     1100,
        # ):
        self.log.error(f"ERROR: {errorCode} {errorString} {contract}")

    def onTimeout(self, idlePeriod: float):
        self.log.debug(f"timeout: {idlePeriod}")

    def onScheduledUpdate(self, time):
        self.log.info(f"pnl: {self.ib.pnl()}")
        summary = [0, 0, 0]
        for _contract, value in self.portfolio_items.items():
            summary[0] += value[0]
            summary[1] += value[1]
            summary[2] += value[2]
        message = (
            f"realized: {summary[0]}, " f"unrealized: {summary[1]}, total: {summary[2]}"
        )
        self.log.info(message)
        positions = [(p.contract.localSymbol, p.position) for p in self.ib.positions()]
        self.log.info(f"POSITIONS: {positions}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
