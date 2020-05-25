import csv
from datetime import datetime
from typing import List

from ib_insync.order import Trade
from ib_insync.objects import Fill, CommissionReport
from logbook import Logger

log = Logger(__name__)


class Blotter:
    def __init__(self, save_to_file: bool = True, filename: str = None,
                 path: str = 'blotter', note: str = ''):
        if filename is None:
            filename = __file__.split('/')[-1][:-3]
        self.file = (f'{path}/{filename}_'
                     f'{datetime.today().strftime("%Y-%m-%d_%H-%M")}{note}.csv')
        self.save_to_file = save_to_file
        self.fieldnames = ['sys_time', 'time', 'contract', 'action', 'amount',
                           'price', 'exec_ids', 'order_id', 'perm_id',
                           'reason', 'commission', 'realizedPNL',
                           # 'comm_reports'
                           ]
        self.blotter = []
        self.unsaved_trades = {}
        self.com_reports = {}
        if self.save_to_file:
            self.create_header()

    def create_header(self):
        with open(self.file, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log_trade(self, trade: Trade, comms: List[CommissionReport],
                  reason: str = ''):
        sys_time = str(datetime.now())
        time = trade.log[-1].time
        contract = trade.contract.localSymbol
        action = trade.order.action
        amount = trade.orderStatus.filled
        price = trade.orderStatus.avgFillPrice
        # ib_insync issue: sometimes fills relate to wrong transaction
        # fill.contract == trade.contract to prevent that
        exec_ids = [fill.execution.execId for fill in trade.fills
                    if fill.contract == trade.contract]
        order_id = trade.order.orderId
        perm_id = trade.order.permId
        reason = reason
        row = {
            'sys_time': sys_time,  # actual system time
            'time': time,  # what ib considers to be current time
            'contract': contract,  # 4 letter symbol string
            'action': action,  # buy or sell
            'amount': amount,  # unsigned amount
            'price': price,
            'exec_ids': exec_ids,  # list of execution ids
            'order_id': order_id,  # non unique
            'perm_id': perm_id,  # unique trade id
            'reason': reason,  # note passed by the trading system
            'commission': sum([comm.commission for comm in comms]),
            'realizedPNL': sum([comm.realizedPNL for comm in comms]),
            # 'comm_reports': comms
        }
        self.save_report(row)
        log.debug(f'trade report saved: {row}')

    def log_commission(self, trade: Trade, fill: Fill,
                       comm_report: CommissionReport, reason: str):
        """
        Get trades that have all CommissionReport filled and log them.
        """
        # bug in ib_insync sometimes causes trade to have fills for
        # unrelated transactions, permId uniquely identifies order
        comms = [fill.commissionReport for fill in trade.fills
                 if fill.commissionReport.execId != ''
                 and fill.execution.permId == trade.order.permId]
        fills = [fill for fill in trade.fills
                 if fill.execution.permId == trade.order.permId]
        if trade.isDone() and (len(comms) == len(fills)):
            self.log_trade(trade, comms, reason)

    def save_report(self, report):
        if self.save_to_file:
            self.write_to_file(report)
        else:
            self.blotter.append(report)

    def write_to_file(self, data: dict):
        with open(self.file, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(data)

    def save(self):
        self.create_header()
        with open(self.file, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            for item in self.blotter:
                writer.writerow(item)
