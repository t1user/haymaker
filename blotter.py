import csv
from datetime import datetime

from ib_insync import util
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
                           'comm_reports']
        self.blotter = []
        self.unsaved_trades = {}
        self.com_reports = {}
        if self.save_to_file:
            self.create_header()

    def create_header(self):
        with open(self.file, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log_trade(self, trade: Trade, reason: str = ''):
        log.debug(f'logging trade (fills only): {trade.fills}')
        sys_time = str(datetime.now())
        time = trade.log[-1].time
        contract = trade.contract.localSymbol
        action = trade.order.action
        amount = trade.orderStatus.filled
        price = trade.orderStatus.avgFillPrice
        exec_ids = [fill.execution.execId for fill in trade.fills]
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
            'commission': 0,  # to be updated subsequently by event
            'realizedPNL': 0,  # to be updated subsequently by event
            'comm_reports': []  # comm reports added as they're being emited
        }
        self.unsaved_trades[perm_id] = row

    def update_commission(self, trade: Trade, fill: Fill,
                          comm_report: CommissionReport):
        log.debug(f'updating commission for trade: {trade}')
        while True:
            report = self.unsaved_trades.get(trade.order.permId)
            if report:
                report['comm_reports'].append(comm_report)
                break
            else:
                util.sleep()

        if len(report['comm_reports']) == len(report['exec_ids']):
            for comm_report in report['comm_reports']:
                report['commission'] += comm_report.commission
                report['realizedPNL'] += comm_report.realizedPNL

            if self.save_to_file:
                self.write_to_file(report)
            else:
                self.blotter.append(report)

            log.debug(f'trade {report} saved')
            del self.unsaved_trades[trade.order.permId]

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
