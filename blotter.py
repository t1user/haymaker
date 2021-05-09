import csv
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Any, Optional

from ib_insync.order import Trade
from ib_insync.objects import Fill, CommissionReport
from ib_insync import util
from logbook import Logger
from pymongo import MongoClient
import motor.motor_asyncio
from arctic import TICK_STORE, Arctic
import pandas as pd

from utilities import default_path

log = Logger(__name__)


class AbstractBaseBlotter(ABC):

    """
    Api for storing blotters.

    Log trade only after all commission reports arrive. Trader
    will log commission after every commission event. It's up to blotter
    to parse through those reports, determine when the trade is ready
    to be logged and filter out some known issues with ib-insync reports.

    Blotter works in one of two modes:
    - tade by trades save to store: suitable for live trading
    - save to store only full blotter: suitable for backtest (save time
      on i/o)
    """

    def __init__(self, save_to_file: bool = True) -> None:
        self.save_to_file = save_to_file
        self.blotter = []
        self.unsaved_trades = {}
        self.com_reports = {}

    def log_trade(self, trade: Trade, comms: List[CommissionReport],
                  reason: str = '') -> None:
        sys_time = datetime.now()
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

    def save_report(self, report: Dict[str, Any]) -> None:
        """
        Choose whether row of data (report) should be written to permanent
        store immediately or just kept in self.blotter for later.
        """
        if self.save_to_file:
            self.write_to_file(report)
        else:
            self.blotter.append(report)

    @abstractmethod
    def write_to_file(self, data: Dict[str, Any]) -> None:
        """
        Write single line of data to the store.
        """
        pass

    @abstractmethod
    def save(self) -> None:
        """
        Write full blotter (all rows) to store.
        """
        pass

    @abstractmethod
    def delete(self, query: Dict) -> str:
        """
        Delete items from blotter.
        """
        pass

    @abstractmethod
    def clear(self):
        """
        Clear all items in the blotter.
        """
        s = input('This will permanently delete all items in the blotter. '
                  'Continue? ').lower()
        if s != 'yes' and s != 'y':
            sys.exit()

    def __repr__(self):
        return (f'{self.__class__.__name__}' + '(' + ', '.join(
            [f'{k}={v}' for k, v in self.__dict__.items()]) + ')')


class CsvBlotter(AbstractBaseBlotter):

    fieldnames = None

    def __init__(self, save_to_file: bool = True, filename: str = None,
                 path: Optional[str] = None, note: str = ''):
        if path is None:
            path = default_path('blotter')
        if filename is None:
            filename = __file__.split('/')[-1][:-3]
        self.file = (f'{path}/{filename}_'
                     f'{datetime.today().strftime("%Y-%m-%d_%H-%M")}{note}.csv')
        super().__init__(save_to_file)

    def create_header(self) -> None:
        with open(self.file, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def write_to_file(self, data: Dict[str, Any]) -> None:
        if not self.fieldnames:
            self.fieldnames = data.keys()
            self.create_header()
        with open(self.file, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(data)

    def save(self) -> None:
        self.fieldnames = self.blotter[0].keys()
        self.create_header()
        with open(self.file, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            for item in self.blotter:
                writer.writerow(item)

    def delete(self, query: Dict) -> str:
        raise NotImplementedError

    def clear(self) -> str:
        raise NotImplementedError


class MongoBlotter(AbstractBaseBlotter):

    def __init__(self, save_to_file: bool = True, host: str = 'localhost',
                 port: int = 27017, db: str = 'blotter',
                 collection: 'str' = 'test_blotter') -> None:
        self.client = MongoClient(host, port)
        self.db = self.client[db]
        self.collection = self.db[collection]
        super().__init__(save_to_file)

    def write_to_file(self, data: Dict[str, Any]) -> None:
        self.collection.insert_one(data)

    def save(self) -> None:
        self.collection.insert_many(self.blotter)

    def read(self) -> pd.DataFrame:
        return util.df([i for i in self.collection.find()])

    def delete(self, querry: Dict) -> str:
        results = self.collection.find(querry)
        for doc in results:
            print(doc)
        s = input('Above documents will be deleted.'
                  'Continue? ').lower()
        if s != 'yes' and s != 'y':
            sys.exit()
        x = self.collection.delete_many(querry)
        return f'Documents deleted: {x.raw_result}'

    def clear(self) -> str:
        print(f'Deleting all items from {self.collection}')
        super().clear()
        x = self.collection.delete_many({})
        return f'Deleted {x.deleted_count} documents.'


class AsyncMongoBlottter(AbstractBaseBlotter):
    """
    NOT TESTED. Clear and delete methods missing. TODO.
    """

    def __init__(self, save_to_file: bool = True, host: str = 'localhost',
                 port: int = 27017, db: str = 'blotter',
                 collection: 'str' = 'test_blotter') -> None:
        self.client = motor.motor_asyncio.AsyncIOMotorClient(host, port)
        self.db = self.client[db]
        self.collection = self.db[collection]
        super().__init__(save_to_file)

    async def _write_to_file(self, data: Dict[str, Any]) -> None:
        await self.collection.insert_one(data)

    def write_to_file(self, data: Dict[str, Any]) -> None:
        util.run(self._write_to_file(data))

    async def _save(self) -> None:
        await self.collection.insert_many(self.blotter)

    def save(self) -> None:
        util.run(self._save())


class TickBlotter(AbstractBaseBlotter):
    def __init__(self, save_to_file: bool = True, host: str = 'localhost',
                 library: str = 'tick_blotter',
                 collection: str = 'test_blotter') -> None:
        self.db = Arctic(host)
        self.db.initialize_library(library, lib_type=TICK_STORE)
        self.store = self.db[library]
        self.collection = collection

    def write_to_file(self, data: Dict[str, Any]) -> None:
        data['index'] = pd.to_datetime(data['time'], utc=True)
        self.store.write(self.collection, [data])

    def save(self) -> None:
        data = []
        for d in self.blotter:
            d.update({'index': pd.to_datetime(d['time'], utc=True)})
            data.append(d)
        self.store.write(self.collection, data)

    def delete(self, querry: Dict) -> str:
        raise NotImplementedError

    def clear(self) -> str:
        raise NotImplementedError
