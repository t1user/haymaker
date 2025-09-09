from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Type, cast

import ib_insync as ibi

from . import misc
from .config import CONFIG
from .saver import AbstractBaseSaver, AsyncSaveManager, CsvSaver, MongoSaver  # noqa

log = logging.getLogger(__name__)


blotter_dict = cast(dict, CONFIG.get("blotter"))
blotter_class = blotter_dict["class"]
blotter_kwargs = blotter_dict["kwds"]
BLOTTER_SAVER = eval(f"{blotter_class}(**{blotter_kwargs})")


class Blotter:
    """
    Log trade only after all commission reports arrive. Trader
    will log commission after every commission event. It's up to blotter
    to parse through those reports, determine when the trade is ready
    to be logged and filter out some known issues with ib-insync reports.

    Blotter works in one of two modes:
    - trade by trade save to store: suitable for live trading
    - save to store only full blotter: suitable for backtest (save time
      on i/o)
    """

    def __init__(
        self,
        save_immediately: bool = True,
        saver: AbstractBaseSaver = BLOTTER_SAVER,
        *args,
        **kwargs,
    ) -> None:
        self.save_immediately = save_immediately  # False for backtester, True otherwise
        self.blotter: list[dict] = []
        self.unsaved_trades: dict = {}
        self.com_reports: dict = {}
        self.done_trades: list[int] = []
        self.saver = saver
        # ensure async saving
        self.save = AsyncSaveManager(saver).save
        log.debug(f"Blotter initiated: {self}")

    def log_trade(
        self, trade: ibi.Trade, comms: list[ibi.CommissionReport], **kwargs
    ) -> None:
        row = {
            "local_time": datetime.now(),
            "sys_time": datetime.now(timezone.utc),  # system time
            "last_fill_time": trade.log[-1].time,
            "contract": (
                trade.contract.localSymbol
                if not isinstance(trade.contract, ibi.Bag)
                else trade.contract.symbol
            ),
            "symbol": trade.contract.symbol,
            "side": trade.order.action,  # buy or sell
            "order_type": trade.order.orderType,  # order type
            "order_price": trade.order.auxPrice or trade.order.lmtPrice,  # order price
            "amount": trade.filled(),  # unsigned amount
            "price": trade.orderStatus.avgFillPrice or misc.trade_fill_price(trade),
            "order_id": trade.order.orderId,  # non unique
            "perm_id": trade.order.permId,  # unique trade id
            "commission": sum([comm.commission for comm in comms]),
            "realizedPNL": sum([comm.realizedPNL for comm in comms]),
            "fills": [fill.execution.dict() for fill in trade.fills],  # type: ignore
        }
        row["trade"] = ibi.util.tree(trade)
        if kwargs:
            row.update(kwargs)
        if trade.order.orderId not in self.done_trades:
            self.save_report(row)
            self.done_trades.append(trade.order.orderId)
        else:
            log.debug(f"Skipping duplicate blotter entry for {trade.order.orderId}")

    def log_commission(
        self,
        trade: ibi.Trade,
        fill: ibi.Fill,
        comm_report: ibi.CommissionReport,
        **kwargs,
    ):
        """
        Get trades that have all CommissionReport filled and log them.
        """

        fills = [
            fill
            for fill in trade.fills
            # empty objects sometimes added here by ib
            if fill.commissionReport.execId != ""
            # rarely, there's a bug in ib data, where unrelated
            # fill appears in the list (probably only in paper acc)
            # it's a precaution
            and fill.execution.permId == trade.order.permId
        ]

        comms = [fill.commissionReport for fill in fills]

        if trade.isDone() and (len(comms) == len(fills)):
            self.log_trade(trade, comms, **kwargs)

    def save_report(self, report: dict[str, Any]) -> None:
        """
        Choose whether row of data (report) should be written to permanent
        store immediately or just kept in self.blotter for later.
        """
        if self.save_immediately:
            self.save(report)
        else:
            self.blotter.append(report)

    def save_many(self) -> None:
        """
        Write full blotter (all rows) to store.
        """
        try:
            self.saver.save_many(self.blotter)  # type: ignore
        except AttributeError:
            log.error(
                f"saver: {self.saver} doesn't support `save_many`, "
                f"use  different saver."
            )

    def __repr__(self):
        return f"Blotter(save_immediately={self.save_immediately}, saver={self.saver})"


def blotter_factory(param: Type[Blotter] | bool | None) -> Blotter | None:
    """
    Instantiate Blotter based on passed param.

    Args:
        param: value read from config `use_blotter` key, which accepts
            either a bool (wheather standard blotter should be used or not) or
            a custom Blotter class
    Returns:
        An instance of :class:`Blotter` or `None`.
    """

    match param:
        case False | None:
            return None
        case True:
            return Blotter()
    try:
        blotter_instance = param()
    except Exception as e:
        log.exception(e)
        return None
    if isinstance(blotter_instance, Blotter):
        return blotter_instance
    else:
        raise TypeError(
            f"Custom Blotter object recevied from config is not a Blotter: {param}"
        )
