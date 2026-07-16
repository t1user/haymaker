from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

import ib_insync as ibi

from . import misc
from .databases import StoreFactory
from .saver import AbstractBaseSaver, AsyncSaveManager, CsvSaver, MongoSaver

log = logging.getLogger(__name__)


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
        saver: AbstractBaseSaver | None = None,
        *args,
        **kwargs,
    ) -> None:
        self.save_immediately = save_immediately  # False for backtester, True otherwise
        self.blotter: list[dict] = []
        self.unsaved_trades: dict = {}
        self.com_reports: dict = {}
        self.done_trades: list[int] = []
        if saver is None:
            saver = CsvSaver(name="blotter", folder="blotter", use_timestamp=False)
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
    ) -> None:
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


def blotter_factory(
    settings: Mapping[str, Any], store_factory: StoreFactory
) -> Blotter | None:
    """Construct a built-in blotter and saver from plain configuration.

    Args:
        settings: Merged ``blotter`` configuration section.
        store_factory: Runtime persistence services used by built-in savers.

    Returns:
        Configured blotter, or ``None`` when disabled.
    """

    config = dict(settings)
    enabled = config.pop("enabled", True)
    saver_config = config.pop("saver", None)
    if config:
        names = ", ".join(sorted(config))
        raise TypeError(f"Unknown blotter configuration: {names}")
    if not enabled:
        return None
    if not isinstance(saver_config, Mapping):
        raise ValueError("Enabled blotter requires saver settings")

    saver_settings = dict(saver_config)
    saver_type = saver_settings.pop("type", None)
    saver_options = saver_settings.pop("options", {})
    if saver_settings:
        names = ", ".join(sorted(saver_settings))
        raise TypeError(f"Unknown blotter saver configuration: {names}")
    if not isinstance(saver_options, Mapping):
        raise TypeError("blotter.saver.options must be a mapping")
    options = dict(saver_options)

    if saver_type == "csv":
        saver: AbstractBaseSaver = CsvSaver(
            **options, base_directory=store_factory.settings.base_directory
        )
    elif saver_type == "mongo":
        saver = MongoSaver(
            **options,
            client=store_factory.mongo_client(),
            database=store_factory.database,
        )
    else:
        raise ValueError("blotter.saver.type must be csv or mongo")
    return Blotter(saver=saver)
