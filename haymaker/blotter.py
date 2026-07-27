from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import ib_insync as ibi
from pymongo import MongoClient  # type: ignore

from . import misc
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
        self._current_records: list[dict[str, Any]] = []
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
        """Record one completed trade and apply the configured save policy.

        The in-memory copy makes a newly completed trade queryable while an
        asynchronous immediate save is still queued. Backtests retain the same
        rows until :meth:`save_many` is called.
        """
        self._current_records.append(report)
        if self.save_immediately:
            self.save(report)
        else:
            self.blotter.append(report)

    def records(self) -> tuple[Mapping[str, Any], ...]:
        """Return persisted and current-process blotter rows without duplicates.

        Returns:
            Completed trade reports. Rows with the same broker permanent or
            local order identity are returned once, preferring the in-memory
            copy produced by the current process.
        """

        persisted: Sequence[Any] = ()
        try:
            if isinstance(self.saver, CsvSaver):
                persisted = self.saver.read(self.saver.name)
            else:
                result = self.saver.read()
                if isinstance(result, Sequence):
                    persisted = result
        except FileNotFoundError:
            pass

        rows: list[Mapping[str, Any]] = []
        positions: dict[tuple[Any, Any], int] = {}
        for candidate in (*persisted, *self._current_records):
            if not isinstance(candidate, Mapping):
                continue
            row = dict(candidate)
            identity = (row.get("perm_id"), row.get("order_id"))
            if identity == (None, None):
                rows.append(row)
                continue
            if identity in positions:
                rows[positions[identity]] = row
            else:
                positions[identity] = len(rows)
                rows.append(row)
        return tuple(rows)

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
    settings: Mapping[str, Any],
    *,
    base_directory: str,
    mongo_client: Callable[[], MongoClient],
    database: str | None,
) -> Blotter | None:
    """Construct a built-in blotter and saver from plain configuration.

    Args:
        settings: Merged ``blotter`` configuration section.
        base_directory: Application data directory used by the CSV saver.
        mongo_client: Lazy accessor for the process-owned Mongo client.
        database: Application database used by the Mongo saver.

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
        saver: AbstractBaseSaver = CsvSaver(**options, base_directory=base_directory)
    elif saver_type == "mongo":
        if not database:
            raise ValueError("storage.mongodb.database is required for Mongo savers")
        saver = MongoSaver(
            **options,
            client=mongo_client(),
            database=database,
        )
    else:
        raise ValueError("blotter.saver.type must be csv or mongo")
    return Blotter(saver=saver)
