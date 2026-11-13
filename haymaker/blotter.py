from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

import ib_insync as ibi
from pymongo import MongoClient  # type: ignore

from . import misc
from .saver import AbstractBaseSaver, AsyncSaveManager, CsvSaver, MongoSaver

log = logging.getLogger(__name__)


def _execution_totals(trade: ibi.Trade) -> tuple[float, float]:
    """Calculate report quantity/price from executions, including leg-only BAGs."""
    if not isinstance(trade.contract, ibi.Bag):
        return trade.filled(), misc.trade_fill_price(trade)
    aggregate = [fill for fill in trade.fills if fill.contract.secType == "BAG"]
    if aggregate:
        clean = replace(trade, fills=aggregate)
        return clean.filled(), misc.trade_fill_price(clean)
    quantities = []
    price = 0.0
    for leg in trade.contract.comboLegs:
        fills = [fill for fill in trade.fills if fill.contract.conId == leg.conId]
        quantity = sum(fill.execution.shares for fill in fills)
        if not quantity or leg.ratio <= 0:
            return 0.0, 0.0
        quantities.append(quantity / leg.ratio)
        price += (
            misc.trade_fill_price(replace(trade, fills=fills))
            * leg.ratio
            * (1 if leg.action == "BUY" else -1)
        )
    if not quantities or len(set(quantities)) != 1:
        return 0.0, 0.0
    return quantities[0], price


def commission_reports(
    trade: ibi.Trade, *, expected_quantity: float | None = None
) -> list[ibi.CommissionReport] | None:
    """Return complete execution commissions, or None while evidence is missing.

    Explicit BAG legs carry the commissions when present. The aggregate BAG
    execution is then quantity/price evidence, not an additional charge.
    """
    quantity, _ = _execution_totals(trade)
    if expected_quantity is None:
        expected_quantity = (
            trade.order.totalQuantity
            if trade.orderStatus.status == ibi.OrderStatus.Filled
            else trade.orderStatus.filled
        )
    if quantity < expected_quantity or quantity > trade.order.totalQuantity:
        return None
    fills = trade.fills
    if isinstance(trade.contract, ibi.Bag):
        legs = [fill for fill in fills if fill.contract.secType != "BAG"]
        if legs:
            for leg in trade.contract.comboLegs:
                leg_quantity = sum(
                    fill.execution.shares
                    for fill in legs
                    if fill.contract.conId == leg.conId
                )
                if not leg_quantity or leg_quantity != quantity * leg.ratio:
                    return None
            fills = legs
    if not fills or any(
        not fill.execution.execId
        or fill.commissionReport.execId != fill.execution.execId
        or not math.isfinite(fill.commissionReport.commission)
        or not math.isfinite(fill.commissionReport.realizedPNL)
        for fill in fills
    ):
        return None
    return [fill.commissionReport for fill in fills]


def _report_identity(perm_id: Any, order_id: Any) -> tuple[Any, Any]:
    # Rebinding can change orderId; resetting numbering can reuse it.
    return (perm_id, None) if perm_id else (None, order_id)


class Blotter:
    """Collect optional trade reports after their commissions arrive.

    Controller supplies each commission callback and signal/position attribution.
    Reporting is independent of Book's authoritative order and Fill persistence;
    disabling the blotter never disables accounting or commission recovery.

    Args:
        save_immediately: Queue each completed trade report when True (the
            live default). When False, retain rows for an explicit save_many().
        saver: Reporting backend; defaults to a CSV blotter. A backend used by
            save_many() must additionally support that bulk-writing method.
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
        self.done_trades: set[tuple[Any, Any]] = set()
        self._loaded_done_trades = False
        if saver is None:
            saver = CsvSaver(name="blotter", folder="blotter", use_timestamp=False)
        self.saver = saver
        # ensure async saving
        self.save = AsyncSaveManager(saver).save
        log.debug(f"Blotter initiated: {self}")

    def log_trade(
        self, trade: ibi.Trade, comms: list[ibi.CommissionReport], **kwargs
    ) -> None:
        if not self._loaded_done_trades:
            self.done_trades.update(
                _report_identity(row.get("perm_id"), row.get("order_id"))
                for row in self.records()
            )
            self._loaded_done_trades = True
        identity = _report_identity(trade.order.permId, trade.order.orderId)
        if identity in self.done_trades:
            return
        quantity, price = _execution_totals(trade)
        row = {
            "local_time": datetime.now(),
            "sys_time": datetime.now(timezone.utc),  # system time
            "last_fill_time": max(fill.time for fill in trade.fills),
            "contract": (
                trade.contract.localSymbol
                if not isinstance(trade.contract, ibi.Bag)
                else trade.contract.symbol
            ),
            "symbol": trade.contract.symbol,
            "side": trade.order.action,  # buy or sell
            "order_type": trade.order.orderType,  # order type
            "order_price": trade.order.auxPrice or trade.order.lmtPrice,  # order price
            "amount": quantity,  # unsigned amount
            "price": price,
            "order_id": trade.order.orderId,  # non unique
            "perm_id": trade.order.permId,  # unique trade id
            "commission": sum([comm.commission for comm in comms]),
            "realizedPNL": sum([comm.realizedPNL for comm in comms]),
            "fills": [fill.execution.dict() for fill in trade.fills],  # type: ignore
        }
        row["trade"] = ibi.util.tree(trade)
        if kwargs:
            row.update(kwargs)
        self.save_report(row)
        self.done_trades.add(identity)

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

        clean = replace(
            trade,
            fills=[fill for fill in trade.fills if misc.valid_trade_fill(trade, fill)],
        )
        comms = commission_reports(clean)
        if clean.isDone() and comms is not None:
            self.log_trade(clean, comms, **kwargs)

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
            identity = _report_identity(row.get("perm_id"), row.get("order_id"))
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
