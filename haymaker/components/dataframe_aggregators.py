"""DataFrame-oriented aggregation components for historical bar pipelines.

This public module complements :mod:`haymaker.components.aggregators`. Its
components maintain and transform complete pandas DataFrames, whereas the bar
aggregators operate on ``ib_insync`` bar objects.
"""

from __future__ import annotations

import asyncio
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import date, datetime, timedelta
from functools import cached_property
from typing import Any, Awaitable, ClassVar, Generator, Literal, cast

import eventkit as ev  # type: ignore
import ib_insync as ibi
import numpy as np
import pandas as pd

from haymaker import misc
from haymaker.async_wrappers import QueueRunner, QueueShutdownPolicy
from haymaker.base import Atom
from haymaker.contract_selector import FutureSelector, custom_bday, utc_now_naive
from haymaker.datastore import AsyncDataStore
from haymaker.details_processor import typical_session_length
from haymaker.durationStr_converters import (
    barSizeSetting_to_timedelta,
    datapoints_to_timedelta,
    durationStr_to_datapoints,
    ensure_duration_str,
    offset_durationStr,
)
from haymaker.enums import ActiveNext
from haymaker.research.numba_tools import volume_grouper
from haymaker.components.streamers import HistoricalDataStreamer, Streamer

from ..stitcher import FuturesStitcher

log = logging.getLogger(__name__)


class MissingStreamerParam(Exception):
    """Required historical-request configuration is absent from the streamer."""

    pass


class WrongStreamer(Exception):
    """An aggregator was connected to an incompatible built-in streamer."""

    pass


@dataclass(eq=False)
class FuturesPandasAggregator(Atom):
    """Maintain and emit a complete DataFrame from historical bar snapshots.

    Use this component between
    :class:`~haymaker.components.HistoricalDataStreamer` and a DataFrame
    consumer such as :class:`~haymaker.components.PandasSignalModel`. It is
    the DataFrame-oriented counterpart to
    :class:`~haymaker.components.BarAggregator`: instead of regrouping
    individual broker bars, it combines each streamed snapshot with
    previously persisted history and emits the complete, monotonically
    indexed DataFrame.

    For futures, the component restricts each contract to its active period,
    obtains missing previous-contract data from the datastore or broker, and
    joins the contracts into a continuous series. The current implementation
    therefore requires a futures Contract and ``FutureSelector``.

    The default datastore and
    ``HistoricalDataStreamer(datastore=True)`` resolve the same runtime-cached
    store. This component persists the maintained DataFrame; the streamer
    consults the persisted endpoint before deciding how much history to request
    from IB.

    Args:
        datastore: ``True`` uses the runtime-default market-data store after
            startup has supplied the connected streamer's request identity.
            Defaults to ``True``; ``False`` is not supported because stored
            history is part of this component's aggregation contract.
        save_frequency: Seconds between periodic saves. Defaults to ``900``;
            zero disables periodic persistence. A tick is skipped while an
            earlier save remains in progress, and failed best-effort saves are
            logged before the next interval retries the complete frame.

    Input:
        Complete ``ib_insync.BarDataList`` snapshots emitted by a
        :class:`~haymaker.components.HistoricalDataStreamer`.

    Emits:
        The complete current ``pandas.DataFrame`` after each processed
        snapshot. Branches receive the maintained DataFrame object and must
        copy it before mutation.

    Attributes:
        store: Resolved awaited datastore. With default configuration it becomes
            available during ``onStart()`` after the streamer request identity
            is known.

    Raises:
        TypeError: If ``datastore`` is ``False`` or ``None``, or if the
            connected streamer does not resolve to a futures Contract.
        WrongStreamer: If connected to an incompatible built-in Streamer.
        MissingStreamerParam: If required historical request parameters are
            unavailable at startup.
    """

    datastore: Literal[True] | AsyncDataStore = True
    save_frequency: int = 900  # in seconds

    # ================================================================================

    _compatible_with: ClassVar[tuple[type[Streamer], ...]] = (HistoricalDataStreamer,)

    _streamer_params: dict[str, Any] = field(
        init=False, repr=False, default_factory=dict
    )
    _df: pd.DataFrame = field(init=False, repr=False, default_factory=pd.DataFrame)
    _queue: QueueRunner = field(init=False, repr=False)
    _store: AsyncDataStore | None = field(init=False, repr=False, default=None)
    _save_timer: ev.Timer | None = field(init=False, repr=False, default=None)
    _timer_task: asyncio.Task | None = field(init=False, repr=False, default=None)
    _save_in_progress: bool = field(init=False, repr=False, default=False)
    _last_data_point: date | datetime | None = field(
        init=False, repr=False, default=None
    )

    def __post_init__(self) -> None:
        if self.datastore is False or self.datastore is None:
            raise TypeError(
                "FuturesPandasAggregator datastore must be True or an AsyncDataStore"
            )
        if self.datastore is not True:
            self._store = self.datastore
        if not isinstance(self.save_frequency, int) or isinstance(
            self.save_frequency, bool
        ):
            raise TypeError(
                f"{self!s} save_frequency must be an int, "
                f"not {type(self.save_frequency)}"
            )
        if self.save_frequency < 0:
            raise ValueError(f"{self!s} save_frequency must not be negative")
        self._queue = QueueRunner(
            self.process_data,
            f"{self!s}",
            shutdown_policy=QueueShutdownPolicy.DISCARD,
        )
        super().__init__()

    def validate_source(self, source: Atom) -> None:
        """Reject structurally incompatible upstream streamer classes."""

        if isinstance(source, Streamer) and not isinstance(
            source, self._compatible_with
        ):
            raise WrongStreamer(
                f"Streamer {type(source).__name__} is not compatible with {self!s}"
            )

    async def set_timer(self) -> None:
        # if many objects created, they shouldn't all save at the same time
        await asyncio.sleep(random.randint(0, 30))
        if self._save_timer is not None:
            return
        log.debug(f"{self!s} setting save timer at {self.save_frequency}secs.")
        timer = ev.Timer(self.save_frequency)
        timer.connect(self.save_data)
        self._save_timer = timer

    def onStart(self, data: Any, source: Atom | None = None) -> Awaitable[None] | None:
        """Synchronize with the upstream streamer and forward startup.

        Args:
            data: Arbitrary mutable startup payload forwarded unchanged.
            source: Immediate upstream Atom. A ``HistoricalDataStreamer`` is
                required for request parameters and contract synchronization.
        """
        self.sync_with_streamer(cast(Atom, source))
        self._resolve_datastore()
        timer_task_done = self._timer_task is None or self._timer_task.done()
        if self._save_timer is None and timer_task_done and self.save_frequency:
            self._timer_task = asyncio.create_task(
                self.set_timer(), name=f"{self!s} timer setter"
            )
        return super().onStart(data, source)

    def sync_with_streamer(self, streamer: Atom) -> None:
        if not is_dataclass(streamer):
            raise TypeError(f"Streamer: {streamer} must be a dataclass")
        # sync contract with streamer
        # these 2 properties together ensure that self.contract
        # will be the same as on streamer
        self.which_contract = streamer.which_contract
        self._contract_blueprint = streamer._contract_blueprint
        if not isinstance(self.contract, ibi.Future):
            raise TypeError(
                f"FuturesPandasAggregator requires a futures Contract, "
                f"not {self.contract!r}"
            )
        self._streamer_params = {
            f.name: getattr(streamer, f.name) for f in fields(streamer)
        }
        log.debug(f"{self!s} streamer params: {self._streamer_params}")

    def _resolve_datastore(self) -> None:
        """Resolve the runtime-default store after streamer synchronization."""

        if self._store is not None:
            return
        try:
            bar_size_setting = self._streamer_params["barSizeSetting"]
            what_to_show = self._streamer_params["whatToShow"]
            use_rth = self._streamer_params["useRTH"]
        except KeyError as exc:
            raise MissingStreamerParam(exc.args[0]) from exc
        self._store = self.runtime.market_data_store_factory(
            bar_size_setting=bar_size_setting,
            what_to_show=what_to_show,
            use_rth=use_rth,
        )

    @property
    def store(self) -> AsyncDataStore:
        """Return the resolved custom or runtime-default datastore.

        Raises:
            RuntimeError: If the runtime default is requested before startup
                has supplied the connected streamer's request identity.
        """

        if self._store is None:
            raise RuntimeError("FuturesPandasAggregator datastore was not initialized")
        return self._store

    async def onData(self, data: ibi.BarDataList, *args: Any) -> None:
        # processing may be slow so queue data before processing
        await self._queue.put(data)

    async def process_data(self, data: ibi.BarDataList) -> None:
        bars = self._unseen_bars(data)
        if not bars:
            return

        incremental_update = self._last_data_point is not None and not self._df.empty
        raw_df = pd.DataFrame(bars).set_index("date")
        current_df = self.process_current_data(raw_df)
        # implicit assumption: if we already have data in `self._df`,
        # together with the newly received data, it should give enough
        # datapoints
        if incremental_update:
            df = self._append_new_data(current_df)
        elif (not self._df.empty) or (len(current_df) >= self.datapoints):
            df = self.append_data(current_df)
        else:
            back_data = await self.acquire_back_data(raw_df)
            if len(back_data) > 0:
                assert back_data.index.is_monotonic_increasing
            df = self.append_data(back_data)
        assert (
            df is not None
        ) and not df.empty, f"{self!s} failed to obtained back data."

        if len(df) < self.datapoints:
            log.warning(
                f"{self!s} acquired too little back data, "
                f"acquired: {len(df)} required: {self.datapoints}"
            )
        self._last_data_point = bars[-1].date
        self.dataEvent.emit(self._df)

    def _unseen_bars(self, data: ibi.BarDataList) -> list[ibi.BarData]:
        """Return the chronologically ordered tail after the watermark.

        Args:
            data: Complete cumulative snapshot from the historical streamer.

        Returns:
            All bars during bootstrap, otherwise only bars newer than the last
            successfully processed bar.
        """

        if self._last_data_point is None or self._df.empty:
            return list(data)

        start = len(data)
        while start and data[start - 1].date > self._last_data_point:
            start -= 1
        return list(data[start:])

    def append_data(self, *dfs: pd.DataFrame) -> pd.DataFrame:
        self._df = misc.concat_dfs(self._df, *dfs)
        return self._df

    def _append_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append an unseen monotonic tail without de-duplicating full history.

        Args:
            df: Newly processed rows for the current contract.

        Returns:
            The complete maintained DataFrame.
        """

        if df.empty:
            return self._df
        if not df.index.is_monotonic_increasing or df.index[0] <= self._df.index[-1]:
            return self.append_data(df)
        self._df = pd.concat((self._df, df))
        return self._df

    async def save_data(self, *args) -> None:
        if self._df.empty:
            return
        if self._save_in_progress:
            log.debug(f"{self!s} skipping save while previous save is in progress.")
            return

        self._save_in_progress = True
        try:
            contract = cast(ibi.Future, self.contract)
            await self.store.append(contract, self._df)
        except Exception:
            log.exception(f"{self!s} failed to save aggregated data.")
        finally:
            self._save_in_progress = False

    def process_current_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Return only the part of the `data` that corresponds to the
        period when current contract is active.
        """
        contract = cast(ibi.Future, self.contract)
        date_range = self._compute_date_range(contract)
        if date_range is None:
            raise RuntimeError(f"No active date range for {contract}")
        start, _ = date_range
        return data.loc[self._tz(start) :]  # type: ignore

    async def acquire_back_data(self, current_df: pd.DataFrame) -> pd.DataFrame:
        """
        This method is being called by :meth:`onData` if there is no
        data in self._df.  This happens if:

            1. this is a fresh start of the system, or

            2. this is a restart and data has been reset after
               contract change

        :meth:`onData` has already received data since last save.  We
        should not re-acquire it here.  In principle all required data
        should be available in database, falling back on broker only
        if it's an empty database (no data for a contract at all).  We
        should never call on broker for current contract data, this
        should be provided by :class:`HistoricalDataStreamer`.

        If it's a new contract after a roll (not saved in database
        yet), :meth:`onData` would have received all necessary data
        for this contract already, but we most likely need to stich it
        with previous contract(s), data for which data is availalbe in
        the database.

        `data` param contains the data for current contract, if there
        is no data for current contract in the database, this is the
        data from broker that we should use to stich with previous
        contracts.

        It may be necessary to pull data for current contract from
        database, in which case it's a regular case, where we stich
        data from database and splice it newer data in :meth:`onData`.
        """
        dfs = {}

        for contract in self._back_contracts():
            date_range_or_none = self._compute_date_range(contract)

            # if range is None it means no more data before
            # `self.offset_durationStr` can be acquired
            if date_range_or_none is None:
                break

            start, stop = date_range_or_none
            #  extra datapoints on either side to ensure (???) overlapping series
            df = await self._acquire_data_for_contract(
                contract, start - timedelta(hours=2), stop + timedelta(hours=2)
            )
            log.debug(
                f"{self!s} acquired back data for contract: {contract.localSymbol}"
                f"from {start} to {stop} {len(df)=}"
            )
            if df.empty and self.contract == contract:
                df = current_df
            elif contract == self.contract:
                df = misc.concat_dfs(df, current_df)
            dfs[contract] = df

        return (
            dfs[cast(ibi.Future, self.contract)]
            if len(dfs) == 1
            else FuturesStitcher(dfs).data
        )

    def _compute_date_range(
        self, contract: ibi.Future
    ) -> tuple[datetime, datetime] | None:
        """
        Return dates between data should be collected for a given
        contract.  We require back data so no limit should be later
        than today.  For any contract, required range should not go
        beyond contract's active period determined by
        `contract_selector` and it shouldn't go back further than
        :meth:`.required_timedelta`
        """
        selector = cast(FutureSelector, self.contract_selector)
        date_ranges = (
            selector.date_ranges_next
            if self.which_contract is ActiveNext.NEXT
            else selector.date_ranges
        )
        start, stop = date_ranges[contract]
        now = utc_now_naive()
        start_date = max(start, self.offset_by_durationStr(now))
        stop_date = min(stop, now)
        # Don't make this timezone aware or datastore will reject it
        return (start_date, stop_date) if (start_date < stop_date) else None

    def _tz(self, dt: datetime) -> datetime:
        return dt.replace(tzinfo=self.contract_details.zone_info)

    async def _historical_data_with_retry(self, **params) -> ibi.BarDataList:
        attempt = 0
        while True:
            try:
                return await self.ib.reqHistoricalDataAsync(**params, timeout=0)
            except Exception:
                log.exception(f"Retry on pulling historical data for {self!s}")
                if attempt == 2:
                    raise
                await asyncio.sleep(2**attempt)
                attempt += 1

    async def _pull_history_from_broker(
        self, contract: ibi.Contract, stop_date: datetime
    ) -> ibi.BarDataList:
        try:
            params = {
                key: self._streamer_params[key]
                for key in (
                    "whatToShow",
                    "barSizeSetting",
                    "useRTH",
                    "formatDate",
                )
            }
        except KeyError as e:
            key = e.args[0]
            raise MissingStreamerParam(key)

        params["endDateTime"] = self.to_datetime(stop_date)
        params["contract"] = contract
        params["durationStr"] = ensure_duration_str(
            self._streamer_params["durationStr"],
            self._streamer_params["barSizeSetting"],
            self.session_length,
        )
        log.warning(f"{self!s} calling broker with params: {params}")
        data_from_broker = await self._historical_data_with_retry(**params)
        if data_from_broker:
            log.debug(
                f"{self!s} {contract.localSymbol} received data from broker from: "
                f"{data_from_broker[0].date} to: {data_from_broker[-1].date}"
            )
        else:
            log.error(f"{self!s} {contract.localSymbol} received no data from broker")
        return data_from_broker

    def _back_contracts(self) -> Generator[ibi.Future, None, None]:
        selector = cast(FutureSelector, self.contract_selector)
        contract = cast(ibi.Future, self.contract)
        expiry = self.expiry_from_contract(contract)

        for wrapper in reversed(selector.all_contracts):
            if self.expiry_from_contract(wrapper.contract) <= expiry:
                yield wrapper.contract

    @staticmethod
    def to_datetime(date):
        try:
            return date.to_pydatetime()
        except Exception:
            return date

    async def _acquire_data_for_contract(
        self, contract: ibi.Contract, start_date: datetime, stop_date: datetime
    ) -> pd.DataFrame:
        log.debug(
            f"{self!s} acquiring back data for contract: {contract.localSymbol} "
            f"{start_date=} {stop_date=}"
        )
        store = self.store
        if (df := await store.read(contract, start_date, stop_date)) is None:
            # don't pull data for current contract from broker, this
            # is :class:`Streamer`'s responsibility; data for previous
            # contracts may be missing if it's a new database and only
            # then it's acceptable; pulling data from broker here is
            # unusual and should be investigated if it happens
            if contract == self.contract:
                df = pd.DataFrame()
            else:
                log.warning(
                    f"{self!s} requesting data from broker for: {contract.localSymbol}"
                )
                bars = await self._pull_history_from_broker(contract, stop_date)

                df = pd.DataFrame(bars).set_index("date")
                try:
                    await store.write(contract, pd.DataFrame(df))
                except Exception:
                    log.exception(
                        "Error while writing data from broker to datastore. "
                        "Data not saved to store."
                    )
        return df

    @cached_property
    def datapoints(self) -> int:
        """
        durationStr can be given either directly as a str acceptable
        by :meth:`ib_insync.IB.reqHistoricalData` or number of
        required datapoints.  If it's given a str, it needs to be
        converted into datapoints.

        This is reverse to what :class:`HistoricalDataStreamer` does,
        which needs to use durationStr directly if it's given as str
        or convert to str if it's given as number of required
        datapoints.

        Data will be converted only once, i.e. either by Streamer or
        by Stitcher.  Converting data twice (eg.  str -> int -> str)
        would be wrong since every conversion entails a degree of
        rounding, which would make us get further away from the the
        duration requested by the user.

        Conversion from str to int is approximated, rounded down.
        Given querks of how many datapoints the broker returns for a
        particular query, the safest option is to assume, it will be a
        day less than really requested.
        """
        durationStr = self._streamer_params["durationStr"]
        if isinstance(durationStr, str):
            return durationStr_to_datapoints(
                durationStr,
                self._streamer_params["barSizeSetting"],
                self.session_length,
                offset_days=-1,
            )

        else:
            return durationStr

    def offset_by_durationStr(self, now: datetime | None = None) -> datetime:
        """Return the earliest required timezone-naive UTC timestamp.

        ``durationStr`` can be either a string accepted by
        :meth:`ib_insync.IB.reqHistoricalData` or a number of required
        datapoints. Both cases are accounted for in this method. The returned
        range includes extra history to facilitate stitching data from
        different contracts.

        Args:
            now: Timezone-naive UTC endpoint. Defaults to the current UTC time.

        Returns:
            Earliest timestamp required for the configured history window.
        """
        durationStr = self._streamer_params["durationStr"]
        if now is None:
            now = utc_now_naive()
        if isinstance(durationStr, str):
            return offset_durationStr(durationStr, now)
        else:
            delta = datapoints_to_timedelta(
                durationStr,
                self._streamer_params["barSizeSetting"],
                self.session_length,
            )
            return now - delta.days * custom_bday - timedelta(seconds=delta.seconds)

    @cached_property
    def session_length(self) -> timedelta:
        """Return length of a typical trading session as a timedelta."""
        return typical_session_length(self.contract_details.trading_hours)

    @staticmethod
    def expiry_from_contract(contract: ibi.Contract) -> datetime:
        return datetime.strptime(contract.lastTradeDateOrContractMonth, "%Y%m%d")

    @cached_property
    def _barSizeSetting_timedelta(self) -> timedelta:
        barSizeSetting = self._streamer_params["barSizeSetting"]
        return barSizeSetting_to_timedelta(barSizeSetting, False)

    def onContractChanged(
        self, old_contract: ibi.Contract, new_contract: ibi.Contract
    ) -> None:
        self._df = pd.DataFrame()
        self._last_data_point = None
        super().onContractChanged(old_contract, new_contract)

    def __str__(self) -> str:
        if self.contract is not None:
            return f"<{self.__class__.__name__} {self.contract.symbol}>"
        else:
            return f"{self!r}"


def _validate_positive_int(value: int, owner: str, parameter: str) -> None:
    """Validate a positive integer grouping threshold.

    Args:
        value: Configured threshold.
        owner: Public component name used in the error message.
        parameter: Public parameter name used in the error message.

    Raises:
        TypeError: If ``value`` is not an integer or is a boolean.
        ValueError: If ``value`` is not positive.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{owner} {parameter} must be an int")
    if value <= 0:
        raise ValueError(f"{owner} {parameter} must be positive")


def _validate_boundary(value: str, owner: str, parameter: str) -> None:
    """Validate a left/right grouping-boundary option.

    Args:
        value: Configured boundary value.
        owner: Public component name used in the error message.
        parameter: Public parameter name used in the error message.

    Raises:
        ValueError: If ``value`` is neither ``"left"`` nor ``"right"``.
    """

    if value not in ("left", "right"):
        raise ValueError(f"{owner} {parameter} must be 'left' or 'right'")


def _aggregation_rules(columns: pd.Index) -> dict[Any, str]:
    """Return standard OHLCV aggregation rules for dataframe columns."""

    rules = {column: "last" for column in columns}
    rules.update(
        {
            column: operation
            for column, operation in {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "barCount": "sum",
            }.items()
            if column in columns
        }
    )
    return rules


def _validate_ohlc(data: pd.DataFrame) -> None:
    """Require the price columns needed to aggregate bars.

    Args:
        data: Source dataframe.

    Raises:
        ValueError: If an OHLC column is absent.
    """

    required = {"open", "high", "low", "close"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"data must contain OHLC columns; missing: {sorted(missing)}")


def _aggregate_labelled_rows(
    data: pd.DataFrame,
    groups: pd.Index,
    label: Literal["left", "right"],
) -> pd.DataFrame:
    """Aggregate source rows identified by positional group labels.

    Args:
        data: Source OHLC dataframe.
        groups: One integer group label per source row.
        label: Whether output uses the first or last source timestamp.

    Returns:
        Complete grouped dataframe, including its possibly incomplete final row.
    """

    _validate_ohlc(data)
    if data.empty:
        return data.copy()

    original_columns = list(data.columns)
    working = data.copy()
    group_column = "__haymaker_group__"
    timestamp_column = "__haymaker_timestamp__"
    weighted_column = "__haymaker_weighted_average__"
    while group_column in working.columns:
        group_column = f"_{group_column}"
    while timestamp_column in working.columns:
        timestamp_column = f"_{timestamp_column}"
    while weighted_column in working.columns:
        weighted_column = f"_{weighted_column}"

    working[group_column] = groups.to_numpy()
    working[timestamp_column] = data.index
    rules = _aggregation_rules(data.columns)
    rules[timestamp_column] = "first" if label == "left" else "last"
    weighted_average = "average" in data.columns and "volume" in data.columns
    if weighted_average:
        working[weighted_column] = working["average"] * working["volume"]
        rules[weighted_column] = "sum"

    grouped = working.groupby(group_column, sort=False).agg(rules)
    if weighted_average:
        grouped["average"] = grouped[weighted_column] / grouped["volume"]
        grouped = grouped.drop(columns=weighted_column)
    grouped = grouped.set_index(timestamp_column)
    grouped.index.name = data.index.name
    return grouped.loc[:, original_columns]


def _add_weighted_average(
    source: pd.DataFrame,
    grouped: pd.DataFrame,
    label: Literal["left", "right"],
) -> pd.DataFrame:
    """Replace grouped ``average`` values with volume-weighted averages.

    Args:
        source: Source rows used by :func:`volume_grouper`.
        grouped: Grouped dataframe returned by :func:`volume_grouper`.
        label: Whether group indexes identify their first or last source row.

    Returns:
        ``grouped`` with corrected averages when both required columns exist.
    """

    if (
        grouped.empty
        or "average" not in source.columns
        or "volume" not in source.columns
    ):
        return grouped

    if label == "left":
        starts = source.index.searchsorted(grouped.index.to_numpy(), side="left")
    else:
        ends = source.index.searchsorted(grouped.index.to_numpy(), side="right")
        starts = np.concatenate(([0], ends[:-1]))

    price_volume = source["average"].to_numpy(dtype=float) * source["volume"].to_numpy(
        dtype=float
    )
    numerators = np.add.reduceat(price_volume, starts)
    result = grouped.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        result["average"] = numerators / result["volume"].to_numpy(dtype=float)
    return result


def _completed_threshold_groups(
    source: pd.DataFrame,
    grouped: pd.DataFrame,
    target: int,
    field: str,
    label: Literal["left", "right"],
) -> pd.DataFrame:
    """Remove a final cumulative group that has not reached its target.

    Args:
        source: Source dataframe.
        grouped: Dataframe grouped by cumulative ``field`` values.
        target: Minimum completed-group total.
        field: Source column accumulated toward ``target``.
        label: Whether group indexes identify their first or last source row.

    Returns:
        Only groups whose cumulative field has reached ``target``.
    """

    if grouped.empty:
        return grouped

    if label == "left":
        start = source.index.searchsorted(grouped.index[-1], side="left")
    elif len(grouped) == 1:
        start = 0
    else:
        start = source.index.searchsorted(grouped.index[-2], side="right")

    last_group_total = source.iloc[start:][field].sum(skipna=False)
    return grouped if last_group_total >= target else grouped.iloc[:-1]


def _resample_rows(
    data: pd.DataFrame,
    rule: str | timedelta,
    label: Literal["left", "right"],
    closed: Literal["left", "right"],
) -> pd.DataFrame:
    """Aggregate rows into non-empty pandas time buckets.

    Args:
        data: Date-indexed source OHLC dataframe.
        rule: Pandas resampling frequency.
        label: Boundary used to label output buckets.
        closed: Boundary included in each output bucket.

    Returns:
        All non-empty time buckets, including the final incomplete bucket.

    Raises:
        TypeError: If the source does not use a ``DatetimeIndex``.
    """

    _validate_ohlc(data)
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("TimeGrouper requires a pandas DatetimeIndex")
    if data.empty:
        return data.copy()

    original_columns = list(data.columns)
    working = data.copy()
    weighted_column = "__haymaker_weighted_average__"
    while weighted_column in working.columns:
        weighted_column = f"_{weighted_column}"

    rules = _aggregation_rules(data.columns)
    weighted_average = "average" in data.columns and "volume" in data.columns
    if weighted_average:
        working[weighted_column] = working["average"] * working["volume"]
        rules[weighted_column] = "sum"

    resampler = working.resample(rule, label=label, closed=closed)
    sizes = resampler.size()
    grouped = resampler.agg(rules).loc[sizes > 0]
    if weighted_average:
        grouped["average"] = grouped[weighted_column] / grouped["volume"]
        grouped = grouped.drop(columns=weighted_column)
    return grouped.loc[:, original_columns]


class _DataFrameGrouper(Atom, ABC):
    """Recalculate complete grouped frames and emit only new completions."""

    def __init__(self) -> None:
        self._last_emitted_point: Any | None = None
        self._initialized = False
        super().__init__()

    def onData(self, data: pd.DataFrame, *args: Any) -> None:
        """Regroup a cumulative dataframe and emit on completion progress.

        Args:
            data: Complete cumulative source dataframe.
            *args: Additional event values, ignored.
        """

        assert isinstance(
            data, pd.DataFrame
        ), f"{self} accepts only pandas DataFrame not {type(data)}"
        completed = self._completed_groups(data)
        if not self._initialized:
            self._initialized = True
            if not completed.empty:
                self._last_emitted_point = completed.index[-1]
            return
        if completed.empty:
            return

        last_completed_point = completed.index[-1]
        if (
            self._last_emitted_point is None
            or last_completed_point > self._last_emitted_point
        ):
            self._last_emitted_point = last_completed_point
            self.dataEvent.emit(completed)

    @abstractmethod
    def _completed_groups(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return the complete frame containing only finished groups."""


@dataclass(eq=False)
class CountGrouper(_DataFrameGrouper):
    """Regroup a DataFrame into rows containing a fixed source-row count.

    Each output uses the first open, maximum high, minimum low, last close,
    summed volume and ``barCount``, and a volume-weighted ``average`` when
    those columns are available. Unknown columns retain their final value.
    The final short group is excluded.

    Args:
        count: Positive number of source rows per output row.
        label: Whether grouped timestamps use the first or last source row.

    Emits:
        The complete DataFrame of finished groups when a new group closes. The
        first input establishes the completion watermark and does not emit.
    """

    count: int
    label: Literal["left", "right"] = "left"

    def __post_init__(self) -> None:
        _validate_positive_int(self.count, type(self).__name__, "count")
        _validate_boundary(self.label, type(self).__name__, "label")
        super().__init__()

    def _completed_groups(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return fixed-count groups except for a final short group."""

        groups = pd.RangeIndex(len(data)) // self.count
        grouped = _aggregate_labelled_rows(data, groups, self.label)
        if data.empty or len(data) % self.count == 0:
            return grouped
        return grouped.iloc[:-1]


@dataclass(eq=False)
class TickGrouper(_DataFrameGrouper):
    """Regroup a DataFrame by cumulative source ``barCount`` values.

    Whole source rows are assigned to a group, so its final ``barCount`` may
    exceed ``count``. The final group is excluded until it reaches the target.

    Args:
        count: Positive minimum tick count for each grouped row.
        label: Whether grouped timestamps use the first or last source row.

    Emits:
        The complete DataFrame of finished tick groups when a new group closes.
        The first input establishes the completion watermark and does not emit.
    """

    count: int
    label: Literal["left", "right"] = "left"

    def __post_init__(self) -> None:
        _validate_positive_int(self.count, type(self).__name__, "count")
        _validate_boundary(self.label, type(self).__name__, "label")
        super().__init__()

    def _completed_groups(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return cumulative tick groups that reached the configured count."""

        grouped = volume_grouper(data, self.count, field="barCount", label=self.label)
        grouped = _add_weighted_average(data, grouped, self.label)
        return _completed_threshold_groups(
            data, grouped, self.count, "barCount", self.label
        )


@dataclass(eq=False)
class TimeGrouper(_DataFrameGrouper):
    """Regroup a date-indexed DataFrame into fixed pandas time buckets.

    Empty buckets are omitted. The final bucket is always treated as
    incomplete and is emitted only after a source row arrives in a later
    bucket, avoiding premature signals without requiring a trading calendar.

    Args:
        rule: Positive pandas resampling frequency such as ``"5min"``.
        label: Boundary used to label output buckets.
        closed: Boundary included in each output bucket.

    Emits:
        The complete DataFrame of finished non-empty time buckets when a new
        bucket closes. The first input establishes the completion watermark and
        does not emit.
    """

    rule: str | timedelta
    label: Literal["left", "right"] = "left"
    closed: Literal["left", "right"] = "left"

    def __post_init__(self) -> None:
        _validate_boundary(self.label, type(self).__name__, "label")
        _validate_boundary(self.closed, type(self).__name__, "closed")
        if isinstance(self.rule, timedelta):
            if self.rule <= timedelta(0):
                raise ValueError("TimeGrouper rule must be positive")
        else:
            try:
                offset = pd.tseries.frequencies.to_offset(self.rule)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"TimeGrouper invalid rule: {self.rule!r}") from exc
            if offset.n <= 0:
                raise ValueError("TimeGrouper rule must be positive")
        super().__init__()

    def _completed_groups(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return non-empty time buckets except for the final bucket."""

        grouped = _resample_rows(data, self.rule, self.label, self.closed)
        return grouped.iloc[:-1]


@dataclass(eq=False)
class VolumeGrouper(_DataFrameGrouper):
    """Regroup a DataFrame into completed equal-volume rows.

    Use this component in a DataFrame aggregation pipeline when volume-based
    rows are preferred to the source bar interval. Unlike
    :class:`~haymaker.components.VolumeBars`, which incrementally groups
    ``ib_insync.BarData`` objects, this component recalculates groups from the
    complete input DataFrame. The final incomplete group is excluded.

    Args:
        volume: Positive target volume for each grouped row.
        group_on: Input column accumulated toward ``volume``.
        label: Whether grouped timestamps use the left or right boundary.

    Emits:
        The complete DataFrame of finished volume groups when a new group has
        closed. The first input establishes the completion watermark and does
        not emit.

    Raises:
        TypeError: If ``volume`` is not an integer.
        ValueError: If ``volume`` is not positive.
        AssertionError: If input is not a dataframe or ``group_on`` is absent.
    """

    volume: int
    group_on: str = "volume"
    label: Literal["left", "right"] = "left"

    def __post_init__(self) -> None:
        _validate_positive_int(self.volume, type(self).__name__, "volume")
        _validate_boundary(self.label, type(self).__name__, "label")
        super().__init__()

    def _completed_groups(self, source: pd.DataFrame) -> pd.DataFrame:
        """Return cumulative volume groups that reached the configured target."""

        assert self.group_on in source.columns, (
            f"{self} attempts to group on {self.group_on}, which is not present "
            f"in passed DataFrame"
        )
        grouped = volume_grouper(
            source, self.volume, field=self.group_on, label=self.label
        )
        grouped = _add_weighted_average(source, grouped, self.label)
        return _completed_threshold_groups(
            source, grouped, self.volume, self.group_on, self.label
        )


__all__ = [
    "CountGrouper",
    "FuturesPandasAggregator",
    "TickGrouper",
    "TimeGrouper",
    "VolumeGrouper",
]
