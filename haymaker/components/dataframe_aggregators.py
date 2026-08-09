"""DataFrame-oriented aggregation components for historical bar pipelines.

This public module complements :mod:`haymaker.components.aggregators`. Its
components maintain and transform complete pandas DataFrames, whereas the bar
aggregators operate on ``ib_insync`` bar objects.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timedelta
from functools import cached_property
from typing import Any, Awaitable, ClassVar, Generator, Literal, cast

import eventkit as ev  # type: ignore
import ib_insync as ibi
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
    offset_durationStr,
)
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
            zero disables periodic persistence.

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

    def __post_init__(self) -> None:
        if self.datastore is False or self.datastore is None:
            raise TypeError(
                "FuturesPandasAggregator datastore must be True or an AsyncDataStore"
            )
        if self.datastore is not True:
            self._store = self.datastore
        assert isinstance(
            self.save_frequency, int
        ), f"{self!s} save_frequency must be an int, not {type(self.save_frequency)}"
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
        if self._save_timer is None and self.save_frequency:
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
        raw_df = pd.DataFrame(data).set_index("date")
        current_df = self.process_current_data(raw_df)
        # implicit assumption: if we already have data in `self._df`,
        # together with the newly received data, it should give enough
        # datapoints
        if (not self._df.empty) or (len(current_df) >= self.datapoints):
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
        self.dataEvent.emit(self._df)

    def append_data(self, *dfs: pd.DataFrame) -> pd.DataFrame:
        self._df = misc.concat_dfs(self._df, *dfs)
        return self._df

    async def save_data(self, *args) -> None:
        contract = cast(ibi.Future, self.contract)
        if not self._df.empty:
            await self.store.append(contract, self._df)

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
        start, stop = selector.date_ranges[contract]
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
        self, contract: ibi.Contract, start_date: datetime, stop_date: datetime
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
        params["durationStr"] = self._streamer_params["durationStr"]
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
                bars = await self._pull_history_from_broker(
                    contract, start_date, stop_date
                )

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
        super().onContractChanged(old_contract, new_contract)

    def __str__(self) -> str:
        if self.contract is not None:
            return f"<{self.__class__.__name__} {self.contract.symbol}>"
        else:
            return f"{self!r}"


@dataclass(eq=False)
class VolumeGrouper(Atom):
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
    _last_emitted_point: pd.Timestamp | None = field(repr=False, default=None)
    _initialized: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        if isinstance(self.volume, bool) or not isinstance(self.volume, int):
            raise TypeError("VolumeGrouper volume must be an int")
        if self.volume <= 0:
            raise ValueError("VolumeGrouper volume must be positive")
        super().__init__()

    def onData(self, data: pd.DataFrame, *args: Any) -> None:
        df = data
        assert isinstance(
            df, pd.DataFrame
        ), f"{self} accepts only pandas DataFrame not {type(df)}"
        assert self.group_on in df.columns, (
            f"{self} attempts to group on {self.group_on}, which is not present "
            f"in passed DataFrame"
        )
        grouped = volume_grouper(df, self.volume, field=self.group_on, label=self.label)
        completed = self._completed_groups(df, grouped)
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

    def _completed_groups(
        self, source: pd.DataFrame, grouped: pd.DataFrame
    ) -> pd.DataFrame:
        """Return grouped rows whose source volume has reached the target."""

        if grouped.empty:
            return grouped

        if self.label == "left":
            start = source.index.searchsorted(grouped.index[-1], side="left")
        elif len(grouped) == 1:
            start = 0
        else:
            start = source.index.searchsorted(grouped.index[-2], side="right")

        last_group_total = source.iloc[start:][self.group_on].sum(skipna=False)
        return grouped if last_group_total >= self.volume else grouped.iloc[:-1]


__all__ = [
    "FuturesPandasAggregator",
    "VolumeGrouper",
]
