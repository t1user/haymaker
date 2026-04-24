import asyncio
import logging
import random
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timedelta
from functools import cached_property
from typing import Any, ClassVar, Generator, Literal, cast

import eventkit as ev  # type: ignore
import ib_insync as ibi
import pandas as pd

from haymaker import misc
from haymaker.async_wrappers import QueueRunner
from haymaker.base import Atom
from haymaker.config import CONFIG
from haymaker.contract_selector import FutureSelector, custom_bday
from haymaker.databases import get_mongo_client
from haymaker.datastore import (
    AsyncAbstractBaseStore,
    AsyncArcticStore,
    CollectionNamerBarsizeSetting,
)
from haymaker.details_processor import typical_session_length
from haymaker.durationStr_converters import (
    barSizeSetting_to_timedelta,
    datapoints_to_timedelta,
    durationStr_to_datapoints,
    offset_durationStr,
)
from haymaker.research.numba_tools import volume_grouper
from haymaker.streamers import Streamer

from .stitcher import FuturesStitcher

log = logging.getLogger(__name__)

AGG_CONFIG = CONFIG.get("dfaggregator", {})
MARKET_DATA_LIB_NAME = AGG_CONFIG.get("market_data_lib", "market_data")
SAVE_FREQUENCY = AGG_CONFIG.get("aggregator_save_frequency", 900)


class MissingStreamerParam(Exception):
    pass


class WrongStreamer(Exception):
    pass


@dataclass
class DfAggregator(Atom):
    """
    Convert recieved data bars into a pandas DataFrame and store
    processed data.  Ensure required minimum amount of data is
    available, calling on database and/or broker if necessary.

    For futures contracts ensure that a conitinuous series is created
    using appropriate back contracts.

    Both arguments can be set either directly while instantiating the
    class or system-wide in config in `dfaggregator` section.

    Args:
    -----

    * datastore: custom datastore can be passed, it needs to handle
    naming contract collections in a manner that can be interpreted by
    streamer; if nothing is passed, default :class:`AsyncArcticStore`
    will be used

    * save_frequency: how often data will be saved to datastore, zero
    means data will not be saved (which maybe useful for testing but
    in a way defies the purpose of the whole object)
    """

    datastore: AsyncAbstractBaseStore | None = None
    save_frequency: int | None = None  # in seconds

    # ================================================================================

    _compatible_with: ClassVar[tuple[str]] = ("HistoricalDataStreamer",)

    _streamer_params: dict[str, Any] = field(
        init=False, repr=False, default_factory=dict
    )
    _df: pd.DataFrame = field(init=False, repr=False, default_factory=pd.DataFrame)
    _queue: QueueRunner = field(init=False, repr=False)
    _save_timer: ev.Timer = field(init=False, repr=False, default=None)
    _timer_task: asyncio.Task | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        if self.save_frequency is None:
            self.save_frequency = SAVE_FREQUENCY
        assert isinstance(
            self.save_frequency, int
        ), f"{self!s} save_frequency must be an int, not {type(self.save_frequency)}"
        self._queue = QueueRunner(self.process_data, f"{self!s}")
        super().__init__()

    @property
    def store(self) -> AsyncAbstractBaseStore:
        assert (barSizeSetting := self._streamer_params.get("barSizeSetting")), (
            f"{self} cannot initialize "
            f" datastore because barSizeSetting is not defined"
        )
        if self.datastore is None:
            assert MARKET_DATA_LIB_NAME, (
                f"{self} cannot initialize datastore because "
                f"MARKET_DATA_LIB_NAME was not given"
            )
            self.datastore = AsyncArcticStore(
                lib=MARKET_DATA_LIB_NAME,
                host=get_mongo_client(),
                collection_namer=CollectionNamerBarsizeSetting(barSizeSetting),
            )
        self.datastore.override_collection_namer(
            CollectionNamerBarsizeSetting(barSizeSetting)
        )
        return self.datastore

    async def set_timer(self) -> None:
        # if many objects created, they shouldn't all save at the same time
        await asyncio.sleep(random.randint(0, 30))
        log.debug(f"{self!s} setting save timer at {self.save_frequency}secs.")
        self._save_timer = ev.Timer(self.save_frequency)
        self._save_timer += self.save_data

    def onStart(self, data: Any, *args: Any) -> None:
        """Syncing contract with streamer."""
        assert args, f"No streamer passed to {self!s}"
        streamer = args[0]
        self.sync_with_streamer(streamer)
        if self._save_timer is None and self.save_frequency:
            self._timer_task = asyncio.create_task(
                self.set_timer(), name=f"{self!s} timer setter"
            )
        super().onStart(data, *args)

    def sync_with_streamer(self, streamer: Streamer) -> None:
        # streamer class used only to verify compatibility
        self.verify_streamer_compatibility(streamer)
        assert is_dataclass(streamer), f"Streamer: {streamer} must be a dataclass."
        self._streamer_params = {
            f.name: getattr(streamer, f.name) for f in fields(streamer)
        }
        # sync contract with streamer
        # these 2 properties together ensure that self.contract
        # will be the same as on streamer
        log.debug(f"{self!s} streamer params: {self._streamer_params}")
        self.which_contract = streamer.which_contract
        self._contract_blueprint = streamer._contract_blueprint

    def verify_streamer_compatibility(self, streamer: Streamer) -> None:
        streamer_class = streamer.__class__.__name__
        if streamer_class not in self._compatible_with:
            raise WrongStreamer(
                f"Streamer {streamer_class} is not compatible with {self!s}"
            )

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
        assert (contract := self.contract), f"Missing contract on {self}"
        if not self._df.empty:
            await self.store.async_append(contract, self._df)

    def process_current_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Return only the part of the `data` that corresponds to the
        period when current contract is active.
        """
        assert isinstance(self.contract, ibi.Future)
        assert (date_range := self._compute_date_range(self.contract)) is not None
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
        assert isinstance(
            self.contract_selector, FutureSelector
        ), f"contract on {self!s} is not a Future: {self.contract}"

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
        assert isinstance(
            self.contract_selector, FutureSelector
        ), f"contract on {self!s} is not a Future: {self.contract}"
        start, stop = self.contract_selector.date_ranges[contract]
        now = datetime.now()
        start_date = max(start, self.offset_by_durationStr())
        stop_date = min(stop, now)
        # Don't make this timezone aware or datastore will reject it
        return (start_date, stop_date) if (start_date < stop_date) else None

    def _tz(self, dt: datetime) -> datetime:
        return dt.replace(tzinfo=self.contract_details.zone_info)

    async def _historical_data_with_retry(self, **params) -> ibi.BarDataList:
        for attempt in range(3):
            try:
                return await self.ib.reqHistoricalDataAsync(**params, timeout=0)
            except Exception:
                log.exception(f"Retry on pulling historical data for {self!s}")
                if attempt == 2:
                    raise
                await asyncio.sleep(2**attempt)

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
        assert isinstance(
            selector := self.contract_selector, FutureSelector
        ), f"Missing contract selector for {self._contract_blueprint}"
        assert isinstance(
            self.contract, ibi.Future
        ), f"{self!s} attempting stiching on a non-Future"

        expiry = self.expiry_from_contract(self.contract)

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
        if (df := await self.store.read(contract, start_date, stop_date)) is None:
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
                    self.store.write(contract, pd.DataFrame(df))
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

    def offset_by_durationStr(self) -> datetime:
        """
        Return durationStr as timedelta, this is how far back we need data.
        durationStr can be given either directly as a str acceptable
        by :meth:`ib_insync.IB.reqHistoricalData` or number of
        required datapoints. Both cases are accounted for in this method.

        Returned timedelta is longer by 10% than strictly necessary
        to facilitate stiching of data for different contracts.
        """
        durationStr = self._streamer_params["durationStr"]
        now = datetime.now()
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


@dataclass
class VolumeGrouper(Atom):
    volume: int
    group_on: str = "volume"
    label: Literal["left", "right"] = "left"
    _last_emitted_point: pd.Timestamp | None = field(repr=False, default=None)

    def __post_init__(self):
        super().__init__()

    def onData(self, df, *args) -> None:
        assert isinstance(
            df, pd.DataFrame
        ), f"{self} accepts only pandas DataFrame not {type(df)}"
        assert self.group_on in df.columns, (
            f"{self} attempts to group on {self.group_on}, which is not present "
            f"in passed DataFrame"
        )
        grouped = volume_grouper(df, self.volume, field=self.group_on, label=self.label)
        if self._last_emitted_point is None:
            self._last_emitted_point = grouped.index[-2]
        elif grouped.index[-2] > self._last_emitted_point:
            self._last_emitted_point = grouped.index[-2]
            self.dataEvent.emit(grouped.iloc[:-1])
