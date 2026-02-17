import asyncio
import logging
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timedelta, timezone
from functools import cached_property
from typing import Any, ClassVar, Generator, cast

import ib_insync as ibi
import pandas as pd

from haymaker import misc
from haymaker.base import Atom
from haymaker.config import CONFIG
from haymaker.contract_selector import FutureSelector
from haymaker.databases import get_mongo_client
from haymaker.datastore import AsyncAbstractBaseStore, AsyncArcticStore
from haymaker.details_processor import typical_session_length
from haymaker.durationStr_converters import (
    datapoints_to_timedelta,
    date_to_delta_wrapper,
    durationStr_to_datapoints,
    durationStr_to_timedelta,
)
from haymaker.streamers import Streamer

from .sticher import FuturesSticher

log = logging.getLogger(__name__)

MARKET_DATA_LIB_NAME = CONFIG.get("market_data_lib", "market_data")


class MissingStreamerParam(Exception):
    pass


class WrongStreamer(Exception):
    pass


@dataclass
class DfAggregator(Atom):
    """
    Covnert recieved data bars into a pandas DataFrame and store
    processed data.  Ensure required minimum amount of data is
    available, calling on database and/or broker if necessary.

    For futures contracts ensure that a conitinuous series is created
    using appropriate back contracts.
    """

    _compatible_with: ClassVar[tuple[str]] = ("HistoricalDataStreamer",)

    _streamer_params: dict[str, Any] = field(repr=False, default_factory=dict)
    _df: pd.DataFrame | None = field(repr=False, default=None)
    store: AsyncAbstractBaseStore = field(init=False)

    def __post_init__(self):
        self.store = AsyncArcticStore(lib=MARKET_DATA_LIB_NAME, host=get_mongo_client())
        super().__init__()

    def onStart(self, data: Any, *args: Any) -> None:
        assert data.get(
            "streamer"
        ), f"{self} didn't receive streamer it is connected to"
        streamer = data.pop("streamer")
        self.sync_with_streamer(streamer)
        super().onStart(data)

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
        self.which_contract = streamer.which_contract
        self._contract_blueprint = streamer._contract_blueprint

    def verify_streamer_compatibility(self, streamer: Streamer) -> None:
        streamer_class = streamer.__class__.__name__
        if streamer_class not in self._compatible_with:
            raise WrongStreamer(
                f"Streamer {streamer_class} is not compatible with {self}"
            )

    async def onData(self, data: ibi.BarDataList, *args: Any) -> None:
        df = await self.process_data(data)
        super().onData(data)
        # pd.DataFrame emitted, make sure next Atom accepts it
        self.dataEvent.emit(df)

    async def process_data(self, data: ibi.BarDataList) -> pd.DataFrame:
        data_df = pd.DataFrame(data).set_index("date")
        df = self._df or await self.acquire_data(data_df)
        assert (
            df is not None
        ) and not df.empty, f"{self} failed to obtained back data."
        last_point = df.index[-1]
        new_df = data_df.loc[last_point:]
        return self.update_df(new_df.iloc[1:] if last_point in new_df.index else new_df)

    def update_df(self, data: pd.DataFrame) -> pd.DataFrame:
        # this is an 'append' function, so only new data
        self.save_data(data)
        if self._df is None:
            self._df = data
        else:
            self._df = misc.concat_dfs(self._df, data)
        return self._df

    def save_data(self, df: pd.DataFrame) -> None:
        assert (contract := self.contract), f"Missing contract on {self}"
        self.store.append(contract, df)

    async def acquire_data(self, data: pd.DataFrame) -> pd.DataFrame:
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
        ), f"contract on {self} is not a Future: {self.contract}"

        dfs = {}
        datapoints = 0
        for contract in self._back_contracts():
            date_range_or_none = self._compute_date_range(contract)
            if date_range_or_none is None:
                continue
            else:
                start, stop = date_range_or_none

            df = await self._acquire_data_for_contract(contract, start, stop)

            # fallback to streamer data if no data for current
            # contract in database
            if df.empty and contract == self.contract:
                df = data

            dfs[contract] = df.loc[start:stop]  # type: ignore[misc]
            datapoints += len(dfs[contract])
            if datapoints >= self.datapoints:
                break

        if datapoints < self.datapoints:
            log.error(f"{self} failed to get required amount of back data.")

        return (
            dfs[cast(ibi.Future, self.contract)]
            if len(dfs) == 1
            else FuturesSticher(dfs).data
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
        ), f"contract on {self} is not a Future: {self.contract}"
        start, stop = self.contract_selector.date_ranges[contract]
        now = datetime.now()
        start_date = max(start, now - self.required_timedelta)
        stop_date = min(stop, now)
        return (
            (self._tz(start_date), self._tz(stop_date))
            if (start_date < stop_date)
            else None
        )

    @staticmethod
    def _tz(dt: datetime) -> datetime:
        return dt.replace(tzinfo=timezone.utc)

    async def _historical_data_with_retry(
        self, **params
    ) -> ibi.BarDataList:  # type: ignore
        for attempt in range(3):
            try:
                return await self.ib.reqHistoricalDataAsync(**params)
            except Exception as e:
                log.warning(
                    f"Retry on pulling historical data for {self} because of {e}"
                )
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

        params["endDateTime"] = stop_date
        params["contract"] = contract
        params["durationStr"] = date_to_delta_wrapper(
            start_date,
            params["barSizeSetting"],
            end_date_or_now=stop_date,
            margin=int(self.datapoints * 0.1),
        )
        return await self._historical_data_with_retry(**params)

    def _back_contracts(self) -> Generator[ibi.Future, None, None]:
        assert isinstance(
            selector := self.contract_selector, FutureSelector
        ), f"Missing contract selector for {self._contract_blueprint}"
        assert isinstance(
            self.contract, ibi.Future
        ), f"{self} attempting stiching on a non-Future"

        expiry = self.expiry_from_contract(self.contract)

        for wrapper in reversed(selector.all_contracts):
            if self.expiry_from_contract(wrapper.contract) <= expiry:
                yield wrapper.contract

    async def _acquire_data_for_contract(
        self, contract: ibi.Contract, start_date: datetime, stop_date: datetime
    ) -> pd.DataFrame:
        if (df := await self.store.read(contract, start_date, stop_date)) is None:
            # don't pull data for current contract from broker, this
            # is :class:`Streamer`'s responsibility; data for previous
            # contracts may be missing if it's a new database and only
            # then it's acceptable; pulling data from broker here is
            # unusual and should be investigated if it happens
            if contract != self.contract:
                log.warning(f"{self} requesting data from broker for: {contract}")
                bars = await self._pull_history_from_broker(
                    contract, start_date, stop_date
                )
                df = pd.DataFrame(bars).set_index("date")
            else:
                # :meth:`acquire_data` needs to deal with this case
                df = pd.DataFrame()
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
        by Sticher.  Converting data twice (eg.  str -> int -> str)
        would be wrong since every conversion entails a degree of
        rounding, which would make us get further away from the the
        duration requested by the user.
        """
        durationStr = self._streamer_params["durationStr"]
        return (
            durationStr_to_datapoints(
                durationStr,
                self._streamer_params["barSizeSetting"],
                self.session_length,
            )
            if isinstance(durationStr, str)
            else durationStr
        )

    @cached_property
    def required_timedelta(self) -> timedelta:
        """
        Return durationStr as timedelta, this is how far back we need data.
        durationStr can be given either directly as a str acceptable
        by :meth:`ib_insync.IB.reqHistoricalData` or number of
        required datapoints. Both cases are accounted for in this method.

        Returned timedelta is longer by 10% than strictly necessary
        to facilitate stiching of data for different contracts.
        """
        durationStr = self._streamer_params["durationStr"]
        return (
            durationStr_to_timedelta(durationStr)
            if isinstance(durationStr, str)
            else datapoints_to_timedelta(
                # adding 10% margin
                int(durationStr * 1.1),
                self._streamer_params["barSizeSetting"],
                self.session_length,
            )
        )

    @cached_property
    def session_length(self) -> timedelta:
        """Return length of a typical trading session as a timedelta."""
        return typical_session_length(self.contract_details.trading_hours)

    @staticmethod
    def expiry_from_contract(contract: ibi.Contract) -> datetime:
        return datetime.strptime(contract.lastTradeDateOrContractMonth, "%Y%m%d")

    def onContractChanged(
        self, old_contract: ibi.Contract, new_contract: ibi.Contract
    ) -> None:
        self._df = None
        super().onContractChanged(old_contract, new_contract)
