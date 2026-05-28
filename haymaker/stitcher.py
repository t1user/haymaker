import logging
import operator as op
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import cached_property
from typing import Literal
from zoneinfo import ZoneInfo

import ib_insync as ibi
import pandas as pd

from . import misc
from .contract_selector import FutureSelector, custom_bday

log = logging.getLogger(__name__)


class NoDataError(Exception):
    pass


class NonOverLappingDfsError(Exception):
    pass


class MissingContract(Exception):
    pass


class CorruptData(Exception):
    pass


@dataclass
class ContractSlice:
    """A single contract's dataframe trimmed to its active period plus buffer."""

    name: str
    df: pd.DataFrame
    roll_day: datetime


@dataclass
class RollRecord:
    """
    Captures all relevant information about a single roll event between
    two consecutive contracts.
    """

    roll_day: datetime
    sync: pd.Timestamp
    old_contract: str
    new_contract: str
    old_close: float
    new_close: float
    individual_offset: float


@dataclass
class FuturesStitcher:
    """
    Construct a continuous back-adjusted futures price series (Panama Canal method)
    from a dict of individual contract dataframes.

    Each contract is adjusted so that prices are continuous at roll points.
    The latest contract is never adjusted; all earlier contracts are shifted
    or scaled by the cumulative roll gap.

    All internal timestamp operations are performed in the native timezone of
    the source dataframes. Roll timestamps are first computed in the instrument's
    local timezone (tz_info) and then converted to the df's native timezone.
    This avoids mixed-timezone comparisons and unnecessary conversions. The
    timezone of each df is read directly from the df argument at the point of
    use, never from a cached source-level property, so synthetic dfs in tests
    behave correctly.

    Parameters
    ----------
    source : dict[ibi.Future, pd.DataFrame]
        Mapping of futures contracts to their OHLCV dataframes. Dataframes
        must have a DatetimeIndex and at minimum a 'close' column. All
        dataframes must be either all tz-aware or all tz-naive.
    adjust_type : 'add' | 'mul' | None
        Adjustment method. 'add' shifts prices by the absolute gap at roll.
        'mul' scales prices by the ratio at roll. None stitches without
        any price adjustment.
    selector : FutureSelector | None
        Optional pre-built selector. If None, one is built from source keys
        using roll_bdays and roll_margin_bdays.
    roll_bdays : int | None
        Business days before last trading day to roll. Passed to FutureSelector
        if selector is None.
    roll_margin_bdays : int | None
        Business days before roll when next contract becomes active. Passed
        to FutureSelector if selector is None.
    tz_info : ZoneInfo
        The instrument's local timezone. Used to interpret roll_hour — e.g.
        roll_hour=10 with tz_info=ZoneInfo("US/Eastern") means "roll at 10am
        Eastern", while tz_info=ZoneInfo("Europe/Berlin") means "roll at 10am
        Frankfurt time". Roll timestamps are computed in this timezone then
        converted to the df's native timezone for slicing. Defaults to
        US/Eastern.
    roll_hour : int
        Hour of day in tz_info timezone at which the roll sync point is
        sought. Defaults to 10.
    buffer_bdays : int
        Number of business days before each contract's start date to include
        in the slice, ensuring overlap between adjacent contracts for sync
        point finding. Defaults to 3.

    Public attributes
    -----------------
    data : pd.DataFrame
        The stitched, back-adjusted continuous price series, in the same
        timezone as the source dataframes.

    Public methods
    --------------
    inspect() -> pd.DataFrame | None
        Returns a DataFrame summarising each roll event: roll day, sync point,
        contract names, prices at sync, and individual offset. Returns None
        if no rolls were processed.
    """

    source: dict[ibi.Future, pd.DataFrame]
    adjust_type: Literal["add", "mul", None] = "add"
    selector: FutureSelector | None = None
    roll_bdays: int | None = None
    roll_margin_bdays: int | None = None
    tz_info: ZoneInfo = ZoneInfo("US/Eastern")
    roll_hour: int = 11
    buffer_bdays: int = 3
    _intraday: bool | None = None

    def __post_init__(self):
        assert self.adjust_type is None or self.adjust_type in [
            "add",
            "mul",
        ], f"Unknown adjust type operator: {self.adjust_type}"

    @cached_property
    def operator(self) -> Callable:
        """Binary operator corresponding to adjust_type (add, mul, or identity)."""
        ops: dict[str | None, Callable] = {
            "add": op.add,
            "mul": op.mul,
            None: lambda x, *args: x,
        }
        return ops[self.adjust_type]

    @cached_property
    def reverse_operator(self) -> Callable:
        """Inverse of operator, used to compute the offset from two prices."""
        ops: dict[str | None, Callable] = {
            "add": op.sub,
            "mul": op.truediv,
            None: lambda x, *args: 0,
        }
        return ops[self.adjust_type]

    @property
    def _buffer_offset(self):
        """CustomBusinessDay offset corresponding to buffer_bdays."""
        return self.buffer_bdays * custom_bday

    @cached_property
    def _selector(self) -> FutureSelector:
        """
        FutureSelector instance determining roll dates and contract order.
        Uses the provided selector if given, otherwise builds one from source keys.
        """
        if self.selector is not None:
            return self.selector
        params = {
            param: p
            for param in ("roll_bdays", "roll_margin_bdays")
            if (p := getattr(self, param)) is not None
        }
        return FutureSelector.from_contracts(list(self.source.keys()), **params)

    @staticmethod
    def _get_df_tz(df: pd.DataFrame) -> ZoneInfo | None:
        """
        Return the timezone of df's DatetimeIndex, or None if tz-naive.
        Read directly from the df argument — never from a cached source property —
        so synthetic test dfs are handled correctly.
        """
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            return df.index.tz  # type: ignore
        return None

    @cached_property
    def _contract_slices(self) -> list[ContractSlice]:
        """
        Load and trim each contract's dataframe to its active period.

        Each df is sliced from (start_date - buffer_bdays) onward. The boundary
        timestamp is computed in tz_info and pandas handles cross-timezone
        comparison against the df index. If a timezone mismatch is detected
        between consecutive dfs, the output resets and only the consistent tail
        is kept. Raises NoDataError if no usable data remains.
        """
        output: list[ContractSlice] = []
        tz_aware: bool | None = None

        for contract, (start_date, roll_day) in self._selector.date_ranges.items():
            try:
                df = self.source[contract]
            except KeyError:
                raise MissingContract(f"No df supplied for {contract.localSymbol}")
            if (aware := misc.is_timezone_aware(df)) is not tz_aware:
                output = []
                tz_aware = aware
            start_ts = self._localize_date(start_date, df)

            if not df.index.is_monotonic_increasing:
                raise CorruptData(
                    f"dataframe for {contract.localSymbol} is not monotonic increasing"
                )

            output.append(
                ContractSlice(
                    name=contract.localSymbol,
                    df=df.loc[start_ts - self._buffer_offset :],
                    roll_day=roll_day,
                )
            )
        if not output:
            raise NoDataError("No suitable data in provided source.")
        if output[-1].df.empty:
            # if there's not data in the last df, we can still use all
            # previous data
            del output[-1]
        return output

    @cached_property
    def _roll_records(self) -> list[RollRecord]:
        """
        Compute roll records for each consecutive contract pair.

        Iterates backward through contract pairs, finds the sync point
        for each roll, and records the prices and individual offset.
        Stops early with a warning if a non-overlapping pair is
        encountered.

        The reason for backward iteration is that if not all data can
        be included in the final series (because of gaps or some data
        being corrupted), we have a preference for the most recent
        data.

        Output data is reversed again so output data is in
        chronological order.  Sync points are in the df's native
        timezone.
        """

        records = []
        for cs0, cs1 in reversed(
            list(zip(self._contract_slices[:-1], self._contract_slices[1:]))
        ):
            try:
                sync = self._sync_point(cs0.df, cs1.df, cs0.roll_day)
            except NonOverLappingDfsError as e:
                w = f"{type(e).__name__}: {e}"
                warnings.warn(w)
                log.warning(w)
                break
            old_close = float(cs0.df.loc[sync].close)  # type: ignore
            new_close = float(cs1.df.loc[sync].close)  # type: ignore
            records.append(
                RollRecord(
                    roll_day=cs0.roll_day,
                    sync=sync,
                    old_contract=cs0.name,
                    new_contract=cs1.name,
                    old_close=old_close,
                    new_close=new_close,
                    individual_offset=self.offset(old_close, new_close),
                )
            )
        return list(reversed(records))

    @cached_property
    def _cumulative_offsets(self) -> list[float]:
        """
        Compute the cumulative adjustment to apply to each contract.

        Iterates roll records in reverse (newest to oldest), accumulating
        offsets so that each earlier contract gets the sum/product of all
        subsequent roll gaps. The latest contract gets identity (0 or 1),
        the second-to-last gets one roll's offset, and so on.
        """
        identity = 0 if self.adjust_type == "add" else 1
        result = []
        acc = identity
        for record in reversed(self._roll_records):
            acc = self.operator(record.individual_offset, acc)
            result.append(acc)
        return list(reversed(result))

    @cached_property
    def data(self) -> pd.DataFrame:
        """
        Build and return the continuous back-adjusted price series.

        Each contract is sliced to its active period (from the previous sync
        point to its own sync point) and adjusted by the cumulative offset.
        The latest contract is appended unadjusted from its sync point onward.
        All timestamps are in the df's native timezone — no conversion is
        performed here.

        Raises NoDataError if no roll records could be computed.
        """
        records = self._roll_records
        if not records:
            raise NoDataError("No compliant data to stitch.")

        slices = self._contract_slices[len(self._contract_slices) - len(records) - 1 :]

        segments = []

        for i, (cs, record, cum_offset) in enumerate(
            zip(slices[:-1], records, self._cumulative_offsets)
        ):
            start = records[i - 1].sync if i > 0 else cs.df.index[0]
            segments.append(self.adjust(cs.df.loc[start : record.sync], cum_offset))

        segments.append(slices[-1].df.loc[records[-1].sync :])

        df = pd.concat(segments)
        return df[~df.index.duplicated()]

    def offset(self, old_price: float, new_price: float) -> float:
        """Compute the individual offset between old and new contract prices at roll."""
        return self.reverse_operator(new_price, old_price)

    def adjust(self, df: pd.DataFrame, offset: float) -> pd.DataFrame:
        """
        Apply offset to all price columns in df, leaving volume-like
        columns (volume, WAP, barCount) unadjusted.
        """
        if self.adjust_type is None:
            return df
        non_adjustable = {"volume", "WAP", "barCount"}
        adjustable_columns = list(set(df.columns) - non_adjustable)
        non_adjustable_columns = list(set(df.columns) - set(adjustable_columns))
        new_df = self.operator(pd.DataFrame(df[adjustable_columns]), offset)
        new_df[non_adjustable_columns] = df[non_adjustable_columns]
        return new_df[df.columns]

    def _localize_date(self, date: datetime, df: pd.DataFrame) -> pd.Timestamp:
        """
        Convert a date into a Timestamp suitable for slicing the df.

        If df is tz-naive, returns a naive Timestamp. If df is tz-aware,
        returns a Timestamp in tz_info (the instrument's local timezone).
        Pandas handles cross-timezone comparison when this timestamp is used
        to slice a df with a different native timezone.

        If date is already tz-aware (future-proofing for when selectors gain
        timezone support), it is converted to tz_info rather than localized.
        """
        tz = self._get_df_tz(df)
        if tz is None:
            if isinstance(date, datetime) and date.tzinfo is not None:
                return pd.Timestamp(date.replace(tzinfo=None))
            return pd.Timestamp(date)
        if isinstance(date, datetime) and date.tzinfo is not None:
            return pd.Timestamp(date).tz_convert(self.tz_info)
        return pd.Timestamp(date, tz=self.tz_info)

    def is_intraday(self, df: pd.DataFrame) -> bool:
        """
        Establish wheather data is intraday only once, subsequent df
        are assumed to be the same.
        """
        if self._intraday is None:
            diffs = pd.Series(df.index.to_series().diff().dropna().values)
            self._intraday = not diffs.empty and pd.Timedelta(
                diffs.mode()[0]
            ) < timedelta(days=1)
        return self._intraday

    def _make_roll_timestamp(
        self, roll_day: datetime, df: pd.DataFrame
    ) -> pd.Timestamp:
        """
        Construct the target sync timestamp for a given roll day, in
        the df's native timezone.

        For intraday data, the roll hour is first applied in the
        instrument's local timezone (tz_info) — e.g. roll_hour=10 with
        tz_info=ZoneInfo("US/Eastern") means 10am Eastern, while
        tz_info=ZoneInfo("Europe/Berlin") means 10am Frankfurt time.
        The result is then converted to the df's native timezone so
        that all sync timestamps are in the same timezone as the df
        index, avoiding mixed-timezone slicing errors.

        The roll should happen on the previous business day, because
        data will be available only up to the roll date (exclusive).

        For daily data, no roll hour is applied to avoid forcing an
        intraday time that won't exist in the index.

        If roll_day is already tz-aware, it is converted to tz_info
        before applying roll_hour.
        """
        df_tz = self._get_df_tz(df)

        if self.is_intraday(df):
            if isinstance(roll_day, datetime) and roll_day.tzinfo is not None:
                ts_instrument = (
                    pd.Timestamp(roll_day)
                    .tz_convert(self.tz_info)
                    .replace(hour=self.roll_hour, minute=0, second=0)
                )
            else:
                ts_instrument = pd.Timestamp(roll_day, tz=self.tz_info).replace(
                    hour=self.roll_hour, minute=0, second=0
                )
            if df_tz is None:
                return ts_instrument.tz_localize(None) - custom_bday
            return ts_instrument.tz_convert(df_tz) - custom_bday
        else:
            return self._localize_date(roll_day, df)

    def _sync_point(
        self, df0: pd.DataFrame, df1: pd.DataFrame, roll_day: datetime
    ) -> pd.Timestamp:
        """
        Find the timestamp at which to sync two consecutive contracts.

        The roll timestamp is computed in the df's native timezone, ensuring
        consistent comparisons against df indices. Prefers the exact roll
        timestamp if present in both dataframes. Falls back to the last
        common timestamp before the roll if not.
        """
        roll_ts = self._make_roll_timestamp(roll_day, df0)
        if roll_ts in df0.index and roll_ts in df1.index:
            return roll_ts
        return self._find_common_index(df0, df1, roll_ts)

    @staticmethod
    def _find_common_index(
        old_df: pd.DataFrame, new_df: pd.DataFrame, sync_index: pd.Timestamp
    ) -> pd.Timestamp:
        """
        Find the last timestamp before sync_index that appears in both dataframes.

        Raises NonOverLappingDfsError if the dataframes share no common timestamps
        at or before sync_index.
        """
        inner = old_df.join(new_df, how="inner", rsuffix="_new")
        if inner.empty:
            raise NonOverLappingDfsError(
                f"Previous last point: "
                f"{old_df.index[-1] if not old_df.empty else 'empty'}, "
                f"next first point: {new_df.index[0] if not new_df.empty else 'empty'} "
                f"sync index: {sync_index}"
            )
        try:
            return inner.loc[:sync_index].index[-1]  # type: ignore
        except IndexError:
            return inner.index[-1]

    def inspect(self) -> pd.DataFrame | None:
        """
        Return a DataFrame summarising each roll event, sorted by sync point.

        Columns: roll_day, sync_point, old_contract, new_contract,
        old_close, new_close, offset, cum_offset (cummulative offset).

        Returns None if no roll records were computed.
        """
        if not self._roll_records:
            return None
        return (
            pd.DataFrame(
                [
                    {
                        "roll_day": r.roll_day,
                        "sync_point": r.sync,
                        "old_contract": r.old_contract,
                        "new_contract": r.new_contract,
                        "old_close": r.old_close,
                        "new_close": r.new_close,
                        "offset": r.individual_offset,
                        "cum_offset": co,
                    }
                    for r, co in zip(self._roll_records, self._cumulative_offsets)
                ]
            )
            .sort_values("sync_point")
            .reset_index(drop=True)
        )
